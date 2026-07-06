(ns clj-pytorch.inference
  (:require [libpython-clj2.python  :refer [py. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.functional :as t]
            [clj-pytorch.nn :as nn]
            [clj-pytorch.tensor :as tensor]
            [clj-pytorch.context :as ctx]))

(require-python '[torch :as torch]
                '[builtins :as builtins])

;; atom-backed cache for models that return a new cache object each step
(defn kv-cache
  ([]     (atom nil))
  ([ctor] (if ctor (atom (nn/call ctor)) (atom nil))))

(defn cache-get  [cache]         @cache)
(defn cache-set! [cache new-val] (reset! cache new-val) cache)

;; array-backed per-layer cache — keys and values stored in Object arrays
;; indexed by layer so concat along the seq dim is cheap
(defn layer-kv-cache [_num-layers]
  {:key-cache (builtins/dict) :value-cache (builtins/dict)})

(defn- py-dict-get [d k]
  (when (py. d __contains__ k) (py/get-item d k)))

(defn kv-num-items [cache]
  (let [kc (:key-cache cache)
        t  (py-dict-get kc (builtins/int 0))]
    (if (nil? t) 0 (nth (t/shape t) 2))))

(defn kv-update! [cache key-states value-states layer-idx]
  (let [idx (builtins/int layer-idx)
        kc  (:key-cache cache)
        vc  (:value-cache cache)
        k   (py-dict-get kc idx)
        v   (py-dict-get vc idx)]
    (if (nil? k)
      (do (py. kc __setitem__ idx key-states)
          (py. vc __setitem__ idx value-states))
      (do (py. kc __setitem__ idx (t/cat (builtins/list [k key-states]) :dim -2))
          (py. vc __setitem__ idx (t/cat (builtins/list [v value-states]) :dim -2))))
    [(py/get-item kc idx) (py/get-item vc idx)]))

(defn kv-get [cache layer-idx]
  (let [idx (builtins/int layer-idx)
        kc  (:key-cache cache)
        vc  (:value-cache cache)]
    [(py/get-item kc idx) (py/get-item vc idx)]))

;; sampling

(defn greedy-sample [logits]
  (t/argmax logits -1 true))

(defn temperature-scale [logits temperature]
  (t/softmax (t/div logits (t/tensor temperature)) :dim -1))

(defn top-p-sample [probs p]
  (let [{:keys [values indices]} (t/sort-tensor probs :dim -1 :descending true)
        cumulative (t/cumsum values -1)
        mask (t/where
              (t/gt (t/sub cumulative values) p)
              (tensor/->tensor 0.0)
              values)
        _ (t/div! mask (t/sum mask -1 true))
        sampled (t/multinomial mask 1)]
    (t/gather indices -1 sampled)))

(defn sample-next-token
  [logits & {:keys [do-sample temperature top-p]
             :or {do-sample false temperature 1.0 top-p 0.9}}]
  (if do-sample
    (-> logits (temperature-scale temperature) (top-p-sample top-p))
    (greedy-sample logits)))

(defn extend-attention-mask
  [attention-mask]
  (t/cat (builtins/list [attention-mask (torch/ones [1 1] :device (t/device-of attention-mask))]) :dim -1))

;; tokenizer helpers
(defn tokenizer-encode
  [tokenizer text & {:keys [return-tensors padding truncation max-length]
                     :or {return-tensors "pt" padding true truncation true}}]
  (let [opts (cond-> {:return_tensors return-tensors :padding padding :truncation truncation}
               max-length (assoc :max_length max-length))
        py-text (if (string? text) (builtins/str text)
                    (builtins/list (mapv builtins/str text)))
        out  (py/call-attr-kw tokenizer "__call__" [py-text] opts)]
    {:input-ids      (py.- out input_ids)
     :attention-mask (py.- out attention_mask)}))

(defn tokenizer-decode
  [tokenizer tokens & {:keys [skip-special-tokens] :or {skip-special-tokens true}}]
  (py/->jvm (py. tokenizer decode tokens :skip_special_tokens skip-special-tokens)))

(defn eos-token-id [tokenizer]
  (py/->jvm (py.- tokenizer eos_token_id)))

;; generation loop
(defn generate
  [model-fn init-state
   & {:keys [max-new-tokens stop-token-id do-sample temperature top-p]
      :or {max-new-tokens 100 do-sample false temperature 1.0 top-p 0.9}}]
  (loop [state init-state
         generated-ids []
         remaining max-new-tokens]
    (if (zero? remaining)
      (assoc state :generated-ids generated-ids)
      (let [outputs    (model-fn state)
            kv-cache   (:kv-cache outputs)
            logits     (-> (:logits outputs) (t/select 1 -1))   ;; select last pos, no CPU index tensor
            next-token (ctx/inference-mode
                        (sample-next-token logits
                                           :do-sample do-sample
                                           :temperature temperature
                                           :top-p top-p))
            token-id   (t/item (t/squeeze next-token))
            new-ids    (conj generated-ids token-id)]
        (if (and stop-token-id (= token-id stop-token-id))
          (assoc state :generated-ids new-ids)
          (recur
           (cond-> state
             true     (assoc :input-ids      next-token           ;; already [batch 1], no unsqueeze
                             :attention-mask (extend-attention-mask (:attention-mask state)))
             kv-cache (assoc :kv-cache kv-cache))
           new-ids
           (dec remaining)))))))

