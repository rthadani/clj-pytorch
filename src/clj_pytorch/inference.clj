(ns clj-pytorch.inference
  (:require [libpython-clj2.python  :refer [py. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.functional :as t]
            [clj-pytorch.nn :as nn]
            [clj-pytorch.tensor :as tensor]
            [clj-pytorch.context :as ctx]))

(require-python '[torch :as torch])

;; atom-backed cache for models that return a new cache object each step
(defn kv-cache
  ([]     (atom nil))
  ([ctor] (if ctor (atom (nn/call ctor)) (atom nil))))

(defn cache-get  [cache]         @cache)
(defn cache-set! [cache new-val] (reset! cache new-val) cache)

;; array-backed per-layer cache — keys and values stored in Object arrays
;; indexed by layer so concat along the seq dim is cheap
(defn layer-kv-cache [num-layers]
  {:key-cache (make-array Object num-layers)
   :value-cache (make-array Object num-layers)})

(defn kv-num-items [cache]
  (let [kc (:key-cache cache)]
    (if (nil? (aget kc 0))
      0
      (nth (t/shape (aget kc 0)) 2))))

(defn kv-update! [cache key-states value-states layer-idx]
  (let [kc (:key-cache cache)
        vc (:value-cache cache)
        k (aget kc layer-idx)
        v (aget vc layer-idx)]
    (if (nil? k)
      (do (aset kc layer-idx key-states)
          (aset vc layer-idx value-states))
      (do (aset kc layer-idx (t/cat [k key-states] :dim -2))
          (aset vc layer-idx (t/cat [v value-states] :dim -2))))
    [(aget kc layer-idx)
     (aget vc layer-idx)]))

(defn kv-get [cache layer-idx]
  [(aget (:key-cache cache) layer-idx)
   (aget (:value-cache cache) layer-idx)])

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

(defn extend-attention-mask [attention-mask]
  (t/cat [attention-mask (torch/ones [1 1] :device (t/device-of attention-mask))] :dim -1))

;; tokenizer helpers

(defn tokenizer-encode
  [tokenizer text & {:keys [return-tensors padding truncation max-length]
                     :or {return-tensors "pt" padding true truncation true}}]
  (let [opts (cond-> {:return_tensors return-tensors :padding padding :truncation truncation}
               max-length (assoc :max_length max-length))
        out (py/call-attr-kw tokenizer "__call__" [text] opts)]
    {:input-ids (py.- out input_ids)
     :attention-mask (py.- out attention_mask)}))

(defn tokenizer-decode
  [tokenizer tokens & {:keys [skip-special-tokens] :or {skip-special-tokens true}}]
  (py/->jvm (py. tokenizer decode tokens :skip_special_tokens skip-special-tokens)))

(defn eos-token-id [tokenizer]
  (py/->jvm (py.- tokenizer eos_token_id)))

;; model loading

(defn load-hf-model
  [model-path & {:keys [device model-class tokenizer-class dtype]}]
  (require-python '[transformers :as transformers])
  (let [xf (py/import-module "transformers")
        dev (or device (tensor/best-device))
        model-cls (or model-class (py.- xf AutoModelForCausalLM))
        tok-cls (or tokenizer-class (py.- xf AutoTokenizer))
        load-opts (cond-> {:device_map dev} dtype (assoc :torch_dtype dtype))
        model (py/call-attr-kw model-cls "from_pretrained" [model-path] load-opts)
        tokenizer (py. tok-cls from_pretrained model-path)]
    (nn/eval! model)
    {:model model :tokenizer tokenizer :device dev}))

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
      (let [outputs (model-fn state)
            kv-cache (:kv-cache outputs)
            logits (-> (:logits outputs)
                       (t/index-select 1 (t/tensor [(dec (first (rest (t/size (:logits outputs)))))]))
                       (t/squeeze 1))
            next-token (ctx/inference-mode
                        (sample-next-token logits
                                           :do-sample do-sample
                                           :temperature temperature
                                           :top-p top-p))
            token-id (t/item (t/squeeze next-token))
            new-ids (conj generated-ids token-id)]
        (if (and stop-token-id (= token-id stop-token-id))
          (assoc state :generated-ids new-ids)
          (recur
           (cond-> state
             true (assoc :input-ids (t/unsqueeze next-token -1)
                         :attention-mask (extend-attention-mask (:attention-mask state)))
             kv-cache (assoc :kv-cache kv-cache))
           new-ids
           (dec remaining)))))))
