(ns clj-pytorch.inference
  "Primitives for autoregressive token generation.

  Covers the patterns from a typical inference loop:
    - Device movement for maps of tensors
    - KV-cache as a Clojure atom wrapping a Python object
    - Greedy and nucleus (top-p) decoding strategies
    - Token generation loop
    - Tokenizer wrappers (encode / decode)
    - HuggingFace model + tokenizer loading"
  (:require [libpython-clj2.python  :refer [py. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.functional  :as t]
            [clj-pytorch.nn          :as nn]
            [clj-pytorch.tensor      :as tensor]
            [clj-pytorch.context     :as ctx]))

(require-python '[torch :as torch])

;; KV Cache
(defn kv-cache
  "Create a KV cache backed by a Python KVCache object.
   Pass your own KVCache class constructor, or nil to use a plain atom
   holding a Python object that gets replaced each step.

   Usage with a custom class:
     (def cache (kv-cache MyKVCache))

   Usage as a bare atom (when the model returns the updated cache):
     (def cache (kv-cache nil))"
  ([]      (atom nil))
  ([ctor]  (if ctor
             (atom (nn/call ctor))
             (atom nil))))

(defn cache-get
  "Dereference the cache atom."
  [cache]
  @cache)

(defn cache-set!
  "Replace the cache contents with a new value."
  [cache new-val]
  (reset! cache new-val)
  cache)

;; Sampling strategies
(defn greedy-sample
  "Pick the highest-probability token.
   logits shape: [batch, vocab]
   Returns token tensor of shape [batch, 1]."
  [logits]
  (t/argmax logits -1 true))

(defn temperature-scale
  "Divide logits by temperature then apply softmax.
   Returns a probability distribution over the vocabulary."
  [logits temperature]
  (t/softmax (t/div logits (t/tensor temperature)) :dim -1))

(defn top-p-sample
  "Nucleus sampling — sample from the smallest set of tokens whose
   cumulative probability exceeds p.

   probs  — probability tensor [batch, vocab] (after softmax)
   p      — nucleus threshold e.g. 0.9
   Returns token tensor of shape [batch, 1]."
  [probs p]
  (let [{:keys [values indices]} (t/sort-tensor probs :dim -1 :descending true)
        ;; cumulative sum of sorted probs
        cumulative               (t/cumsum values -1)
        ;; mask tokens whose cumulative prob exceeds p
        ;; shift by subtracting values so the token that pushes past p is kept
        mask                     (t/where
                                  (t/gt (t/sub cumulative values) p)
                                  (tensor/->tensor 0.0)
                                  values)
        ;; renormalise
        _                        (t/div! mask (t/sum mask -1 true))
        ;; sample one token index from the filtered distribution
        sampled-idx              (t/multinomial mask 1)
        ;; map back to vocabulary index
        next-token               (t/gather indices -1 sampled-idx)]
    next-token))

(defn sample-next-token
  "Given logits [batch, vocab], return next token [batch, 1].
   Options:
     :do-sample   — true for sampling, false for greedy (default false)
     :temperature — temperature for softmax scaling (default 1.0)
     :top-p       — nucleus threshold (default 0.9)"
  [logits & {:keys [do-sample temperature top-p]
             :or   {do-sample false temperature 1.0 top-p 0.9}}]
  (if do-sample
    (-> logits
        (temperature-scale temperature)
        (top-p-sample top-p))
    (greedy-sample logits)))

;; Attention mask

(defn extend-attention-mask
  "Append a column of ones to attention-mask for the newly generated token.
   mask shape: [batch, seq] -> [batch, seq+1]"
  [attention-mask]
  (let [dev   (t/device-of attention-mask)
        ones  (torch/ones [1 1] :device dev)]
    (t/cat [attention-mask ones] :dim -1)))

;; Tokenizer wrappers

(defn tokenizer-encode
  "Encode a string or seq of strings to input-ids tensor.
   Returns a map with at least :input-ids and :attention-mask."
  [tokenizer text & {:keys [return-tensors padding truncation max-length]
                     :or   {return-tensors "pt" padding true truncation true}}]
  (let [opts {:return_tensors return-tensors
              :padding        padding
              :truncation     truncation}
        opts (if max-length (assoc opts :max_length max-length) opts)
        out  (py/call-attr-kw tokenizer "__call__" [text] opts)]
    {:input-ids      (py.- out input_ids)
     :attention-mask (py.- out attention_mask)}))

(defn tokenizer-decode
  "Decode a token tensor or seq of token ids back to a string.
   skip-special-tokens defaults to true."
  [tokenizer tokens & {:keys [skip-special-tokens] :or {skip-special-tokens true}}]
  (py/->jvm
   (py. tokenizer decode tokens :skip_special_tokens skip-special-tokens)))

(defn eos-token-id
  "Return the end-of-sequence token id for a tokenizer."
  [tokenizer]
  (py/->jvm (py.- tokenizer eos_token_id)))

;; Model loading (HuggingFace)

(defn load-hf-model
  "Load a HuggingFace model and tokenizer from a local path or HF hub id.

   Returns {:model m :tokenizer t} with the model already in eval mode
   on the requested device.

   Options:
     :device       — \"cpu\", \"cuda\", \"mps\" (default: auto-detect)
     :model-class  — HF model class to use (default: AutoModelForCausalLM)
     :tokenizer-class — HF tokenizer class (default: AutoTokenizer)
     :dtype        — torch dtype e.g. torch/bfloat16 (default: auto)"
  [model-path & {:keys [device model-class tokenizer-class dtype]
                 :or   {device nil}}]
  (require-python '[transformers :as transformers])
  (let [xf           (py/import-module "transformers")
        dev          (or device (tensor/best-device))
        model-cls    (or model-class (py.- xf AutoModelForCausalLM))
        tok-cls      (or tokenizer-class (py.- xf AutoTokenizer))
        load-opts    (cond-> {:device_map dev}
                       dtype (assoc :torch_dtype dtype))
        model        (py/call-attr-kw model-cls "from_pretrained" [model-path] load-opts)
        tokenizer    (py. tok-cls from_pretrained model-path)]
    (nn/eval! model)
    {:model     model
     :tokenizer tokenizer
     :device    dev}))

;; Generation loop

(defn generate
  "Autoregressive token generation loop.

   model-fn    — (fn [state] outputs-map) where state is the current
                 generation state map and outputs-map must contain
                 :logits and optionally :kv-cache.

   init-state  — map with at least:
                   :input-ids      [batch seq]
                   :attention-mask [batch seq]
                 and any other keys your model-fn needs (e.g. :pixel-values).

   Options:
     :max-new-tokens    — max tokens to generate (default 100)
     :stop-token-id     — generation stops when this token is produced
     :do-sample         — true for sampling, false for greedy (default false)
     :temperature       — temperature (default 1.0)
     :top-p             — nucleus threshold (default 0.9)

   Returns the final state map with :generated-ids added
   (a vector of token id scalars)."
  [model-fn init-state
   & {:keys [max-new-tokens stop-token-id do-sample temperature top-p]
      :or   {max-new-tokens 100
             do-sample      false
             temperature    1.0
             top-p          0.9}}]
  (loop [state           init-state
         generated-ids   []
         steps-remaining max-new-tokens]
    (if (zero? steps-remaining)
      (assoc state :generated-ids generated-ids)
      (let [outputs    (model-fn state)
            kv-cache   (:kv-cache outputs)
            ;; logits: [batch, seq, vocab] — take last position
            logits     (t/index-select
                        (:logits outputs)
                        1
                        (t/tensor [(dec (first (rest (t/size (:logits outputs)))))]))
            logits     (t/squeeze logits 1)    ;; [batch, vocab]
            next-token (ctx/inference-mode
                        (sample-next-token logits
                                           :do-sample   do-sample
                                           :temperature temperature
                                           :top-p       top-p))
            token-id   (t/item (t/squeeze next-token))
            new-ids    (conj generated-ids token-id)]
        (if (and stop-token-id (= token-id stop-token-id))
          (assoc state :generated-ids new-ids)
          (recur
           (cond-> state
             true       (assoc :input-ids      (t/unsqueeze next-token -1)
                               :attention-mask (extend-attention-mask
                                                (:attention-mask state)))
             kv-cache   (assoc :kv-cache kv-cache))
           new-ids
           (dec steps-remaining)))))))
