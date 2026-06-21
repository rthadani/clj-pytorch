(ns clj-pytorch.nn
  "Module definition DSL and layer constructors.

  Primary entry points:
    defmodule  — define an nn.Module subclass in idiomatic Clojure
    call       — invoke a module (goes through __call__, fires hooks)
    get-layer  — retrieve a registered submodule from self by keyword
    module?    — check if an object is an nn.Module instance

  Layer constructors:
    linear, conv2d, conv1d, embedding, layer-norm, batch-norm,
    dropout, relu, gelu, sigmoid, tanh, sequential, module-list

  Container helpers:
    module-list-seq     — iterate a registered ModuleList as a Clojure seq
    expand-sequential-output — grow a Sequential's final Linear layer

  Loss functions:
    cross-entropy-loss, mse-loss, bce-loss, bce-with-logits, focal-loss"

  (:require [libpython-clj2.python  :refer [py. py.- py..] :as py]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.tensor    :as t]
            [clj-pytorch.functional :as f]
            [clj-pytorch.context   :as ctx]))

(require-python '[torch.nn :as nn]
                '[torch.nn.functional :as F]
                '[builtins :as builtins])

;; require-python :as creates a namespace alias, not a Clojure var, so `nn`
;; can't be resolved in syntax-quote. Cache torch.nn.Module as a real var.
(def -nn-Module nn/Module)

;; Internal helpers
(defn kw->str [k]
  (-> (name k) (clojure.string/replace "-" "_")))
;; Module wrapper — makes any nn.Module callable as a Clojure fn
(deftype Module [py-module]
  clojure.lang.IFn
  (invoke [_]                        (py/call-attr py-module "__call__"))
  (invoke [_ a]                      (py/call-attr py-module "__call__" a))
  (invoke [_ a b]                    (py/call-attr py-module "__call__" a b))
  (invoke [_ a b c]                  (py/call-attr py-module "__call__" a b c))
  (invoke [_ a b c d]                (py/call-attr py-module "__call__" a b c d))
  (invoke [_ a b c d e]              (py/call-attr py-module "__call__" a b c d e))
  (invoke [_ a b c d e f]            (py/call-attr py-module "__call__" a b c d e f))
  (invoke [_ a b c d e f g]          (py/call-attr py-module "__call__" a b c d e f g))
  (invoke [_ a b c d e f g h]        (py/call-attr py-module "__call__" a b c d e f g h))
  (invoke [_ a b c d e f g h i]      (py/call-attr py-module "__call__" a b c d e f g h i))
  (invoke [_ a b c d e f g h i j]    (py/call-attr py-module "__call__" a b c d e f g h i j))
  (invoke [_ a b c d e f g h i j k]  (py/call-attr py-module "__call__" a b c d e f g h i j k))
  (invoke [_ a b c d e f g h i j k l]
    (py/call-attr py-module "__call__" a b c d e f g h i j k l))
  (invoke [_ a b c d e f g h i j k l m]
    (py/call-attr py-module "__call__" a b c d e f g h i j k l m))
  (invoke [_ a b c d e f g h i j k l m n]
    (py/call-attr py-module "__call__" a b c d e f g h i j k l m n))
  (invoke [_ a b c d e f g h i j k l m n o]
    (py/call-attr py-module "__call__" a b c d e f g h i j k l m n o))
  (invoke [_ a b c d e f g h i j k l m n o p]
    (py/call-attr py-module "__call__" a b c d e f g h i j k l m n o p))
  (applyTo [_ args]                  (apply py/call-attr py-module "__call__" args)))

(defn- ->py
  "Unwrap a Module to its underlying Python object, or pass through."
  [m]
  (if (instance? Module m) (.py-module m) m))

(defn module?
  "Returns true if obj is a PyTorch nn.Module instance."
  [obj]
  (try
    (when (and obj (not (map? obj)))
      (builtins/isinstance (->py obj) nn/Module))
    (catch Exception _ false)))

;; Module call / attribute access
(defn call
  "Call a module or any Python callable — equivalent to module(args) in Python.
   Always goes through __call__ so forward hooks fire."
  [module & args]
  (apply py/call-attr (->py module) "__call__" args))

(defn get-layer
  "Get a registered submodule / attribute from self by keyword.
   :my-layer -> self.my_layer"
  [self kw]
  (->Module (py/get-attr (->py self) (kw->str kw))))

(defn set-layer!
  "Register a submodule on self by keyword.
   :my-layer -> self.my_layer = layer"
  [self kw layer]
  (py/set-attr! (->py self) (kw->str kw) layer))

(defn register-buffer!
  "Register a non-parameter tensor buffer on self."
  [self attr-name tensor & {:keys [persistent] :or {persistent true}}]
  (py. (->py self) register_buffer (kw->str attr-name) tensor :persistent persistent))

(defn parameters
  "Returns an iterator over module parameters — pass to an optimizer."
  [module]
  (py. (->py module) parameters))

(defn train!
  "Set module to training mode."
  [module]
  (py. (->py module) train)
  module)

(defn eval!
  "Set module to eval mode."
  [module]
  (py. (->py module) eval)
  module)

(defn state-dict      [module]       (py. (->py module) state_dict))
(defn load-state-dict! [module sd]   (py. (->py module) load_state_dict sd))

(defn training?
  "Returns true if module is in training mode."
  [module]
  (py.- (->py module) training))

;; defmodule macro
;; Usage:
;;
;;   (defmodule MyNet [in-features hidden out-features]
;;     :layers {:fc1  (linear in-features hidden)
;;              :relu (relu)
;;              :fc2  (linear hidden out-features)}
;;     :init   (fn [self]
;;               (register-buffer! self :mask (t/zeros [hidden])))
;;     :forward (fn [self x]
;;                (-> x
;;                    (->> (call (get-layer self :fc1)))
;;                    (->> (call (get-layer self :relu)))
;;                    (->> (call (get-layer self :fc2))))))
;;
;; Defines a constructor fn `MyNet` that takes [in-features hidden out-features]
;; and returns a real nn.Module instance.

(defmacro defmodule
  [sym params & {:keys [layers init forward]}]
  {:pre [(symbol? sym)
         (vector? params)
         (some? forward)]}
  `(defn ~sym ~params
     (let [layer-map#  ~layers
           extra-init# ~init
           cls#
           (py/create-class
            ~(name sym)
            [clj-pytorch.nn/-nn-Module]
            {"__init__"
             (py/make-instance-fn
              (fn [self#]
                (py/call-attr clj-pytorch.nn/-nn-Module "__init__" self#)
                (doseq [[k# v#] layer-map#]
                  (py/set-attr! self# (kw->str k#) v#))
                (when extra-init#
                  (extra-init# self#))
                nil))
             "forward"
             (py/make-instance-fn ~forward)})]
       (->Module (cls#)))))

;; Layer constructors
(defn linear
  "nn.Linear(in-features, out-features, bias=true)"
  [in out & {:keys [bias] :or {bias true}}]
  (nn/Linear in out :bias bias))

(defn conv2d
  "nn.Conv2d"
  [in-ch out-ch kernel & {:keys [stride padding dilation groups bias]
                           :or   {stride 1 padding 0 dilation 1 groups 1 bias true}}]
  (nn/Conv2d in-ch out-ch kernel
             :stride stride :padding padding
             :dilation dilation :groups groups :bias bias))

(defn conv1d
  "nn.Conv1d"
  [in-ch out-ch kernel & {:keys [stride padding] :or {stride 1 padding 0}}]
  (nn/Conv1d in-ch out-ch kernel :stride stride :padding padding))

;; Embeddings / norms
(defn embedding
  "nn.Embedding(num-embeddings, embedding-dim)"
  [num-embeddings embed-dim]
  (nn/Embedding num-embeddings embed-dim))

(defn layer-norm
  "nn.LayerNorm(normalized-shape, eps=1e-6)"
  [normalized-shape & {:keys [eps] :or {eps 1e-6}}]
  (nn/LayerNorm normalized-shape :eps eps))

(defn batch-norm1d
  "nn.BatchNorm1d"
  [num-features & {:keys [eps momentum] :or {eps 1e-5 momentum 0.1}}]
  (nn/BatchNorm1d num-features :eps eps :momentum momentum))

(defn batch-norm2d
  "nn.BatchNorm2d"
  [num-features & {:keys [eps momentum] :or {eps 1e-5 momentum 0.1}}]
  (nn/BatchNorm2d num-features :eps eps :momentum momentum))

;; Activations (as modules)
(defn relu    [] (nn/ReLU))
(defn gelu    [] (nn/GELU))
(defn sigmoid [] (nn/Sigmoid))
(defn tanh-m  [] (nn/Tanh))
(defn silu    [] (nn/SiLU))

;; Dropout
(defn dropout
  "nn.Dropout(p)"
  [p] (nn/Dropout p))

(defn dropout2d [p] (nn/Dropout2d p))

;; Containers
(defn sequential
  "nn.Sequential(*layers)"
  [& layers]
  (apply nn/Sequential layers))

(defn module-list
  "nn.ModuleList(modules) — preserves parameter registration for all submodules."
  [modules]
  (nn/ModuleList modules))

(defn module-list-seq
  "Convert a registered ModuleList attribute into a lazy Clojure seq.
   Uses py/as-jvm (shallow bridge) so the modules remain live Python
   objects you can call — not deep-converted JVM values.

   Example:
     (reduce (fn [h layer] (call layer h))
             inputs
             (module-list-seq self :layers))"
  [self kw]
  (seq (py/as-jvm (get-layer self kw))))

(defn module-dict
  "nn.ModuleDict(mapping)"
  [m]
  (nn/ModuleDict m))

;; Loss functions
(defn cross-entropy-loss [] (nn/CrossEntropyLoss))
(defn mse-loss           [] (nn/MSELoss))
(defn bce-loss           [] (nn/BCELoss))
(defn bce-with-logits    [] (nn/BCEWithLogitsLoss))

(defmodule FocalLoss [gamma ignore-index]
  :layers {}
  :forward (fn [self logits targets]
             (let [ce-loss      (F/cross_entropy logits targets :reduction "none" :ignore_index ignore-index)
                   probs        (F/softmax logits :dim 1)
                   safe-targets (py. targets masked_fill (py. targets eq ignore-index) 0)
                   targets-exp  (py. safe-targets unsqueeze 1)
                   p-t          (py.. probs (gather 1 targets-exp) (squeeze 1))
                   focal-weight (py.. (py. p-t neg) (add 1) (pow gamma))
                   weighted     (py. ce-loss mul focal-weight)
                   mask         (py. targets ne ignore-index)
                   num-valid    (py.. mask sum (clamp :min 1))]
               (py.. weighted sum (div num-valid)))))

(defn focal-loss
  "Create a FocalLoss module."
  [& {:keys [gamma ignore-index] :or {gamma 2.0 ignore-index -100}}]
  (FocalLoss gamma ignore-index))

;; Utilities
(defn expand-sequential-output
  "Expand the output size of a Sequential's final Linear layer, copying
   existing weights into the new layer. New class neurons initialise randomly."
  [seq-module new-out-features]
  (let [device   (t/best-device)
        modules  (vec (py/as-jvm (py. (->py seq-module) children) :iterable? true))
        last-mod (last modules)]
    (if-not (builtins/isinstance last-mod nn/Linear)
      (throw (ex-info "Last layer of Sequential is not Linear"
                      {:last-layer-type (str (type last-mod))}))
      (let [in-features (py.- last-mod in_features)
            old-out     (py.- last-mod out_features)
            new-linear  (f/to-device (nn/Linear in-features new-out-features) device)
            new-modules (conj (vec (butlast modules)) new-linear)
            _           (ctx/no-grad
                          (let [old-w (py.- last-mod weight)
                                old-b (py.- last-mod bias)
                                new-w (py.- new-linear weight)
                                new-b (py.- new-linear bias)]
                            (py. (py/get-item new-w (builtins/slice 0 old-out)) copy_ old-w)
                            (py. (py/get-item new-b (builtins/slice 0 old-out)) copy_ old-b)))]
        (f/to-device (apply sequential new-modules) device)))))
