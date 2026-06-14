(ns clj-pytorch.tensor
  "Interop between PyTorch tensors and Clojure/JVM types.

  ->clj    — tensor to nested Clojure vector
  ->tensor — Clojure data to tensor
  ->numpy  — tensor to numpy array (stays in Python)"

  (:require [libpython-clj2.python  :refer [py. py.. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.functional   :as f]))

(require-python '[numpy :as np]
                '[torch :as torch])

(defn ->numpy
  "Detach tensor and return a numpy array."
  [t]
  (py.. t (detach) (cpu) (numpy)))

(defn ->clj
  "Convert a tensor to a nested Clojure vector via numpy.
   Uses py/as-jvm to bridge the numpy array without deep-copying Python objects."
  [t]
  (-> (->numpy t) py/as-jvm vec))

(defn item
  "Extract a scalar from a single-element tensor as a Clojure number."
  [t]
  (py/->jvm (py. t item)))

;; Clojure -> Tensor

(defn ->tensor
  "Convert Clojure data (nested vectors, numbers) to a torch tensor.
   Options: :dtype :device"
  [data & {:keys [dtype device]}]
  (cond-> (torch/tensor data)
    dtype  (py. to dtype)
    device (f/to-device device)))

(defn ->float [t] (py. t float))
(defn ->long  [t] (py. t long))
(defn ->int   [t] (py. t int))
(defn ->half  [t] (py. t half))

(defn detach
  "Detach a tensor from the computation graph."
  [t]
  (py. t detach))

(defn requires-grad!
  "Set requires_grad on a tensor."
  [t requires-grad]
  (py/set-attr! t "requires_grad" requires-grad)
  t)

;; Device helpers
(defn cuda-available? []
  (py. torch/cuda is_available))

(defn mps-available? []
  (py.. torch/backends mps (is_available)))

(defn device
  "Return a torch.device for the given string: \"cuda\", \"cpu\", \"mps\""
  [device-str]
  (torch/device device-str))

(defn best-device
  "Return \"cuda\" if available, else \"mps\" if available, else \"cpu\"."
  []
  (cond
    (cuda-available?)  "cuda"
    (mps-available?) "mps"
    :else "cpu"))

(defn move-to-device
  "Move every tensor value in a map to device-str.
   Non-tensor values are passed through unchanged.
   Equivalent to Python's {k: v.to(device) for k, v in inputs.items()}"
  [inputs device-str]
  (reduce-kv
   (fn [m k v]
     (assoc m k (try (f/to-device v device-str)
                     (catch Exception _ v))))
   {}
   inputs))

;; Checkpointing
(defn save!
  "Save a state-dict or any Python object to path."
  [obj path]
  (torch/save obj path))

(defn load!
  "Load a previously saved object from path."
  [path & {:keys [map-location] :or {map-location "cpu"}}]
  (torch/load path :map_location (device map-location)))
