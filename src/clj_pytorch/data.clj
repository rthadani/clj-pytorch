(ns clj-pytorch.data
  "Dataset and DataLoader wrappers.

  dataloader   — wrap a Python dataset into a lazy Clojure seq of [x y] batches
  tensor-dataset — build an in-memory TensorDataset from Clojure vectors"
  (:require [libpython-clj2.python  :as py]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.tensor      :as tensor]))

(require-python '[torch.utils.data :as data])

(defn tensor-dataset
  "Build a TensorDataset from tensors (already on correct device/dtype).
   (tensor-dataset x-tensor y-tensor)"
  [& tensors]
  (apply data/TensorDataset tensors))

(defn dataloader
  "Wrap a dataset in a DataLoader.
   Returns a Python iterable — use ->batches to get a Clojure seq.

   Options:
     :batch-size  (default 32)
     :shuffle     (default false)
     :num-workers (default 0)
     :drop-last   (default false)"
  [dataset & {:keys [batch-size shuffle num-workers drop-last]
              :or   {batch-size 32 shuffle false num-workers 0 drop-last false}}]
  (data/DataLoader dataset
                   :batch_size  batch-size
                   :shuffle     shuffle
                   :num_workers num-workers
                   :drop_last   drop-last))

(defn ->batches
  "Convert a DataLoader into a lazy Clojure seq of [x y] vectors.
   Each element is a Python tensor — pass through your model directly."
  [loader]
  (->> (py/as-jvm loader)
       (map (fn [batch]
              ;; DataLoader returns a Python list [x, y]
              [(py/get-item batch 0)
               (py/get-item batch 1)]))))

(defn clj->dataloader
  "Build a DataLoader directly from Clojure vectors of data and labels.
   Converts to tensors then wraps in TensorDataset + DataLoader.

   (clj->dataloader xs ys :batch-size 64 :shuffle true)"
  [xs ys & opts]
  (let [x-t (tensor/->tensor xs)
        y-t (tensor/->tensor ys)
        ds  (tensor-dataset x-t y-t)]
    (apply dataloader ds opts)))
