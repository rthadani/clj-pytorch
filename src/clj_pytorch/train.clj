(ns clj-pytorch.train
  "Training loop primitives.

  train-step  — single forward + backward + step
  eval-step   — single forward without grad
  train-epoch — one full pass over batches
  fit         — full training loop returning per-epoch metrics"
  (:require [clj-pytorch.nn      :as nn]
            [clj-pytorch.optimizer :as optim]
            [clj-pytorch.tensor    :as tensor]
            [clj-pytorch.context   :as ctx]))

;; Single step
(defn train-step
  "One forward + backward + optimizer step.
   Returns the loss as a Clojure number.

   (train-step model optimizer loss-fn x y)"
  [model optimizer loss-fn x y]
  (optim/zero-grad! optimizer)
  (let [pred (nn/call model x)
        loss (nn/call loss-fn pred y)]
    (optim/backward! loss)
    (optim/step! optimizer)
    (tensor/item loss)))

(defn eval-step
  "One forward pass without gradient tracking.
   Returns [pred loss-scalar] as Clojure values.

   (eval-step model loss-fn x y)"
  [model loss-fn x y]
  (ctx/no-grad
   (let [pred (nn/call model x)
         loss (nn/call loss-fn pred y)]
     {:pred  pred
      :loss  (tensor/item loss)})))

;; Epoch loop
(defn train-epoch
  "Run one training epoch over a seq of [x y] batches.
   Returns a vector of per-batch loss scalars."
  [model optimizer loss-fn batches]
  (nn/train! model)
  (mapv (fn [[x y]]
          (train-step model optimizer loss-fn x y))
        batches))

(defn eval-epoch
  "Run one eval epoch over a seq of [x y] batches.
   Returns a vector of per-batch loss scalars."
  [model loss-fn batches]
  (nn/eval! model)
  (mapv (fn [[x y]]
          (:loss (eval-step model loss-fn x y)))
        batches))

;; Full training loop
(defn mean [xs]
  (/ (reduce + xs) (count xs)))

(defn fit
  "Full training loop.

   Config map keys:
     :model         — nn.Module instance
     :optimizer     — TorchOptimizer (from clj-pytorch.optim)
     :loss-fn       — loss module
     :epochs        — number of epochs
     :train-batches — 0-arity fn returning seq of [x y] pairs each epoch
     :val-batches   — (optional) 0-arity fn returning seq of [x y] pairs
     :on-epoch      — (optional) callback (fn [epoch metrics]) for logging
     :scheduler     — (optional) lr scheduler — stepped after each epoch

   Returns a vector of per-epoch metric maps:
     {:epoch N :train-loss F :val-loss F}"
  [{:keys [model optimizer loss-fn epochs
           train-batches val-batches
           on-epoch scheduler]}]
  (vec
   (for [epoch (range epochs)]
     (let [train-losses (train-epoch model optimizer loss-fn (train-batches))
           val-losses   (when val-batches
                          (eval-epoch model loss-fn (val-batches)))
           metrics      (cond-> {:epoch      epoch
                                 :train-loss (mean train-losses)}
                          val-losses (assoc :val-loss (mean val-losses)))]
       (when scheduler
         (optim/step-scheduler! scheduler))
       (when on-epoch
         (on-epoch epoch metrics))
       metrics))))
