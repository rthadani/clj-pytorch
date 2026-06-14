(ns clj-pytorch.optimizer
  "Optimizer constructors and a simple Optimizer protocol.

  Usage:
    (def opt (adam model :lr 1e-3))
    (zero-grad! opt)
    (backward! loss)
    (step! opt)"
  (:require [libpython-clj2.python :refer [py.]]
            [libpython-clj2.require :refer [require-python]]
            [clj-pytorch.nn :as nn]))

(require-python '[torch.optim :as optim]
                '[torch.optim.lr_scheduler :as lr-scheduler])

(defprotocol Optimizer
  (step!      [this] "Run one optimiser step.")
  (zero-grad! [this] "Zero all parameter gradients."))

(defrecord TorchOptimizer [opt]
  Optimizer
  (step!      [_] (py. opt step))
  (zero-grad! [_] (py. opt zero_grad)))

(defn backward!
  "Call loss.backward(). Optionally pass retain-graph=true."
  [loss & {:keys [retain-graph] :or {retain-graph false}}]
  (py. loss backward :retain_graph retain-graph))

;; Optimiser constructors
(defn adam
  "Adam optimiser.
   (adam model :lr 1e-3 :betas [0.9 0.999] :eps 1e-8 :wd 0)"
  [model & {:keys [lr betas eps wd]
            :or   {lr 1e-3 betas [0.9 0.999] eps 1e-8 wd 0}}]
  (->TorchOptimizer
   (optim/Adam (nn/parameters model)
               :lr lr :betas betas :eps eps :weight_decay wd)))

(defn adamw
  "AdamW optimiser (Adam with decoupled weight decay)."
  [model & {:keys [lr betas eps wd]
            :or   {lr 1e-3 betas [0.9 0.999] eps 1e-8 wd 1e-2}}]
  (->TorchOptimizer
   (optim/AdamW (nn/parameters model)
                :lr lr :betas betas :eps eps :weight_decay wd)))

(defn sgd
  "SGD optimiser."
  [model & {:keys [lr momentum wd nesterov]
            :or   {lr 0.01 momentum 0.9 wd 0 nesterov false}}]
  (->TorchOptimizer
   (optim/SGD (nn/parameters model)
              :lr lr :momentum momentum
              :weight_decay wd :nesterov nesterov)))

(defn rmsprop
  "RMSProp optimiser."
  [model & {:keys [lr alpha eps wd momentum]
            :or   {lr 1e-2 alpha 0.99 eps 1e-8 wd 0 momentum 0}}]
  (->TorchOptimizer
   (optim/RMSprop (nn/parameters model)
                  :lr lr :alpha alpha :eps eps
                  :weight_decay wd :momentum momentum)))

;; LR Schedulers - Call (step-scheduler! s) after each epoch
(defn step-lr
  "StepLR — decay lr by gamma every step-size epochs."
  [optimizer & {:keys [step-size gamma] :or {step-size 10 gamma 0.1}}]
  (lr-scheduler/StepLR (:opt optimizer) :step_size step-size :gamma gamma))

(defn cosine-annealing-lr
  "CosineAnnealingLR."
  [optimizer t-max & {:keys [eta-min] :or {eta-min 0}}]
  (lr-scheduler/CosineAnnealingLR (:opt optimizer) :T_max t-max :eta_min eta-min))

(defn step-scheduler!
  "Step a lr scheduler forward one epoch."
  [scheduler]
  (py. scheduler step))
