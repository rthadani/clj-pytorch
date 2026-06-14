(ns clj-pytorch.optimizer-test
  (:require [clojure.test :refer :all]
            [clj-pytorch.nn        :as nn]
            [clj-pytorch.optimizer :as optim]
            [clj-pytorch.tensor    :as tensor]))

(defn- simple-model [] (nn/linear 4 1))

(deftest optimizer-constructors
  (let [model (simple-model)]
    (testing "adam"   (is (some? (optim/adam   model :lr 1e-3))))
    (testing "adamw"  (is (some? (optim/adamw  model :lr 1e-3))))
    (testing "sgd"    (is (some? (optim/sgd    model :lr 0.01))))
    (testing "rmsprop" (is (some? (optim/rmsprop model :lr 1e-2))))))

(deftest zero-grad-and-step
  (testing "zero-grad! and step! run without error"
    (let [model (simple-model)
          opt   (optim/adam model :lr 1e-3)
          x     (tensor/->tensor [[1.0 2.0 3.0 4.0]])
          loss  (nn/call (nn/mse-loss)
                         (nn/call model x)
                         (tensor/->tensor [[0.0]]))]
      (optim/zero-grad! opt)
      (optim/backward! loss)
      (optim/step! opt)
      (is true))))

(deftest backward!-test
  (testing "backward! runs on a scalar loss"
    (let [model (simple-model)
          opt   (optim/adam model :lr 1e-3)
          x     (tensor/->tensor [[1.0 2.0 3.0 4.0]])
          loss  (nn/call (nn/mse-loss)
                         (nn/call model x)
                         (tensor/->tensor [[0.0]]))]
      (optim/zero-grad! opt)
      (optim/backward! loss)
      (is true))))

(deftest lr-schedulers
  (let [model (simple-model)
        opt   (optim/adam model :lr 1e-3)]
    (testing "step-lr"
      (let [sched (optim/step-lr opt :step-size 5 :gamma 0.1)]
        (optim/step-scheduler! sched)
        (is (some? sched))))
    (testing "cosine-annealing-lr"
      (let [sched (optim/cosine-annealing-lr opt 10)]
        (optim/step-scheduler! sched)
        (is (some? sched))))))
