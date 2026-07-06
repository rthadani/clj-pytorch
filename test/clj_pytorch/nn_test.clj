(ns clj-pytorch.nn-test
  (:require [clojure.test :refer :all]
            [libpython-clj2.python :refer [py.-]]
            [clj-pytorch.nn        :as nn]
            [clj-pytorch.functional :as f]
            [clj-pytorch.tensor    :as tensor]))

(deftest layer-constructors
  (testing "linear layer has parameters"
    (let [layer (nn/linear 4 8)]
      (is (some? layer))
      (is (some? (nn/parameters layer)))))
  (testing "conv2d"
    (is (some? (nn/conv2d 1 4 3))))
  (testing "conv1d"
    (is (some? (nn/conv1d 1 4 3))))
  (testing "embedding"
    (is (some? (nn/embedding 10 16))))
  (testing "embedding with padding-idx"
    (is (some? (nn/embedding 10 16 :padding-idx 0))))
  (testing "layer-norm"
    (is (some? (nn/layer-norm [8]))))
  (testing "batch-norm1d"
    (is (some? (nn/batch-norm1d 8))))
  (testing "batch-norm2d"
    (is (some? (nn/batch-norm2d 8)))))

(deftest activation-modules
  (testing "relu module"   (is (some? (nn/relu))))
  (testing "gelu module"   (is (some? (nn/gelu))))
  (testing "sigmoid module" (is (some? (nn/sigmoid))))
  (testing "silu module"   (is (some? (nn/silu)))))

(deftest dropout-modules
  (testing "dropout"   (is (some? (nn/dropout 0.5))))
  (testing "dropout2d" (is (some? (nn/dropout2d 0.5)))))

(deftest container-modules
  (testing "sequential"
    (let [net (nn/sequential (nn/linear 4 8) (nn/relu))]
      (is (some? net))))
  (testing "module-list"
    (is (some? (nn/module-list [(nn/linear 4 8) (nn/linear 8 4)]))))
  (testing "module-list-seq returns a seq of layers"
    (let [parent (nn/linear 2 2)
          ml (nn/module-list [(nn/linear 4 8) (nn/relu)])]
      (nn/set-layer! parent :layers ml)
      (is (= 2 (count (nn/module-list-seq parent :layers))))))
  (testing "module-list registers parameters"
    (let [ml (nn/module-list [(nn/linear 4 8) (nn/linear 8 2)])]
      (is (some? (nn/parameters ml))))))

(deftest loss-functions
  (testing "cross-entropy-loss" (is (some? (nn/cross-entropy-loss))))
  (testing "mse-loss"           (is (some? (nn/mse-loss))))
  (testing "bce-with-logits"    (is (some? (nn/bce-with-logits)))))

(deftest call-and-forward
  (testing "call linear with input"
    (let [layer (nn/linear 4 8)
          x     (tensor/->tensor [[1.0 2.0 3.0 4.0]])]
      (is (= [1 8] (f/shape (nn/call layer x))))))
  (testing "call sequential"
    (let [net (nn/sequential (nn/linear 4 8) (nn/relu))
          x   (tensor/->tensor [[1.0 2.0 3.0 4.0]])]
      (is (= [1 8] (f/shape (nn/call net x)))))))

(deftest train-eval-mode
  (let [layer (nn/linear 4 8)]
    (testing "starts in training mode"
      (is (true? (boolean (nn/training? layer)))))
    (testing "eval! switches to eval mode"
      (nn/eval! layer)
      (is (false? (boolean (nn/training? layer)))))
    (testing "train! switches back"
      (nn/train! layer)
      (is (true? (boolean (nn/training? layer)))))))

(nn/defmodule TwoLayerNet [in-features hidden out-features]
  :layers {:fc1  (nn/linear in-features hidden)
           :relu (nn/relu)
           :fc2  (nn/linear hidden out-features)}
  :forward (fn [self x]
             (-> x
                 ((nn/get-layer self :fc1))
                 ((nn/get-layer self :relu))
                 ((nn/get-layer self :fc2)))))

(deftest custom-module
  (testing "defmodule creates an nn.Module"
    (let [net (TwoLayerNet 4 8 2)]
      (is (nn/module? net))))
  (testing "custom module forward pass produces correct output shape"
    (let [net (TwoLayerNet 4 8 2)
          x   (tensor/->tensor [[1.0 2.0 3.0 4.0]])]
      (is (= [1 2] (f/shape (net x))))))
  (testing "custom module has parameters"
    (let [net (TwoLayerNet 4 8 2)]
      (is (some? (nn/parameters net))))))

(deftest parameter-tests
  (testing "parameter wraps tensor as learnable parameter"
    (let [t   (tensor/->tensor [[1.0 2.0] [3.0 4.0]])
          p   (nn/parameter t)]
      (is (some? p))))
  (testing "parameter requires-grad is true by default"
    (let [p (nn/parameter (tensor/->tensor [1.0 2.0]))]
      (is (true? (boolean (py.- p requires_grad))))))
  (testing "register-parameter! makes parameter visible in module parameters"
    (let [mod (nn/linear 2 2)
          t   (tensor/->tensor [1.0 2.0])]
      (nn/register-parameter! mod :my-param t)
      (is (some? (nn/parameters mod))))))

(deftest state-dict-roundtrip
  (testing "state-dict and load-state-dict!"
    (let [layer1 (nn/linear 4 8)
          layer2 (nn/linear 4 8)
          sd     (nn/state-dict layer1)]
      (nn/load-state-dict! layer2 sd)
      (is (some? sd)))))
