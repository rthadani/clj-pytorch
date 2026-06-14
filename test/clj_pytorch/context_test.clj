(ns clj-pytorch.context-test
  (:require [clojure.test :refer :all]
            [clj-pytorch.context   :as ctx]
            [clj-pytorch.tensor    :as tensor]
            [clj-pytorch.functional :as f]))

(deftest no-grad-test
  (testing "body evaluates and returns its value"
    (let [result (ctx/no-grad (+ 1 2))]
      (is (= 3 result))))
  (testing "tensor ops work inside no-grad"
    (let [t      (tensor/->tensor [1.0 2.0 3.0])
          result (ctx/no-grad (f/sum t))]
      (is (= 6.0 (f/item result))))))

(deftest inference-mode-test
  (testing "body evaluates and returns its value"
    (let [result (ctx/inference-mode (* 6 7))]
      (is (= 42 result))))
  (testing "tensor ops work inside inference-mode"
    (let [t      (tensor/->tensor [1.0 2.0 3.0])
          result (ctx/inference-mode (f/mean t))]
      (is (= 2.0 (f/item result))))))

(deftest enable-grad-test
  (testing "enable-grad body evaluates"
    (let [result (ctx/no-grad
                  (ctx/enable-grad (+ 1 1)))]
      (is (= 2 result)))))

(deftest no-grad-exception-safety
  (testing "exceptions propagate out of no-grad"
    (is (thrown? Exception
                 (ctx/no-grad (throw (ex-info "test" {})))))))
