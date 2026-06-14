(ns clj-pytorch.train-test
  (:require [clojure.test :refer :all]
            [clj-pytorch.nn        :as nn]
            [clj-pytorch.optimizer :as optim]
            [clj-pytorch.train     :as train]
            [clj-pytorch.tensor    :as tensor]))

(defn- make-fixtures []
  (let [model   (nn/linear 2 1)
        loss-fn (nn/mse-loss)
        opt     (optim/adam model :lr 1e-2)]
    {:model model :loss-fn loss-fn :opt opt}))

(defn- batch [] [(tensor/->tensor [[1.0 2.0]]) (tensor/->tensor [[1.0]])])

(deftest train-step-test
  (testing "train-step returns a number"
    (let [{:keys [model loss-fn opt]} (make-fixtures)
          [x y] (batch)
          loss  (train/train-step model opt loss-fn x y)]
      (is (number? loss))
      (is (not (Double/isNaN loss))))))

(deftest eval-step-test
  (testing "eval-step returns a map with :pred and :loss"
    (let [{:keys [model loss-fn]} (make-fixtures)
          [x y] (batch)
          result (train/eval-step model loss-fn x y)]
      (is (map? result))
      (is (contains? result :pred))
      (is (contains? result :loss))
      (is (number? (:loss result))))))

(deftest train-epoch-test
  (testing "train-epoch returns a vector of loss values"
    (let [{:keys [model loss-fn opt]} (make-fixtures)
          batches (repeat 3 (batch))
          losses  (train/train-epoch model opt loss-fn batches)]
      (is (vector? losses))
      (is (= 3 (count losses)))
      (is (every? number? losses)))))

(deftest eval-epoch-test
  (testing "eval-epoch returns a vector of loss values"
    (let [{:keys [model loss-fn]} (make-fixtures)
          batches (repeat 3 (batch))
          losses  (train/eval-epoch model loss-fn batches)]
      (is (vector? losses))
      (is (= 3 (count losses)))
      (is (every? number? losses)))))

(deftest fit-test
  (testing "fit runs epochs and returns metrics"
    (let [{:keys [model loss-fn opt]} (make-fixtures)
          batches (repeat 4 (batch))
          metrics (train/fit {:model         model
                              :optimizer     opt
                              :loss-fn       loss-fn
                              :epochs        3
                              :train-batches (constantly batches)})]
      (is (= 3 (count metrics)))
      (doseq [m metrics]
        (is (contains? m :epoch))
        (is (contains? m :train-loss))
        (is (number? (:train-loss m))))))
  (testing "fit with validation batches includes :val-loss"
    (let [{:keys [model loss-fn opt]} (make-fixtures)
          batches (repeat 2 (batch))
          metrics (train/fit {:model         model
                              :optimizer     opt
                              :loss-fn       loss-fn
                              :epochs        2
                              :train-batches (constantly batches)
                              :val-batches   (constantly batches)})]
      (is (every? #(contains? % :val-loss) metrics)))))
