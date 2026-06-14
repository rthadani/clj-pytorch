(ns clj-pytorch.data-test
  (:require [clojure.test :refer :all]
            [clj-pytorch.data   :as data]
            [clj-pytorch.tensor :as tensor]
            [clj-pytorch.functional :as f]))

(deftest tensor-dataset-test
  (testing "tensor-dataset wraps tensors"
    (let [xs (tensor/->tensor [[1.0 2.0] [3.0 4.0]])
          ys (tensor/->tensor [0.0 1.0])
          ds (data/tensor-dataset xs ys)]
      (is (some? ds)))))

(deftest dataloader-test
  (testing "dataloader wraps a dataset"
    (let [xs (tensor/->tensor [[1.0 2.0] [3.0 4.0] [5.0 6.0] [7.0 8.0]])
          ys (tensor/->tensor [0.0 1.0 0.0 1.0])
          ds (data/tensor-dataset xs ys)
          dl (data/dataloader ds :batch-size 2)]
      (is (some? dl))))
  (testing "->batches returns pairs"
    (let [xs (tensor/->tensor [[1.0 2.0] [3.0 4.0]])
          ys (tensor/->tensor [0.0 1.0])
          ds (data/tensor-dataset xs ys)
          dl (data/dataloader ds :batch-size 1)
          batches (data/->batches dl)]
      (is (seq batches))
      (is (= 2 (count batches)))
      (doseq [[x y] batches]
        (is (some? x))
        (is (some? y))))))

(deftest clj->dataloader-test
  (testing "end-to-end Clojure data to DataLoader"
    (let [xs [[1.0 2.0] [3.0 4.0] [5.0 6.0] [7.0 8.0]]
          ys [0.0 1.0 0.0 1.0]
          dl (data/clj->dataloader xs ys :batch-size 2)
          batches (data/->batches dl)]
      (is (= 2 (count batches)))
      (doseq [[x y] batches]
        (is (= [2 2] (f/shape x)))
        (is (= [2]   (f/shape y)))))))
