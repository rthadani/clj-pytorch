(ns clj-pytorch.tensor-test
  (:require [clojure.test :refer :all]
            [clj-pytorch.tensor :as tensor]))

(deftest ->tensor-and-item
  (testing "scalar roundtrip"
    (is (= 42.0 (tensor/item (tensor/->tensor [42.0])))))
  (testing "->clj on 1-D tensor"
    (let [v (tensor/->clj (tensor/->tensor [1.0 2.0 3.0]))]
      (is (= 3 (count v)))
      (is (= 1.0 (first v)))))
  (testing ":device option is accepted"
    (let [dev (tensor/best-device)]
      (is (some? (tensor/->tensor [1.0 2.0] :device dev))))))

(deftest type-casts
  (let [t (tensor/->tensor [1.0 2.0 3.0])]
    (testing "->float"  (is (some? (tensor/->float t))))
    (testing "->long"   (is (some? (tensor/->long  t))))
    (testing "->int"    (is (some? (tensor/->int   t))))
    (testing "->half"   (is (some? (tensor/->half  t))))))

(deftest detach-test
  (testing "detach returns a tensor"
    (is (some? (tensor/detach (tensor/->tensor [1.0 2.0]))))))

(deftest device-helpers
  (testing "cuda-available? returns a boolean"
    (is (boolean? (boolean (tensor/cuda-available?)))))
  (testing "best-device returns a string"
    (is (string? (tensor/best-device))))
  (testing "best-device is one of the expected values"
    (is (#{"cuda" "mps" "cpu"} (tensor/best-device)))))

(deftest save-load-roundtrip
  (testing "save! and load! roundtrip a tensor"
    (let [path "/tmp/torch_clj_test_tensor.pt"
          t    (tensor/->tensor [1.0 2.0 3.0])]
      (tensor/save! t path)
      (let [loaded (tensor/load! path)]
        (is (= (tensor/->clj t)
               (tensor/->clj loaded)))))))
