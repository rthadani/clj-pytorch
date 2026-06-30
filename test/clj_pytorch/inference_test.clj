(ns clj-pytorch.inference-test
  (:require [clojure.test :refer :all]
            [clj-pytorch.inference  :as inf]
            [clj-pytorch.functional :as f]
            [clj-pytorch.tensor     :as tensor]))

;; KV tensors shaped [batch=1, heads=1, seq=1, head-dim=4]
(defn- kv-tensor [] (tensor/->tensor [[[[1.0 2.0 3.0 4.0]]]]))

(deftest layer-kv-cache-test
  (testing "creates cache with correct number of slots"
    (let [c (inf/layer-kv-cache 3)]
      (is (= 3 (alength (:key-cache c))))
      (is (= 3 (alength (:value-cache c))))))
  (testing "slots start as nil"
    (let [c (inf/layer-kv-cache 2)]
      (is (nil? (aget (:key-cache c) 0)))
      (is (nil? (aget (:value-cache c) 0))))))

(deftest kv-num-items-test
  (testing "returns 0 when cache is empty"
    (is (= 0 (inf/kv-num-items (inf/layer-kv-cache 2))))))

(deftest kv-update-test
  (testing "first update stores tensors at the given layer"
    (let [c   (inf/layer-kv-cache 2)
          k   (kv-tensor)
          v   (kv-tensor)
          [rk rv] (inf/kv-update! c k v 0)]
      (is (= [1 1 1 4] (f/shape rk)))
      (is (= [1 1 1 4] (f/shape rv)))))
  (testing "kv-num-items reflects stored sequence length"
    (let [c (inf/layer-kv-cache 1)]
      (inf/kv-update! c (kv-tensor) (kv-tensor) 0)
      (is (= 1 (inf/kv-num-items c)))))
  (testing "second update concatenates along seq dim"
    (let [c (inf/layer-kv-cache 1)]
      (inf/kv-update! c (kv-tensor) (kv-tensor) 0)
      (let [[rk rv] (inf/kv-update! c (kv-tensor) (kv-tensor) 0)]
        (is (= [1 1 2 4] (f/shape rk)))
        (is (= [1 1 2 4] (f/shape rv)))))))

(deftest kv-get-test
  (testing "returns stored tensors after update"
    (let [c (inf/layer-kv-cache 2)
          k (kv-tensor)
          v (kv-tensor)]
      (inf/kv-update! c k v 1)
      (let [[rk rv] (inf/kv-get c 1)]
        (is (= [1 1 1 4] (f/shape rk)))
        (is (= [1 1 1 4] (f/shape rv)))))))
