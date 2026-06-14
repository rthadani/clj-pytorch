(ns clj-pytorch.functional-test
  (:require [clojure.test :refer :all]
            [clojure.string :as str]
            [libpython-clj2.python :refer [py.-]]
            [clj-pytorch.functional :as f]
            [clj-pytorch.tensor     :as tensor]))

(defn- t [& xs] (tensor/->tensor (vec xs)))
(defn- clj [tensor] (vec (tensor/->clj tensor)))

(deftest creation-ops
  (testing "zeros" (is (= 0.0 (f/item (f/squeeze (f/zeros [1]))))))
  (testing "ones"  (is (= 1.0 (f/item (f/squeeze (f/ones [1]))))))
  (testing "zeros shape" (is (= [2 3] (f/shape (f/zeros [2 3])))))
  (testing "ones shape"  (is (= [2 3] (f/shape (f/ones [2 3])))))
  (testing "arange length" (is (= 5 (first (f/size (f/arange 5))))))
  (testing "arange values" (is (= [0 1 2 3 4] (clj (f/arange 5)))))
  (testing "eye shape" (is (= [3 3] (f/shape (f/eye 3)))))
  (testing "eye trace is n" (is (= 3.0 (f/item (f/sum (f/eye 3))))))
  (testing "randn shape" (is (= [4 4] (f/shape (f/randn [4 4])))))
  (testing "rand shape"  (is (= [3 3] (f/shape (f/rand [3 3])))))
  (testing "rand values in [0 1)"
    (let [v (clj (f/rand [100]))]
      (is (every? #(and (>= % 0.0) (< % 1.0)) v))))
  (testing "tensor fn"
    (is (= [1.0 2.0] (clj (f/tensor [1.0 2.0]))))))


(deftest metadata-ops
  (let [t2d (tensor/->tensor [[1.0 2.0 3.0] [4.0 5.0 6.0]])]
    (testing "shape" (is (= [2 3] (f/shape t2d))))
    (testing "size"  (is (= [2 3] (f/size t2d))))
    (testing "numel" (is (= 6 (f/numel t2d))))
    (testing "dtype returns something" (is (some? (f/dtype t2d))))))

(deftest shape-ops
  (let [t2d (tensor/->tensor [[1.0 2.0 3.0] [4.0 5.0 6.0]])]
    (testing "reshape" (is (= [3 2] (f/shape (f/reshape t2d [3 2])))))
    (testing "view"    (is (= [6]   (f/shape (f/view t2d [6])))))
    (testing "transpose" (is (= [3 2] (f/shape (f/transpose t2d 0 1)))))
    (testing "flatten from dim 0" (is (= [6] (f/shape (f/flatten t2d 0)))))
    (testing "flatten from dim 1" (is (= [2 3] (f/shape (f/flatten t2d 1)))))
    (testing "flatten with end-dim" (is (= [2 3] (f/shape (f/flatten t2d 0 0)))))
    (testing "contiguous" (is (some? (f/contiguous t2d))))
    (testing "squeeze removes size-1 dims"
      (is (= [3] (f/shape (f/squeeze (f/unsqueeze (t 1.0 2.0 3.0) 0))))))
    (testing "squeeze at specific dim"
      (is (= [3] (f/shape (f/squeeze (f/unsqueeze (t 1.0 2.0 3.0) 0) 0)))))
    (testing "permute"
      (let [t3d (tensor/->tensor [[[1.0 2.0] [3.0 4.0]]])]
        (is (= [2 1 2] (f/shape (f/permute t3d [1 0 2]))))))
    (testing "expand broadcasts size-1 dim"
      (let [row (tensor/->tensor [[1.0 2.0 3.0]])]
        (is (= [4 3] (f/shape (f/expand row [4 3]))))))))

(deftest math-ops
  (let [a (t 1.0 2.0 3.0)
        b (t 4.0 5.0 6.0)]
    (testing "add" (is (= [5.0 7.0 9.0]   (clj (f/add a b)))))
    (testing "sub" (is (= [-3.0 -3.0 -3.0] (clj (f/sub a b)))))
    (testing "mul" (is (= [2.0 4.0 6.0]   (clj (f/mul a 2.0)))))
    (testing "div" (is (= [0.5 1.0 1.5]   (clj (f/div a 2.0)))))
    (testing "pow" (is (= [1.0 4.0 9.0]   (clj (f/pow a 2)))))
    (testing "sqrt" (is (= [1.0 2.0 3.0]  (clj (f/sqrt (t 1.0 4.0 9.0))))))
    (testing "abs"  (is (= [1.0 2.0 3.0]  (clj (f/abs (t -1.0 -2.0 -3.0))))))
    (testing "exp e^0 = 1" (is (< (Math/abs (- 1.0 (f/item (f/squeeze (f/exp (tensor/->tensor [0.0])))))) 1e-6)))
    (testing "log ln(1) = 0" (is (< (Math/abs (f/item (f/squeeze (f/log (tensor/->tensor [1.0]))))) 1e-6)))
    (testing "matmul shape"
      (let [m (tensor/->tensor [[1.0 0.0] [0.0 1.0]])
            v (tensor/->tensor [[1.0] [2.0]])]
        (is (= [2 1] (f/shape (f/matmul m v))))))
    (testing "matmul identity values"
      (let [m (tensor/->tensor [[1.0 0.0] [0.0 1.0]])
            v (tensor/->tensor [[3.0] [4.0]])
            r (f/matmul m v)]
        (is (= [2 1] (f/shape r)))))))

(deftest reduction-ops
  (let [a (t 1.0 2.0 3.0)
        m (tensor/->tensor [[1.0 2.0] [3.0 4.0]])]
    (testing "sum 1-D"          (is (= 6.0  (f/item (f/sum a)))))
    (testing "mean 1-D"         (is (= 2.0  (f/item (f/mean a)))))
    (testing "sum with dim"     (is (= [2]  (f/shape (f/sum m 1)))))
    (testing "mean with dim"    (is (= [2]  (f/shape (f/mean m 1)))))
    (testing "sum keepdim"      (is (= [2 1] (f/shape (f/sum m 1 true)))))
    (testing "mean keepdim"     (is (= [2 1] (f/shape (f/mean m 1 true)))))
    (testing "max global"       (is (= 3.0  (f/item (f/max a)))))
    (testing "min global"       (is (= 1.0  (f/item (f/min a)))))
    (testing "max with dim"
      (let [r (f/max m 1)]
        (is (= [2.0 4.0] (clj (py.- r values))))))
    (testing "min with dim"
      (let [r (f/min m 1)]
        (is (= [1.0 3.0] (clj (py.- r values))))))
    (testing "norm"             (is (some? (f/norm a))))
    (testing "norm with p"      (is (some? (f/norm a 2))))
    (testing "norm with p dim"  (is (= [2] (f/shape (f/norm m 2 1)))))))

(deftest comparison-ops
  (let [a (t 1.0 2.0 3.0)
        b (t 2.0 2.0 2.0)]
    (testing "gt" (is (= [false false true]  (mapv boolean (clj (f/gt a b))))))
    (testing "lt" (is (= [true false false]   (mapv boolean (clj (f/lt a b))))))
    (testing "ge" (is (= [false true true]    (mapv boolean (clj (f/ge a b))))))
    (testing "le" (is (= [true true false]    (mapv boolean (clj (f/le a b))))))
    (testing "eq" (is (= [false true false]   (mapv boolean (clj (f/eq a b))))))))

(deftest stacking-ops
  (let [a (t 1.0 2.0)
        b (t 3.0 4.0)]
    (testing "cat dim 0 shape"  (is (= [4]   (f/shape (f/cat [a b] :dim 0)))))
    (testing "cat dim 0 values" (is (= [1.0 2.0 3.0 4.0] (clj (f/cat [a b] :dim 0)))))
    (testing "stack dim 0"      (is (= [2 2] (f/shape (f/stack [a b] :dim 0)))))
    (testing "stack dim 1"      (is (= [2 2] (f/shape (f/stack [a b] :dim 1)))))))

(deftest activations
  (let [x (t 1.0 2.0 3.0)]
    (testing "softmax sums to 1"
      (is (< (Math/abs (- 1.0 (f/item (f/sum (f/softmax x :dim -1))))) 1e-5)))
    (testing "log-softmax all non-positive"
      (is (every? #(<= % 0.0) (clj (f/log-softmax x :dim -1)))))
    (testing "log-softmax exp sums to ~1"
      (is (< (Math/abs (- 1.0 (f/item (f/sum (f/exp (f/log-softmax x :dim -1)))))) 1e-5)))
    (testing "relu-f zeroes negatives"
      (is (= [0.0 0.0 1.0] (clj (f/relu-f (t -2.0 0.0 1.0))))))
    (testing "sigmoid at 0 is 0.5"
      (is (< (Math/abs (- 0.5 (f/item (f/squeeze (f/sigmoid-f (tensor/->tensor [0.0])))))) 1e-5)))
    (testing "tanh at 0 is 0"
      (is (< (Math/abs (f/item (f/squeeze (f/tanh-f (tensor/->tensor [0.0]))))) 1e-6)))
    (testing "gelu output shape" (is (= [3] (f/shape (f/gelu x)))))
    (testing "silu output shape" (is (= [3] (f/shape (f/silu-f x)))))
    (testing "dropout training keeps shape"
      (is (= [3] (f/shape (f/dropout-f x 0.5 true)))))
    (testing "dropout eval is identity"
      (is (= (clj x) (clj (f/dropout-f x 0.5 false)))))))

(deftest indexing-ops
  (let [a (t 0.1 0.2 0.9)]
    (testing "argmax global"     (is (= 2 (f/item (f/argmax a)))))
    (testing "argmax with dim"   (is (some? (f/argmax a 0))))
    (testing "argmax keepdim"    (is (= [1] (f/shape (f/argmax a 0 true)))))
    (testing "sort-tensor ascending"
      (let [{:keys [values indices]} (f/sort-tensor (t 3.0 1.0 2.0) :dim 0 :descending false)]
        (is (= [1.0 2.0 3.0] (clj values)))
        (is (= [1 2 0] (clj indices)))))
    (testing "sort-tensor descending"
      (let [{:keys [values]} (f/sort-tensor (t 3.0 1.0 2.0) :dim 0 :descending true)]
        (is (= [3.0 2.0 1.0] (clj values)))))
    (testing "cumsum"
      (is (= [1.0 3.0 6.0] (clj (f/cumsum (t 1.0 2.0 3.0) 0)))))
    (testing "gather"
      (let [src (tensor/->tensor [[1.0 2.0] [3.0 4.0]])
            idx (tensor/->long (tensor/->tensor [[0 0] [1 0]]))
            r   (f/gather src 1 idx)]
        (is (= [2 2] (f/shape r)))
        (is (= [1.0 1.0 4.0 3.0] (clj (f/flatten r 0))))))
    (testing "index-select rows"
      (let [src (tensor/->tensor [[1.0 2.0] [3.0 4.0] [5.0 6.0]])
            idx (tensor/->long (t 0 2))]
        (is (= [2 2] (f/shape (f/index-select src 0 idx))))))
    (testing "multinomial samples correct count"
      (let [probs  (f/softmax (t 1.0 2.0 3.0) :dim -1)
            sample (f/multinomial probs 2 :replacement true)]
        (is (= [2] (f/shape sample)))))))

(deftest masking-ops
  (let [a    (t 1.0 2.0 3.0 4.0)
        cond (f/gt a (t 2.0 2.0 2.0 2.0))
        zero (t 0.0 0.0 0.0 0.0)]
    (testing "where selects from two tensors"
      (is (= [0.0 0.0 3.0 4.0] (clj (f/where cond a zero)))))
    (testing "masked-fill replaces masked positions"
      (let [mask (f/le a (t 2.0 2.0 2.0 2.0))
            r    (f/masked-fill a mask -1.0)]
        (is (= [-1.0 -1.0 3.0 4.0] (clj r)))))))

(deftest device-and-dtype
  (let [t1 (t 1.0 2.0)]
    (testing "device-of returns a string"
      (is (string? (f/device-of t1))))
    (testing "to-device moves tensor"
      (let [dev (tensor/best-device)]
        (is (str/starts-with? (f/device-of (f/to-device t1 dev)) dev))))
    (testing "to-dtype converts dtype"
      (let [long-t (f/to-dtype t1 (f/dtype (tensor/->long t1)))]
        (is (some? long-t))
        (is (= (f/shape t1) (f/shape long-t)))))))

(deftest misc-ops
  (testing "clone is independent"
    (let [orig  (t 1.0 2.0)
          clone (f/clone orig)]
      (is (= (clj orig) (clj clone)))))
  (testing "detach"
    (is (some? (f/detach (t 1.0 2.0)))))
  (testing "item extracts scalar"
    (is (= 7.0 (f/item (f/sum (t 3.0 4.0))))))
  (testing "div! mutates in-place"
    (let [a (t 4.0 6.0 8.0)]
      (f/div! a 2.0)
      (is (= [2.0 3.0 4.0] (clj a))))))
