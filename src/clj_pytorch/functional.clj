(ns clj-pytorch.functional
  "Tensor operations and functional activations.
  Shape ops:   flatten, transpose, reshape, view, expand,
               contiguous, permute, squeeze, unsqueeze
  Math ops:    matmul, add, sub, mul, div, pow, sqrt,
               mean, sum, max, min, abs, exp, log
  Activations: softmax, log-softmax, gelu, relu, sigmoid, tanh, silu
  Stacking:    cat, stack
  Misc:        dropout, size, shape, dtype, numel"
  (:require [libpython-clj2.python  :refer [py. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]))


(require-python '[torch :as torch]
                '[torch.nn.functional :as F])

(declare to-device)

;; Shape ops
(defn flatten
  "Flatten tensor from start-dim. Optionally to end-dim."
  ([t start-dim]           (py. t flatten start-dim))
  ([t start-dim end-dim]   (py. t flatten start-dim end-dim)))

(defn transpose
  [t dim0 dim1]
  (py. t transpose dim0 dim1))

(defn permute
  "Permute dimensions. dims is a vector."
  [t dims]
  (apply py/call-attr t "permute" dims))

(defn reshape
  "Reshape tensor. shape is a vector."
  [t shape]
  (apply py/call-attr t "reshape" shape))

(defn view
  "View tensor with new shape. shape is a vector."
  [t shape]
  (apply py/call-attr t "view" shape))

(defn expand
  "Expand tensor to shape. shape is a vector."
  [t shape]
  (apply py/call-attr t "expand" shape))

(defn contiguous
  "Return a contiguous tensor in memory."
  [t]
  (py. t contiguous))

(defn squeeze
  "Remove dimensions of size 1."
  ([t]     (py. t squeeze))
  ([t dim] (py. t squeeze dim)))

(defn unsqueeze
  "Insert a dimension at dim."
  [t dim]
  (py. t unsqueeze dim))

;; Tensor metadata
(defn size
  "Return size as a Clojure vector."
  [t]
  (vec (py/->jvm (py. t size))))

(defn shape
  "Return shape as a Clojure vector."
  [t]
  (vec (py/->jvm (py.- t shape))))

(defn dtype
  "Return the dtype of a tensor."
  [t]
  (py.- t dtype))

(defn numel
  "Return the total number of elements."
  [t]
  (py/->jvm (py. t numel)))

;; Creation ops
(defn arange
  "torch.arange(n)"
  [n]
  (torch/arange n))

(defn zeros
  "torch.zeros(shape)"
  [shape]
  (torch/zeros shape))

(defn ones
  "torch.ones(shape)"
  [shape]
  (torch/ones shape))

(defn randn
  "torch.randn(shape)"
  [shape]
  (torch/randn shape))

(defn rand
  "torch.rand(shape)"
  [shape]
  (torch/rand shape))

(defn tensor
  "torch.tensor(data)"
  [data & {:keys [dtype device]}]
  (cond-> (torch/tensor data)
    dtype  (py. to dtype)
    device (to-device device)))

(defn eye
  "torch.eye(n)"
  [n]
  (torch/eye n))

;; Math ops
(defn matmul  [a b]       (torch/matmul a b))
(defn add     [a b]       (torch/add a b))
(defn sub     [a b]       (torch/sub a b))
(defn gt      [a b]       (torch/gt a b))
(defn lt      [a b]       (torch/lt a b))
(defn ge      [a b]       (torch/ge a b))
(defn le      [a b]       (torch/le a b))
(defn eq      [a b]       (torch/eq a b))
(defn mul     [t scalar]  (py. t __mul__ scalar))
(defn div     [a b]       (torch/div a b))
(defn pow     [t exp]     (py. t __pow__ exp))
(defn sqrt    [t]         (torch/sqrt t))
(defn abs     [t]         (torch/abs t))
(defn exp     [t]         (torch/exp t))
(defn log     [t]         (torch/log t))

(defn mean
  ([t]        (py. t mean))
  ([t dim]    (py. t mean dim))
  ([t dim kd] (py. t mean dim :keepdim kd)))

(defn sum
  ([t]        (py. t sum))
  ([t dim]    (py. t sum dim))
  ([t dim kd] (py. t sum dim :keepdim kd)))

(defn max
  ([t]        (py. t max))
  ([t dim]    (py. t max dim)))

(defn min
  ([t]        (py. t min))
  ([t dim]    (py. t min dim)))

(defn norm
  ([t]         (py. t norm))
  ([t p]       (py. t norm p))
  ([t p dim]   (py. t norm p dim)))

;; Stacking / concatenation
(defn cat
  "torch.cat(tensors, dim=0)"
  [tensors & {:keys [dim] :or {dim 0}}]
  (torch/cat tensors :dim dim))

(defn stack
  "torch.stack(tensors, dim=0)"
  [tensors & {:keys [dim] :or {dim 0}}]
  (torch/stack tensors :dim dim))

;;activations
(defn softmax
  "F.softmax(t, dim=-1, dtype=nil)"
  [t & {:keys [dim dtype] :or {dim -1}}]
  (if dtype
    (F/softmax t :dim dim :dtype dtype)
    (F/softmax t :dim dim)))

(defn log-softmax
  "F.log_softmax(t, dim=-1)"
  [t & {:keys [dim] :or {dim -1}}]
  (F/log_softmax t :dim dim))

(defn gelu
  "F.gelu(t, approximate='tanh')"
  [t & {:keys [approximate] :or {approximate "tanh"}}]
  (F/gelu t :approximate approximate))

(defn relu-f
  "F.relu(t)"
  [t]
  (F/relu t))

(defn sigmoid-f
  "F.sigmoid(t)"
  [t]
  (F/sigmoid t))

(defn tanh-f
  "F.tanh(t)"
  [t]
  (F/tanh t))

(defn silu-f
  "F.silu(t)"
  [t]
  (F/silu t))

;; Dropout needs training flag
(defn dropout-f
  "F.dropout(t, p, training)"
  [t p training]
  (F/dropout t :p p :training training))

;; Indexing / masking
(defn where
  "torch.where(condition, x, y)"
  [condition x y]
  (torch/where condition x y))

(defn masked-fill
  "t.masked_fill(mask, value)"
  [t mask value]
  (py. t masked_fill mask value))

;; Device / dtype
(defn to-device
  [t device-str]
  (py. t to (torch/device device-str)))

(defn to-dtype
  [t dtype]
  (py. t to dtype))

(defn detach
  [t]
  (py. t detach))

(defn clone
  [t]
  (py. t clone))

(defn item
  "Extract a scalar value from a single-element tensor."
  [t]
  (py/->jvm (py. t item)))

;; Sorting / cumulative ops — needed for nucleus (top-p) sampling
(defn sort-tensor
  "torch.sort(t, dim, descending).
   Returns a map {:values sorted-tensor :indices index-tensor}."
  [t & {:keys [dim descending] :or {dim -1 descending false}}]
  (let [result (torch/sort t :dim dim :descending descending)]
    {:values  (py.- result values)
     :indices (py.- result indices)}))

(defn cumsum
  "torch.cumsum(t, dim)"
  [t dim]
  (torch/cumsum t dim))

(defn multinomial
  "torch.multinomial(probs, num-samples, replacement=false)"
  [probs num-samples & {:keys [replacement] :or {replacement false}}]
  (torch/multinomial probs num-samples :replacement replacement))

(defn argmax
  "torch.argmax(t) or torch.argmax(t, dim, keepdim)"
  ([t]             (torch/argmax t))
  ([t dim]         (torch/argmax t :dim dim))
  ([t dim keepdim] (torch/argmax t :dim dim :keepdim keepdim)))

(defn div!
  "In-place division: t.div_(other). Mutates t, returns t."
  [t other]
  (py. t div_ other))

(defn gather
  "torch.gather(input, dim, index)"
  [input dim index]
  (torch/gather input dim index))

(defn index-select
  "torch.index_select(input, dim, index)"
  [input dim index]
  (torch/index_select input dim index))

(defn device-of
  "Return the device a tensor lives on as a string."
  [t]
  (str (py.- t device)))
