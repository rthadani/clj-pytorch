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
                '[torch.nn.functional :as F]
                '[builtins :as builtins])

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

(defn select
  "Select a single slice along dim at index, squeezing that dimension."
  [tensor dim index]
  (py. tensor select dim index))

(defn narrow
  "Narrow tensor along dim from start for length elements.
   Equivalent to tensor[start:start+length] along dim."
  [tensor dim start length]
  (torch/narrow tensor dim start length))

(defn index-select
  "torch.index_select(input, dim, index)"
  [input dim index]
  (torch/index_select input dim index))

(defn- apply-dim-slice [t dim bound dim-size]
  (cond
    (= bound :all) t
    (number? bound) (narrow t dim bound 1)
    (vector? bound)
    (let [[start end step] bound
          start (or start 0)
          end (or end dim-size)
          step (or step 1)]
      (if (= step 1)
        (narrow t dim start (- end start))
        (torch/index_select t dim (torch/arange start end step))))
    :else (throw (IllegalArgumentException. (str "Invalid slice bound: " bound)))))

(defn slice
  "Slice a tensor along each dimension using a spec vector.
   Each entry corresponds to one dimension:
     :all            — entire dimension
     n               — single element at n, keeps the dimension (length 1)
     [start end]     — start to end exclusive, step 1 (uses narrow, returns a view)
     [start end step] — start to end exclusive with step (uses index-select, returns a copy)"
  [tensor slice-spec]
  (let [shapes (vec (py/->jvm (py. tensor size)))]
    (reduce-kv (fn [t dim spec]
                 (apply-dim-slice t dim spec (nth shapes dim)))
               tensor
               (vec slice-spec))))

(defn chunk
  "Split t into chunks equal-sized pieces along dim. Returns a Clojure seq of tensors."
  [t chunks & {:keys [dim] :or {dim 0}}]
  (torch/chunk t chunks :dim dim))

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

(defn tensor-get
  "Get the scalar value at indices from a tensor of any rank.
   (tensor-get t 1 2) is equivalent to t[1, 2] in Python."
  [t & indices]
  (py. (py/get-item t (builtins/tuple indices)) item))

(defn numel
  "Return the total number of elements."
  [t]
  (py/->jvm (py. t numel)))

(declare to-dtype to-device)

;; Creation ops
(defn arange
  ([n]                    (torch/arange n))
  ([start end]            (torch/arange start end))
  ([start end step]       (torch/arange start end step))
  ([start end step dtype] (torch/arange start end step :dtype dtype)))

(defn zeros [shape & {:keys [dtype device]}]
  (cond-> (torch/zeros shape)
    dtype  (to-dtype dtype)
    device (to-device device)))

(defn ones [shape & {:keys [dtype device]}]
  (cond-> (torch/ones shape)
    dtype  (to-dtype dtype)
    device (to-device device)))

(defn zeros-like [t] (torch/zeros_like t))

(defn ones-like  [t] (torch/ones_like  t))

(defn full [shape fill-value & {:keys [dtype device]}]
  (cond-> (torch/full shape fill-value)
    dtype  (to-dtype dtype)
    device (to-device device)))

(defn full-like [t fill-value]
  (torch/full_like t fill-value))

(defn manual-seed
  "Set the random seed for reproducibility."
  [seed]
  (torch/manual_seed seed))

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
(defn matmul [a b] (torch/matmul a b))
(defn add [a b] (torch/add a b))
(defn sub [a b] (torch/sub a b))
(defn gt  [a b] (torch/gt a b))
(defn lt  [a b] (torch/lt a b))
(defn ge  [a b] (torch/ge a b))
(defn le  [a b] (torch/le a b))
(defn eq  [a b] (torch/eq a b))
(defn ne [a b] (torch/ne a b))
(defn mul [t scalar] (py. t __mul__ scalar))
(defn div [a b] (torch/div a b))
(defn pow [t exp] (py. t __pow__ exp))
(defn scalar-pow [base exp] (torch/pow base exp))
(defn sqrt [t] (torch/sqrt t))
(defn rsqrt [t] (torch/rsqrt t))
(defn abs [t] (torch/abs t))
(defn exp [t] (torch/exp t))
(defn log [t] (torch/log t))

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

(defn logical-and [a b] (torch/logical_and a b))

(defn cos [t] (torch/cos t))
(defn sin [t] (torch/sin t))

;; Stacking / concatenation

(defn cat
  "torch.cat(tensors, dim=0)"
  [tensors & {:keys [dim] :or {dim 0}}]
  (torch/cat (builtins/list tensors) :dim dim))

(defn stack
  "torch.stack(tensors, dim=0)"
  [tensors & {:keys [dim] :or {dim 0}}]
  (torch/stack (builtins/list tensors) :dim dim))

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

(defn masked-fill!
  "In-place fill: sets positions where mask is true to val. Mutates t, returns t."
  [t mask val]
  (py. t masked_fill_ mask val))

(defn masked-scatter
  "Copy elements from src into t at positions where mask is true."
  [t mask src]
  (py. t masked_scatter mask src))

;; Device / dtype
(defn to-device
  [t device-str]
  (py. t to (torch/device (name device-str))))

(defn to-dtype
  [t dtype]
  (py. t to dtype))

(defn type-as
  "Cast tensor t to the same dtype as other."
  [t other]
  (py. t type_as other))

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

(defn device-of
  "Return the device a tensor lives on as a string."
  [t]
  (str (py.- t device)))

;; dtype constants instead of torch/int64...
(def int64  torch/int64)
(def float32 torch/float32)
(def long torch/long)
