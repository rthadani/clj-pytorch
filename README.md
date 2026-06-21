# clj-pytorch

A Clojure wrapper around PyTorch via [libpython-clj](https://github.com/clj-python/libpython-clj). This is basically a collection of all the PyTorch patterns I kept rewriting over the past three years — tensor ops, training loops, dataloaders, inference utilities — bundled into something reusable. If you stumble across this and find it useful, awesome.

Requires a Python environment with PyTorch installed and libpython-clj pointing at it. Tested with `torch 2.9.1+cu130`, `torchaudio 2.9.1+cu130`, and `torchvision 0.24.1+cu130`.

## What's in here

- `clj-pytorch.tensor` — create tensors, convert to/from Clojure, device helpers
- `clj-pytorch.functional` — math ops, shape ops, activations, indexing, masking
- `clj-pytorch.nn` — layer constructors and the `defmodule` macro for defining custom modules
- `clj-pytorch.optimizer` — Adam, SGD, lr schedulers
- `clj-pytorch.context` — `no-grad`, `inference-mode`, `enable-grad`, `autocast` macros
- `clj-pytorch.train` — `train-step`, `eval-step`, `train-epoch`, `eval-epoch`, `fit`
- `clj-pytorch.data` — `tensor-dataset`, `dataloader`, `->batches`, `clj->dataloader`
- `clj-pytorch.inference` — top-p sampling, KV cache, HuggingFace model loading

## Defining modules

The `defmodule` macro is the main thing here that's non-obvious. It wraps `py/create-class` to build a real `nn.Module` subclass from Clojure:

```clojure
(require '[clj-pytorch.nn :as nn]
         '[clj-pytorch.functional :as f])

(nn/defmodule MLP [in-features hidden out-features]
  :layers {:fc1  (nn/linear in-features hidden)
           :act  (nn/relu)
           :fc2  (nn/linear hidden out-features)}
  :forward (fn [self x]
             (-> x
                 ((nn/get-layer self :fc1))
                 ((nn/get-layer self :act))
                 ((nn/get-layer self :fc2)))))

;; MLP is now a regular constructor fn
(def model (MLP 128 256 10))
```

`defmodule` takes:

- `sym` — the name, becomes the constructor fn
- `params` — constructor args
- `:layers` — map of keyword → layer, registered on `self` so PyTorch tracks parameters
- `:forward` — fn of `[self & inputs]`, becomes the module's `forward` method
- `:init` — optional fn of `[self]` for extra setup (buffers, etc.)

Modules defined with `defmodule` implement `clojure.lang.IFn`, so you can call them directly like a function — no need for `nn/call` at the call site:

```clojure
(model x)           ;; equivalent to (nn/call model x)
(nn/train! model)   ;; all nn/ helpers still work transparently
(optim/adam model)  ;; passes through to the underlying Python module
```

To access a sublayer inside `forward`:

```clojure
(nn/get-layer self :fc1)   ;; :fc1 -> self.fc1
```

## Training

```clojure
(require '[clj-pytorch.train :as train]
         '[clj-pytorch.optimizer :as optim])

(def opt (optim/adam model :lr 1e-3))
(def loss-fn (nn/cross-entropy-loss))

(train/fit {:model         model
            :optimizer     opt
            :loss-fn       loss-fn
            :epochs        10
            :train-batches (fn [] (->batches train-loader))
            :val-batches   (fn [] (->batches val-loader))})
```

## Running tests

```sh
clj -M:test -m cognitect.test-runner
```
