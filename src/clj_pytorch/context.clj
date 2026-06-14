(ns clj-pytorch.context
  "PyTorch context managers as Clojure macros.

  no-grad      — disable gradient tracking
  enable-grad  — re-enable gradient tracking inside a no-grad block
  inference    — inference mode (stronger than no-grad, no version tracking)
  autocast     — automatic mixed precision (AMP)"
  (:require [libpython-clj2.python :as py :refer [py.]]
            [libpython-clj2.require :refer [require-python]]))

(require-python '[torch :as torch])

(defmacro no-grad
  "Run body without gradient tracking.
   Equivalent to Python's `with torch.no_grad():`"
  [& body]
  `(let [ctx# (torch/no_grad)]
     (libpython-clj2.python/call-attr ctx# "__enter__")
     (try
       (do ~@body)
       (finally
         (libpython-clj2.python/call-attr ctx# "__exit__" nil nil nil)))))

(defmacro enable-grad
  "Re-enable gradient tracking — use inside a no-grad block."
  [& body]
  `(let [ctx# (torch/enable_grad)]
     (libpython-clj2.python/call-attr ctx# "__enter__")
     (try
       (do ~@body)
       (finally
         (libpython-clj2.python/call-attr ctx# "__exit__" nil nil nil)))))

(defmacro inference-mode
  "Run body in inference mode — no gradient tracking, no version counter.
   Stronger than no-grad and faster for pure inference."
  [& body]
  `(let [ctx# (torch/inference_mode)]
     (libpython-clj2.python/call-attr ctx# "__enter__")
     (try
       (do ~@body)
       (finally
         (libpython-clj2.python/call-attr ctx# "__exit__" nil nil nil)))))

(defmacro autocast
  "Run body with automatic mixed precision.
   device-type is a string: \"cuda\", \"cpu\", \"mps\".
   dtype defaults to torch.float16 on cuda, bfloat16 on cpu."
  [device-type & body]
  `(let [ctx# (libpython-clj2.python/call-attr torch/amp "autocast" ~device-type)]
     (libpython-clj2.python/call-attr ctx# "__enter__")
     (try
       (do ~@body)
       (finally
         (libpython-clj2.python/call-attr ctx# "__exit__" nil nil nil)))))
