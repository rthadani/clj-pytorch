(ns hooks.clj-pytorch.nn
  (:require [clj-kondo.hooks-api :as api]))

(defn defmodule [{:keys [node]}]
  (let [[_ sym params & _rest] (:children node)]
    {:node (api/list-node
            [(api/token-node 'defn)
             sym
             params])}))
