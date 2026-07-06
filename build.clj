(ns build
  (:refer-clojure :exclude [test])
  (:require [clojure.tools.deps :as t]
            [clojure.tools.build.api :as b]
            [deps-deploy.deps-deploy :as dd]))

(def lib 'io.github.rthadani/clj-pytorch)
(def default-version "0.1.0-SNAPSHOT")
(def class-dir "target/classes")
(def basis (delay (b/create-basis {:project "deps.edn"})))

(defn- jar-file [version]
  (format "target/%s-%s.jar" (name lib) version))

(defn test [opts]
  (println "\nRunning tests...")
  (let [basis    (b/create-basis {:aliases [:test]})
        combined (t/combine-aliases basis [:test])
        cmds     (b/java-command
                  {:basis basis
                   :java-opts (:jvm-opts combined)
                   :main 'clojure.main
                   :main-args ["-m" "cognitect.test-runner"]})
        {:keys [exit]} (b/process cmds)]
    (when-not (zero? exit) (throw (ex-info "Tests failed" {}))))
  opts)

(defn jar [{:keys [version] :or {version default-version} :as opts}]
  (b/write-pom {:class-dir class-dir
                :lib lib
                :version version
                :basis @basis
                :src-dirs ["src"]})
  (b/copy-dir {:src-dirs ["src" "resources"] :target-dir class-dir})
  (b/jar {:class-dir class-dir :jar-file (jar-file version)})
  opts)

(defn install [{:keys [version] :or {version default-version} :as opts}]
  (jar opts)
  (b/install {:basis @basis
              :lib lib
              :version version
              :jar-file (jar-file version)
              :class-dir class-dir}))

(defn ci [opts]
  (test opts)
  (b/delete {:path "target"})
  (jar opts))

(defn deploy [{:keys [version] :or {version default-version} :as opts}]
  (jar opts)
  (dd/deploy {:installer :remote
              :artifact (jar-file version)
              :pom-file (b/pom-path {:lib lib :class-dir class-dir})}))
