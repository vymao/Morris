load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _github_archive(repo, commit, **kwargs):
    repo_name = repo.split("/")[-1]
    http_archive(
        urls = [repo + "/archive/" + commit + ".zip"],
        strip_prefix = repo_name + "-" + commit,
        **kwargs
    )

def libtorch():
    http_archive(
        name = "libtorch",
        strip_prefix = "libtorch",
        type = "zip",
        urls = ["https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip"],
        build_file = Label("//external:BUILD.libtorch")
    )

def whisper():
    maybe(
        new_git_repository,
        name = "com_github_ggerganov_whisper",
        build_file = Label("//external:BUILD.whisper"),
        init_submodules = True,
        recursive_init_submodules = True,
        remote = "https://github.com/ggerganov/whisper.cpp.git",
        tag = "v1.4.3",
    )

    git_repository(
        name = "rules_foreign_cc",
        remote = "https://github.com/vymao/rules_foreign_cc.git",
        commit = "7eb6db89e59a1397584d147ad2bc63e89f613d81",
    )

def extra_tools():
    maybe(
        git_repository,
        name = "nlohmann_json",
        remote = "https://github.com/nlohmann/json.git",
        tag = "v3.11.3"
    )

def extra_bazel_deps():
    maybe(
        http_archive,
        name = "rules_cc",
        sha256 = "3d9e271e2876ba42e114c9b9bc51454e379cbf0ec9ef9d40e2ae4cec61a31b40",
        strip_prefix = "rules_cc-0.0.6",
        urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.6/rules_cc-0.0.6.tar.gz"],
    )

    maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        ],
    )

    http_archive(
        name = "rules_python",
        sha256 = "9d04041ac92a0985e344235f5d946f71ac543f1b1565f2cdbc9a2aaee8adf55b",
        strip_prefix = "rules_python-0.26.0",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.26.0/rules_python-0.26.0.tar.gz",
    )

def sdl():
    new_git_repository(
        name = "com_github_libsdl_sdl2",
        remote = "https://github.com/libsdl-org/SDL.git",
        build_file = Label("//external:BUILD.sdl2"),
        tag = "release-2.28.5",
    )

def onnx():
    maybe(
        http_archive,
        name = "onnx_runtime",
        strip_prefix = "onnxruntime-osx-x86_64-1.16.2",
        type = "tgz",
        urls = ["https://github.com/microsoft/onnxruntime/releases/download/v1.16.2/onnxruntime-osx-x86_64-1.16.2.tgz"],
        build_file = Label("//external:BUILD.onnx_runtime"),
    )

    #new_git_repository(
    #    name = "onnx_runtime_extensions",
    #    build_file = "//external:BUILD.onnx_runtime_extensions",
    #    remote = "https://github.com/microsoft/onnxruntime-extensions",
    #    commit = "81e7799c69044c745239202085eb0a98f102937b",
    #    patches=["//external:onnx_ext.patch"],
    #)

    maybe(
        new_git_repository,
        name = "onnx_runtime_extensions",
        remote = "https://github.com/microsoft/onnxruntime-extensions.git",
        tag = "v0.9.0",
        build_file = Label("//external:BUILD.onnx_runtime_extensions")
    )

def pybind():
    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-master",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/master.zip"],
    )
            # We still require the pybind library.
    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-2.11.1",
        urls = ["https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz"],
    )

    native.new_local_repository(
        name = "python_macos",
        path = "/Users/victor/anaconda3/envs/transformers-v2",
        build_file = "//external:BUILD.python"
    )

def torch():
     maybe(
        git_repository,
        name = "pytorch",
        remote = "https://github.com/pytorch/pytorch.git",
        tag = "v2.1.1-rc4",
    )

def torch_deps():
    http_archive(
        name = "rules_cuda",
        strip_prefix = "runtime-b1c7cce21ba4661c17ac72421c6a0e2015e7bef3/third_party/rules_cuda",
        urls = ["https://github.com/tensorflow/runtime/archive/b1c7cce21ba4661c17ac72421c6a0e2015e7bef3.tar.gz"],
    )

    maybe(
        git_repository,
        name = "pytorch",
        remote = "https://github.com/pytorch/pytorch.git",
        tag = "v2.1.1-rc4",
    )

    http_archive(
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-2e5f4a6beece3b92d2f87744f305eb52b6852aa9",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/2e5f4a6beece3b92d2f87744f305eb52b6852aa9.zip"],
    )

    http_archive(
        name = "pybind11",
        build_file = "@pybind11_bazel//:pybind11.BUILD",
        strip_prefix = "pybind11-2.11.1",
        urls = ["https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz"],
    )

    maybe(
        new_git_repository,
        name = "kineto",
        build_file = Label("//external:BUILD.kineto"),
        remote = "https://github.com/pytorch/kineto.git",
        tag = "v0.4.0",
    )

    maybe(
        new_git_repository,
        name = "fmt",
        build_file = Label("//external:BUILD.fmt"),
        remote = "https://github.com/fmtlib/fmt.git",
        tag = "10.1.1",
    )

    maybe(
        new_git_repository,
        name = "tensorpipe",
        build_file = Label("//external:BUILD.tensorpipe"),
        remote = "https://github.com/pytorch/tensorpipe.git",
        commit = "bb1473a4b38b18268e8693044afdb8635bc8351b",
    )

    maybe(
        git_repository,
        name = "com_github_google_flatbuffers",
        remote = "https://github.com/google/flatbuffers.git",
        tag = "v23.5.26",
    )

    maybe(
        git_repository,
        name = "com_google_protobuf",
        remote = "https://github.com/protocolbuffers/protobuf.git",
        tag = "v25.0",
    )

    maybe(
        new_git_repository,
        name = "foxi",
        build_file = Label("//external:BUILD.foxi"),
        remote = "https://github.com/houseroad/foxi.git",
        commit = "c278588e34e535f0bb8f00df3880d26928038cad",
    )

    maybe(
        new_git_repository,
        name = "gloo",
        build_file = Label("//external:BUILD.gloo"),
        remote = "https://github.com/facebookincubator/gloo.git",
        commit = "2cbcef29a6aff241896a86c719195f1757bfd1b8",
    )

    maybe(
        new_git_repository,
        name = "onnx",
        build_file = Label("//external:BUILD.onnx"),
        remote = "https://github.com/onnx/onnx.git",
        tag = "v1.15.0",
    )

    maybe(
        git_repository,
        name = "fbgemm",
        remote = "https://github.com/pytorch/FBGEMM.git",
        tag = "v0.5.0",
        repo_mapping = {"@cpuinfo": "@org_pytorch_cpuinfo"},
    )

    maybe(
        new_git_repository,
        name = "eigen",
        build_file = Label("//external:BUILD.eigen"),
        remote = "https://gitlab.com/libeigen/eigen.git",
        tag = "3.4.0",
    )

    maybe(
        git_repository,
        name = "aspect_rules_js",
        build_file = Label("//external:BUILD.eigen"),
        remote = "https://github.com/aspect-build/rules_js.git",
        tag = "v1.33.1",
    )

    http_archive(
        name = "aspect_bazel_lib",
        sha256 = "a185ccff9c1b8589c63f66d7eb908de15c5d6bb05562be5f46336c53e7a7326a",
        strip_prefix = "bazel-lib-2.0.0-rc1",
        url = "https://github.com/aspect-build/bazel-lib/releases/download/v2.0.0-rc1/bazel-lib-v2.0.0-rc1.tar.gz",
    )

    http_archive(
        name = "bazel_features",
        sha256 = "f3082bfcdca73dc77dcd68faace806135a2e08c230b02b1d9fbdbd7db9d9c450",
        strip_prefix = "bazel_features-0.1.0",
        url = "https://github.com/bazel-contrib/bazel_features/releases/download/v0.1.0/bazel_features-v0.1.0.tar.gz",
    )

    http_archive(
        name = "rules_pkg",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
        ],
        sha256 = "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
    )

    _github_archive(
        name = "rules_ruby",
        repo = "https://github.com/protocolbuffers/rules_ruby",
        commit = "b7f3e9756f3c45527be27bc38840d5a1ba690436",
        sha256 = "347927fd8de6132099fcdc58e8f7eab7bde4eb2fd424546b9cd4f1c6f8f8bad8",
    )

    _github_archive(
            name = "com_google_absl",
            repo = "https://github.com/abseil/abseil-cpp",
            commit = "fb3621f4f897824c0dbe0615fa94543df6192f30",  # Abseil LTS 20230802.1
            sha256 = "aa768256d0567f626334fcbe722f564c40b281518fc8423e2708a308e5f983ea",
        )
    
    _github_archive(
            name = "utf8_range",
            repo = "https://github.com/protocolbuffers/utf8_range",
            commit = "d863bc33e15cba6d873c878dcca9e6fe52b2f8cb",
            sha256 = "568988b5f7261ca181468dba38849fabf59dd9200fb2ed4b2823da187ef84d8c",
        )

    http_archive(
            name = "zlib",
            build_file = Label("//external:BUILD.zlib"),
            sha256 = "d14c38e313afc35a9a8760dadf26042f51ea0f5d154b0630a31da0540107fb98",
            strip_prefix = "zlib-1.2.13",
            urls = [
                "https://github.com/madler/zlib/releases/download/v1.2.13/zlib-1.2.13.tar.xz",
                "https://zlib.net/zlib-1.2.13.tar.xz",
            ],
        )

    maybe(
        git_repository,
        name = "org_pytorch_cpuinfo",
        remote = "https://github.com/pytorch/cpuinfo.git",
        commit = "d6860c477c99f1fce9e28eb206891af3c0e1a1d7",
    )

    maybe(
        git_repository,
        name = "ideep",
        build_file = Label("//external:BUILD.ideep"),
        remote = "https://github.com/intel/ideep.git",
        commit = "d6860c477c99f1fce9e28eb206891af3c0e1a1d7",
    )
