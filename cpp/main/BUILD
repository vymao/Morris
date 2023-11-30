load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

COPTS = [
    "-O2",
    "-pthread",
    "-std=c++17",
    "-fPIC",
    "-Wall",
] 

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    copts = COPTS,
    deps = [
        "@com_github_ggerganov_whisper//:whisper",
        "@com_github_libsdl_sdl2//:include",
        "@onnx_runtime//:onnx_runtime",
        #"@onnx_runtime_extensions//:headers",
        "//main/lib/util:feature_extractor",
        "@pybind11//:pybind11_embed",
        "@pybind11",
        "@python_macos//:python-lib",
        ":run"
    ]
)

cc_library(
    name = "run",
    srcs = ["run.cc"],
    hdrs = ["run.h"],
    deps = [
        "@com_github_ggerganov_whisper//:whisper",
        "@com_github_ggerganov_whisper//:common",
        "@com_github_ggerganov_whisper//:common-sdl",
        "@com_github_libsdl_sdl2//:include",
        "@com_github_ggerganov_whisper//:ggml",
        "@libtorch//:torch_cpu",
        "@onnx_runtime//:onnx_runtime"
    ]
)