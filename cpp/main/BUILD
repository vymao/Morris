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
        "@onnx_runtime//:onnx_runtime",
        "@onnx_runtime_extensions//:onnx_runtime_ext",
        "@onnx_runtime_extensions//:headers",
        "//main/lib/util:feature_extractor",
        "//main/lib/wake-classifier:wake_classifier",
        "//main/lib/audio-transcriber:audio_transcriber",
        "//main/lib/model:model_base",
        "//main/lib/audio-streamer:audio_stream",
        "//main/lib/util:common",
        "//main/lib/util:sdl",
        "@pybind11//:pybind11_embed",
        "@pybind11",
        "@python_macos//:python-lib",
    ]
)

cc_library(
    name = "run",
    srcs = ["run.cc"],
    hdrs = ["run.h"],
    deps = [
        "//main/lib/util:sdl",
        "@libtorch//:torch_cpu",
        "@onnx_runtime//:onnx_runtime"
    ]
)