load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "audio_transcriber",
    srcs = ["audio_transcriber.cc"],
    hdrs = ["audio_transcriber.h"],
    deps = [
        "@pybind11//:pybind11_embed",
        "@pybind11",
        "@python_macos//:python-lib",
        "@onnx_runtime//:onnx_runtime",
        "//main/lib/util:feature_extractor",
        "//main/lib/util:common",
        "//main/lib/model:model_base",
        "@nlohmann_json//:json"
    ]
)