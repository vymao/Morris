load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "model_base",
    srcs = ["model_base.cc"],
    hdrs = ["model_base.h"],
    deps = [
        "@pybind11//:pybind11_embed",
        "@pybind11",
        "@python_macos//:python-lib",
        "@onnx_runtime//:onnx_runtime",
        "//main/lib/util:feature_extractor",
        "//main/lib/util:common",
    ]
)