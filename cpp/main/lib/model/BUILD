load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

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
        ":audio_tensor"
    ]
)

cc_library(
    name = "audio_tensor",
    srcs = ["audio_tensor.cc"],
    hdrs = ["audio_tensor.h"],
    deps = [
        "@pybind11//:pybind11_embed",
        "@pybind11",
        "@python_macos//:python-lib",
        "@onnx_runtime//:onnx_runtime",
        "@libtorch//:torch_cpu",
    ]
)

cc_test(
  name = "audio_tensor_test",
  size = "small",
  srcs = ["audio_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@libtorch//:torch_cpu",
    "@onnx_runtime//:onnx_runtime",
    "@pybind11",
    "@pybind11//:pybind11_embed",
    ":audio_tensor"
],
)