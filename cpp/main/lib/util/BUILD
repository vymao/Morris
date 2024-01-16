load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
package(default_visibility = ["//visibility:public"])

cc_test(
  name = "common_test",
  size = "small",
  srcs = ["common_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@libtorch//:torch_cpu",
    "@onnx_runtime//:onnx_runtime",
    "@pybind11",
    "@pybind11//:pybind11_embed",
    ":common"
],
)

cc_library(
    name = "feature_extractor",
    srcs = [
        "feature_extractor.cc",
    ],
    hdrs = [
        "feature_extractor.h",
    ],
    deps = [
        "@pybind11//:pybind11_embed",
        "@pybind11",
        "@python_macos//:python-lib",
        "@onnx_runtime//:onnx_runtime",
    ]
)

cc_library(
    name = "sdl",
    srcs = [
        "common_sdl.cc",
    ],
    hdrs = [
        "common_sdl.h"
    ],
    deps = [
        "@com_github_libsdl_sdl2//:include",
    ]
)

cc_library(
    name = "common",
    hdrs = ["common.h"],
    srcs = ["common.cc"],
    deps = [
        "@libtorch//:torch_cpu",
        "@onnx_runtime//:onnx_runtime",
        "@nlohmann_json//:json",
        "@pybind11",
        "@pybind11//:pybind11_embed",
        "@python_macos//:python-lib",
    ]
)
