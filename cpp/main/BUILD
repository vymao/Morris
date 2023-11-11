load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

COPTS = [
    "-O3",
    "-pthread",
    "-std=c++11",
    "-fPIC",
    "-Wall",
] + selects.with_or({
    "//conditions:default": [],
    "@bazel_tools//src/conditions:linux_x86_64": [
        "-mavx",
        "-mavx2",
        "-mfma",
        "-mf16c",
        "-msse3",
    ],
})

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        "@com_github_ggerganov_whisper//:whisper",
        "@com_github_libsdl_sdl2//:include",
        "@onnx_runtime//:onnx_runtime",
        "@onnx_runtime_extensions//:onnx_extensions",
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