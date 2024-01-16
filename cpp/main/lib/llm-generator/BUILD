load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "generator",
    srcs = ["generation.cc"],
    hdrs = ["generation.h"],
    deps = [
        "//main/lib/util:common",
        "@nlohmann_json//:json",
        "@cpr"
    ]
)

cc_test(
  name = "generator_test",
  size = "small",
  srcs = ["generation_test.cc"],
  deps = [
    "@com_google_googletest//:gtest_main",
    "@cpr",
    ":generator"
],
)