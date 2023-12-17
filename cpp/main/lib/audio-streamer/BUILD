load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "audio_stream",
    srcs = [
        "audio_stream.cc",
    ],
    hdrs = [
        "audio_stream.h"
    ],
    deps = [
        "@com_github_libsdl_sdl2//:include",
        "//main/lib/util:sdl",
        "//main/lib/util:common"
    ]
)
