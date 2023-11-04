load("@bazel_skylib//lib:selects.bzl", "selects")

config_setting(
    name = "cuda",
    define_values = {
        "cuda": "true",
    },
)

# Even when building with --config=cuda, host targets should be built with cuda disabled
# as these targets will run on CI machines that have no GPUs.
selects.config_setting_group(
    name = "cuda_enabled_and_capable",
    match_all = [
        ":cuda",
    ],
)