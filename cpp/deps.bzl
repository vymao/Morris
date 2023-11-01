load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def whisper():
    maybe(
        new_git_repository,
        name = "com_github_ggerganov_whisper",
        build_file = Label("//external:BUILD.whisper"),
        init_submodules = True,
        recursive_init_submodules = True,
        remote = "https://github.com/ggerganov/whisper.cpp.git",
        tag = "v1.4.0"
    )

    http_archive(
        name = "com_github_libsdl_sdl2",
        build_file = Label("//external:BUILD.sdl2"),
        sha256 = "e2ac043bd2b67be328f875043617b904a0bb7d277ba239fe8ac6b9c94b85cbac",
        strip_prefix = "SDL-dca3fd8307c2c9ebda8d8ea623bbbf19649f5e22",
        urls = ["https://github.com/libsdl-org/SDL/archive/dca3fd8307c2c9ebda8d8ea623bbbf19649f5e22.zip"],
    )

def _data_deps_extension_impl(_):
    whisper()
    

data_deps_ext = module_extension(
    implementation = _data_deps_extension_impl,
)