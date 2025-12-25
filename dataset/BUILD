load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "download",
    srcs = ["download.py"],
    deps = [
        requirement("kagglehub"),
    ],
)
