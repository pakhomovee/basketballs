load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "visualize",
    srcs = ["visualize.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("opencv-python"),
    ],
)

