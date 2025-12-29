load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "utils",
    srcs = ["utils.py", "datasets.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("torch"),
        requirement("pandas"),
        requirement("opencv-python"),
        requirement("matplotlib"),
        requirement("IPython"),
    ],
)

