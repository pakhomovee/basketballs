load("@rules_python//python:defs.bzl", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "detector_model",
    srcs = ["detector_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        requirement("opencv-python"),
        requirement("numpy"),
    ],
)

