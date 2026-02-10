load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "court_detector",
    srcs = ["court_detector.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        requirement("opencv-python"),
        requirement("numpy"),
    ],
)
