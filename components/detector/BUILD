load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "detector_model",
    srcs = ["detector_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//common/utils:utils",
        requirement("torch"),
        requirement("torchvision"),
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("nvidia-cudnn-cu12"),
        requirement("nvidia-cublas-cu12"),
        requirement("nvidia-cuda-runtime-cu12"),
    ],
)

py_binary(
    name = "train_model",
    srcs = ["train_model.py"],
    deps = [
        "//components/detector:detector_model",
    ],
)

