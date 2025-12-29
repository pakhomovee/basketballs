load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_test(
    name = "test_detector",
    srcs = ["test_detector.py"],
    data = ["//common/data/train/sample_item:img1.mp4"],
    size = "medium",
    deps = [
        "//components/detector:detector_model",
        "//visualization:visualize",
        "//common/utils:utils",
        requirement("pytest"),
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("torch"),
        requirement("torchvision"),
    ],
)
