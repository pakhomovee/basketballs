load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_test(
    name = "test_detector",
    srcs = ["test_detector.py"],
    data = ["//components/detector/tests/data:Q2_side_540-570.mp4"],
    size = "medium",
    deps = [
        "//components/detector:detector_model",
        "//visualization:visualize",
        requirement("pytest"),
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("torch"),
        requirement("torchvision"),
    ],
)
