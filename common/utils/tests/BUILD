load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

py_test(
    name = "test_datasets",
    srcs = ["test_datasets.py"],
    data = ["//common/data/train/sample_item:img1.mp4", "//common/data/train/sample_item:seqinfo.ini", "//common/data/train/sample_item:gt/gt.txt"],
    size = "medium",
    deps = [
        "//common/utils:utils",
        requirement("pytest"),
        requirement("pandas"),
        requirement("opencv-python"),
    ],
)
