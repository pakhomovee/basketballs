load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "tracker_lib",
    srcs = ["tracker.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("ultralytics"),
        requirement("torch"),
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("lap"),
        "//common/utils:utils",
    ],
)

py_binary(
    name = "tracker",
    srcs = ["tracker.py"],
    main = "tracker.py",
    deps = [
        ":tracker_lib",
    ],
)
