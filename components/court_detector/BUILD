load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip//:requirements.bzl", "requirement")

py_library(
    name = "court_constants_lib",
    srcs = ["court_constants.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "court_detector_lib",
    srcs = ["court_detector.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":court_constants_lib",
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("torch"),
        requirement("tqdm"),
        requirement("ultralytics"),
    ],
)

py_binary(
    name = "court_detector",
    srcs = ["court_detector.py"],
    visibility = ["//visibility:public"],
    data = ["//:court_detection_model"],
    deps = [
        ":court_detector_lib",
        requirement("torch"),
        requirement("torchvision"),
        requirement("opencv-python"),
        requirement("numpy"),
        requirement("tqdm"),
        requirement("ultralytics"),
    ],
)

py_binary(
    name = "court_detector_visualizer",
    srcs = ["visualizer.py"],
    main = "visualizer.py",
    visibility = ["//visibility:public"],
    data = ["//:court_detection_model"],
    deps = [
        ":court_constants_lib",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("opencv-python"),
        requirement("torch"),
        requirement("ultralytics"),
        requirement("tqdm"),
        requirement("PyQt5"),
    ],
)
