load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = ["//..."],
)

filegroup(
    name = "court_detection_model",
    srcs = ["models/court_detection_model.pt"],
    visibility = ["//visibility:public"],
)

alias(
    name = "court_detector_visualier",
    actual = "//components/court_detector:court_detector_visualier",
)
