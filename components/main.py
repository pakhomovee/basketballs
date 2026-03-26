import logging
from pathlib import Path

from config import AppConfig, load_app_config
from run_pipeline import TOTAL_STAGES, NullStageLogger, run_pipeline
from visualization import make_side_by_side_video, write_2d_court_video

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

COMPONENTS_DIR = Path(__file__).resolve().parent


def main(cfg: AppConfig):
    """
    Full pipeline: detect, court detect, team cluster, generate 2D video.

    Args:
        cfg: pipeline configuration from YAML/CLI.
    """
    main_cfg = cfg.main
    video_path = main_cfg.video_path
    if video_path is None:
        raise ValueError("main.video_path must be set in config or via CLI override")
    output_2d_path = main_cfg.output_2d_path
    output_both = main_cfg.output_both
    result = run_pipeline(video_path, cfg, stage_logger=NullStageLogger(total_stages=TOTAL_STAGES))

    # Visualization

    if output_2d_path is None:
        stem = Path(video_path).stem
        output_2d_path = str(Path(video_path).parent / f"{stem}_2d.mp4")

    write_2d_court_video(result.players_detections, output_2d_path, result.court_type, video_path)
    print(f"Saved 2D video to {output_2d_path}")

    if output_both is not None:
        if not Path(output_both).suffix:
            output_both += ".mp4"
        make_side_by_side_video(
            video_path,
            output_2d_path,
            output_both,
            detections=result.players_detections,
            ball_detections=result.ball_detections,
            passes=result.pass_events,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", help="Path to input video (overrides config.main.video_path)")
    parser.add_argument(
        "output_both", nargs="?", help="Path to output side-by-side video (overrides config.main.output_both)"
    )
    parser.add_argument("--config", default=str(COMPONENTS_DIR / "configs" / "main.yaml"), help="Path to YAML config")
    parser.add_argument("--output", "-o", default=None, help="Output 2D video path")
    parser.add_argument("--court_type", choices=["nba", "fiba"], default=None)
    parser.add_argument(
        "--with-pose", action=argparse.BooleanOptionalAction, default=None, help="Enrich players with pose skeletons"
    )
    parser.add_argument(
        "--enable-smoothing", action=argparse.BooleanOptionalAction, default=None, help="Enable smoothing"
    )
    parser.add_argument("--no-reid", action=argparse.BooleanOptionalAction, default=None, help="Disable ReID")
    args = parser.parse_args()

    app_cfg = load_app_config(args.config).model_copy(deep=True)
    cfg = app_cfg.main
    if args.video_path is not None:
        cfg.video_path = args.video_path
    if args.output_both is not None:
        cfg.output_both = args.output_both
    if args.output is not None:
        cfg.output_2d_path = args.output
    if args.court_type is not None:
        cfg.court_type = args.court_type
    if args.with_pose is not None:
        cfg.with_pose = args.with_pose
    if args.enable_smoothing is not None:
        cfg.enable_smoothing = args.enable_smoothing
    if args.no_reid is not None:
        cfg.no_reid = args.no_reid

    main(app_cfg)
