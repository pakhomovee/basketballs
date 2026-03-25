import logging
import os
from pathlib import Path
import cv2

from court_detector.court_detector import CourtDetector
from team_clustering.embedding import PlayerEmbedder
from team_clustering.mock_detector import MockDetector
from team_clustering.team_clustering import TeamClustering
from tracking import FlowTracker
from visualization import write_2d_court_video, make_side_by_side_video
from smoother import smooth_detection_coordinates
from reidentification import extract_reid_embeddings
from common.classes import CourtType
from common.logger import get_logger
from common.utils.utils import download
from config import AppConfig, load_app_config
from detector import Detector, enrich_detections_with_numbers, enrich_players_with_pose
from detector.interpolate_ball_detections import linear_interpolate_ball_detections
from ball_detector.detector import WASBBallDetector
from actions.ball_possession import (
    assign_ball_possession_soft_dribble,
    apply_possession_segments,
    greedy_possession_segments_soft_dribble,
)
from actions.passes import find_team_passes

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

COMPONENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMPONENTS_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
COURT_MODEL_PATH = MODELS_DIR / "court_detection_model.pt"
DETECTOR_MODEL_PATH = MODELS_DIR / "detector_model.pt"
PARSEQ_MODEL_PATH = MODELS_DIR / "parseq_flex.ckpt"
REID_MODEL_PATH = MODELS_DIR / "reid_model.pth"
SEG_MODEL_PATH = MODELS_DIR / "seg-model.pt"
WASB_MODEL_PATH = MODELS_DIR / "wasb_basketball_best.pth.tar"
DEFAULT_REID_WEIGHTS = str(REID_MODEL_PATH)


def _ensure_default_models() -> None:
    if not COURT_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/VRabl680FfKBog", COURT_MODEL_PATH.name, str(MODELS_DIR))
    if not DETECTOR_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/KdtL0zaQQlI5fg", DETECTOR_MODEL_PATH.name, str(MODELS_DIR))
    if not PARSEQ_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/QucoCUmnUbLMHw", PARSEQ_MODEL_PATH.name, str(MODELS_DIR))
    if not REID_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/Ak2skkMBdVCqmQ", REID_MODEL_PATH.name, str(MODELS_DIR))
    if not SEG_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/dpjzmKkadg-nZg", SEG_MODEL_PATH.name, str(MODELS_DIR))
    if not WASB_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/JZQN5HEOKOegog", WASB_MODEL_PATH.name, str(MODELS_DIR))


def _extract_embeddings(video_path, detections, enable_reid=True):
    """Extract color histograms (team clustering) and optionally ReID features (tracking)."""
    PlayerEmbedder().extract_player_embeddings(video_path, detections)

    if enable_reid and os.path.isfile(DEFAULT_REID_WEIGHTS):
        from common.utils.utils import get_device

        device = get_device()
        print(f"Extracting ReID features for tracking ({DEFAULT_REID_WEIGHTS})")
        extract_reid_embeddings(video_path, detections, DEFAULT_REID_WEIGHTS, device=device)
    elif enable_reid:
        print(f"ReID weights not found at {DEFAULT_REID_WEIGHTS}, tracker will use color histograms")


def _get_video_frame_width(video_path: str) -> float | None:
    """Return video frame width in pixels, or None if unavailable."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return width if width > 0 else None
    except Exception:
        return None


def _get_video_fps(video_path: str) -> float:
    """Return FPS of the video, or 25.0 if unavailable."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 25.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return fps if fps > 0 else 25.0
    except Exception:
        return 25.0


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
    court_type = CourtType.NBA if main_cfg.court_type == "nba" else CourtType.FIBA

    get_logger().clear()
    _ensure_default_models()

    # Detector

    detector = Detector(conf_threshold=cfg.detector.initial_threshold)
    all_detections = detector.detect_video(video_path)
    players_detections, referees_detections, numbers_detections = enrich_detections_with_numbers(
        video_path,
        all_detections,
        player_conf_threshold=cfg.detector.player_conf_threshold,
        referee_conf_threshold=cfg.detector.referee_conf_threshold,
        number_conf_threshold=cfg.detector.number_conf_threshold,
        ocr_conf_threshold=cfg.detector.ocr_conf_threshold,
    )

    if main_cfg.with_pose:
        enrich_players_with_pose(video_path, players_detections)

    # Homography

    court_detector = CourtDetector(cfg=cfg)
    court_detector.run(video_path, players_detections)

    # Ball detection (WASB)
    wasb_detector = WASBBallDetector(cfg=cfg)
    ball_detections = wasb_detector.detect_video(video_path)

    interpolated_ball_by_frame = linear_interpolate_ball_detections(ball_detections)
    possession_ball_detections = {frame_id: [ball] for frame_id, ball in interpolated_ball_by_frame.items()}

    # Tracker
    _extract_embeddings(video_path, players_detections, enable_reid=not main_cfg.no_reid)

    video_fps = _get_video_fps(video_path)
    frame_width = _get_video_frame_width(video_path)
    tracker = FlowTracker(cfg=cfg, frame_width=frame_width, fps=video_fps)
    tracker.track(players_detections)

    team_clustering = TeamClustering()
    team_clustering.run(players_detections)

    # Possession

    assign_ball_possession_soft_dribble(players_detections, possession_ball_detections)
    possession_segments = greedy_possession_segments_soft_dribble(players_detections, fps=video_fps)
    apply_possession_segments(players_detections, possession_segments)
    pass_events = find_team_passes(possession_segments, players_detections)

    # Smoothing

    if main_cfg.enable_smoothing:
        smooth_detection_coordinates(players_detections)

    # Visualization

    if output_2d_path is None:
        stem = Path(video_path).stem
        output_2d_path = str(Path(video_path).parent / f"{stem}_2d.mp4")

    write_2d_court_video(players_detections, output_2d_path, court_type, video_path)
    print(f"Saved 2D video to {output_2d_path}")

    if output_both is not None:
        if not Path(output_both).suffix:
            output_both += ".mp4"
        make_side_by_side_video(
            video_path,
            output_2d_path,
            output_both,
            detections=players_detections,
            ball_detections=possession_ball_detections,
            passes=pass_events,
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
