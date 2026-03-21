import logging
import os
from pathlib import Path
import cv2

from cache import load_detections_cache, save_detections_cache
from court_detector.court_detector import CourtDetector
from team_clustering.embedding import DEFAULT_SEG_MODEL, extract_player_embeddings
from team_clustering.mock_detector import MockDetector
from team_clustering.team_clustering import TeamClustering
from tracking import FlowTracker
from visualization import write_2d_court_video, make_side_by_side_video
from smoother import smooth_detection_coordinates
from reidentification import extract_reid_embeddings
from common.classes import CourtType
from common.logger import get_logger
from common.utils.utils import download
from detector import Detector, enrich_detections_with_numbers, enrich_players_with_pose, get_video_ball_detections
from detector.remove_bad_ball_detections import remove_bad_ball_detections
from detector.interpolate_ball_detections import linear_interpolate_ball_detections
from actions import (
    assign_ball_possession_soft_dribble,
    greedy_possession_segments_soft_dribble,
    apply_possession_segments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

COMPONENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMPONENTS_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
COURT_MODEL_PATH = MODELS_DIR / "court_detection_model.pt"
DETECTOR_MODEL_PATH = MODELS_DIR / "best-4.pt"
PARSEQ_MODEL_PATH = MODELS_DIR / "parseq_flex.ckpt"
REID_MODEL_PATH = MODELS_DIR / "reid_model.pth"
SEG_MODEL_PATH = MODELS_DIR / "seg-model.pt"
DEFAULT_REID_WEIGHTS = str(REID_MODEL_PATH)


def _ensure_default_models() -> None:
    if not COURT_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/VRabl680FfKBog", COURT_MODEL_PATH.name, str(MODELS_DIR))
    if not DETECTOR_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/MAGAbYxRFEvX6w", DETECTOR_MODEL_PATH.name, str(MODELS_DIR))
    if not PARSEQ_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/QucoCUmnUbLMHw", PARSEQ_MODEL_PATH.name, str(MODELS_DIR))
    if not REID_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/Ak2skkMBdVCqmQ", REID_MODEL_PATH.name, str(MODELS_DIR))
    if not SEG_MODEL_PATH.exists():
        download("https://disk.yandex.ru/d/dpjzmKkadg-nZg", SEG_MODEL_PATH.name, str(MODELS_DIR))


def _extract_embeddings(video_path, detections, enable_reid=True):
    """Extract color histograms (team clustering) and optionally ReID features (tracking)."""
    extract_player_embeddings(video_path, detections)

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


def _get_video_frame_size(video_path: str) -> tuple[int, int] | None:
    """Return (width, height) in pixels, or None if unavailable."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w <= 0 or h <= 0:
            return None
        return w, h
    except Exception:
        return None


def main(
    video_path: str,
    gt_path: str | None = None,
    output_2d_path: str | None = None,
    court_type: CourtType = CourtType.NBA,
    output_both: str | None = None,
    with_pose: bool = False,
    enable_smoothing: bool = True,
    no_cache: bool = False,
    no_cache_detector: bool = False,
    no_cache_embeds: bool = False,
    no_reid: bool = False,
):
    """
    Full pipeline: detect, court detect, team cluster, generate 2D video.

    Args:
        video_path: Path to input video.
        gt_path: Path to ground-truth annotations (MOT format).
        output_2d_path: Path for output 2D court video. Default: <video_stem>_2d.mp4
        league: Court type (NBA or FIBA) for flattener.
        no_reid: Disable ReID features and use only color embeddings.
    """
    get_logger().clear()
    _ensure_default_models()
    kept_ball_detections = {}
    possession_ball_detections = {}

    # Detector

    detector = Detector()
    all_detections = detector.detect_video(video_path)
    players_detections, referees_detections, numbers_detections = enrich_detections_with_numbers(
        video_path, all_detections
    )

    if with_pose:
        enrich_players_with_pose(video_path, players_detections)

    raw_ball_detections = get_video_ball_detections(all_detections)

    # Homography

    court_detector = CourtDetector()
    homographies = court_detector.run(video_path, players_detections, court_type)

    # Ball detection
    frame_size = _get_video_frame_size(video_path)

    kept_ball_by_frame = remove_bad_ball_detections(
        raw_ball_detections,
        frame_size=frame_size,
        homographies=homographies,
    )

    interpolated_ball_by_frame = linear_interpolate_ball_detections(kept_ball_by_frame)
    possession_ball_detections = {frame_id: [ball] for frame_id, ball in interpolated_ball_by_frame.items()}

    # Tracker
    _extract_embeddings(video_path, players_detections, enable_reid=not no_reid)

    frame_width = _get_video_frame_width(video_path)
    tracker = FlowTracker(num_tracks=10, frame_width=frame_width)
    tracker.track(players_detections)

    team_clustering = TeamClustering()
    team_clustering.run(players_detections)

    # Possession

    if possession_ball_detections:
        assign_ball_possession_soft_dribble(players_detections, possession_ball_detections)
        possession_segments = greedy_possession_segments_soft_dribble(players_detections)
        apply_possession_segments(players_detections, possession_segments)

    # Smoothing

    if enable_smoothing:
        smooth_detection_coordinates(players_detections)

    # Visualization

    if output_2d_path is None:
        stem = Path(video_path).stem
        output_2d_path = str(Path(video_path).parent / f"{stem}_2d.mp4")

    write_2d_court_video(players_detections, output_2d_path, court_type, video_path)
    print(f"Saved 2D video to {output_2d_path}")

    if output_both is not None:
        make_side_by_side_video(
            video_path,
            output_2d_path,
            output_both,
            detections=players_detections,
            ball_detections=possession_ball_detections,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("output_both", help="Path to output both video path")
    parser.add_argument("--gt_path", default=None, help="Path to ground-truth annotations (MOT format)")
    parser.add_argument("--output", "-o", default=None, help="Output 2D video path")
    parser.add_argument("--with-pose", action="store_true", help="Enrich players with pose skeletons for visualization")
    parser.add_argument("--court_type", choices=["nba", "fiba"], default="nba")
    parser.add_argument("--no_smoothing", type=bool, default=False, help="Disable smoothing")
    parser.add_argument("--no-cache", action="store_true", help="Disable all caching (don't load, don't save)")
    parser.add_argument("--no-cache-detector", action="store_true", help="Don't use cache for detector output")
    parser.add_argument("--no-cache-embeds", action="store_true", help="Don't use cache for embeddings")
    parser.add_argument("--no-reid", action="store_true", help="Force color-histogram embeddings (skip ReID)")
    args = parser.parse_args()

    court_type = CourtType.NBA if args.court_type == "nba" else CourtType.FIBA
    main(
        args.video_path,
        args.gt_path,
        output_2d_path=args.output,
        court_type=court_type,
        output_both=args.output_both,
        with_pose=args.with_pose,
        enable_smoothing=not args.no_smoothing,
        no_cache=args.no_cache,
        no_cache_detector=args.no_cache_detector,
        no_cache_embeds=args.no_cache_embeds,
        no_reid=args.no_reid,
    )
