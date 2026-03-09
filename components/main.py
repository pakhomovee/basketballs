import logging
import os
from pathlib import Path

from cache import load_detections_cache, save_detections_cache
from court_detector.court_detector import CourtDetector
from team_clustering.embedding import extract_player_embeddings
from team_clustering.mock_detector import MockDetector
from team_clustering.team_clustering import TeamClustering
from tracking import PlayerTracker, FlowTracker
from visualization import write_2d_court_video, make_side_by_side_video
from smoother import smooth_detection_coordinates
from reidentification import extract_reid_embeddings
from common.classes import CourtType
from common.logger import get_logger
from common.utils.utils import download
from detector import Detector, get_video_players_detections

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _extract_embeddings(video_path, detections, seg_model, reid_weights):
    """Extract color histograms (team clustering) and optionally ReID features (tracking)."""
    extract_player_embeddings(video_path, detections, seg_model=seg_model)

    if reid_weights and os.path.isfile(reid_weights):
        from common.utils.utils import get_device

        device = get_device()
        print(f"Extracting ReID features for tracking ({reid_weights})")
        extract_reid_embeddings(video_path, detections, reid_weights, device=device)
    elif reid_weights:
        print(f"ReID weights not found at {reid_weights}, tracker will use color histograms")


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


def main(
    video_path: str,
    gt_path: str | None = None,
    seg_model: str = "yolov8n-seg.pt",
    output_2d_path: str | None = None,
    court_type: CourtType = CourtType.NBA,
    k_frames: int = 30,
    output_both: str | None = None,
    enable_smoothing: bool = True,
    max_track_length: int | None = 60,
    no_cache: bool = False,
    no_cache_detector: bool = False,
    no_cache_embeds: bool = False,
    tracker_type: str = "flow",
):
    """
    Full pipeline: detect, court detect, team cluster, generate 2D video.

    Args:
        video_path: Path to input video.
        gt_path: Path to ground-truth annotations (MOT format).
        seg_model: YOLO segmentation model for team clustering.
        output_2d_path: Path for output 2D court video. Default: <video_stem>_2d.mp4
        league: Court type (NBA or FIBA) for flattener.
        k_frames: Sample every k-th frame for team clustering.
        max_track_length: Cap track length in frames (~2 sec at 30fps) for dense crowds.
                         None = no limit.
    """
    get_logger().clear()
    if not os.path.exists("../models/court_detection_model.pt"):
        download("https://disk.yandex.ru/d/o7lVmeYl0xmn4g", "court_detection_model.pt", "../models")
    if not os.path.exists("../models/best-4.pt"):
        download("https://disk.yandex.ru/d/MAGAbYxRFEvX6w", "best-4.pt", "../models")
    if not os.path.exists("../models/reid_model.pth"):
        download("https://disk.yandex.ru/d/Ak2skkMBdVCqmQ", "reid_model.pth", "../models")
    if gt_path is None:
        # no_cache overrides the other two (full disable)
        cache_detector = not (no_cache or no_cache_detector)
        cache_embeds = not (no_cache or no_cache_embeds)
        save_cache = not no_cache

        detections = load_detections_cache(
            video_path,
            seg_model,
            use_detector_cache=cache_detector,
            use_embeddings_cache=cache_embeds,
        )
        if detections is None:
            detector = Detector()
            all_detections = detector.detect_video(video_path)
            detections = get_video_players_detections(all_detections)
            court_detector = CourtDetector()
            court_detector.run(video_path, detections)
            _extract_embeddings(video_path, detections, seg_model, "../models/reid_model.pth")
            if save_cache:
                save_detections_cache(
                    video_path,
                    detections,
                    seg_model,
                    use_detector_cache=cache_detector,
                    use_embeddings_cache=cache_embeds,
                )
        else:
            if no_cache_embeds and not no_cache:
                # Loaded bbox + court, but recompute embeddings
                for players in detections.values():
                    for p in players:
                        p.embedding = None
                _extract_embeddings(video_path, detections, seg_model, "../models/reid_model.pth")
                save_detections_cache(video_path, detections, seg_model)
            else:
                parts = []
                if cache_detector:
                    parts.append("detections")
                if cache_embeds:
                    parts.append("embeddings")
                print(f"Loaded {' and '.join(parts)} from cache")
    else:
        detector = MockDetector(gt_path, normalized=True)
        detections = detector.detect(video_path)
        court_detector = CourtDetector()
        court_detector.run(video_path, detections)
        _extract_embeddings(video_path, detections, seg_model, "../models/reid_model.pth")

    if tracker_type == "flow":
        frame_width = _get_video_frame_width(video_path)
        tracker = FlowTracker(num_tracks=10, frame_width=frame_width)
        tracker.track(detections)
    else:
        tracker = PlayerTracker(max_track_length=max_track_length)
        tracker.track(detections)

    team_clustering = TeamClustering()
    team_clustering.run(detections, k_frames=k_frames)

    if tracker_type != "flow":
        for threshold in [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9]:
            track_id_to_team = {
                p.player_id: p.team_id
                for players in detections.values()
                for p in players
                if p.player_id >= 0 and p.team_id is not None
            }
            tracker.merge_tracks(
                max_tracks=10, detections=detections, track_id_to_team=track_id_to_team, cost_threshold=threshold
            )
            team_clustering.run(detections, k_frames=k_frames)
            if len(tracker.tracks) <= 10:
                break

    if enable_smoothing:
        smooth_detection_coordinates(detections)

    if output_2d_path is None:
        stem = Path(video_path).stem
        output_2d_path = str(Path(video_path).parent / f"{stem}_2d.mp4")

    write_2d_court_video(detections, output_2d_path, court_type, video_path)
    print(f"Saved 2D video to {output_2d_path}")

    if output_both is not None:
        make_side_by_side_video(video_path, output_2d_path, output_both, detections=detections)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--gt_path", default=None, help="Path to ground-truth annotations (MOT format)")
    parser.add_argument("--seg-model", default="yolov8n-seg.pt", help="YOLO segmentation model")
    parser.add_argument("--output", "-o", default=None, help="Output 2D video path")
    parser.add_argument("--output_both", default=None, help="Output side by side 2D video path")
    parser.add_argument("--court_type", choices=["nba", "fiba"], default="nba")
    parser.add_argument("--k-frames", type=int, default=30, help="Sample every k frames for clustering")
    parser.add_argument("--max-track-length", type=int, default=60, help="Max track length. 0 = no limit")
    parser.add_argument("--no_smoothing", type=bool, default=False, help="Disable smoothing")
    parser.add_argument("--no-cache", action="store_true", help="Disable all caching (don't load, don't save)")
    parser.add_argument("--no-cache-detector", action="store_true", help="Don't use cache for detector output")
    parser.add_argument("--no-cache-embeds", action="store_true", help="Don't use cache for embeddings")
    parser.add_argument("--no-reid", action="store_true", help="Force color-histogram embeddings (skip ReID)")
    parser.add_argument(
        "--tracker",
        choices=["online", "flow"],
        default="flow",
        help="Tracker type: 'online' (Kalman+Hungarian) or 'flow' (min-cost-max-flow)",
    )
    args = parser.parse_args()

    court_type = CourtType.NBA if args.court_type == "nba" else CourtType.FIBA
    main(
        args.video_path,
        args.gt_path,
        seg_model=args.seg_model,
        output_2d_path=args.output,
        court_type=court_type,
        k_frames=args.k_frames,
        output_both=args.output_both,
        enable_smoothing=not args.no_smoothing,
        max_track_length=args.max_track_length or None,
        no_cache=args.no_cache,
        no_cache_detector=args.no_cache_detector,
        no_cache_embeds=args.no_cache_embeds,
        tracker_type=args.tracker,
    )
