import os
from pathlib import Path

from team_clustering.mock_detector import MockDetector
from court_detector.court_detector import CourtDetector
from team_clustering.team_clustering import TeamClustering
from visualization import write_2d_court_video, make_side_by_side_video
from smoother import smooth_detection_coordinates
from common.classes import CourtType
from common.logger import get_logger
from common.utils.utils import download
from detector import Detector, get_video_players_detections


def main(
    video_path: str,
    gt_path: str | None = None,
    seg_model: str = "yolov8n-seg.pt",
    output_2d_path: str | None = None,
    court_type: CourtType = CourtType.NBA,
    k_frames: int = 30,
    output_both: str | None = None,
    enable_smoothing: bool = True,
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
    """
    get_logger().clear()
    if not os.path.exists("../models/court_detection_model.pt"):
        download("https://disk.yandex.ru/d/o7lVmeYl0xmn4g", "court_detection_model.pt", "../models")
    if not os.path.exists("../models/yolo26m_object_detection.pt"):
        download("https://disk.yandex.ru/d/MAGAbYxRFEvX6w", "yolo26m_object_detection.pt", "../models")
    # detector = MockDetector(gt_path, normalized=True)
    # detections = detector.detect(video_path)
    # print(detections)
    clever_detector = Detector()
    all_detections = clever_detector.detect_video(video_path)
    detections = get_video_players_detections(all_detections)

    court_detector = CourtDetector()
    court_detector.run(video_path, detections)

    team_clustering = TeamClustering(seg_model)
    team_clustering.run(video_path, detections, k_frames=k_frames)

    if enable_smoothing:
        smooth_detection_coordinates(detections)

    if output_2d_path is None:
        stem = Path(video_path).stem
        output_2d_path = str(Path(video_path).parent / f"{stem}_2d.mp4")

    write_2d_court_video(detections, output_2d_path, court_type, video_path)
    print(f"Saved 2D video to {output_2d_path}")

    if output_both is not None:
        make_side_by_side_video(video_path, output_2d_path, output_both)


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
    parser.add_argument("--no_smoothing", type=bool, default=False, help="Disable smoothing")
    args = parser.parse_args()

    court_type = CourtType.NBA if args.court_type == "nba" else CourtType.FIBA
    print(1)
    main(
        args.video_path,
        args.gt_path,
        seg_model=args.seg_model,
        output_2d_path=args.output,
        court_type=court_type,
        k_frames=args.k_frames,
        output_both=args.output_both,
        enable_smoothing=not args.no_smoothing,
    )
