import sys
from pathlib import Path

# Ensure components dir is first in path (avoids shadowing by system 'common' package)
_components_dir = Path(__file__).resolve().parent
if str(_components_dir) not in sys.path:
    sys.path.insert(0, str(_components_dir))

import os

from team_clustering.mock_detector import MockDetector
from court_detector.court_detector import CourtDetector
from team_clustering.team_clustering import TeamClustering
from visualization.court_2d import write_2d_court_video
from common.classes import CourtType
from common.utils.utils import download


def main(
    video_path: str,
    gt_path: str,
    seg_model: str = "yolov8n-seg.pt",
    output_2d_path: str | None = None,
    league: CourtType = CourtType.NBA,
    k_frames: int = 30,
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
    if not os.path.exists('../models/court_detection_model.pt'):
        download('https://disk.yandex.ru/d/_dHiheOwN2-R_w', 'court_detection_model.pt', '../models')
    detector = MockDetector(gt_path, normalized=True)
    detections = detector.detect(video_path)

    court_detector = CourtDetector()
    court_detector.run(video_path, detections)

    team_clustering = TeamClustering(seg_model)
    team_clustering.run(video_path, detections, k_frames=k_frames)

    if output_2d_path is None:
        stem = Path(video_path).stem
        output_2d_path = str(Path(video_path).parent / f"{stem}_2d.mp4")

    write_2d_court_video(detections, output_2d_path, league, video_path)
    print(f"Saved 2D video to {output_2d_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("gt_path", help="Path to ground-truth annotations (MOT format)")
    parser.add_argument("--seg-model", default="yolov8n-seg.pt", help="YOLO segmentation model")
    parser.add_argument("--output", "-o", default=None, help="Output 2D video path")
    parser.add_argument("--league", choices=["nba", "fiba"], default="nba")
    parser.add_argument("--k-frames", type=int, default=30, help="Sample every k frames for clustering")
    args = parser.parse_args()

    league = CourtType.NBA if args.league == "nba" else CourtType.FIBA
    main(
        args.video_path,
        args.gt_path,
        seg_model=args.seg_model,
        output_2d_path=args.output,
        league=league,
        k_frames=args.k_frames,
    )
