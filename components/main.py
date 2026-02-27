import sys
sys.path.append('/home/pakhomovee/basketballs')
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from team_clustering.mock_detector import MockDetector
from court_detector.court_detector import CourtDetector
from team_clustering.team_clustering import TeamClustering
from flattener.flattener import Flattener
from classes.classes import CourtType


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
    detector = MockDetector(gt_path, normalized=False)
    detections = detector.detect(video_path)

    court_detector = CourtDetector()
    court_detector.run(video_path, detections)

    team_clustering = TeamClustering(seg_model)
    team_clustering.run(video_path, detections, k_frames=k_frames)

    if output_2d_path is None:
        stem = Path(video_path).stem
        output_2d_path = str(Path(video_path).parent / f"{stem}_2d.mp4")

    _write_2d_video(detections, output_2d_path, league, video_path)
    print(f"Saved 2D video to {output_2d_path}")


def _write_2d_video(
    detections: dict[int, list],
    output_path: str,
    league: CourtType,
    video_path: str,
):
    """Render 2D court view for each frame and write to video."""
    flattener = Flattener(league)
    base_frame = flattener.get_frame(
        np.empty((0, 2)),
        np.empty((0, 2)),
        (0.0, 0.0),
    )
    h, w = base_frame.shape[:2]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_id in tqdm(range(1, total_frames + 1), desc="Writing 2D video"):
        players = detections.get(frame_id, [])
        team1_xy = []
        team2_xy = []
        for p in players:
            if p.court_position is None or p.team_id is None:
                continue
            x_m, y_m = p.court_position
            xy = np.array([[x_m, y_m]])
            if p.team_id == 0:
                team1_xy.append(xy)
            else:
                team2_xy.append(xy)

        team1_xy = np.vstack(team1_xy) if team1_xy else np.empty((0, 2))
        team2_xy = np.vstack(team2_xy) if team2_xy else np.empty((0, 2))
        ball_xy = (0.0, 0.0)

        frame = flattener.get_frame(team1_xy, team2_xy, ball_xy)
        out.write(frame)

    out.release()


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
