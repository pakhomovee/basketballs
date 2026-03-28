from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ball_detector.detector import WASBBallDetector
from common.classes import CourtType
from common.utils.models import get_model_paths
from config import AppConfig, load_app_config, load_default_config
from court_detector.court_constants import CourtConstants
from court_detector.court_detector import CourtDetector
from detector import Detector, get_video_rim_detections
from shot_detector.shot_detector import SHOT_CHECKPOINT_FILENAME, ShotDetector


CLASS_LABELS = {
    0: "nothing",
    1: "shot",
    2: "make",
}

CLASS_COLORS = {
    0: (220, 220, 220),
    1: (0, 200, 255),
    2: (0, 220, 0),
}


def _draw_bbox(
    frame: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def visualize_predictions(
    input_video: str | Path,
    output_video: str | Path,
    model_path: str | Path | None = None,
    *,
    cfg: AppConfig | None = None,
    rim_conf_threshold: float = 0.25,
    ball_step: int = 3,
) -> None:
    if cfg is None:
        cfg = load_default_config()

    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    ct = CourtType.NBA if cfg.main.court_type == "nba" else CourtType.FIBA
    court_constants = CourtConstants(ct)

    print("Loading shot detector...")
    shot_detector = ShotDetector(model_path, cfg=cfg)

    print("Running rim detector...")
    detector = Detector(conf_threshold=cfg.detector.initial_threshold)
    video_detections = detector.detect_video(str(input_video))
    rim_detections = get_video_rim_detections(video_detections, conf_threshold=rim_conf_threshold)

    print("Running ball detector...")
    ball_detector = WASBBallDetector(cfg=cfg, step=ball_step)
    ball_detections = ball_detector.detect_video(str(input_video))

    print("Estimating homographies...")
    court_detector = CourtDetector(cfg=cfg)
    homographies, frame_sizes, _, _ = court_detector.extract_homographies_from_video_v2(
        str(input_video),
        court_constants,
        smoothness_cost=cfg.court_detector.smoothness_cost,
        keypoint_eps=cfg.court_detector.keypoint_eps,
        smoothing_num_epochs=cfg.court_detector.smoothing_num_epochs,
        smoothing_lr=cfg.court_detector.smoothing_lr,
    )

    n_frames = len(frame_sizes)
    if n_frames == 0:
        raise RuntimeError("No frames were read from video")

    print("Running shot prediction...")
    preds = shot_detector.predict_from_detections(
        ball_detections,
        rim_detections,
        homographies,
        frame_sizes=frame_sizes,
        num_frames=n_frames,
    )

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video frame size")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or n_frames)
    total = min(total, n_frames, len(preds))

    for frame_id in tqdm(range(total), desc="Writing visualization", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        ball = ball_detections.get(frame_id)
        if ball is not None and ball.bbox and len(ball.bbox) >= 4:
            conf = 0.0 if ball.confidence is None else float(ball.confidence)
            _draw_bbox(frame, ball.bbox, (0, 255, 0), f"ball {conf:.2f}")

        rims = rim_detections.get(frame_id, [])
        for r in rims:
            _draw_bbox(frame, r.get_bbox(), (255, 255, 0), f"rim {float(r.confidence):.2f}")

        cls = int(preds[frame_id])
        cls_name = CLASS_LABELS.get(cls, str(cls))
        cls_color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.putText(
            frame,
            f"class: {cls_name}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            cls_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame: {frame_id}",
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved visualization to {output_video}")


def _default_shot_model_path() -> str:
    return str(get_model_paths(load_default_config()).shot_detection)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize shot-detector predictions on video")
    p.add_argument("input_video", type=str, help="Path to input video")
    p.add_argument("output_video", type=str, help="Path to output video")
    p.add_argument(
        "--model-path",
        type=str,
        default=_default_shot_model_path(),
        help=f"Path to shot checkpoint .pt (default: models/{SHOT_CHECKPOINT_FILENAME} from config)",
    )
    p.add_argument("--config", type=str, default=None, help="Optional config YAML path")
    p.add_argument("--court-type", type=str, default="nba", choices=["nba", "fiba"])
    p.add_argument("--rim-conf", type=float, default=0.25, help="Rim confidence threshold")
    p.add_argument("--ball-step", type=int, default=3, help="WASB triplet step")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_default_config() if args.config is None else load_app_config(args.config)
    cfg = cfg.model_copy(
        update={"main": cfg.main.model_copy(update={"court_type": args.court_type})},
    )
    visualize_predictions(
        input_video=args.input_video,
        output_video=args.output_video,
        model_path=args.model_path,
        cfg=cfg,
        rim_conf_threshold=args.rim_conf,
        ball_step=args.ball_step,
    )
