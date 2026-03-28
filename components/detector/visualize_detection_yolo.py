"""
Visualize YOLO player detections (bboxes only) on video.

Self-contained: loads a YOLO model, runs detection, draws player bboxes,
writes the output video. No external detector/pipeline dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


COLOR_PLAYER = (255, 165, 0)  # BGR orange
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def visualize_yolo_players(
    input_path: str,
    output_path: str,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.2,
    player_class_id: int = 0,
) -> str:
    default_model = Path(__file__).parent.parent.parent / "models" / "yolo-players-50.pt"
    model = YOLO(str(model_path or default_model))

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    try:
        for frame_id in tqdm(range(total_frames), desc="YOLO player viz", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False, conf=conf_threshold)[0]

            for box in results.boxes:
                cls = int(box.cls.item())
                if cls != player_class_id:
                    continue

                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().reshape(-1))

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PLAYER, THICKNESS)

                label = f"{conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), COLOR_PLAYER, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 2), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO player detections (bboxes only)")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to output video")
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Path to YOLO checkpoint (.pt). Default: models/detector_model.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold (default: 0.2)",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class id for player (default: 0)",
    )
    args = parser.parse_args()

    out = visualize_yolo_players(
        args.input, args.output, model_path=args.model, conf_threshold=args.conf, player_class_id=args.class_id
    )
    print(f"Saved YOLO player visualization to {out}")
