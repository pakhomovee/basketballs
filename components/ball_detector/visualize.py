"""
Visualise WASB ball detection results on a video.

Usage:
    python -m ball_detector.visualize input.mp4 output.mp4
    python -m ball_detector.visualize input.mp4 output.mp4 --step 1
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ball_detector.detector import WASBBallDetector
from config import load_app_config


def visualize(input_path: str, output_path: str, step: int = 3):
    cfg = load_app_config(Path(__file__).resolve().parent.parent / "configs" / "main.yaml")
    detector = WASBBallDetector(cfg=cfg, step=step)
    print(f"Running WASB ball detection (step={step}) on {input_path} …")
    results = detector.detect_video(input_path)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    detected_count = 0

    for i in tqdm(range(total), desc="Drawing"):
        ret, frame = cap.read()
        if not ret:
            break

        # frame number overlay
        label = f"frame: {i}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(frame, (4, 4), (tw + 14, th + 14), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, th + 8), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        ball = results.get(i)
        if ball is not None:
            detected_count += 1
            x1, y1, x2, y2 = ball.bbox[:4]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = 12
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1, cv2.LINE_AA)
            score_txt = f"{ball.confidence:.1f}" if ball.confidence else ""
            cv2.putText(frame, score_txt, (cx + r + 4, cy + 4), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Done. Ball detected on {detected_count}/{total} frames. Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="WASB ball detection visualizer")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to output video")
    parser.add_argument(
        "--step", type=int, default=1, help="Frame step for the sliding window (1 = overlap, 3 = no overlap)"
    )
    args = parser.parse_args()
    visualize(args.input, args.output, step=args.step)


if __name__ == "__main__":
    main()
