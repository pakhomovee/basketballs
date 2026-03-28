from __future__ import annotations

import argparse
from pathlib import Path

from detector.detector_rfdetr import DetectorRFDETR, visualize_players_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RF-DETR player detections (bboxes only).")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to output video")
    parser.add_argument(
        "--model-path",
        default=str(Path(__file__).parent.parent.parent / "models" / "checkpoint_best_ema.pth"),
        help="Path to RF-DETR checkpoint (.ckpt/.pth)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.2,
        help="Confidence threshold for detections",
    )
    args = parser.parse_args()

    detector = DetectorRFDETR(model_path=args.model_path, conf_threshold=args.conf_threshold)
    out = visualize_players_detections(
        args.input,
        args.output,
        detector=detector,
        conf_threshold=args.conf_threshold,
    )
    print(f"Saved RF-DETR player visualization to {out}")


if __name__ == "__main__":
    main()
