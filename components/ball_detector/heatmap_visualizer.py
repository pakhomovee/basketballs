"""
Visualize WASB ball heatmaps on a video.

Outputs a video where the predicted heatmap is warped back to the original
frame coordinates and alpha-blended over the original frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ball_detector.detector import MODEL_INPUT_WH, WASBBallDetector, get_affine_transform
from config import load_default_config
from tqdm.auto import tqdm


def _overlay_heatmap_on_frame(
    frame_bgr: np.ndarray,
    hm: np.ndarray,
    trans_inv: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Warp heatmap (model input coords) onto the original frame and blend."""
    if hm is None:
        return frame_bgr

    hm_max = float(hm.max())
    if hm_max <= 0:
        return frame_bgr

    hm_norm = (hm / (hm_max + 1e-12)).astype(np.float32)
    hm_u8 = (hm_norm * 255.0).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)  # (H_in, W_in, 3)
    hm_warp = cv2.warpAffine(hm_color, trans_inv, (frame_bgr.shape[1], frame_bgr.shape[0]), flags=cv2.INTER_LINEAR)

    # Blend: heatmap on top of the original frame.
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, hm_warp, alpha, 0.0)


def visualize_heatmaps(
    input_path: str | Path,
    output_path: str | Path,
    step: int = 3,
    alpha: float = 0.45,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    cfg = load_default_config()
    detector = WASBBallDetector(cfg=cfg, step=step)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_bgr: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    n = len(frames_bgr)
    if n == 0:
        raise RuntimeError(f"Empty/invalid video: {input_path}")

    # Affine mapping: original frame coords <-> model input coords.
    center = np.array([frame_w / 2.0, frame_h / 2.0], dtype=np.float32)
    scale = float(max(frame_h, frame_w))
    trans = get_affine_transform(center, scale, MODEL_INPUT_WH)
    trans_inv = get_affine_transform(center, scale, MODEL_INPUT_WH, inv=True)

    # Precompute heatmaps only for frames that appear in the same triplets
    # as detector.detect_video() (step stride).
    heatmaps: list[np.ndarray | None] = [None] * n
    for start in tqdm(range(0, n, step), desc="Heatmap inference", unit="triplet"):
        triplet_indices = [min(start + k, n - 1) for k in range(3)]
        triplet_frames = [frames_bgr[i] for i in triplet_indices]
        hms = detector.detect_heatmap(triplet_frames, trans=trans)  # (3, 288, 512)
        for k, fidx in enumerate(triplet_indices):
            heatmaps[fidx] = hms[k]

    if not output_path.suffix:
        output_path = output_path.with_suffix(".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    for i in tqdm(range(n), desc="Writing frames", unit="frame"):
        frame = frames_bgr[i]
        hm = heatmaps[i]
        overlay = _overlay_heatmap_on_frame(frame, hm, trans_inv=trans_inv, alpha=alpha)
        writer.write(overlay)

    writer.release()
    print(f"Saved heatmap visualization to {output_path} (frames={n})")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_path", help="Path to input video")
    p.add_argument("output_path", help="Path to output video")
    p.add_argument("--step", type=int, default=3, help="Triplet stride for heatmap inference")
    p.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    visualize_heatmaps(args.input_path, args.output_path, step=args.step, alpha=args.alpha)

