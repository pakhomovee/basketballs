"""
WASB ball detector: loads the pretrained HRNet checkpoint, runs inference on
triplets of consecutive frames, and returns ball position candidates per frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torchvision import transforms

from ball_detector.model import WASBHRNet
from common.classes.ball import Ball
from common.utils.utils import get_device
from config import load_default_config

if TYPE_CHECKING:
    from config import AppConfig

MODEL_INPUT_WH = (512, 288)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BALL_BBOX_RADIUS = 10

_to_tensor = transforms.ToTensor()
_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ---------------------------------------------------------------------------
# Affine helpers (same logic as the original CenterNet / HRNet family)
# ---------------------------------------------------------------------------


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return b + np.array([-d[1], d[0]], dtype=np.float32)


def get_affine_transform(
    center: np.ndarray, scale: float, output_size: tuple[int, int], inv: bool = False
) -> np.ndarray:
    """Build a 2x3 affine matrix that maps the original image (described by
    *center* and *scale*) to *output_size* (w, h), or the inverse."""
    src_w = scale
    dst_w, dst_h = output_size

    src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2] = _get_3rd_point(src[0], src[1])
    dst[2] = _get_3rd_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def affine_transform_pt(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix *t* to a single 2-D point."""
    return (t @ np.array([pt[0], pt[1], 1.0], dtype=np.float32))[:2]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_frame(frame_bgr: np.ndarray, trans: np.ndarray) -> torch.Tensor:
    """Warp a BGR frame using the affine matrix, convert to normalised tensor."""
    warped = cv2.warpAffine(frame_bgr, trans, MODEL_INPUT_WH, flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    from PIL import Image

    pil = Image.fromarray(rgb)
    return _normalize(_to_tensor(pil))


# ---------------------------------------------------------------------------
# Postprocessing (connected-component blob detection on heatmap)
# ---------------------------------------------------------------------------


def postprocess_heatmap(hm: np.ndarray, trans_inv: np.ndarray, threshold: float):
    """
    Extract ball candidates from a single-channel heatmap (after sigmoid).

    Returns list of dicts: [{"xy": np.array([x, y]), "score": float}, ...]
    where (x, y) are in the original image coordinate system.
    """
    candidates = []
    if hm.max() <= threshold:
        return candidates
    _, hm_th = cv2.threshold(hm, threshold, 1, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
    for m in range(1, n_labels):
        ys, xs = np.where(labels == m)
        weights = hm[ys, xs]
        score = float(weights.sum())
        x = float(np.sum(xs * weights) / weights.sum())
        y = float(np.sum(ys * weights) / weights.sum())
        xy_orig = affine_transform_pt(np.array([x, y], dtype=np.float32), trans_inv)
        candidates.append({"xy": xy_orig, "score": score})
    return candidates


# ---------------------------------------------------------------------------
# Simple online tracker (picks highest-score candidate not too far from prev)
# ---------------------------------------------------------------------------


class SimpleTracker:
    def __init__(self, max_disp: float = 300.0):
        self.max_disp = max_disp
        self.prev_xy: np.ndarray | None = None

    def update(self, candidates: list[dict]) -> dict | None:
        if self.prev_xy is not None:
            candidates = [c for c in candidates if np.linalg.norm(c["xy"] - self.prev_xy) < self.max_disp]
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c["score"])
        self.prev_xy = best["xy"]
        return best

    def reset(self):
        self.prev_xy = None


# ---------------------------------------------------------------------------
# High-level detector
# ---------------------------------------------------------------------------


class WASBBallDetector:
    def __init__(
        self,
        weights_path: str | Path = Path(__file__).parent.parent.parent / "models" / "wasb_basketball_best.pth.tar",
        cfg: "AppConfig | None" = None,
        device: str | None = None,
        step: int = 3,
    ):
        if cfg is None:
            cfg = load_default_config()
        self.cfg = cfg
        if device is None:
            device = get_device()
        self.device = device
        self.step = step

        self.model = WASBHRNet()
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        state = ckpt["model_state_dict"]
        # strip "module." prefix left by DataParallel
        cleaned = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(cleaned)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def detect_video(self, video_path: str | Path) -> dict[int, Ball]:
        """
        Run detection on every frame of the video.

        Returns ``dict[int, Ball]`` — only frames where the ball was
        detected are present.  ``Ball.bbox`` is a small square centred
        on the detected position.
        """
        score_threshold = self.cfg.ball_detector.score_threshold
        max_disp_ratio = self.cfg.ball_detector.max_disp_ratio

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames_bgr: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
        cap.release()

        n = len(frames_bgr)
        if n == 0:
            return {}

        h, w = frames_bgr[0].shape[:2]
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale = float(max(h, w))
        trans = get_affine_transform(center, scale, MODEL_INPUT_WH)
        trans_inv = get_affine_transform(center, scale, MODEL_INPUT_WH, inv=True)

        per_frame_candidates: list[list[dict]] = [[] for _ in range(n)]

        tensors_cache: dict[int, torch.Tensor] = {}

        def _get_tensor(idx: int) -> torch.Tensor:
            if idx not in tensors_cache:
                tensors_cache[idx] = preprocess_frame(frames_bgr[idx], trans)
            return tensors_cache[idx]

        for start in range(0, n, self.step):
            triplet_indices = [min(start + k, n - 1) for k in range(3)]
            inp = torch.cat([_get_tensor(i) for i in triplet_indices], dim=0)
            inp = inp.unsqueeze(0).to(self.device)

            logits = self.model(inp)  # (1, 3, 288, 512)
            hms = logits.sigmoid().cpu().numpy()[0]  # (3, 288, 512)

            for k, fidx in enumerate(triplet_indices):
                cands = postprocess_heatmap(hms[k], trans_inv, threshold=score_threshold)
                per_frame_candidates[fidx].extend(cands)

            for idx in list(tensors_cache):
                if idx < start:
                    del tensors_cache[idx]

        max_disp = max_disp_ratio * max(h, w)
        tracker = SimpleTracker(max_disp=max_disp)
        ball_detections: dict[int, Ball] = {}
        for frame_id, cands in enumerate(per_frame_candidates):
            best = tracker.update(cands)
            if best is not None:
                x, y = best["xy"]
                r = BALL_BBOX_RADIUS
                ball_detections[frame_id] = Ball(
                    bbox=[int(round(x - r)), int(round(y - r)), int(round(x + r)), int(round(y + r))],
                    confidence=best["score"],
                )
        return ball_detections
