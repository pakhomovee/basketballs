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
    def detect_heatmap(
        self,
        frames_bgr: np.ndarray | list[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray],
        trans: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Run the HRNet model and return predicted heatmaps.

        Parameters
        ----------
        frames_bgr
            Either a single BGR frame (then it is replicated 3 times) or a
            triplet of consecutive BGR frames. In the triplet case the model
            outputs 3 heatmaps in the same order.
        trans
            Optional precomputed affine transform mapping the original frame to
            model input resolution `MODEL_INPUT_WH`.

        Returns
        -------
        np.ndarray
            Heatmaps in model input coordinates:
            shape `(3, 288, 512)` (one per input frame).
        """
        if isinstance(frames_bgr, np.ndarray):
            frames = [frames_bgr, frames_bgr, frames_bgr]
        else:
            frames = list(frames_bgr)
            if len(frames) != 3:
                raise ValueError(f"frames_bgr must be a single frame or a triplet, got len={len(frames)}")

        frame_h, frame_w = frames[1].shape[:2]
        if trans is None:
            center = np.array([frame_w / 2.0, frame_h / 2.0], dtype=np.float32)
            scale = float(max(frame_h, frame_w))
            trans = get_affine_transform(center, scale, MODEL_INPUT_WH)

        # Preprocess each frame to the model input resolution and stack them.
        inp = torch.cat([preprocess_frame(f, trans) for f in frames], dim=0).unsqueeze(0).to(self.device)

        logits = self.model(inp)  # (1, 3, 288, 512)
        hms = logits.sigmoid().detach().cpu().numpy()[0].astype(np.float32)
        return hms

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def linear_interpolate_ball_detections(
        ball_detections: dict[int, Ball],
        *,
        max_gap: int = 50,
        interpolated_confidence: float = 0.0,
    ) -> dict[int, Ball]:
        """
        Fill missing frames between known ball detections with linear interpolation.

        Interpolates bbox corners [x1, y1, x2, y2] directly.
        Original detections are preserved; only missing frames are inserted.
        """
        if not ball_detections:
            return {}

        sorted_frames = sorted(ball_detections.keys())
        out: dict[int, Ball] = {f: ball_detections[f] for f in sorted_frames}

        for i in range(len(sorted_frames) - 1):
            f1 = sorted_frames[i]
            f2 = sorted_frames[i + 1]
            gap = f2 - f1 - 1
            if gap <= 0:
                continue
            if gap > max_gap:
                continue

            b1 = ball_detections[f1]
            b2 = ball_detections[f2]
            if len(b1.bbox) < 4 or len(b2.bbox) < 4:
                continue

            x1a, y1a, x2a, y2a = b1.bbox[:4]
            x1b, y1b, x2b, y2b = b2.bbox[:4]

            for k in range(1, gap + 1):
                t = k / float(gap + 1)
                frame_id = f1 + k

                x1 = int(round(WASBBallDetector._lerp(x1a, x1b, t)))
                y1 = int(round(WASBBallDetector._lerp(y1a, y1b, t)))
                x2 = int(round(WASBBallDetector._lerp(x2a, x2b, t)))
                y2 = int(round(WASBBallDetector._lerp(y2a, y2b, t)))

                # Ensure valid bbox order after rounding.
                if x2 <= x1:
                    x2 = x1 + 1
                if y2 <= y1:
                    y2 = y1 + 1

                out[frame_id] = Ball(
                    bbox=[x1, y1, x2, y2],
                    confidence=interpolated_confidence,
                )

        return dict(sorted(out.items(), key=lambda kv: kv[0]))

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

        for start in range(0, n, self.step):
            triplet_indices = [min(start + k, n - 1) for k in range(3)]
            triplet_frames = [frames_bgr[i] for i in triplet_indices]
            hms = self.detect_heatmap(triplet_frames, trans=trans)  # (3, 288, 512)

            for k, fidx in enumerate(triplet_indices):
                cands = postprocess_heatmap(hms[k], trans_inv, threshold=score_threshold)
                per_frame_candidates[fidx].extend(cands)

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
        return WASBBallDetector.linear_interpolate_ball_detections(ball_detections)
