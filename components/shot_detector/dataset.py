"""
PyTorch Dataset for shot-detection training.

Loads precomputed ``.npz`` features produced by ``build_training_dataset``
and generates per-frame embeddings + labels on-the-fly via
:class:`~shot_detector.shot_embedder.ShotEmbedder`.

Three mutually-exclusive labels:
  0 – background (nothing)
  1 – shot
  2 – make  (overrides shot when both apply)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common.classes import CourtType
from common.classes.ball import Ball
from common.classes.detections import Detection
from shot_detector.shot_embedder import ShotEmbedder

LABEL_NOTHING = 0
LABEL_SHOT = 1
LABEL_MAKE = 2
NUM_CLASSES = 3


def _build_labels(
    T: int,
    shot_start: int | None,
    shot_end: int | None,
    make_start: int | None,
    make_end: int | None,
) -> np.ndarray:
    labels = np.zeros(T, dtype=np.int64)
    if shot_start is not None and shot_end is not None:
        s, e = max(shot_start, 0), min(shot_end, T - 1)
        labels[s : e + 1] = LABEL_SHOT
    if make_start is not None and make_end is not None:
        s, e = max(make_start, 0), min(make_end, T - 1)
        labels[s : e + 1] = LABEL_MAKE
    return labels


def _reconstruct_from_npz(
    data: np.lib.npyio.NpzFile,
) -> tuple[
    dict[int, Ball],
    dict[int, list[Detection]],
    dict[int, np.ndarray | None],
    int,
    int,
    int,
]:
    T = int(data["frame_count"])

    homography_arr = data["homography"]
    homography_mask = data["homography_mask"]
    homographies: dict[int, np.ndarray | None] = {}
    for f in range(T):
        homographies[f] = homography_arr[f] if homography_mask[f] > 0.5 else None

    ball_bbox = data["ball_bbox"]
    ball_conf = data["ball_conf"]
    ball_detections: dict[int, Ball] = {}
    for f in range(T):
        if ball_conf[f] > 1e-6:
            bbox = [int(v) for v in ball_bbox[f]]
            ball_detections[f] = Ball(bbox=bbox, confidence=float(ball_conf[f]))

    rim_bbox_arr = data["rim_bbox"]
    rim_conf_arr = data["rim_conf"]
    rim_detections: dict[int, list[Detection]] = {}
    for f in range(T):
        if rim_conf_arr[f] > 1e-6:
            x1, y1, x2, y2 = (int(v) for v in rim_bbox_arr[f])
            d = Detection(x1, y1, x2, y2, class_id=0, confidence=float(rim_conf_arr[f]))
            rim_detections[f] = [d]

    if "frame_w" not in data or "frame_h" not in data:
        raise KeyError(
            "npz must contain 'frame_w' and 'frame_h' (rebuild features with current build_training_dataset)"
        )
    fw = int(data["frame_w"])
    fh = int(data["frame_h"])

    return ball_detections, rim_detections, homographies, fw, fh, T


class ShotDataset(Dataset):
    """
    Variable-length dataset: each item is one video clip.

    Parameters
    ----------
    features_dir : path to the directory produced by ``build_training_dataset``
        (must contain ``index.jsonl`` and ``samples/``).
    court_type : NBA or FIBA.
    fliplr, random_scale, random_shift, random_rotate :
        Augmentation params forwarded to :class:`ShotEmbedder`.
    skip_prob : float
        If ``> 0``, each frame is dropped independently with this probability
        (temporal subsampling). At least one frame is always kept. Use on train
        only; keep ``0`` for validation.
    random_crop_ratio : float
        If ``< 1``, before *skip_prob* a contiguous temporal window is taken:
        its length is ``u * T`` with ``u ~ Uniform(random_crop_ratio, 1)``,
        start uniform over valid offsets. Use ``1.0`` to disable. Train only;
        use ``1.0`` for validation / class-weight counting.
    """

    def __init__(
        self,
        features_dir: str | Path,
        court_type: CourtType = CourtType.NBA,
        *,
        fliplr: bool = False,
        random_scale: float = 1.0,
        random_shift: float = 0.0,
        random_rotate: float = 0.0,
        skip_prob: float = 0.0,
        random_crop_ratio: float = 1.0,
    ):
        features_dir = Path(features_dir)
        index_path = features_dir / "index.jsonl"

        self.samples: list[dict] = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                abs_path = rec.get("sample_path_abs")
                if abs_path and Path(abs_path).exists():
                    self.samples.append(rec)
                    continue
                rel_path = rec.get("sample_path")
                if rel_path:
                    p = features_dir / rel_path
                    if p.exists():
                        rec["sample_path_abs"] = str(p)
                        self.samples.append(rec)

        self.skip_prob = max(0.0, min(float(skip_prob), 1.0))
        self.random_crop_ratio = float(random_crop_ratio)
        self.embedder = ShotEmbedder(
            court_type=court_type,
            fliplr=fliplr,
            random_scale=random_scale,
            random_shift=random_shift,
            random_rotate=random_rotate,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.samples[idx]
        data = np.load(rec["sample_path_abs"], allow_pickle=False)

        ball_dets, rim_dets, homos, fw, fh, T = _reconstruct_from_npz(data)

        embedding = self.embedder.build_embedding(
            ball_dets,
            rim_dets,
            homos,
            frame_width=float(fw),
            frame_height=float(fh),
            num_frames=T,
        )  # (T, FLAT_DIM)

        shot_start = int(data["shot_start_frame"])
        shot_end = int(data["shot_end_frame"])
        make_start = int(data["make_start_frame"])
        make_end = int(data["make_end_frame"])

        labels = _build_labels(
            T,
            shot_start if shot_start >= 0 else None,
            shot_end if shot_end >= 0 else None,
            make_start if make_start >= 0 else None,
            make_end if make_end >= 0 else None,
        )

        if self.random_crop_ratio < 1.0 - 1e-9 and T > 0:
            lo = min(1.0, max(0.0, self.random_crop_ratio))
            hi = 1.0
            if lo >= hi - 1e-9:
                u = 1.0
            else:
                u = random.uniform(lo, hi)
            seg_len = max(1, min(T, int(round(u * T))))
            start = 0 if seg_len >= T else random.randint(0, T - seg_len)
            end = start + seg_len
            embedding = embedding[start:end]
            labels = labels[start:end]
            T = seg_len

        if self.skip_prob > 0.0 and T > 0:
            keep = np.random.rand(T) >= self.skip_prob
            if not np.any(keep):
                keep[np.random.randint(0, T)] = True
            sel = np.flatnonzero(keep)
            embedding = embedding[sel]
            labels = labels[sel]
            T = int(embedding.shape[0])

        return (
            torch.from_numpy(embedding),  # (T', 112)
            torch.from_numpy(labels),  # (T',)
            T,
        )


class StackedShotDataset(Dataset):
    """
    Train-only: each item is ``k`` clips from *base* concatenated in time,
    with ``k ~ Uniform({1, …, max_stack})``. The first clip is always
    ``base[idx]``; the remaining ``k-1`` indices are uniform random (with
    replacement). *base* must yield ``(features, labels, T)`` tensors.
    """

    def __init__(self, base: Dataset, max_stack: int):
        if max_stack < 2:
            raise ValueError(
                "StackedShotDataset expects max_stack >= 2; use the base dataset directly for max_stack == 1"
            )
        self.base = base
        self.max_stack = max_stack

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        k = random.randint(1, self.max_stack)
        n = len(self.base)
        picks = [idx] + [random.randrange(n) for _ in range(k - 1)]

        f0, l0, _ = self.base[picks[0]]
        feats = [f0]
        labs = [l0]
        for j in picks[1:]:
            feat, lab, _ = self.base[j]
            feats.append(feat)
            labs.append(lab)
        feat = torch.cat(feats, dim=0)
        lab = torch.cat(labs, dim=0)
        return feat, lab, int(feat.shape[0])


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length clips to the longest sequence in the batch."""
    features_list, labels_list, lengths = zip(*batch)
    max_len = max(lengths)
    feat_dim = features_list[0].shape[1]
    B = len(batch)

    features = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    labels = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (f, lab, length) in enumerate(zip(features_list, labels_list, lengths)):
        features[i, :length] = f
        labels[i, :length] = lab
        mask[i, :length] = True

    return features, labels, mask
