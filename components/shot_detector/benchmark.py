"""
Benchmark a trained shot detector on the validation split.

Reports:
- frame-level accuracy for each class (background / shot / make)
- segmental F1@{10,25,50} based on IoU overlap threshold
  (reported separately for shot and make)

Validation split is built exactly like in ``shot_detector.train``:
indices are shuffled with ``random.Random(seed)`` and first
``max(1, int(n_total * val_fraction))`` are used for validation.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from common.classes import CourtType
from common.utils.utils import get_device
from shot_detector.dataset import LABEL_MAKE, LABEL_NOTHING, LABEL_SHOT, ShotDataset
from shot_detector.model import MultiStageTCN
from shot_detector.shot_detector import load_shot_checkpoint, shot_events_from_frame_labels

CLASS_NAMES = {
    LABEL_NOTHING: "background",
    LABEL_SHOT: "shot",
    LABEL_MAKE: "make",
}
F1_THRESHOLDS = (10, 25, 50)


def _build_val_indices(n_total: int, val_fraction: float, seed: int) -> list[int]:
    indices = list(range(n_total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n_total * val_fraction))
    return indices[:n_val]


def _segments_from_events(labels: np.ndarray, class_id: int) -> list[tuple[int, int]]:
    """
    Build event segments via shot_events_from_frame_labels.
    - SHOT: full shot events [frame_start, frame_end]
    - MAKE: make sub-segments [make_start, make_end] for make events
    """
    events = shot_events_from_frame_labels(labels)
    if class_id == LABEL_SHOT:
        return [(int(e.frame_start), int(e.frame_end)) for e in events]
    if class_id == LABEL_MAKE:
        out: list[tuple[int, int]] = []
        for e in events:
            if e.is_make and e.make_start is not None and e.make_end is not None:
                out.append((int(e.make_start), int(e.make_end)))
        return out
    return []


def _segment_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    if inter <= 0:
        return 0.0
    len_a = a1 - a0 + 1
    len_b = b1 - b0 + 1
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0


def _match_segments_iou(
    gt_segments: list[tuple[int, int]],
    pred_segments: list[tuple[int, int]],
    overlap_percent: int,
) -> tuple[int, int, int]:
    """
    Segmental matching by IoU threshold (one GT segment can be matched once).
    Returns (tp, fp, fn).
    """
    thr = float(overlap_percent) / 100.0
    gt_used = np.zeros(len(gt_segments), dtype=bool)
    tp = 0
    fp = 0

    for pred in pred_segments:
        best_i = -1
        best_iou = 0.0
        for gi, gt in enumerate(gt_segments):
            iou = _segment_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_i = gi
        if best_i >= 0 and (best_iou >= thr) and not gt_used[best_i]:
            tp += 1
        else:
            fp += 1

    fn = int(len(gt_segments) - tp)
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


@torch.no_grad()
def benchmark(
    features_dir: Path,
    model_path: Path,
    *,
    court_type: str = "nba",
    val_fraction: float = 0.2,
    seed: int = 42,
    device: str | None = None,
) -> None:
    ct = CourtType.NBA if court_type.lower() == "nba" else CourtType.FIBA
    dataset = ShotDataset(features_dir, court_type=ct)
    n_total = len(dataset)
    if n_total == 0:
        raise ValueError(f"No samples found in {features_dir}")

    val_indices = _build_val_indices(n_total, val_fraction, seed)
    val_subset = Subset(dataset, val_indices)

    meta, state_dict = load_shot_checkpoint(model_path)
    model = MultiStageTCN(
        input_dim=meta["input_dim"],
        n_classes=meta["n_classes"],
        n_stages=meta["n_stages"],
        n_filters=meta["n_filters"],
        n_layers=meta["n_layers"],
    )
    model.load_state_dict(state_dict)
    use_device = device or get_device()
    model.to(use_device).eval()

    per_class_correct = np.zeros(3, dtype=np.int64)
    per_class_total = np.zeros(3, dtype=np.int64)
    f1_counts: dict[int, dict[int, np.ndarray]] = {
        LABEL_SHOT: {t: np.zeros(3, dtype=np.int64) for t in F1_THRESHOLDS},
        LABEL_MAKE: {t: np.zeros(3, dtype=np.int64) for t in F1_THRESHOLDS},
    }
    # f1_counts[class_id][threshold] = [tp, fp, fn]

    for i in range(len(val_subset)):
        features, labels, T = val_subset[i]
        features = features[:T].unsqueeze(0).permute(0, 2, 1).to(use_device)  # (1, D, T)
        mask = torch.ones((1, int(T)), dtype=torch.bool, device=use_device)
        logits = model(features, mask)[-1]
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64, copy=False)
        gt = labels[:T].cpu().numpy().astype(np.int64, copy=False)

        for c in (LABEL_NOTHING, LABEL_SHOT, LABEL_MAKE):
            c_mask = gt == c
            per_class_total[c] += int(c_mask.sum())
            per_class_correct[c] += int((preds[c_mask] == c).sum())

        for cls in (LABEL_SHOT, LABEL_MAKE):
            gt_segments = _segments_from_events(gt, cls)
            pred_segments = _segments_from_events(preds, cls)
            for thr in F1_THRESHOLDS:
                tp, fp, fn = _match_segments_iou(gt_segments, pred_segments, thr)
                f1_counts[cls][thr] += np.array([tp, fp, fn], dtype=np.int64)

    per_class_acc = np.where(
        per_class_total > 0,
        per_class_correct / per_class_total,
        0.0,
    )

    print(f"Validation split: total={n_total}, val={len(val_subset)}, val_fraction={val_fraction}, seed={seed}")
    print(f"Model: {model_path}")
    print("\nPer-class accuracy:")
    for c in (LABEL_NOTHING, LABEL_SHOT, LABEL_MAKE):
        print(
            f"  {CLASS_NAMES[c]:10s}  acc={per_class_acc[c]:.4f}  "
            f"correct={int(per_class_correct[c])}  total={int(per_class_total[c])}"
        )

    print("\nSegmental F1 by IoU overlap threshold:")
    for cls in (LABEL_SHOT, LABEL_MAKE):
        print(f"  Class: {CLASS_NAMES[cls]}")
        for thr in F1_THRESHOLDS:
            tp, fp, fn = (int(x) for x in f1_counts[cls][thr])
            print(f"    F1@{thr:>2d}: {_f1(tp, fp, fn):.4f}  (tp={tp}, fp={fp}, fn={fn})")


def _parse_args() -> argparse.Namespace:
    default_features = Path(__file__).resolve().parent / "dataset_features"
    default_model = Path(__file__).resolve().parent / "trained_model" / "shot_detection_model.pt"
    p = argparse.ArgumentParser(description="Benchmark shot detector on validation split")
    p.add_argument("--features-dir", type=Path, default=default_features)
    p.add_argument("--model", type=Path, default=default_model, help="Path to shot checkpoint (.pt)")
    p.add_argument("--court-type", type=str, default="nba", choices=["nba", "fiba"])
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42, help="Validation split seed (same logic as train)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    benchmark(
        features_dir=args.features_dir,
        model_path=args.model,
        court_type=args.court_type,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
    )
