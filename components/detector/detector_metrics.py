"""
Bbox detection metrics (players and ball) on the test split of a YOLO-format dataset.

Accepts path to dataset (data.yaml or root dir containing data.yaml) and optional model path.
Computes metrics on the test split (test/) for class "player" (classes 2–8) or "ball" (class 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import yaml

from common.classes.detections import FrameDetections
from .detector import Detector, get_frame_players_detections, get_frame_ball_detections


PLAYER_CLASS_NAME = "player"
BALL_CLASS_NAME = "ball"

# For mAP50-95: IoU thresholds from 0.5 to 0.95 step 0.05 (COCO-style)
IOU_THRESHOLDS_MAP50_95 = (0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def _resolve_dataset_path(dataset_path: str | Path) -> Path:
    """Resolve path to data.yaml: if a directory is given, look for data.yaml inside it."""
    p = Path(dataset_path).expanduser().resolve()
    if p.is_file():
        return p
    if (p / "data.yaml").exists():
        return p / "data.yaml"
    raise FileNotFoundError(f"data.yaml not found: {p} or {p / 'data.yaml'}")


def _class_id_from_yaml(data_yaml: Path, class_name: str) -> int:
    """Index of class class_name in names from data.yaml."""
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names") or []
    if isinstance(names, dict):
        for k, v in names.items():
            if v == class_name:
                return int(k)
        raise ValueError(f"Class '{class_name}' not found in names: {names}")
    try:
        return names.index(class_name)
    except ValueError:
        raise ValueError(f"Class '{class_name}' not found in names: {names}")


def _player_class_id(data_yaml: Path) -> int:
    """Index of class 'player' in names from data.yaml."""
    return _class_id_from_yaml(data_yaml, PLAYER_CLASS_NAME)


def _ball_class_id(data_yaml: Path) -> int:
    """Index of class 'ball' in names from data.yaml."""
    return _class_id_from_yaml(data_yaml, BALL_CLASS_NAME)


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """IoU of two bboxes in format (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    total = area_a + area_b - inter
    return inter / total if total > 0 else 0.0


def load_test_gt_and_predictions(
    dataset_path: str | Path,
    model_path: str | Path | None,
    conf_threshold: float,
    player_cls_id: int | None = None,
) -> tuple[list[list[tuple[int, int, int, int]]], list[list[tuple[int, int, int, int, float]]], int]:
    """
    Load test split, run detector, return GT and predictions per frame.

    Returns:
        (all_gt_boxes, all_pred_boxes, n_images)
        - all_gt_boxes[i]: list of GT bbox (x1,y1,x2,y2) for i-th image (player class only).
        - all_pred_boxes[i]: list of pred (x1,y1,x2,y2, conf) for i-th image.
    """
    data_yaml = _resolve_dataset_path(dataset_path)
    root = data_yaml.parent.resolve()
    test_images_dir = root / "test" / "images"
    test_labels_dir = root / "test" / "labels"
    if not test_images_dir.is_dir() or not test_labels_dir.is_dir():
        raise FileNotFoundError(f"test/images or test/labels not found in dataset: {root}")
    if player_cls_id is None:
        player_cls_id = _player_class_id(data_yaml)

    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(p for p in test_images_dir.iterdir() if p.suffix.lower() in ext)
    if not image_paths:
        raise FileNotFoundError(f"No images in {test_images_dir}")

    if model_path is not None:
        detector = Detector(model_path=str(model_path), conf_threshold=conf_threshold)
    else:
        detector = Detector(conf_threshold=conf_threshold)

    all_gt_boxes: list[list[tuple[int, int, int, int]]] = []
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]] = []

    for img_path in image_paths:
        lbl_path = test_labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes: list[tuple[int, int, int, int]] = []
        if lbl_path.is_file():
            with open(lbl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(parts[0])
                        x_c, y_c, bw, bh = map(float, parts[1:5])
                    except (ValueError, IndexError):
                        continue
                    if cls < 2 or cls > 8:
                        continue
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        gt_boxes.append((x1, y1, x2, y2))

        detections = detector.detect_frame(img)
        frame_detections = FrameDetections(frame_id=0, detections=detections)
        players = get_frame_players_detections(frame_detections, conf_threshold=conf_threshold)
        pred_boxes = [(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3], p.confidence) for p in players]

        all_gt_boxes.append(gt_boxes)
        all_pred_boxes.append(pred_boxes)

    return all_gt_boxes, all_pred_boxes, len(image_paths)


def load_test_gt_and_predictions_ball(
    dataset_path: str | Path,
    model_path: str | Path | None,
    conf_threshold: float,
    ball_cls_id: int = 0,
) -> tuple[list[list[tuple[int, int, int, int]]], list[list[tuple[int, int, int, int, float]]], int]:
    """
    Load test split, run detector, return GT and predictions per frame for ball.

    Returns:
        (all_gt_boxes, all_pred_boxes, n_images) — ball class only (index from data.yaml).
    """
    data_yaml = _resolve_dataset_path(dataset_path)
    root = data_yaml.parent.resolve()
    test_images_dir = root / "test" / "images"
    test_labels_dir = root / "test" / "labels"
    if not test_images_dir.is_dir() or not test_labels_dir.is_dir():
        raise FileNotFoundError(f"test/images or test/labels not found in dataset: {root}")

    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(p for p in test_images_dir.iterdir() if p.suffix.lower() in ext)
    if not image_paths:
        raise FileNotFoundError(f"No images in {test_images_dir}")

    if model_path is not None:
        detector = Detector(model_path=str(model_path), conf_threshold=conf_threshold)
    else:
        detector = Detector(conf_threshold=conf_threshold)

    all_gt_boxes: list[list[tuple[int, int, int, int]]] = []
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]] = []

    for img_path in image_paths:
        lbl_path = test_labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes: list[tuple[int, int, int, int]] = []
        if lbl_path.is_file():
            with open(lbl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(parts[0])
                        x_c, y_c, bw, bh = map(float, parts[1:5])
                    except (ValueError, IndexError):
                        continue
                    if cls != ball_cls_id:
                        continue
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
                    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        gt_boxes.append((x1, y1, x2, y2))

        detections = detector.detect_frame(img)
        frame_detections = FrameDetections(frame_id=0, detections=detections)
        balls = get_frame_ball_detections(frame_detections, conf_threshold=conf_threshold)
        # dets_ball = [d for d in detections if d.class_id == ball_cls_id and d.confidence >= conf_threshold]
        pred_boxes = [(b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], b.confidence) for b in balls]

        all_gt_boxes.append(gt_boxes)
        all_pred_boxes.append(pred_boxes)

    return all_gt_boxes, all_pred_boxes, len(image_paths)


def compute_tp_fp_fn(
    all_gt_boxes: list[list[tuple[int, int, int, int]]],
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """
    Match predictions to GT by IoU. One GT — at most one pred.

    Returns:
        (tp, fp, fn)
    """
    tp, fp, fn = 0, 0, 0
    for gt_list, pred_list in zip(all_gt_boxes, all_pred_boxes):
        pred_sorted = sorted(pred_list, key=lambda x: x[4], reverse=True)
        matched_gt = set()
        for pred in pred_sorted:
            pred_box = (pred[0], pred[1], pred[2], pred[3])
            best_iou, best_gt_idx = 0.0, -1
            for gt_idx, gt_box in enumerate(gt_list):
                if gt_idx in matched_gt:
                    continue
                iou = _iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        fn += len(gt_list) - len(matched_gt)
    return tp, fp, fn


def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, Recall, F1 from TP, FP, FN."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _ap_at_iou(
    all_gt_boxes: list[list[tuple[int, int, int, int]]],
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]],
    iou_threshold: float,
    n_gt: int,
) -> float:
    """
    Average Precision at a single IoU threshold (11-point interpolated).
    all_pred_boxes is per-frame; we sort all predictions by conf and build P-R curve.
    """
    all_preds: list[tuple[int, int, int, int, float, int]] = []
    for idx, pred_list in enumerate(all_pred_boxes):
        for x1, y1, x2, y2, conf in pred_list:
            all_preds.append((x1, y1, x2, y2, conf, idx))
    all_preds.sort(key=lambda x: x[4], reverse=True)

    gt_matched_by_img = [set() for _ in all_gt_boxes]
    tp_cumul, fp_cumul = 0, 0
    precisions: list[float] = []
    recalls: list[float] = []

    for px1, py1, px2, py2, _, img_idx in all_preds:
        pred_box = (px1, py1, px2, py2)
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx, gt_box in enumerate(all_gt_boxes[img_idx]):
            if gt_idx in gt_matched_by_img[img_idx]:
                continue
            iou = _iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_cumul += 1
            gt_matched_by_img[img_idx].add(best_gt_idx)
        else:
            fp_cumul += 1
        p = tp_cumul / (tp_cumul + fp_cumul) if (tp_cumul + fp_cumul) > 0 else 0.0
        r = tp_cumul / n_gt if n_gt > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    if n_gt == 0 or not precisions:
        return 0.0
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = [r >= t for r in recalls]
        if not any(mask):
            continue
        ap += max(p for p, m in zip(precisions, mask) if m)
    return ap / 11


def compute_map50(
    all_gt_boxes: list[list[tuple[int, int, int, int]]],
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]],
) -> float:
    """mAP at IoU = 0.5 (single threshold)."""
    n_gt = sum(len(g) for g in all_gt_boxes)
    return _ap_at_iou(all_gt_boxes, all_pred_boxes, 0.5, n_gt)


def compute_map50_95(
    all_gt_boxes: list[list[tuple[int, int, int, int]]],
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]],
) -> float:
    """mAP@0.5:0.95 — mean AP over IoU 0.5, 0.55, ..., 0.95 (10 thresholds, COCO-style)."""
    n_gt = sum(len(g) for g in all_gt_boxes)
    aps = [_ap_at_iou(all_gt_boxes, all_pred_boxes, iou_t, n_gt) for iou_t in IOU_THRESHOLDS_MAP50_95]
    return float(np.mean(aps))


class PlayerDetectionMetrics(NamedTuple):
    """Player detection metrics."""

    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    n_images: int
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int


def compute_player_detection_metrics(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> PlayerDetectionMetrics:
    """
    Compute all player detection metrics on the test split.

    Loads data and predictions, then calls individual functions for
    precision/recall/F1, mAP50 and mAP50-95.
    """
    all_gt_boxes, all_pred_boxes, n_images = load_test_gt_and_predictions(dataset_path, model_path, conf_threshold)
    n_gt = sum(len(g) for g in all_gt_boxes)
    n_pred = sum(len(p) for p in all_pred_boxes)

    tp, fp, fn = compute_tp_fp_fn(all_gt_boxes, all_pred_boxes, iou_threshold)
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
    map50 = compute_map50(all_gt_boxes, all_pred_boxes)
    map50_95 = compute_map50_95(all_gt_boxes, all_pred_boxes)

    return PlayerDetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        map50=map50,
        map50_95=map50_95,
        n_images=n_images,
        n_gt=n_gt,
        n_pred=n_pred,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def print_metrics(metrics: PlayerDetectionMetrics) -> None:
    """Print player detection metrics to console."""
    _print_detection_metrics(metrics, "players (player)")


class BallDetectionMetrics(NamedTuple):
    """Ball detection metrics."""

    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    n_images: int
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int


def compute_ball_detection_metrics(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> BallDetectionMetrics:
    """
    Compute ball detection metrics on the test split (same metrics: P, R, F1, mAP50, mAP50-95).
    """
    all_gt_boxes, all_pred_boxes, n_images = load_test_gt_and_predictions_ball(dataset_path, model_path, conf_threshold)
    n_gt = sum(len(g) for g in all_gt_boxes)
    n_pred = sum(len(p) for p in all_pred_boxes)

    tp, fp, fn = compute_tp_fp_fn(all_gt_boxes, all_pred_boxes, iou_threshold)
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
    map50 = compute_map50(all_gt_boxes, all_pred_boxes)
    map50_95 = compute_map50_95(all_gt_boxes, all_pred_boxes)

    return BallDetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        map50=map50,
        map50_95=map50_95,
        n_images=n_images,
        n_gt=n_gt,
        n_pred=n_pred,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def print_ball_metrics(metrics: BallDetectionMetrics) -> None:
    """Print ball detection metrics to console."""
    _print_detection_metrics(metrics, "ball")


def _print_detection_metrics(
    metrics: PlayerDetectionMetrics | BallDetectionMetrics,
    title: str,
) -> None:
    """Common routine to print metrics to console."""
    print(f"=== Detection metrics: {title} ===")
    print(f"  Images:     {metrics.n_images}")
    print(f"  GT boxes:   {metrics.n_gt}")
    print(f"  Pred boxes: {metrics.n_pred}")
    print(f"  TP / FP / FN: {metrics.tp} / {metrics.fp} / {metrics.fn}")
    print(f"  Precision:  {metrics.precision:.4f}")
    print(f"  Recall:     {metrics.recall:.4f}")
    print(f"  F1:         {metrics.f1:.4f}")
    print(f"  mAP@0.5:    {metrics.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.map50_95:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detection metrics (players or ball) on YOLO dataset test split")
    parser.add_argument("dataset", type=str, nargs="?", default=None, help="Path to data.yaml or dataset root")
    parser.add_argument("--ball", "-b", action="store_true", help="Compute ball metrics (default: players)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to .pt model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for P/R")
    args = parser.parse_args()

    dataset = args.dataset
    if args.ball:
        metrics = compute_ball_detection_metrics(
            dataset,
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        print_ball_metrics(metrics)
    else:
        metrics = compute_player_detection_metrics(
            dataset,
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        print_metrics(metrics)
