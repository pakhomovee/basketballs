"""
Bbox detection metrics (players, ball, rim) on the test split of a YOLO-format dataset.

Accepts path to dataset (data.yaml or root dir containing data.yaml) and optional model path.
Computes metrics on the test split (test/) for class "player" (classes 2–8), "ball" (class 0), or "rim".
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from common.classes.detections import FrameDetections
from .detector import Detector, get_frame_players_detections, get_frame_ball_detections, get_frame_rim_detections


PLAYER_CLASS_NAME = "player"
BALL_CLASS_NAME = "ball"
RIM_CLASS_NAME = "rim"

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


def _rim_class_id(data_yaml: Path) -> int:
    """Index of class 'rim' in names from data.yaml."""
    return _class_id_from_yaml(data_yaml, RIM_CLASS_NAME)


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

    for img_path in tqdm(image_paths, desc="Metrics: test images", unit="img"):
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


def load_test_gt_and_predictions_rfdetr_players(
    dataset_path: str | Path,
    model_path: str | Path | None,
    conf_threshold: float,
) -> tuple[list[list[tuple[int, int, int, int]]], list[list[tuple[int, int, int, int, float]]], int]:
    """
    Load test split, run RF-DETR detector, return GT and predictions per frame for players.
    """
    from .detector_rfdetr import DetectorRFDETR, get_frame_players_detections as get_frame_players_detections_rfdetr

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
        detector = DetectorRFDETR(model_path=str(model_path), conf_threshold=conf_threshold)
    else:
        detector = DetectorRFDETR(conf_threshold=conf_threshold)

    all_gt_boxes: list[list[tuple[int, int, int, int]]] = []
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]] = []

    for img_path in tqdm(image_paths, desc="Metrics: test images", unit="img"):
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
                    # Player classes in dataset labels for the old taxonomy: 2..8.
                    # RF-DETR player extraction later handles detector-side class mapping.
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
        players = get_frame_players_detections_rfdetr(frame_detections, conf_threshold=conf_threshold)
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

    for img_path in tqdm(image_paths, desc="Metrics: test images", unit="img"):
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


def load_test_gt_and_predictions_ball_rfdetr(
    dataset_path: str | Path,
    model_path: str | Path | None,
    conf_threshold: float,
    ball_cls_id: int = 0,
) -> tuple[list[list[tuple[int, int, int, int]]], list[list[tuple[int, int, int, int, float]]], int]:
    """
    Load test split, run RF-DETR detector, return GT and predictions per frame for ball.
    """
    from .detector_rfdetr import DetectorRFDETR, get_frame_ball_detections as get_frame_ball_detections_rfdetr

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
        detector = DetectorRFDETR(model_path=str(model_path), conf_threshold=conf_threshold)
    else:
        detector = DetectorRFDETR(conf_threshold=conf_threshold)

    all_gt_boxes: list[list[tuple[int, int, int, int]]] = []
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]] = []

    for img_path in tqdm(image_paths, desc="Metrics: test images", unit="img"):
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
        balls = get_frame_ball_detections_rfdetr(frame_detections, conf_threshold=conf_threshold)
        pred_boxes = [(b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3], b.confidence) for b in balls]

        all_gt_boxes.append(gt_boxes)
        all_pred_boxes.append(pred_boxes)

    return all_gt_boxes, all_pred_boxes, len(image_paths)


def load_test_gt_and_predictions_rim(
    dataset_path: str | Path,
    model_path: str | Path | None,
    conf_threshold: float,
    rim_cls_id: int | None = None,
) -> tuple[list[list[tuple[int, int, int, int]]], list[list[tuple[int, int, int, int, float]]], int]:
    """
    Load test split, run detector, return GT and predictions per frame for rim.

    Returns:
        (all_gt_boxes, all_pred_boxes, n_images) — rim class only.
    """
    data_yaml = _resolve_dataset_path(dataset_path)
    root = data_yaml.parent.resolve()
    test_images_dir = root / "test" / "images"
    test_labels_dir = root / "test" / "labels"
    if not test_images_dir.is_dir() or not test_labels_dir.is_dir():
        raise FileNotFoundError(f"test/images or test/labels not found in dataset: {root}")
    if rim_cls_id is None:
        rim_cls_id = _rim_class_id(data_yaml)

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

    for img_path in tqdm(image_paths, desc="Metrics: test images", unit="img"):
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
                    if cls != rim_cls_id:
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
        rims = get_frame_rim_detections(frame_detections, conf_threshold=conf_threshold)
        pred_boxes = [(r.x1, r.y1, r.x2, r.y2, r.confidence) for r in rims]

        all_gt_boxes.append(gt_boxes)
        all_pred_boxes.append(pred_boxes)

    return all_gt_boxes, all_pred_boxes, len(image_paths)


def load_test_gt_and_predictions_rim_rfdetr(
    dataset_path: str | Path,
    model_path: str | Path | None,
    conf_threshold: float,
    rim_cls_id: int | None = None,
) -> tuple[list[list[tuple[int, int, int, int]]], list[list[tuple[int, int, int, int, float]]], int]:
    """
    Load test split, run RF-DETR detector, return GT and predictions per frame for rim.
    """
    from .detector_rfdetr import DetectorRFDETR, get_frame_rim_detections as get_frame_rim_detections_rfdetr

    data_yaml = _resolve_dataset_path(dataset_path)
    root = data_yaml.parent.resolve()
    test_images_dir = root / "test" / "images"
    test_labels_dir = root / "test" / "labels"
    if not test_images_dir.is_dir() or not test_labels_dir.is_dir():
        raise FileNotFoundError(f"test/images or test/labels not found in dataset: {root}")
    if rim_cls_id is None:
        rim_cls_id = _rim_class_id(data_yaml)

    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(p for p in test_images_dir.iterdir() if p.suffix.lower() in ext)
    if not image_paths:
        raise FileNotFoundError(f"No images in {test_images_dir}")

    if model_path is not None:
        detector = DetectorRFDETR(model_path=str(model_path), conf_threshold=conf_threshold)
    else:
        detector = DetectorRFDETR(conf_threshold=conf_threshold)

    all_gt_boxes: list[list[tuple[int, int, int, int]]] = []
    all_pred_boxes: list[list[tuple[int, int, int, int, float]]] = []

    for img_path in tqdm(image_paths, desc="Metrics: test images", unit="img"):
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
                    if cls != rim_cls_id:
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
        rims = get_frame_rim_detections_rfdetr(frame_detections, conf_threshold=conf_threshold)
        pred_boxes = [(r.x1, r.y1, r.x2, r.y2, r.confidence) for r in rims]

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
    for t in np.linspace(0, 1, 101):
        mask = [r >= t for r in recalls]
        if not any(mask):
            continue
        ap += max(p for p, m in zip(precisions, mask) if m)
    return ap / 101


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
    conf_threshold: float = 0.001,
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


def compute_player_detection_metrics_rfdetr(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.5,
) -> PlayerDetectionMetrics:
    """
    Compute player detection metrics on the test split using RF-DETR backend.
    """
    all_gt_boxes, all_pred_boxes, n_images = load_test_gt_and_predictions_rfdetr_players(
        dataset_path, model_path, conf_threshold
    )
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


class RimDetectionMetrics(NamedTuple):
    """Rim detection metrics."""

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
    conf_threshold: float = 0.001,
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


def compute_ball_detection_metrics_rfdetr(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.5,
) -> BallDetectionMetrics:
    """
    Compute ball detection metrics on the test split using RF-DETR backend.
    """
    all_gt_boxes, all_pred_boxes, n_images = load_test_gt_and_predictions_ball_rfdetr(
        dataset_path, model_path, conf_threshold
    )
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


def compute_rim_detection_metrics(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.5,
) -> RimDetectionMetrics:
    """
    Compute rim detection metrics on the test split (P, R, F1, mAP50, mAP50-95).
    """
    all_gt_boxes, all_pred_boxes, n_images = load_test_gt_and_predictions_rim(dataset_path, model_path, conf_threshold)
    n_gt = sum(len(g) for g in all_gt_boxes)
    n_pred = sum(len(p) for p in all_pred_boxes)

    tp, fp, fn = compute_tp_fp_fn(all_gt_boxes, all_pred_boxes, iou_threshold)
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
    map50 = compute_map50(all_gt_boxes, all_pred_boxes)
    map50_95 = compute_map50_95(all_gt_boxes, all_pred_boxes)

    return RimDetectionMetrics(
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


def compute_rim_detection_metrics_rfdetr(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.5,
) -> RimDetectionMetrics:
    """
    Compute rim detection metrics on the test split using RF-DETR backend.
    """
    all_gt_boxes, all_pred_boxes, n_images = load_test_gt_and_predictions_rim_rfdetr(
        dataset_path, model_path, conf_threshold
    )
    n_gt = sum(len(g) for g in all_gt_boxes)
    n_pred = sum(len(p) for p in all_pred_boxes)

    tp, fp, fn = compute_tp_fp_fn(all_gt_boxes, all_pred_boxes, iou_threshold)
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
    map50 = compute_map50(all_gt_boxes, all_pred_boxes)
    map50_95 = compute_map50_95(all_gt_boxes, all_pred_boxes)

    return RimDetectionMetrics(
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


def print_rim_metrics(metrics: RimDetectionMetrics) -> None:
    """Print rim detection metrics to console."""
    _print_detection_metrics(metrics, "rim")


class YoloBuiltinMetrics(NamedTuple):
    """Metrics computed with ultralytics ap_per_class (same algorithm as model.val)."""

    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    n_images: int
    n_gt: int
    n_pred: int


def compute_player_metrics_yolo_builtin(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    split: str = "test",
    conf_threshold: float = 0.001,
) -> YoloBuiltinMetrics:
    """
    Compute player metrics exactly as YOLO does, but using our detection pipeline.

    Runs the multi-class detector on each image, extracts player boxes via
    ``get_frame_players_detections`` (classes 2-8, NMS, cap at 10), then evaluates
    against GT from a single-class player dataset using ``ap_per_class`` and
    ``match_predictions`` from ``ultralytics.utils.metrics``.
    """
    import torch
    from ultralytics.utils.metrics import ap_per_class, box_iou

    data_yaml = _resolve_dataset_path(dataset_path)
    root = data_yaml.parent.resolve()
    split_images_dir = root / split / "images"
    split_labels_dir = root / split / "labels"
    if not split_images_dir.is_dir() or not split_labels_dir.is_dir():
        raise FileNotFoundError(f"{split}/images or {split}/labels not found in dataset: {root}")

    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(p for p in split_images_dir.iterdir() if p.suffix.lower() in ext)
    if not image_paths:
        raise FileNotFoundError(f"No images in {split_images_dir}")

    if model_path is not None:
        detector = Detector(model_path=str(model_path), conf_threshold=conf_threshold)
    else:
        detector = Detector(conf_threshold=conf_threshold)

    iou_thresholds = torch.linspace(0.5, 0.95, 10)

    all_tp: list[torch.Tensor] = []
    all_conf: list[torch.Tensor] = []
    all_pred_cls: list[torch.Tensor] = []
    all_target_cls: list[torch.Tensor] = []
    total_pred = 0

    for img_path in tqdm(image_paths, desc="Metrics (YOLO built-in): test images", unit="img"):
        lbl_path = split_labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes_list: list[list[float]] = []
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
                        x_c, y_c, bw, bh = map(float, parts[1:5])
                    except (ValueError, IndexError):
                        continue
                    x1 = (x_c - bw / 2) * w
                    y1 = (y_c - bh / 2) * h
                    x2 = (x_c + bw / 2) * w
                    y2 = (y_c + bh / 2) * h
                    gt_boxes_list.append([x1, y1, x2, y2])

        detections = detector.detect_frame(img)
        frame_detections = FrameDetections(frame_id=0, detections=detections)
        players = get_frame_players_detections(frame_detections, conf_threshold=conf_threshold)

        n_gt = len(gt_boxes_list)
        n_pred = len(players)
        total_pred += n_pred

        if n_gt > 0:
            all_target_cls.append(torch.zeros(n_gt))

        if n_pred == 0:
            continue

        pred_boxes = torch.tensor(
            [[p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]] for p in players],
            dtype=torch.float32,
        )
        pred_conf = torch.tensor([p.confidence for p in players], dtype=torch.float32)
        pred_cls = torch.zeros(n_pred)

        if n_gt == 0:
            all_tp.append(torch.zeros(n_pred, len(iou_thresholds), dtype=torch.bool))
            all_conf.append(pred_conf)
            all_pred_cls.append(pred_cls)
            continue

        gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32)
        gt_cls = torch.zeros(n_gt)

        # box_iou returns (n_pred, n_gt); YOLO matching expects (n_gt, n_pred)
        iou_matrix = box_iou(gt_boxes, pred_boxes)

        correct = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
        correct_class = (gt_cls[:, None] == pred_cls).cpu().numpy()
        iou_np = (iou_matrix * torch.tensor(correct_class, dtype=iou_matrix.dtype)).cpu().numpy()
        for t_idx, threshold in enumerate(iou_thresholds.tolist()):
            matches = np.nonzero(iou_np >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), t_idx] = True
        correct = torch.tensor(correct, dtype=torch.bool)

        all_tp.append(correct)
        all_conf.append(pred_conf)
        all_pred_cls.append(pred_cls)

    if not all_conf:
        return YoloBuiltinMetrics(0.0, 0.0, 0.0, 0.0, 0.0, len(image_paths), 0, 0)

    tp = torch.cat(all_tp)
    conf = torch.cat(all_conf)
    pred_cls = torch.cat(all_pred_cls)
    target_cls = torch.cat(all_target_cls) if all_target_cls else torch.zeros(0)

    results = ap_per_class(
        tp.numpy(), conf.numpy(), pred_cls.numpy(), target_cls.numpy(),
        names={0: "player"},
    )
    p, r, f1_arr, ap = results[2], results[3], results[4], results[5]

    mp = float(p.mean()) if len(p) else 0.0
    mr = float(r.mean()) if len(r) else 0.0
    mf1 = float(f1_arr.mean()) if len(f1_arr) else 0.0
    map50 = float(ap[:, 0].mean()) if ap.shape[0] > 0 else 0.0
    map50_95 = float(ap.mean()) if ap.size > 0 else 0.0
    n_gt_total = int(target_cls.shape[0])

    return YoloBuiltinMetrics(
        precision=mp,
        recall=mr,
        f1=mf1,
        map50=map50,
        map50_95=map50_95,
        n_images=len(image_paths),
        n_gt=n_gt_total,
        n_pred=total_pred,
    )


def compute_player_united_metrics_yolo_builtin(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    split: str = "test",
    conf_threshold: float = 0.001,
    gt_class_ids: tuple[int, ...] = (2, 3),
    pred_class_ids: tuple[int, ...] = (2, 3),
) -> YoloBuiltinMetrics:
    """
    Compute merged-player metrics using YOLO matching algorithm.

    Takes a multi-class dataset (e.g. ball, number, player, player-dribble, referee, rim),
    merges specified GT classes and prediction classes into a single "player" evaluation.

    Uses raw detector output with class filtering only (no NMS, no cap).

    Args:
        gt_class_ids: label indices in the dataset that count as "player" GT.
        pred_class_ids: model class indices whose predictions count as "player".
    """
    import torch
    from ultralytics.utils.metrics import ap_per_class, box_iou

    data_yaml = _resolve_dataset_path(dataset_path)
    root = data_yaml.parent.resolve()
    split_images_dir = root / split / "images"
    split_labels_dir = root / split / "labels"
    if not split_images_dir.is_dir() or not split_labels_dir.is_dir():
        raise FileNotFoundError(f"{split}/images or {split}/labels not found in dataset: {root}")

    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(p for p in split_images_dir.iterdir() if p.suffix.lower() in ext)
    if not image_paths:
        raise FileNotFoundError(f"No images in {split_images_dir}")

    if model_path is not None:
        detector = Detector(model_path=str(model_path), conf_threshold=conf_threshold)
    else:
        detector = Detector(conf_threshold=conf_threshold)

    gt_class_set = set(gt_class_ids)
    pred_class_set = set(pred_class_ids)
    iou_thresholds = torch.linspace(0.5, 0.95, 10)

    all_tp: list[torch.Tensor] = []
    all_conf: list[torch.Tensor] = []
    all_pred_cls: list[torch.Tensor] = []
    all_target_cls: list[torch.Tensor] = []
    total_pred = 0

    for img_path in tqdm(image_paths, desc="Metrics (player-united): test images", unit="img"):
        lbl_path = split_labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes_list: list[list[float]] = []
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
                    if cls not in gt_class_set:
                        continue
                    x1 = (x_c - bw / 2) * w
                    y1 = (y_c - bh / 2) * h
                    x2 = (x_c + bw / 2) * w
                    y2 = (y_c + bh / 2) * h
                    gt_boxes_list.append([x1, y1, x2, y2])

        detections = detector.detect_frame(img)
        player_dets = [
            d for d in detections
            if d.class_id in pred_class_set and d.confidence >= conf_threshold
        ]
        player_dets.sort(key=lambda d: -d.confidence)

        n_gt = len(gt_boxes_list)
        n_pred = len(player_dets)
        total_pred += n_pred

        if n_gt > 0:
            all_target_cls.append(torch.zeros(n_gt))

        if n_pred == 0:
            continue

        pred_boxes = torch.tensor(
            [[d.x1, d.y1, d.x2, d.y2] for d in player_dets],
            dtype=torch.float32,
        )
        pred_conf = torch.tensor([d.confidence for d in player_dets], dtype=torch.float32)
        pred_cls = torch.zeros(n_pred)

        if n_gt == 0:
            all_tp.append(torch.zeros(n_pred, len(iou_thresholds), dtype=torch.bool))
            all_conf.append(pred_conf)
            all_pred_cls.append(pred_cls)
            continue

        gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32)
        gt_cls = torch.zeros(n_gt)

        iou_matrix = box_iou(gt_boxes, pred_boxes)

        correct = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
        correct_class = (gt_cls[:, None] == pred_cls).cpu().numpy()
        iou_np = (iou_matrix * torch.tensor(correct_class, dtype=iou_matrix.dtype)).cpu().numpy()
        for t_idx, threshold in enumerate(iou_thresholds.tolist()):
            matches = np.nonzero(iou_np >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), t_idx] = True
        correct = torch.tensor(correct, dtype=torch.bool)

        all_tp.append(correct)
        all_conf.append(pred_conf)
        all_pred_cls.append(pred_cls)

    if not all_conf:
        return YoloBuiltinMetrics(0.0, 0.0, 0.0, 0.0, 0.0, len(image_paths), 0, 0)

    tp = torch.cat(all_tp)
    conf = torch.cat(all_conf)
    pred_cls = torch.cat(all_pred_cls)
    target_cls = torch.cat(all_target_cls) if all_target_cls else torch.zeros(0)

    results = ap_per_class(
        tp.numpy(), conf.numpy(), pred_cls.numpy(), target_cls.numpy(),
        names={0: "player-united"},
    )
    p, r, f1_arr, ap = results[2], results[3], results[4], results[5]

    mp = float(p.mean()) if len(p) else 0.0
    mr = float(r.mean()) if len(r) else 0.0
    mf1 = float(f1_arr.mean()) if len(f1_arr) else 0.0
    map50 = float(ap[:, 0].mean()) if ap.shape[0] > 0 else 0.0
    map50_95 = float(ap.mean()) if ap.size > 0 else 0.0
    n_gt_total = int(target_cls.shape[0])

    return YoloBuiltinMetrics(
        precision=mp,
        recall=mr,
        f1=mf1,
        map50=map50,
        map50_95=map50_95,
        n_images=len(image_paths),
        n_gt=n_gt_total,
        n_pred=total_pred,
    )


def print_yolo_builtin_metrics(metrics: YoloBuiltinMetrics) -> None:
    """Print YOLO built-in metrics to console."""
    print(f"\n{'=' * 50}")
    print("Player detection metrics (YOLO built-in)")
    print(f"{'=' * 50}")
    print(f"  Images         : {metrics.n_images}")
    print(f"  GT boxes       : {metrics.n_gt}")
    print(f"  Pred boxes     : {metrics.n_pred}")
    print(f"  Precision      : {metrics.precision:.4f}")
    print(f"  Recall         : {metrics.recall:.4f}")
    print(f"  F1             : {metrics.f1:.4f}")
    print(f"  mAP@0.5        : {metrics.map50:.4f}")
    print(f"  mAP@0.5:0.95   : {metrics.map50_95:.4f}")
    print(f"{'=' * 50}\n")


def _print_detection_metrics(
    metrics: PlayerDetectionMetrics | BallDetectionMetrics | RimDetectionMetrics,
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

    parser = argparse.ArgumentParser(description="Detection metrics (players, ball, rim) on YOLO dataset test split")
    parser.add_argument("dataset", type=str, nargs="?", default=None, help="Path to data.yaml or dataset root")
    parser.add_argument("--ball", "-b", action="store_true", help="Compute ball metrics (default: players)")
    parser.add_argument("--rim", "-r", action="store_true", help="Compute rim metrics (default: players)")
    parser.add_argument("--rfdetr", action="store_true", help="Use RF-DETR backend (players/ball/rim)")
    parser.add_argument(
        "--yolo-builtin", action="store_true",
        help="Use YOLO matching for player metrics (single-class player dataset)",
    )
    parser.add_argument(
        "--player-united", action="store_true",
        help="Merge player + player-dribble as one class (multi-class dataset)",
    )
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to detector checkpoint")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for P/R")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (for --yolo-builtin / --player-united)")
    parser.add_argument(
        "--gt-classes", type=str, default=None,
        help="Comma-separated GT class ids to merge as player (for --player-united, default: 2,3)",
    )
    parser.add_argument(
        "--pred-classes", type=str, default=None,
        help="Comma-separated pred class ids to merge as player (for --player-united, default: 2,3)",
    )
    args = parser.parse_args()

    dataset = args.dataset

    if args.player_united:
        gt_ids = tuple(int(x) for x in args.gt_classes.split(",")) if args.gt_classes else (2, 3)
        pred_ids = tuple(int(x) for x in args.pred_classes.split(",")) if args.pred_classes else (2, 3)
        metrics = compute_player_united_metrics_yolo_builtin(
            dataset,
            model_path=args.model,
            split=args.split,
            conf_threshold=args.conf,
            gt_class_ids=gt_ids,
            pred_class_ids=pred_ids,
        )
        print_yolo_builtin_metrics(metrics)
        raise SystemExit(0)

    if args.yolo_builtin:
        metrics = compute_player_metrics_yolo_builtin(
            dataset,
            model_path=args.model,
            split=args.split,
            conf_threshold=args.conf,
        )
        print_yolo_builtin_metrics(metrics)
        raise SystemExit(0)

    if args.rim:
        if args.rfdetr:
            metrics = compute_rim_detection_metrics_rfdetr(
                dataset,
                model_path=args.model,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
        else:
            metrics = compute_rim_detection_metrics(
                dataset,
                model_path=args.model,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
        print_rim_metrics(metrics)
    elif args.ball:
        if args.rfdetr:
            metrics = compute_ball_detection_metrics_rfdetr(
                dataset,
                model_path=args.model,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
        else:
            metrics = compute_ball_detection_metrics(
                dataset,
                model_path=args.model,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
        print_ball_metrics(metrics)
    else:
        if args.rfdetr:
            metrics = compute_player_detection_metrics_rfdetr(
                dataset,
                model_path=args.model,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
        else:
            metrics = compute_player_detection_metrics(
                dataset,
                model_path=args.model,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
            )
        print_metrics(metrics)
