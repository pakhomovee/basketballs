"""
Detection metrics using built-in library evaluation:

* COCO dataset  → RF-DETR predict + pycocotools COCOeval
* YOLO dataset  → ultralytics model.val(split="test")
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# COCO format  →  RF-DETR + pycocotools
# ---------------------------------------------------------------------------

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_rfdetr_coco(
    annotations_json: str | Path,
    images_root: str | Path,
    model_path: str | Path,
    conf_threshold: float = 0.001,
    cat_ids: list[int] | None = None,
) -> COCOeval:
    """
    Run RF-DETR on every image in a COCO dataset and evaluate with COCOeval.

    Args:
        annotations_json: path to COCO instances JSON (ground truth).
        images_root: directory containing the images referenced by file_name.
        model_path: RF-DETR checkpoint (.pth / .ckpt).
        conf_threshold: minimum confidence for predictions.
        cat_ids: if set, evaluate only these COCO category ids.

    Returns:
        COCOeval object (already summarized). Access .stats for the 12 COCO metrics.
    """
    from .detector_rfdetr import DetectorRFDETR

    annotations_json = Path(annotations_json).expanduser().resolve()
    images_root = Path(images_root).expanduser().resolve()

    coco_gt = COCO(str(annotations_json))
    img_ids = sorted(coco_gt.getImgIds())

    detector = DetectorRFDETR(model_path=str(model_path), conf_threshold=conf_threshold)

    results: list[dict] = []

    for img_info in tqdm(coco_gt.loadImgs(img_ids), desc="RF-DETR inference", unit="img"):
        img_id = img_info["id"]
        path = images_root / img_info["file_name"]
        if not path.is_file():
            path = images_root / Path(img_info["file_name"]).name

        img = cv2.imread(str(path))
        if img is None:
            continue

        preds = detector.model.predict(img, threshold=conf_threshold)
        xyxy, confidence, class_id = detector._normalize_predictions(preds)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w, h = float(x2 - x1), float(y2 - y1)
            results.append(
                {
                    "image_id": img_id,
                    "category_id": int(class_id[i]),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(confidence[i]),
                }
            )

    if not results:
        print("No predictions — cannot evaluate.")
        return None

    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if cat_ids:
        coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print_coco_metrics(coco_eval, coco_gt)

    return coco_eval


def print_coco_metrics(coco_eval: COCOeval, coco_gt: COCO) -> None:
    """Pretty-print COCOeval results in a YOLO-like table."""
    s = coco_eval.stats

    print(f"\n{'=' * 60}")
    print("COCO evaluation results (RF-DETR)")
    print(f"{'=' * 60}")
    print(f"  mAP@0.5       : {s[1]:.4f}")
    print(f"  mAP@0.5:0.95  : {s[0]:.4f}")
    print(f"  mAP@0.75      : {s[2]:.4f}")
    print(f"  AR@100         : {s[8]:.4f}")

    precision = coco_eval.eval.get("precision")
    if precision is None:
        print(f"{'=' * 60}\n")
        return

    cat_ids = coco_eval.params.catIds
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    recall_arr = coco_eval.eval.get("recall")
    recall_thresholds = np.linspace(0, 1, 101)

    # precision shape: [T, R, K, A, M]
    # T=10 IoU thresholds, R=101 recall, K=categories, A=4 areas, M=3 maxDets
    # area=0 (all), maxDets=2 (100)
    print(f"\n  {'Class':<25} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print(f"  {'-' * 61}")

    all_p, all_r = [], []
    for k_idx, cat_id in enumerate(cat_ids):
        pr_all = precision[:, :, k_idx, 0, 2]  # [T, R] area=all, maxDets=100
        pr_50 = precision[0, :, k_idx, 0, 2]  # [R] IoU=0.5

        valid_all = pr_all[pr_all > -1]
        valid_50 = pr_50[pr_50 > -1]

        ap5095 = float(valid_all.mean()) if len(valid_all) > 0 else float("nan")
        ap50 = float(valid_50.mean()) if len(valid_50) > 0 else float("nan")

        # P and R at max-F1 operating point (IoU=0.5), like YOLO
        p_i = float("nan")
        r_i = float("nan")
        best_f1 = -1.0
        for ri, r_val in enumerate(recall_thresholds):
            p_val = pr_50[ri]
            if p_val < 0:
                continue
            f1 = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                p_i = float(p_val)
                r_i = float(r_val)
        all_p.append(p_i)
        all_r.append(r_i)

        name = cat_names.get(cat_id, str(cat_id))
        print(f"  {name:<25} {p_i:>8.4f} {r_i:>8.4f} {ap50:>8.4f} {ap5095:>10.4f}")

    mean_p = float(np.nanmean(all_p)) if all_p else float("nan")
    mean_r = float(np.nanmean(all_r)) if all_r else float("nan")
    print(f"  {'-' * 61}")
    print(f"  {'Mean':<25} {mean_p:>8.4f} {mean_r:>8.4f} {s[1]:>8.4f} {s[0]:>10.4f}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# YOLO format  →  ultralytics model.val()
# ---------------------------------------------------------------------------

from ultralytics import YOLO


def evaluate_yolo(
    data_yaml: str | Path,
    model_path: str | Path,
    split: str = "test",
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    imgsz: int = 640,
):
    """
    Run ultralytics YOLO validation on the given split.

    Args:
        data_yaml: path to data.yaml.
        model_path: YOLO checkpoint (.pt).
        split: dataset split to evaluate ("val" or "test").
        conf_threshold: confidence threshold.
        iou_threshold: NMS IoU threshold.
        imgsz: inference image size.

    Returns:
        ultralytics Results object with .box (mAP50, mAP50-95, etc.).
    """
    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        split=split,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
    )
    return results


def print_yolo_metrics(results) -> None:
    """Pretty-print ultralytics validation results."""
    b = results.box

    names = results.names
    class_count = len(names) if names else 0

    print(f"\n{'=' * 60}")
    print("YOLO validation results")
    print(f"{'=' * 60}")
    print(f"  mAP@0.5       : {b.map50:.4f}")
    print(f"  mAP@0.5:0.95  : {b.map:.4f}")
    print(f"  mAP@0.75      : {b.map75:.4f}")
    print(f"  Mean Precision : {b.mp:.4f}")
    print(f"  Mean Recall    : {b.mr:.4f}")

    if class_count > 0 and hasattr(b, "maps") and b.maps is not None and len(b.maps) > 0:
        print(f"\n  {'Class':<25} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
        print(f"  {'-' * 61}")
        ap50 = b.all_ap[:, 0] if hasattr(b, "all_ap") and b.all_ap is not None else None
        for i, (cls_id, cls_name) in enumerate(sorted(names.items())):
            p_i = b.p[i] if hasattr(b, "p") and i < len(b.p) else float("nan")
            r_i = b.r[i] if hasattr(b, "r") and i < len(b.r) else float("nan")
            m50_i = ap50[i] if ap50 is not None and i < len(ap50) else float("nan")
            m5095_i = b.maps[i] if i < len(b.maps) else float("nan")
            print(f"  {cls_name:<25} {p_i:>8.4f} {r_i:>8.4f} {m50_i:>8.4f} {m5095_i:>10.4f}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detection metrics via built-in library evaluation")
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- coco sub-command ---
    p_coco = sub.add_parser("coco", help="COCO dataset + RF-DETR → pycocotools COCOeval")
    p_coco.add_argument("annotations", type=str, help="Path to COCO instances JSON")
    p_coco.add_argument("images", type=str, help="Directory with images")
    p_coco.add_argument("--model", "-m", type=str, required=True, help="RF-DETR checkpoint")
    p_coco.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    p_coco.add_argument(
        "--cat-ids", type=str, default=None, help="Comma-separated category ids to evaluate (default: all)"
    )

    # --- yolo sub-command ---
    p_yolo = sub.add_parser("yolo", help="YOLO dataset + ultralytics model.val()")
    p_yolo.add_argument("data_yaml", type=str, help="Path to data.yaml")
    p_yolo.add_argument("--model", "-m", type=str, required=True, help="YOLO checkpoint (.pt)")
    p_yolo.add_argument("--split", type=str, default="test", help="Split to evaluate (default: test)")
    p_yolo.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    p_yolo.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p_yolo.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()

    if args.mode == "coco":
        cat_ids = None
        if args.cat_ids:
            cat_ids = [int(x.strip()) for x in args.cat_ids.split(",") if x.strip()]
        evaluate_rfdetr_coco(
            annotations_json=args.annotations,
            images_root=args.images,
            model_path=args.model,
            conf_threshold=args.conf,
            cat_ids=cat_ids,
        )

    elif args.mode == "yolo":
        results = evaluate_yolo(
            data_yaml=args.data_yaml,
            model_path=args.model,
            split=args.split,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
        )
        print_yolo_metrics(results)
