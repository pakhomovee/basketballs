"""
Tracking benchmark runner.

Tests the tracker in isolation: GT bboxes from annotations are fed to
FlowTracker (without detector), and the assigned track IDs are compared
against the annotated ground truth IDs.

Expected dataset layout::

    tracking_data/
        video/
            clip_01.mp4
            clip_02.mp4
        annotation/
            clip_01_tracking.txt
            clip_02_tracking.txt

Each annotation is in YOLO MOT format (as produced by ``tracking.annotate``):
    frame_id track_id class_id cx cy w h   (coords normalised to [0, 1])

Usage
-----
    cd components
    python -m tracking.benchmark tracking_data/ [--iou-threshold 0.5]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from common.classes.player import Player, PlayersDetections
from common.classes import CourtType
from common.utils.datasets import ensure_dataset
from common.utils.models import ensure_models
from config import load_default_config
from tracking.evaluation import evaluate, load_yolo_mot, remap_pred_ids
from video_reader import VideoReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Distinct BGR colors for up to 20 track IDs (same palette as annotate.py)
_TRACK_COLORS_RGB = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a given track ID."""
    r, g, b = _TRACK_COLORS_RGB[track_id % len(_TRACK_COLORS_RGB)]
    return (b, g, r)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_video_dims(video_path: str) -> tuple[int, int]:
    """Return (width, height) from a video file."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _gt_to_player_detections(
    gt: dict[int, list[dict]],
) -> PlayersDetections:
    """
    Build PlayersDetections from GT bboxes with IDs stripped (player_id = -1).

    The tracker will assign new IDs; GT IDs are only used for evaluation.
    """
    detections: PlayersDetections = {}
    for frame_id, dets in gt.items():
        detections[frame_id] = [Player(bbox=[int(v) for v in d["bbox"]], player_id=-1) for d in dets]
    return detections


def _detections_to_pred(
    detections: PlayersDetections,
) -> dict[int, list[dict]]:
    """Convert PlayersDetections (after tracking) to evaluation format."""
    out: dict[int, list[dict]] = {}
    for frame_id, players in detections.items():
        dets = [{"id": p.player_id, "bbox": list(p.bbox)} for p in players if p.player_id >= 0]
        if dets:
            out[frame_id] = dets
    return out


_FP_COLOR = (0, 0, 220)  # red — predicted bbox that matched no GT (FP)
_FN_COLOR = (220, 0, 0)  # blue — GT bbox that matched no pred (FN)


def write_tracking_visual(
    video_path: str,
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    output_path: str,
    iou_threshold: float = 0.5,
) -> None:
    """
    Write a side-by-side comparison video: left = GT, right = tracker pred.

    Color scheme (applied consistently to both panels):
    - Each GT track ID has a fixed color from the palette.
    - GT bboxes are colored by their GT ID.
    - Pred bboxes are colored by the GT ID they IoU-match on that frame.
      This means: if tracking is perfect, both panels show the same colors.
      If the tracker swaps two players, the pred panel shows the wrong color
      for those boxes, immediately highlighting the confusion.
    - Unmatched pred bboxes (FP) are drawn in red.
    - Unmatched GT bboxes (FN) are drawn in blue on the GT panel.

    Each bbox label shows the track ID (GT or pred) for that detection.
    """
    from tracking.evaluation import match_frame as _match_frame

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video for visualization: %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = src_w // 2
    panel_h = int(src_h * panel_w / src_w)
    out_w = panel_w * 2
    out_h = panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    font = cv2.FONT_HERSHEY_SIMPLEX

    def _draw_box(
        panel: np.ndarray,
        bbox: list,
        color: tuple[int, int, int],
        label: str,
        sx: float,
        sy: float,
        thickness: int = 2,
    ) -> None:
        x1 = int(bbox[0] * sx)
        y1 = int(bbox[1] * sy)
        x2 = int(bbox[2] * sx)
        y2 = int(bbox[3] * sy)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
        cv2.rectangle(panel, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(panel, label, (x1 + 1, y1 - 3), font, 0.55, (0, 0, 0), 2)

    frame_id = 0
    log.info("Writing visualization: %s", output_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gt_dets = gt.get(frame_id, [])
        pred_dets = pred.get(frame_id, [])

        # Per-frame IoU matching to determine which GT each pred box corresponds to
        matches, um_gt_ids, _ = _match_frame(gt_dets, pred_dets, iou_threshold)
        pred_to_gt: dict[int, int] = {pred_id: gt_id for gt_id, pred_id in matches}
        um_gt_set: set[int] = set(um_gt_ids)

        sx = panel_w / src_w
        sy = panel_h / src_h

        gt_panel = cv2.resize(frame, (panel_w, panel_h))
        pred_panel = cv2.resize(frame.copy(), (panel_w, panel_h))

        # Left panel — GT boxes
        for d in gt_dets:
            color = _FN_COLOR if d["id"] in um_gt_set else _color_for_id(d["id"])
            _draw_box(gt_panel, d["bbox"], color, str(d["id"]), sx, sy)

        # Right panel — pred boxes colored by their matched GT ID
        for d in pred_dets:
            matched_gt = pred_to_gt.get(d["id"])
            if matched_gt is None:
                color = _FP_COLOR  # FP: no GT match
            else:
                color = _color_for_id(matched_gt)  # color = matched GT's color
            # Label: "pred_id" or "pred_id≠gt_id" when wrong ID assigned
            if matched_gt is not None and matched_gt != d["id"]:
                label = f"{d['id']}≠{matched_gt}"
            else:
                label = str(d["id"])
            _draw_box(pred_panel, d["bbox"], color, label, sx, sy)

        cv2.putText(gt_panel, "GT", (6, 22), font, 0.7, (255, 255, 255), 2)
        cv2.putText(pred_panel, "PRED (color=matched GT)", (6, 22), font, 0.55, (255, 255, 255), 2)

        combined = np.hstack([gt_panel, pred_panel])
        writer.write(combined)
        frame_id += 1

    cap.release()
    writer.release()
    log.info("Visualization saved: %s", output_path)


def discover_sequences(data_dir: str | Path) -> list[tuple[str, str]]:
    """
    Discover (video_path, annotation_path) pairs.

    Looks for ``video/<name>.mp4`` and ``annotation/<name>_tracking.txt``.
    """
    data_dir = Path(data_dir)
    video_dir = data_dir / "video"
    ann_dir = data_dir / "annotation"

    if not video_dir.is_dir():
        log.error("Video directory not found: %s", video_dir)
        return []
    if not ann_dir.is_dir():
        log.error("Annotation directory not found: %s", ann_dir)
        return []

    pairs = []
    for vp in sorted(video_dir.glob("*.mp4")):
        ann = ann_dir / f"{vp.stem}_tracking.txt"
        if ann.exists():
            pairs.append((str(vp), str(ann)))
        else:
            log.warning("No annotation for %s (expected %s)", vp.name, ann.name)
    return pairs


# ── Per-sequence evaluation ──────────────────────────────────────────────────


def evaluate_sequence(
    video_path: str,
    annotation_path: str,
    iou_threshold: float = 0.5,
    court_type: CourtType = CourtType.NBA,
    use_court: bool = True,
    visual_path: str | None = None,
) -> dict[str, float | int]:
    """
    Evaluate tracker performance on one sequence.

    1. Load GT bboxes + IDs from annotation.
    2. Build PlayersDetections from GT bboxes (IDs stripped).
    3. Optionally run court detection (provides homography for field-coord gating).
    4. Run embedding extraction on those boxes.
    5. Run FlowTracker to assign track IDs.
    6. Remap pred IDs to the optimal bijection onto GT IDs, then compute metrics.
    7. Optionally write a side-by-side GT vs pred visualization video.
    """
    from team_clustering.embedding import PlayerEmbedder
    from tracking.flow_tracker import FlowTracker

    img_w, img_h = _get_video_dims(video_path)

    log.info("Loading GT annotations: %s", annotation_path)
    gt = load_yolo_mot(annotation_path, img_w, img_h)

    # Build detections from GT bboxes (tracker gets the correct boxes, not IDs)
    detections = _gt_to_player_detections(gt)

    with VideoReader(video_path) as vr:
        if use_court:
            from court_detector.court_detector import CourtDetector

            log.info("Running court detection on GT boxes...")
            _cfg = load_default_config()
            _cfg.main.court_type = court_type
            court_detector = CourtDetector(cfg=_cfg)
            court_detector.run(vr, detections)
        else:
            log.info("Court detection disabled — tracker uses pixel-space costs only")

        log.info("Extracting embeddings...")
        PlayerEmbedder().extract_player_embeddings(vr, detections)

    log.info("Running FlowTracker...")
    tracker = FlowTracker(frame_width=float(img_w))
    tracker.track(detections)

    pred = _detections_to_pred(detections)
    pred_remapped = remap_pred_ids(gt, pred, iou_threshold)
    metrics = evaluate(gt, pred, iou_threshold)

    if visual_path is not None:
        write_tracking_visual(video_path, gt, pred_remapped, visual_path)

    return metrics


# ── Benchmark runner ─────────────────────────────────────────────────────────


def run_benchmark(
    data_dir: str,
    iou_threshold: float = 0.5,
    court_type: CourtType = CourtType.NBA,
    use_court: bool = True,
    write_visuals: bool = True,
) -> dict[str, float | int]:
    """
    Run tracker + evaluation on all sequences in *data_dir*.

    Returns per-sequence and aggregated metrics.
    Writes side-by-side GT vs pred videos to ``<data_dir>/visuals/`` unless
    *write_visuals* is False.
    """
    pairs = discover_sequences(data_dir)
    if not pairs:
        log.error("No sequences found in %s", data_dir)
        return {}

    all_results: list[tuple[str, dict]] = []
    agg = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "IDSW": 0,
        "total_gt": 0,
        "sum_iou": 0.0,
    }
    hota_vals, deta_vals, assa_vals = [], [], []
    idf1_idp_idr: list[dict] = []

    for video_path, ann_path in pairs:
        name = Path(video_path).stem
        log.info("═══ Sequence: %s ═══", name)

        visual_path: str | None = None
        if write_visuals:
            visuals_dir = Path(data_dir) / "visuals"
            visual_path = str(visuals_dir / f"{name}_visual.mp4")

        metrics = evaluate_sequence(
            video_path,
            ann_path,
            iou_threshold=iou_threshold,
            court_type=court_type,
            use_court=use_court,
            visual_path=visual_path,
        )
        all_results.append((name, metrics))

        for k in ("TP", "FP", "FN", "IDSW", "total_gt", "sum_iou"):
            agg[k] += metrics.get(k, 0)

        hota_vals.append(metrics.get("HOTA", 0.0))
        deta_vals.append(metrics.get("DetA", 0.0))
        assa_vals.append(metrics.get("AssA", 0.0))
        idf1_idp_idr.append(
            {
                "IDF1": metrics.get("IDF1", 0.0),
                "IDP": metrics.get("IDP", 0.0),
                "IDR": metrics.get("IDR", 0.0),
            }
        )

        _print_metrics(name, metrics)

    # Aggregate CLEAR from summed counts
    tgt = agg["total_gt"]
    if tgt > 0:
        agg["MOTA"] = 1.0 - (agg["FP"] + agg["FN"] + agg["IDSW"]) / tgt
        agg["MOTP"] = agg["sum_iou"] / agg["TP"] if agg["TP"] else 0.0
    else:
        agg["MOTA"] = 0.0
        agg["MOTP"] = 0.0

    # Average HOTA across sequences
    agg["HOTA"] = float(np.mean(hota_vals)) if hota_vals else 0.0
    agg["DetA"] = float(np.mean(deta_vals)) if deta_vals else 0.0
    agg["AssA"] = float(np.mean(assa_vals)) if assa_vals else 0.0

    # Average IDF1 across sequences
    agg["IDF1"] = float(np.mean([d["IDF1"] for d in idf1_idp_idr])) if idf1_idp_idr else 0.0
    agg["IDP"] = float(np.mean([d["IDP"] for d in idf1_idp_idr])) if idf1_idp_idr else 0.0
    agg["IDR"] = float(np.mean([d["IDR"] for d in idf1_idp_idr])) if idf1_idp_idr else 0.0

    print()
    _print_metrics("AGGREGATE", agg)

    agg["per_sequence"] = all_results
    return agg


def _print_metrics(name: str, m: dict) -> None:
    print(f"\n{'═' * 50}")
    print(f"  {name}")
    print(f"{'═' * 50}")
    print(f"  MOTA:  {m.get('MOTA', 0):.4f}    MOTP: {m.get('MOTP', 0):.4f}")
    print(f"  HOTA:  {m.get('HOTA', 0):.4f}    DetA: {m.get('DetA', 0):.4f}    AssA: {m.get('AssA', 0):.4f}")
    print(f"  IDF1:  {m.get('IDF1', 0):.4f}    IDP:  {m.get('IDP', 0):.4f}    IDR:  {m.get('IDR', 0):.4f}")
    print(
        f"  TP: {m.get('TP', 0)}  FP: {m.get('FP', 0)}  FN: {m.get('FN', 0)} "
        f" IDSW: {m.get('IDSW', 0)}  GT: {m.get('total_gt', 0)}"
    )
    print(f"{'═' * 50}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    cfg = load_default_config()
    ensure_models(cfg)

    parser = argparse.ArgumentParser(
        description="Run tracking benchmark on annotated data",
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Root directory (with video/ and annotation/ subdirs). Defaults to the path configured in main.yaml.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)",
    )
    parser.add_argument(
        "--court-type",
        choices=["nba", "fiba"],
        default="nba",
    )
    parser.add_argument(
        "--no-court",
        action="store_true",
        help="Disable court detection (no homography; tracker uses pixel-space costs only)",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Skip writing side-by-side GT vs pred visualization videos",
    )
    args = parser.parse_args()

    court_type = CourtType.NBA if args.court_type == "nba" else CourtType.FIBA
    data_dir = args.data_dir or str(ensure_dataset(cfg.benchmarks.tracking.dataset))
    run_benchmark(
        data_dir,
        iou_threshold=args.iou_threshold,
        court_type=court_type,
        use_court=not args.no_court,
        write_visuals=not args.no_visual,
    )


if __name__ == "__main__":
    main()
