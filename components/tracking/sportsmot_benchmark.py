"""
SportsMOT tracking benchmark (basketball only).

Evaluates a tracker in isolation on MOT-format image sequences laid out like
``dataset/SportsMOT_basketball``::

    SportsMOT_basketball/
        splits_txt/
            basketball.txt          # sequence names we care about
        dataset/
            train/<seq>/
                seqinfo.ini         # name, frameRate, seqLength, imWidth/Height
                img1/000001.jpg ...
                gt/gt.txt           # MOT: frame,id,x,y,w,h,conf,class,vis
            val/<seq>/...
            test/<seq>/...          # GT withheld — skipped automatically

Only the sequences listed in ``splits_txt/basketball.txt`` are used. SportsMOT's
**test** split ships without ``gt/gt.txt`` (labels are withheld for the eval
server), so GT-fed evaluation runs on **train + val** only; GT-less folders are
skipped automatically.

GT boxes are fed to the tracker (no detector); assigned IDs are compared against
the annotated GT IDs, exactly like :mod:`tracking.benchmark`. The two benchmarks
share :func:`tracking.benchmark.apply_tracker`, so tracker configuration is
identical; only the input adapter (image sequences + MOT GT) differs.

Comparing tracker variants (``--compare-court``)
------------------------------------------------
Perception — court detection, colour embeddings, ReID — is identical regardless
of the tracker's spatial-cost mode, so it is run **once per clip** and both
configs are evaluated on the same features:

  - ``baseline`` — ``use_court_spatial=False`` (pixel bottom-centre distance).
  - ``court``    — ``use_court_spatial=True``  (camera-invariant court distance).

This is an exact A/B: any metric difference is attributable to the spatial cost
alone, not to perception variance.

Frame rate
----------
SportsMOT basketball is **25 fps** but the pipeline is tuned for 30 fps. We run
at native fps (no resampling) and pass it to the tracker, which rescales its
pixel-distance thresholds by ``30 / fps`` — keeping pixels-of-motion-per-frame
calibrated. cv2 reads the numbered frames via the ``img1/%06d.jpg`` pattern, so
no transcoding is needed.

The dataset auto-downloads from Yandex.Disk on first use (like the models and
the other benchmarks) when no root is passed — so a fresh clone can run on GPU
with a single command, no manual data placement. Configure the share URL at
``cfg.benchmarks.tracking.sportsmot``.

Usage
-----
    cd components
    # A/B the initial vs court-projection flow tracker on train+val
    # (auto-downloads the dataset if missing):
    python -m tracking.sportsmot_benchmark --compare-court
    # single config:
    python -m tracking.sportsmot_benchmark --tracker hungarian
    python -m tracking.sportsmot_benchmark --court-spatial
    # quick smoke on 2 clips, or point at an explicit root:
    python -m tracking.sportsmot_benchmark --compare-court --limit 2
    python -m tracking.sportsmot_benchmark /path/to/SportsMOT_basketball --compare-court
"""

from __future__ import annotations

import argparse
import configparser
import copy
import logging
from pathlib import Path

import cv2
import numpy as np

from common.classes import CourtType
from common.classes.player import PlayersDetections
from config import load_default_config
from tracking.benchmark import (
    _detections_to_pred,
    _gt_to_player_detections,
    _print_metrics,
    apply_tracker,
    write_tracking_visual,
)
from tracking.evaluation import evaluate, remap_pred_ids

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Dataset parsing ──────────────────────────────────────────────────────────


def parse_seqinfo(seqinfo_path: str | Path) -> dict:
    """Parse a MOT ``seqinfo.ini`` into a plain dict with typed fields."""
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path)
    seq = parser["Sequence"]
    return {
        "name": seq.get("name", Path(seqinfo_path).parent.name),
        "imDir": seq.get("imDir", "img1"),
        "imExt": seq.get("imExt", ".jpg"),
        "frameRate": float(seq.get("frameRate", "25")),
        "seqLength": int(seq.get("seqLength", "0")),
        "imWidth": int(seq.get("imWidth", "0")),
        "imHeight": int(seq.get("imHeight", "0")),
    }


def load_mot_gt(gt_path: str | Path) -> dict[int, list[dict]]:
    """Load a MOT ``gt.txt`` into ``{frame_id: [{id, bbox}, ...]}``.

    Format per line: ``frame, id, x, y, w, h, conf, class, visibility`` with
    ``(x, y)`` the top-left corner and ``(w, h)`` the box size, all in pixels.
    Returns bboxes as ``[x1, y1, x2, y2]``.

    MOT frame numbers are 1-based (``000001.jpg`` is frame 1); we convert to the
    0-based index that cv2's sequential ``read()`` produces, which is what the
    frame consumers use to look up detections.
    """
    result: dict[int, list[dict]] = {}
    for line in Path(gt_path).read_text().strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        track_id = int(parts[1])
        if track_id < 0:
            continue
        # conf == 0 marks an ignore region in MOT; SportsMOT uses 1 throughout.
        if len(parts) > 6 and float(parts[6]) == 0:
            continue
        x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        result.setdefault(frame - 1, []).append(
            {"id": track_id, "bbox": [x, y, x + w, y + h]}
        )
    return result


def discover_basketball_sequences(
    root: str | Path,
    split: str = "all",
    require_gt: bool = True,
) -> list[Path]:
    """Sequence dirs listed in ``splits_txt/basketball.txt`` that exist on disk.

    Parameters
    ----------
    split : {"train", "val", "test", "trainval", "all"}
        Which dataset split(s) to scan.
    require_gt : bool
        Skip sequences without ``gt/gt.txt`` (SportsMOT's test split ships none).
    """
    root = Path(root)
    names_file = root / "splits_txt" / "basketball.txt"
    if not names_file.is_file():
        log.error("basketball split not found: %s", names_file)
        return []

    split_map = {
        "train": ("train",),
        "val": ("val",),
        "test": ("test",),
        "trainval": ("train", "val"),
        "all": ("train", "val", "test"),
    }
    wanted_splits = split_map.get(split, ("train", "val", "test"))

    names = [n.strip() for n in names_file.read_text().splitlines() if n.strip()]
    seqs: list[Path] = []
    skipped_no_gt = 0
    for name in names:
        for sp in wanted_splits:
            seq_dir = root / "dataset" / sp / name
            if not seq_dir.is_dir():
                continue
            if require_gt and not (seq_dir / "gt" / "gt.txt").is_file():
                skipped_no_gt += 1
                break
            seqs.append(seq_dir)
            break
    if skipped_no_gt:
        log.info("Skipped %d basketball sequence(s) without gt/gt.txt (e.g. test split)", skipped_no_gt)
    return seqs


# ── Perception (run once per clip, shared across tracker configs) ─────────────


def extract_features(
    seq_dir: Path,
    cfg,
    use_court: bool,
) -> tuple[dict, dict[int, list[dict]], PlayersDetections]:
    """Load GT and populate detections with court positions + embeddings + ReID.

    Returns ``(seqinfo, gt, detections)``. ``detections`` carry ``player_id=-1``
    (ready for tracking) plus ``court_position`` / ``embedding`` / ``reid_embedding``.
    This is the expensive part and is independent of the tracker's spatial mode,
    so callers run it once and evaluate multiple tracker configs on copies.
    """
    from common.utils.models import get_model_paths
    from reidentification.extract import extract_reid_embeddings
    from team_clustering.embedding import PlayerEmbedder

    info = parse_seqinfo(seq_dir / "seqinfo.ini")
    pattern = str(seq_dir / info["imDir"] / f"%06d{info['imExt']}")

    gt = load_mot_gt(seq_dir / "gt" / "gt.txt")
    detections = _gt_to_player_detections(gt)

    cap = cv2.VideoCapture(pattern)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open image sequence: {pattern}")
    try:
        if use_court:
            from court_detector.court_detector import CourtDetector

            log.info("Running court detection on GT boxes...")
            CourtDetector(cfg=cfg).run(cap, detections)
        else:
            log.info("Court detection disabled — court-space costs unavailable")

        log.info("Extracting colour embeddings...")
        PlayerEmbedder().extract_player_embeddings(cap, detections)

        paths = get_model_paths(cfg)
        if Path(paths.reid).is_file():
            from common.utils.utils import get_device

            log.info("Extracting ReID embeddings...")
            extract_reid_embeddings(cap, detections, str(paths.reid), device=get_device())
        else:
            log.warning("ReID model not found at %s — using colour embeddings only", paths.reid)
    finally:
        cap.release()

    return info, gt, detections


def _track_and_eval(
    detections: PlayersDetections,
    gt: dict[int, list[dict]],
    tracker_type: str,
    cfg,
    img_w: float,
    fps: float,
    iou_threshold: float,
    visual_path: str | None = None,
    visual_pattern: str | None = None,
) -> dict[str, float | int]:
    """Run a tracker over a *copy* of detections and evaluate against GT."""
    dets = copy.deepcopy(detections)
    apply_tracker(dets, tracker_type, cfg, frame_width=img_w, fps=fps)
    pred = _detections_to_pred(dets)
    metrics = evaluate(gt, pred, iou_threshold)
    if visual_path is not None and visual_pattern is not None:
        pred_remapped = remap_pred_ids(gt, pred, iou_threshold)
        write_tracking_visual(visual_pattern, gt, pred_remapped, visual_path)
    return metrics


# ── Per-sequence evaluation ──────────────────────────────────────────────────


def evaluate_sequence_mot(
    seq_dir: str | Path,
    iou_threshold: float = 0.5,
    court_type: CourtType = CourtType.NBA,
    use_court: bool = True,
    use_court_spatial: bool = False,
    visual_path: str | None = None,
    tracker_type: str = "flow",
) -> dict[str, float | int]:
    """Evaluate one SportsMOT image-sequence for a single tracker config."""
    seq_dir = Path(seq_dir)
    cfg = load_default_config()
    cfg.main.court_type = court_type
    cfg.tracker.use_court_spatial = use_court_spatial

    info, gt, detections = extract_features(seq_dir, cfg, use_court)
    pattern = str(seq_dir / info["imDir"] / f"%06d{info['imExt']}")
    return _track_and_eval(
        detections,
        gt,
        tracker_type,
        cfg,
        img_w=info["imWidth"],
        fps=info["frameRate"],
        iou_threshold=iou_threshold,
        visual_path=visual_path,
        visual_pattern=pattern,
    )


# ── Aggregation ──────────────────────────────────────────────────────────────


def _aggregate(per_clip: list[dict]) -> dict[str, float | int]:
    """Aggregate per-clip metrics: pooled CLEAR counts, averaged HOTA/IDF1."""
    agg = {"TP": 0, "FP": 0, "FN": 0, "IDSW": 0, "total_gt": 0, "sum_iou": 0.0}
    hota, deta, assa, idf1, idp, idr = [], [], [], [], [], []
    for m in per_clip:
        for k in ("TP", "FP", "FN", "IDSW", "total_gt", "sum_iou"):
            agg[k] += m.get(k, 0)
        hota.append(m.get("HOTA", 0.0))
        deta.append(m.get("DetA", 0.0))
        assa.append(m.get("AssA", 0.0))
        idf1.append(m.get("IDF1", 0.0))
        idp.append(m.get("IDP", 0.0))
        idr.append(m.get("IDR", 0.0))

    tgt = agg["total_gt"]
    agg["MOTA"] = 1.0 - (agg["FP"] + agg["FN"] + agg["IDSW"]) / tgt if tgt > 0 else 0.0
    agg["MOTP"] = agg["sum_iou"] / agg["TP"] if agg["TP"] else 0.0
    agg["HOTA"] = float(np.mean(hota)) if hota else 0.0
    agg["DetA"] = float(np.mean(deta)) if deta else 0.0
    agg["AssA"] = float(np.mean(assa)) if assa else 0.0
    agg["IDF1"] = float(np.mean(idf1)) if idf1 else 0.0
    agg["IDP"] = float(np.mean(idp)) if idp else 0.0
    agg["IDR"] = float(np.mean(idr)) if idr else 0.0
    return agg


# ── Benchmark runners ────────────────────────────────────────────────────────


def run_sportsmot_benchmark(
    root: str,
    iou_threshold: float = 0.5,
    court_type: CourtType = CourtType.NBA,
    use_court: bool = True,
    use_court_spatial: bool = False,
    write_visuals: bool = False,
    tracker_type: str = "flow",
    split: str = "trainval",
    limit: int | None = None,
) -> dict[str, float | int]:
    """Run a single tracker config on every GT-labelled basketball sequence."""
    seqs = discover_basketball_sequences(root, split=split, require_gt=True)
    if limit:
        seqs = seqs[:limit]
    if not seqs:
        log.error("No GT-labelled basketball sequences found under %s (split=%s)", root, split)
        return {}

    log.info("Evaluating %d sequence(s) [split=%s]", len(seqs), split)
    per_clip: list[dict] = []
    for seq_dir in seqs:
        log.info("═══ %s ═══", seq_dir.name)
        visual_path = (
            str(Path(root) / "visuals" / f"{seq_dir.name}_visual.mp4") if write_visuals else None
        )
        metrics = evaluate_sequence_mot(
            seq_dir,
            iou_threshold=iou_threshold,
            court_type=court_type,
            use_court=use_court,
            use_court_spatial=use_court_spatial,
            visual_path=visual_path,
            tracker_type=tracker_type,
        )
        if metrics:
            per_clip.append(metrics)
            _print_metrics(seq_dir.name, metrics)

    agg = _aggregate(per_clip)
    print()
    _print_metrics(f"AGGREGATE ({tracker_type}, court_spatial={use_court_spatial})", agg)
    return agg


def run_compare_court(
    root: str,
    iou_threshold: float = 0.5,
    court_type: CourtType = CourtType.NBA,
    tracker_type: str = "flow",
    split: str = "trainval",
    limit: int | None = None,
) -> dict[str, dict]:
    """A/B the initial (pixel) vs fixed (court-projection) spatial cost.

    Perception runs once per clip; both configs are evaluated on the same
    features. Returns ``{"baseline": agg, "court": agg}``.
    """
    seqs = discover_basketball_sequences(root, split=split, require_gt=True)
    if limit:
        seqs = seqs[:limit]
    if not seqs:
        log.error("No GT-labelled basketball sequences found under %s (split=%s)", root, split)
        return {}

    log.info("A/B comparison over %d sequence(s) [split=%s, tracker=%s]", len(seqs), split, tracker_type)

    configs = [("baseline", False), ("court", True)]
    per_clip: dict[str, list[dict]] = {name: [] for name, _ in configs}

    cfg = load_default_config()
    cfg.main.court_type = court_type

    for seq_dir in seqs:
        log.info("═══ %s ═══", seq_dir.name)
        # Court detection must run so court_position exists for the 'court' config.
        info, gt, detections = extract_features(seq_dir, cfg, use_court=True)
        for name, flag in configs:
            cfg.tracker.use_court_spatial = flag
            metrics = _track_and_eval(
                detections,
                gt,
                tracker_type,
                cfg,
                img_w=info["imWidth"],
                fps=info["frameRate"],
                iou_threshold=iou_threshold,
            )
            per_clip[name].append(metrics)
            log.info(
                "  %-9s HOTA=%.4f IDF1=%.4f MOTA=%.4f IDSW=%d",
                name,
                metrics.get("HOTA", 0.0),
                metrics.get("IDF1", 0.0),
                metrics.get("MOTA", 0.0),
                metrics.get("IDSW", 0),
            )

    aggs = {name: _aggregate(per_clip[name]) for name, _ in configs}
    print()
    for name, _ in configs:
        _print_metrics(f"AGGREGATE — {name} ({tracker_type})", aggs[name])
    _print_court_delta(aggs["baseline"], aggs["court"], tracker_type, len(seqs))
    return aggs


def _print_court_delta(baseline: dict, court: dict, tracker_type: str, n_clips: int) -> None:
    """Compact baseline-vs-court comparison with deltas (↑ better, except IDSW)."""
    print(f"\n{'═' * 60}")
    print(f"  COURT-PROJECTION A/B — {tracker_type}, {n_clips} clips")
    print(f"{'═' * 60}")
    print(f"  {'metric':<8}{'baseline':>12}{'court':>12}{'Δ':>12}")
    print(f"  {'-' * 44}")
    for key in ("HOTA", "AssA", "IDF1", "MOTA"):
        b, c = baseline.get(key, 0.0), court.get(key, 0.0)
        print(f"  {key:<8}{b:>12.4f}{c:>12.4f}{c - b:>+12.4f}")
    bi, ci = baseline.get("IDSW", 0), court.get("IDSW", 0)
    print(f"  {'IDSW':<8}{bi:>12d}{ci:>12d}{ci - bi:>+12d}   (lower better)")
    print(f"{'═' * 60}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = load_default_config()
    from common.utils.models import ensure_models

    ensure_models(cfg)

    parser = argparse.ArgumentParser(description="SportsMOT basketball tracking benchmark")
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="SportsMOT root (with splits_txt/ and dataset/). "
        "Default: auto-download via cfg.benchmarks.tracking.sportsmot",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--court-type", choices=["nba", "fiba"], default="nba")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "trainval", "all"],
        default="trainval",
        help="Which split(s) to evaluate (GT-less sequences are skipped). Default: trainval",
    )
    parser.add_argument("--tracker", choices=["flow", "hungarian", "appearance"], default="flow")
    parser.add_argument(
        "--compare-court",
        action="store_true",
        help="A/B the initial (pixel) vs court-projection spatial cost on the same perception",
    )
    parser.add_argument(
        "--court-spatial",
        action="store_true",
        help="Single run with use_court_spatial=True (ignored when --compare-court is set)",
    )
    parser.add_argument(
        "--no-court",
        action="store_true",
        help="Disable court detection (no homography; court-space costs unavailable)",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Write side-by-side GT vs pred videos (single-config runs only)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N clips (smoke test)")
    args = parser.parse_args()

    court_type = CourtType.NBA if args.court_type == "nba" else CourtType.FIBA

    # Resolve the dataset root, auto-downloading the basketball SportsMOT subset
    # from Yandex.Disk on first use (mirrors ensure_models / the other benchmarks).
    if args.root:
        root = args.root
    else:
        from common.utils.datasets import ensure_dataset

        root = str(ensure_dataset(cfg.benchmarks.tracking.sportsmot))

    if args.compare_court:
        run_compare_court(
            root,
            iou_threshold=args.iou_threshold,
            court_type=court_type,
            tracker_type=args.tracker,
            split=args.split,
            limit=args.limit,
        )
    else:
        run_sportsmot_benchmark(
            root,
            iou_threshold=args.iou_threshold,
            court_type=court_type,
            use_court=not args.no_court,
            use_court_spatial=args.court_spatial,
            write_visuals=args.visual,
            tracker_type=args.tracker,
            split=args.split,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
