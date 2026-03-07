"""
Evaluation utilities: MOTA, IDF1, benchmark runner.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import load_detections_csv
from .tracker import PlayerTracker

log = logging.getLogger(__name__)


def match_objects(gt_list, tracked_list, distance_threshold=2.0):
    """
    Bipartite match of GT <-> tracked objects by Euclidean field distance.

    Returns
    -------
    matched_pairs : list of (gt_id, tracked_id)
    unmatched_gt_ids : list
    unmatched_tracked_ids : list
    """
    if not gt_list or not tracked_list:
        return [], [g["id"] for g in gt_list], [t["id"] for t in tracked_list]

    ng, nt = len(gt_list), len(tracked_list)
    cost = np.full((ng, nt), 1e9)
    for i, g in enumerate(gt_list):
        gp = np.asarray(g["pos"])
        for j, t in enumerate(tracked_list):
            d = np.linalg.norm(gp - np.asarray(t["pos"]))
            if d <= distance_threshold:
                cost[i, j] = d

    rows, cols = linear_sum_assignment(cost)
    matches, mg, mt = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] < 1e9:
            matches.append((gt_list[r]["id"], tracked_list[c]["id"]))
            mg.add(r)
            mt.add(c)

    um_g = [gt_list[i]["id"] for i in range(ng) if i not in mg]
    um_t = [tracked_list[j]["id"] for j in range(nt) if j not in mt]
    return matches, um_g, um_t


def evaluate_tracking(
    tracker: PlayerTracker,
    detections: list[dict],
    distance_threshold: float = 2.0,
):
    """
    Compute MOTA and IDF1 from tracker results vs detection ground truth.

    Parameters
    ----------
    tracker : PlayerTracker
        A tracker whose ``run_tracking`` has already been called.
    detections : list[dict]
        The same detection dicts that were fed to the tracker, used as GT.
    distance_threshold : float
        Maximum field distance for a match to count as correct.

    Returns
    -------
    dict
        ``{MOTA, IDF1, FP, FN, IDSW, TP, total_gt}``
    """
    gt_by_frame: dict[int, list] = {}
    for d in detections:
        gt_by_frame.setdefault(d["frame_id"], []).append(
            {"id": d["detection_id"], "pos": d["field_coords"]}
        )

    tr_by_frame: dict[int, list] = {}
    for t in tracker.tracks:
        for fid, pos in zip(t.frame_ids, t.history):
            tr_by_frame.setdefault(fid, []).append({"id": t.track_id, "pos": pos})

    all_fids = sorted(set(gt_by_frame) | set(tr_by_frame))

    fp = fn = idsw = tp = total_gt = 0
    prev_map: dict = {}

    for fid in all_fids:
        cg = gt_by_frame.get(fid, [])
        ct = tr_by_frame.get(fid, [])
        total_gt += len(cg)

        m, ug, ut = match_objects(cg, ct, distance_threshold)
        fp += len(ut)
        fn += len(ug)
        tp += len(m)

        cur_map = {}
        for gid, tid in m:
            if gid in prev_map and prev_map[gid] != tid:
                idsw += 1
            cur_map[gid] = tid
        prev_map = cur_map

    mota = 1.0 - (fp + fn + idsw) / total_gt if total_gt else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    idf1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "MOTA": mota,
        "IDF1": idf1,
        "FP": fp,
        "FN": fn,
        "IDSW": idsw,
        "TP": tp,
        "total_gt": total_gt,
    }


def run_benchmark(
    tracking_dir: str,
    distance_threshold: float = 6.0,
    max_time_since_update: int = 3,
    eval_distance: float = 2.0,
):
    """
    Run the tracker + evaluation loop over all segment folders in *tracking_dir*.

    Each subfolder is expected to contain ``annotations.csv``.

    Returns
    -------
    dict
        Aggregated ``{MOTA, IDF1, FP, FN, IDSW, TP, total_gt}``
    """
    agg = {"FP": 0, "FN": 0, "IDSW": 0, "TP": 0, "total_gt": 0}

    for folder_name in sorted(os.listdir(tracking_dir)):
        segment_path = os.path.join(tracking_dir, folder_name)
        if not os.path.isdir(segment_path):
            continue

        csv_path = os.path.join(segment_path, "annotations.csv")
        if not os.path.exists(csv_path):
            log.warning("annotations.csv not found in %s — skipping", folder_name)
            continue

        log.info("Processing segment: %s", folder_name)
        detections = load_detections_csv(csv_path)

        tracker = PlayerTracker(
            field_gate=distance_threshold,
            max_age=max_time_since_update,
        )
        tracker.run_tracking(detections)

        metrics = evaluate_tracking(tracker, detections, eval_distance)
        for k in agg:
            agg[k] += metrics[k]

        log.info(
            "  %s  MOTA=%.3f  IDF1=%.3f  IDSW=%d",
            folder_name,
            metrics["MOTA"],
            metrics["IDF1"],
            metrics["IDSW"],
        )

    tgt = agg["total_gt"]
    if tgt > 0:
        agg["MOTA"] = 1.0 - (agg["FP"] + agg["FN"] + agg["IDSW"]) / tgt
        prec = agg["TP"] / (agg["TP"] + agg["FP"]) if (agg["TP"] + agg["FP"]) else 0.0
        rec = agg["TP"] / (agg["TP"] + agg["FN"]) if (agg["TP"] + agg["FN"]) else 0.0
        agg["IDF1"] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    else:
        agg["MOTA"] = 0.0
        agg["IDF1"] = 0.0

    return agg
