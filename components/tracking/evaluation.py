"""
Tracking evaluation metrics via TrackEval (HOTA, CLEAR/MOTA, Identity/IDF1).

Thin adapter from our YOLO MOT format to TrackEval's sequence data dict.
Works with YOLO MOT format (normalised bboxes) produced by the annotation tool.
All matching is IoU-based in pixel space.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.distances import bbox_iou

log = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_yolo_mot(
    txt_path: str | Path,
    img_w: int,
    img_h: int,
) -> dict[int, list[dict]]:
    """
    Load YOLO MOT txt into ``{frame_id: [{id, bbox}, ...]}``.

    Format per line: ``frame_id track_id class_id cx cy w h``
    with cx, cy, w, h in [0, 1].

    Returns bbox as [x1, y1, x2, y2] in pixels.
    """
    result: dict[int, list[dict]] = {}
    for line in Path(txt_path).read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        frame_id = int(parts[0])
        track_id = int(parts[1])
        if track_id < 0:
            continue
        cx, cy, w, h = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        result.setdefault(frame_id, []).append(
            {
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
            }
        )
    return result


# ── IoU matching ─────────────────────────────────────────────────────────────


def match_frame(
    gt_dets: list[dict],
    pred_dets: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Bipartite IoU match of GT ↔ predicted detections on a single frame.

    Returns
    -------
    matches : list of (gt_id, pred_id)
    unmatched_gt : list of gt_id
    unmatched_pred : list of pred_id
    """
    if not gt_dets or not pred_dets:
        return (
            [],
            [g["id"] for g in gt_dets],
            [p["id"] for p in pred_dets],
        )

    ng, np_ = len(gt_dets), len(pred_dets)
    cost = np.ones((ng, np_), dtype=float)
    for i, g in enumerate(gt_dets):
        for j, p in enumerate(pred_dets):
            iou = bbox_iou(g["bbox"], p["bbox"])
            if iou >= iou_threshold:
                cost[i, j] = 1.0 - iou

    rows, cols = linear_sum_assignment(cost)
    matches, mg, mp = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] < 1.0:  # was actually above threshold
            matches.append((gt_dets[r]["id"], pred_dets[c]["id"]))
            mg.add(r)
            mp.add(c)

    um_g = [gt_dets[i]["id"] for i in range(ng) if i not in mg]
    um_p = [pred_dets[j]["id"] for j in range(np_) if j not in mp]
    return matches, um_g, um_p


# ── TrackEval adapter ────────────────────────────────────────────────────────


def _build_trackeval_data(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
) -> dict:
    """
    Convert our ``{frame_id: [{id, bbox}]}`` dicts into TrackEval's sequence
    data dict.

    TrackEval requires:
    - 0-indexed contiguous integer IDs per sequence
    - per-frame arrays of GT / tracker IDs
    - per-frame IoU similarity matrices (shape: n_gt × n_pred)
    """
    all_frames = sorted(set(gt) | set(pred))

    gt_id_set = sorted({d["id"] for dets in gt.values() for d in dets})
    pred_id_set = sorted({d["id"] for dets in pred.values() for d in dets})
    gt_id_map = {v: i for i, v in enumerate(gt_id_set)}
    pred_id_map = {v: i for i, v in enumerate(pred_id_set)}

    gt_ids_list: list[np.ndarray] = []
    tracker_ids_list: list[np.ndarray] = []
    similarity_scores_list: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for fid in all_frames:
        g_dets = gt.get(fid, [])
        p_dets = pred.get(fid, [])

        gt_ids_t = np.array([gt_id_map[d["id"]] for d in g_dets], dtype=np.int64)
        tracker_ids_t = np.array([pred_id_map[d["id"]] for d in p_dets], dtype=np.int64)

        sim = np.zeros((len(g_dets), len(p_dets)), dtype=np.float64)
        for i, g in enumerate(g_dets):
            for j, p in enumerate(p_dets):
                sim[i, j] = bbox_iou(g["bbox"], p["bbox"])

        gt_ids_list.append(gt_ids_t)
        tracker_ids_list.append(tracker_ids_t)
        similarity_scores_list.append(sim)
        num_gt_dets += len(g_dets)
        num_tracker_dets += len(p_dets)

    return {
        "num_timesteps": len(all_frames),
        "num_gt_ids": len(gt_id_set),
        "num_tracker_ids": len(pred_id_set),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids_list,
        "tracker_ids": tracker_ids_list,
        "similarity_scores": similarity_scores_list,
    }


# ── Optimal ID remapping ────────────────────────────────────────────────────


def remap_pred_ids(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict[int, list[dict]]:
    """
    Remap pred track IDs to the optimal bijection onto GT track IDs.

    The tracker may output any permutation of IDs (e.g. tracker assigns
    IDs 0-9 while GT uses IDs 1-10 in a different order).  Without remapping,
    CLEAR-MOT IDSW would count a spurious switch on the very first matched
    frame for every track whose ID number differs from GT.

    We find the globally optimal GT→pred ID bijection by maximising total
    co-occurrence (frames where the same bbox is matched to both a GT track
    and a pred track), then relabel pred so IDs align with GT.
    Pred tracks with no GT match and extra GT tracks keep/get unique IDs.
    """
    all_frames = sorted(set(gt) | set(pred))

    # Collect co-occurrence counts between gt IDs and pred IDs
    gt_ids: list[int] = sorted({d["id"] for dets in gt.values() for d in dets})
    pred_ids: list[int] = sorted({d["id"] for dets in pred.values() for d in dets})
    if not gt_ids or not pred_ids:
        return pred

    gt_idx = {v: i for i, v in enumerate(gt_ids)}
    pred_idx = {v: i for i, v in enumerate(pred_ids)}
    cooccur = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)

    for fid in all_frames:
        g_dets = gt.get(fid, [])
        p_dets = pred.get(fid, [])
        matches, _, _ = match_frame(g_dets, p_dets, iou_threshold)
        for gt_id, pred_id in matches:
            if gt_id in gt_idx and pred_id in pred_idx:
                cooccur[gt_idx[gt_id], pred_idx[pred_id]] += 1

    # Hungarian on negative co-occurrence → maximise overlap
    rows, cols = linear_sum_assignment(-cooccur)
    remap: dict[int, int] = {}
    for r, c in zip(rows, cols):
        if cooccur[r, c] > 0:
            remap[pred_ids[c]] = gt_ids[r]

    # Pred IDs with no GT match get fresh IDs beyond max(all IDs)
    next_id = max(max(gt_ids), max(pred_ids)) + 1
    for pid in pred_ids:
        if pid not in remap:
            remap[pid] = next_id
            next_id += 1

    # Apply remap
    remapped: dict[int, list[dict]] = {}
    for fid, dets in pred.items():
        remapped[fid] = [{"id": remap.get(d["id"], d["id"]), "bbox": d["bbox"]} for d in dets]
    return remapped


# ── Evaluate ─────────────────────────────────────────────────────────────────


def evaluate(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """
    Compute all tracking metrics for one sequence using TrackEval.

    Pred IDs are first remapped to the optimal bijection onto GT IDs so that
    IDSW reflects true track fragmentation, not arbitrary ID numbering.

    Returns dict with MOTA, MOTP, HOTA, DetA, AssA, IDF1, IDP, IDR,
    TP, FP, FN, IDSW, total_gt, sum_iou.
    """
    from trackeval.metrics import CLEAR, HOTA, Identity

    pred_remapped = remap_pred_ids(gt, pred, iou_threshold)
    data = _build_trackeval_data(gt, pred_remapped)

    threshold_cfg = {"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}
    hota_res = HOTA({"PRINT_CONFIG": False}).eval_sequence(data)
    clear_res = CLEAR(threshold_cfg).eval_sequence(data)
    id_res = Identity(threshold_cfg).eval_sequence(data)

    return {
        "MOTA": float(clear_res["MOTA"]),
        "MOTP": float(clear_res["MOTP"]),
        "TP": int(clear_res["CLR_TP"]),
        "FP": int(clear_res["CLR_FP"]),
        "FN": int(clear_res["CLR_FN"]),
        "IDSW": int(clear_res["IDSW"]),
        "total_gt": int(data["num_gt_dets"]),
        "sum_iou": float(clear_res["MOTP_sum"]),
        "IDF1": float(id_res["IDF1"]),
        "IDP": float(id_res["IDP"]),
        "IDR": float(id_res["IDR"]),
        "HOTA": float(np.mean(hota_res["HOTA"])),
        "DetA": float(np.mean(hota_res["DetA"])),
        "AssA": float(np.mean(hota_res["AssA"])),
    }
