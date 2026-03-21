"""
Tracking evaluation metrics: MOTA, HOTA, IDF1, and supporting utilities.

Works with YOLO MOT format (normalised bboxes) produced by the annotation tool.
All matching is IoU-based in pixel space.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
        cx, cy, w, h = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        result.setdefault(frame_id, []).append({
            "id": track_id,
            "bbox": [x1, y1, x2, y2],
        })
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


# ── CLEAR MOT metrics ───────────────────────────────────────────────────────

def compute_clear_mot(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """
    CLEAR MOT metrics: MOTA, MOTP, TP, FP, FN, IDSW.

    Parameters
    ----------
    gt, pred : {frame_id: [{id, bbox}, ...]}
    iou_threshold : IoU gate for matching.
    """
    all_frames = sorted(set(gt) | set(pred))
    tp = fp = fn = idsw = 0
    total_gt = 0
    sum_iou = 0.0
    prev_map: dict[int, int] = {}  # gt_id → pred_id from previous frame

    for fid in all_frames:
        g_dets = gt.get(fid, [])
        p_dets = pred.get(fid, [])
        total_gt += len(g_dets)

        matches, um_g, um_p = match_frame(g_dets, p_dets, iou_threshold)
        fp += len(um_p)
        fn += len(um_g)
        tp += len(matches)

        # Compute MOTP IoU sum for matched pairs
        bbox_by_id_g = {d["id"]: d["bbox"] for d in g_dets}
        bbox_by_id_p = {d["id"]: d["bbox"] for d in p_dets}
        for gt_id, pred_id in matches:
            sum_iou += bbox_iou(bbox_by_id_g[gt_id], bbox_by_id_p[pred_id])
            if gt_id in prev_map and prev_map[gt_id] != pred_id:
                idsw += 1
            prev_map[gt_id] = pred_id

    mota = 1.0 - (fp + fn + idsw) / total_gt if total_gt else 0.0
    motp = sum_iou / tp if tp else 0.0

    return {
        "MOTA": mota,
        "MOTP": motp,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "IDSW": idsw,
        "total_gt": total_gt,
        "sum_iou": sum_iou,
    }


# ── IDF1 ─────────────────────────────────────────────────────────────────────

def compute_idf1(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """
    ID F1 score (IDF1).

    Computes the optimal global ID assignment then measures how consistently
    predicted IDs match ground truth IDs across all frames.
    """
    all_frames = sorted(set(gt) | set(pred))

    # Count co-occurrences between gt and pred track IDs
    gt_ids: set[int] = set()
    pred_ids: set[int] = set()
    gt_len: dict[int, int] = defaultdict(int)
    pred_len: dict[int, int] = defaultdict(int)
    pair_count: dict[tuple[int, int], int] = defaultdict(int)

    for fid in all_frames:
        g_dets = gt.get(fid, [])
        p_dets = pred.get(fid, [])
        for g in g_dets:
            gt_ids.add(g["id"])
            gt_len[g["id"]] += 1
        for p in p_dets:
            pred_ids.add(p["id"])
            pred_len[p["id"]] += 1

        matches, _, _ = match_frame(g_dets, p_dets, iou_threshold)
        for gt_id, pred_id in matches:
            pair_count[(gt_id, pred_id)] += 1

    if not gt_ids or not pred_ids:
        return {"IDF1": 0.0, "IDP": 0.0, "IDR": 0.0}

    gt_id_list = sorted(gt_ids)
    pred_id_list = sorted(pred_ids)
    ng, np_ = len(gt_id_list), len(pred_id_list)
    gt_idx = {v: i for i, v in enumerate(gt_id_list)}
    pred_idx = {v: i for i, v in enumerate(pred_id_list)}

    # Cost matrix: how many frames are NOT matched if we assign gt_i → pred_j
    cost = np.zeros((ng, np_), dtype=float)
    for i, gid in enumerate(gt_id_list):
        for j, pid in enumerate(pred_id_list):
            c = pair_count.get((gid, pid), 0)
            cost[i, j] = gt_len[gid] + pred_len[pid] - 2 * c

    rows, cols = linear_sum_assignment(cost)

    idtp = 0
    for r, c in zip(rows, cols):
        gid = gt_id_list[r]
        pid = pred_id_list[c]
        idtp += pair_count.get((gid, pid), 0)

    total_gt_dets = sum(gt_len.values())
    total_pred_dets = sum(pred_len.values())

    idp = idtp / total_pred_dets if total_pred_dets else 0.0
    idr = idtp / total_gt_dets if total_gt_dets else 0.0
    idf1 = 2 * idp * idr / (idp + idr) if (idp + idr) else 0.0

    return {"IDF1": idf1, "IDP": idp, "IDR": idr}


# ── HOTA ─────────────────────────────────────────────────────────────────────

def _compute_hota_at_alpha(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    alpha: float,
) -> dict[str, float]:
    """HOTA components at a single IoU threshold alpha."""
    all_frames = sorted(set(gt) | set(pred))

    # Per-frame matches at this alpha
    frame_matches: list[list[tuple[int, int]]] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for fid in all_frames:
        g_dets = gt.get(fid, [])
        p_dets = pred.get(fid, [])
        matches, um_g, um_p = match_frame(g_dets, p_dets, alpha)
        frame_matches.append(matches)
        total_tp += len(matches)
        total_fn += len(um_g)
        total_fp += len(um_p)

    # Detection accuracy
    det_a = total_tp / (total_tp + total_fn + total_fp) if (total_tp + total_fn + total_fp) else 0.0

    # Association accuracy: for each TP pair (gt_id, pred_id), measure how
    # consistently they are associated across all frames.
    # TPA(c) = frames where both gt_id and pred_id are TP-matched to each other
    # FPA(c) = frames where pred_id is TP-matched to a different gt_id
    # FNA(c) = frames where gt_id is TP-matched to a different pred_id
    pair_tpa: dict[tuple[int, int], int] = defaultdict(int)
    gt_in_tp: dict[int, int] = defaultdict(int)    # gt_id → total times as TP
    pred_in_tp: dict[int, int] = defaultdict(int)  # pred_id → total times as TP

    for matches in frame_matches:
        for gt_id, pred_id in matches:
            pair_tpa[(gt_id, pred_id)] += 1
            gt_in_tp[gt_id] += 1
            pred_in_tp[pred_id] += 1

    ass_sum = 0.0
    for (gt_id, pred_id), tpa in pair_tpa.items():
        fpa = pred_in_tp[pred_id] - tpa  # pred matched to other gt
        fna = gt_in_tp[gt_id] - tpa      # gt matched to other pred
        ass_sum += tpa / (tpa + fpa + fna) * tpa  # weight by TPA

    ass_a = ass_sum / total_tp if total_tp else 0.0

    hota = np.sqrt(det_a * ass_a)
    return {"HOTA": hota, "DetA": det_a, "AssA": ass_a}


def compute_hota(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    alphas: list[float] | None = None,
) -> dict[str, float]:
    """
    HOTA (Higher Order Tracking Accuracy).

    Averages detection and association accuracy over multiple IoU thresholds.
    Default: 19 thresholds from 0.05 to 0.95.
    """
    if alphas is None:
        alphas = [0.05 * i for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95

    hota_vals, deta_vals, assa_vals = [], [], []
    for alpha in alphas:
        r = _compute_hota_at_alpha(gt, pred, alpha)
        hota_vals.append(r["HOTA"])
        deta_vals.append(r["DetA"])
        assa_vals.append(r["AssA"])

    return {
        "HOTA": float(np.mean(hota_vals)),
        "DetA": float(np.mean(deta_vals)),
        "AssA": float(np.mean(assa_vals)),
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
    used_gt: set[int] = set()
    for r, c in zip(rows, cols):
        if cooccur[r, c] > 0:
            remap[pred_ids[c]] = gt_ids[r]
            used_gt.add(gt_ids[r])

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


# ── Aggregate all metrics ───────────────────────────────────────────────────

def evaluate(
    gt: dict[int, list[dict]],
    pred: dict[int, list[dict]],
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """
    Compute all tracking metrics for one sequence.

    Pred IDs are first remapped to the optimal bijection onto GT IDs so that
    IDSW reflects true track fragmentation, not arbitrary ID numbering.

    Returns dict with MOTA, MOTP, HOTA, DetA, AssA, IDF1, IDP, IDR,
    TP, FP, FN, IDSW, total_gt.
    """
    pred_remapped = remap_pred_ids(gt, pred, iou_threshold)
    clear = compute_clear_mot(gt, pred_remapped, iou_threshold)
    idf1 = compute_idf1(gt, pred_remapped, iou_threshold)
    hota = compute_hota(gt, pred_remapped)
    return {**clear, **idf1, **hota}
