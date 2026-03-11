"""
Evaluation utilities: MOTA, IDF1, benchmark runner.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import load_detections_csv
from .flow_tracker import FlowTracker

log = logging.getLogger(__name__)


def _get_video_frame_width(video_path: str) -> float | None:
    """Return video frame width in pixels, or None if unavailable."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return width if width > 0 else None
    except Exception:
        return None


def _flow_tracker_results_to_tracks(det_list: list[tuple[int, object]]) -> list:
    """Build track-like objects (frame_ids, history) from FlowTracker output for evaluation."""
    from types import SimpleNamespace

    track_data: dict[int, list[tuple[int, list[float]]]] = {}
    for frame_id, player in det_list:
        pid = player.player_id
        if pid < 0:
            continue
        pos = player.court_position
        if pos is None:
            pos = (0.0, 0.0)
        track_data.setdefault(pid, []).append((frame_id, list(pos)))

    tracks = []
    for track_id, points in sorted(track_data.items()):
        points.sort(key=lambda x: x[0])
        t = SimpleNamespace(
            track_id=track_id,
            frame_ids=[p[0] for p in points],
            history=[p[1] for p in points],
        )
        tracks.append(t)
    return tracks


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
    tracker,
    detections: list[dict],
    distance_threshold: float = 2.0,
):
    """
    Compute MOTA and IDF1 from tracker results vs detection ground truth.

    Parameters
    ----------
    tracker : object with .tracks attribute
        Each track has .frame_ids and .history (list of positions).
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
        gt_by_frame.setdefault(d["frame_id"], []).append({"id": d["detection_id"], "pos": d["field_coords"]})

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


def _csv_to_player_detections(data: list[dict]):
    """Convert benchmark CSV format to dict[frame_id, list[Player]] for FlowTracker."""
    from common.classes.player import Player

    detections: dict[int, list[Player]] = {}
    for d in data:
        fid = d["frame_id"]
        p = Player(
            bbox=d["bbox"],
            court_position=tuple(d["field_coords"]) if d.get("field_coords") else None,
        )
        detections.setdefault(fid, []).append(p)
    return detections


def _extract_embeddings_for_segment(
    video_path: str,
    player_dets: dict[int, list],
    seg_model: str = "yolov8n-seg.pt",
    reid_weights: str | None = None,
) -> None:
    """Extract color + ReID embeddings from video for benchmark segment."""
    from team_clustering.embedding import extract_player_embeddings

    # Embedding extractors expect 0-based frame ids; CSV may be 1-based
    min_fid = min(player_dets) if player_dets else 0
    remapped = {fid - min_fid: players for fid, players in player_dets.items()}
    extract_player_embeddings(video_path, remapped, seg_model=seg_model)
    # Copy embeddings back to original keys (same Player objects)
    if reid_weights and os.path.isfile(reid_weights):
        from common.utils.utils import get_device
        from reidentification import extract_reid_embeddings

        extract_reid_embeddings(video_path, remapped, reid_weights, device=get_device())


def _find_segment_video(segment_path: str) -> str | None:
    """Find video in entry folder. Tries video.mp4, video.avi, etc. or any .mp4/.avi file."""
    for name in ("video", "clip"):
        for ext in (".mp4", ".avi", ".mkv", ".mov"):
            path = os.path.join(segment_path, name + ext)
            if os.path.isfile(path):
                return path
    for f in os.listdir(segment_path):
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            return os.path.join(segment_path, f)
    return None


def run_benchmark(
    tracking_dir: str,
    distance_threshold: float = 6.0,
    max_time_since_update: int = 3,
    eval_distance: float = 2.0,
    reid_weights: str | None = None,
    seg_model: str = "yolov8n-seg.pt",
):
    """
    Run the FlowTracker + evaluation loop over all entry folders in *tracking_dir*.

    Dataset format: tracking_dir/entry/[annotations.csv, homography, video]
    Each entry folder contains annotations, homography, and video.

    Parameters
    ----------
    reid_weights : str, optional
        Path to ReID model weights. Used when video is present.
    seg_model : str
        YOLO seg model for color embeddings.

    Returns
    -------
    dict
        Aggregated ``{MOTA, IDF1, FP, FN, IDSW, TP, total_gt}``
    """
    from types import SimpleNamespace

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

        player_dets = _csv_to_player_detections(detections)
        video_path = _find_segment_video(segment_path)
        frame_width = None
        if video_path:
            log.info("  Extracting embeddings from %s", video_path)
            _extract_embeddings_for_segment(
                video_path,
                player_dets,
                seg_model=seg_model,
                reid_weights=reid_weights,
            )
            frame_width = _get_video_frame_width(video_path)
        else:
            log.warning("  No video in %s — FlowTracker uses spatial only (no embeddings)", segment_path)
        tracker_obj = FlowTracker(num_tracks=10, frame_width=frame_width)
        tracker_obj.track(player_dets)
        det_list = [(fid, p) for fid in sorted(player_dets) for p in player_dets[fid]]
        tracks = _flow_tracker_results_to_tracks(det_list)
        tracker = SimpleNamespace(tracks=tracks)

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
