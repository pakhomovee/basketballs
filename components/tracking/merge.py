"""
Track merging: reduce N tracks to max_tracks (10) via greedy merge.

Heap-based greedy merge with cost recomputation after each merge step.
Cost combines appearance similarity, spatial distance at transition points,
and IoU with adaptive weighting based on temporal gap.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq

import numpy as np
from tqdm import tqdm

from common.distances import (
    bbox_bottom_mid_distance,
    bbox_iou,
    cosine_dist,
    bbox_size_ratio,
)


@dataclass
class _MergeGroup:
    """Aggregate properties of a merged group for cost computation."""

    indices: list[int] = field(default_factory=list)
    frames: set[int] = field(default_factory=set)
    gallery: list[np.ndarray] = field(default_factory=list)
    first_bbox: np.ndarray | None = None
    last_bbox: np.ndarray | None = None
    first_frame: int = 0
    last_frame: int = 0
    team: int | None = None
    _mean_emb: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_track(cls, track, idx: int, team: int | None = None):
        fids = list(getattr(track, "frame_ids", []))
        bbox = getattr(track, "bbox", None)
        first_b = getattr(track, "_first_bbox", bbox)
        last_b = getattr(track, "_last_bbox", bbox)
        return cls(
            indices=[idx],
            frames=set(fids),
            gallery=list(getattr(track, "gallery", []) or []),
            first_bbox=(np.asarray(first_b, dtype=float).copy() if first_b is not None else None),
            last_bbox=(np.asarray(last_b, dtype=float).copy() if last_b is not None else None),
            first_frame=min(fids) if fids else 0,
            last_frame=max(fids) if fids else 0,
            team=team,
        )

    @property
    def mean_embedding(self) -> np.ndarray | None:
        if self._mean_emb is None and self.gallery:
            self._mean_emb = np.mean(self.gallery, axis=0)
        return self._mean_emb

    def absorb(self, other: _MergeGroup):
        """Merge *other* group into this one."""
        self.indices.extend(other.indices)
        self.frames |= other.frames
        self.gallery.extend(other.gallery)
        self._mean_emb = None
        if other.first_frame < self.first_frame:
            self.first_bbox = other.first_bbox
            self.first_frame = other.first_frame
        if other.last_frame > self.last_frame:
            self.last_bbox = other.last_bbox
            self.last_frame = other.last_frame


def _group_cost(
    ga: _MergeGroup,
    gb: _MergeGroup,
    *,
    bbox_gate: float,
    frame_gap_scale: float,
    w_app: float,
    w_dist: float,
    w_iou: float,
    cost_threshold: float,
) -> float:
    """
    Cost of merging two groups (lower = more similar).

    Components (sequential case):
    - Appearance: cosine distance between mean embeddings
    - Distance: bbox feet distance at transition, gate scaled by sqrt(frame_gap)
    - IoU: transition IoU (end-bbox vs start-bbox), faded for large gaps

    Returns inf for invalid pairs (team mismatch, frame overlap, gated out).
    """
    INF = float("inf")

    if ga.team is not None and gb.team is not None and ga.team != gb.team:
        return INF
    if ga.frames & gb.frames:
        return INF

    # Temporal relationship
    if ga.last_frame < gb.first_frame:
        end_bbox, start_bbox = ga.last_bbox, gb.first_bbox
        frame_gap = max(0, gb.first_frame - ga.last_frame - 1)
        sequential = True
    elif gb.last_frame < ga.first_frame:
        end_bbox, start_bbox = gb.last_bbox, ga.first_bbox
        frame_gap = max(0, ga.first_frame - gb.last_frame - 1)
        sequential = True
    else:
        sequential = False
        end_bbox = ga.last_bbox
        start_bbox = gb.first_bbox
        frame_gap = 0

    # Appearance (cosine distance between cached mean embeddings)
    ma, mb = ga.mean_embedding, gb.mean_embedding
    app_c = cosine_dist(ma, mb) if ma is not None and mb is not None else 0.5

    if sequential:
        effective_gate = bbox_gate * (1 + frame_gap_scale * np.sqrt(frame_gap))

        if (
            end_bbox is not None
            and start_bbox is not None
            and len(np.asarray(end_bbox).ravel()) >= 4
            and len(np.asarray(start_bbox).ravel()) >= 4
        ):
            dist = bbox_bottom_mid_distance(end_bbox, start_bbox)
            if dist > effective_gate:
                return INF
            dist_c = min(1.0, dist / effective_gate)

            sr = bbox_size_ratio(end_bbox, start_bbox)
            if sr < 0.3:
                return INF

            iou_c = 1.0 - bbox_iou(
                np.asarray(end_bbox).ravel()[:4],
                np.asarray(start_bbox).ravel()[:4],
            )
        else:
            dist_c = 0.5
            iou_c = 0.5

        iou_fade = max(0.0, 1.0 - frame_gap / 10.0)
        w_iou_eff = w_iou * iou_fade
        extra = w_iou - w_iou_eff
        w_app_adj = w_app + extra * 0.6
        w_dist_adj = w_dist + extra * 0.4

        cost = w_app_adj * app_c + w_dist_adj * dist_c + w_iou_eff * iou_c
    else:
        # Interleaved non-overlapping frames: appearance-dominated cost
        cost = app_c * 0.8 + 0.2

    return cost if cost <= cost_threshold else INF


def _frames_set(track) -> set[int]:
    """Set of frame IDs where this track appears."""
    return set(getattr(track, "frame_ids", []))


def greedy_merge(
    tracks: list,
    max_tracks: int,
    bbox_gate: float,
    w_app: float,
    w_dist: float,
    w_iou: float,
    frame_gap_scale: float = 0.5,
    track_id_to_team: dict[int, int] | None = None,
    cost_threshold: float = 0.7,
) -> list[list[int]]:
    """
    Heap-based greedy merge with cost recomputation.

    Merges lowest-cost non-overlapping groups until max_tracks remain
    or no valid merge exists below cost_threshold.

    After each merge the affected group's properties (gallery, frame coverage,
    bboxes) are updated and new candidate costs are pushed into the heap,
    giving downstream merges accurate cost information.
    """
    n = len(tracks)
    if n <= max_tracks:
        return [[i] for i in range(n)]

    # One group per track
    groups: dict[int, _MergeGroup] = {}
    for i in range(n):
        team = track_id_to_team.get(getattr(tracks[i], "track_id", None)) if track_id_to_team else None
        groups[i] = _MergeGroup.from_track(tracks[i], i, team=team)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cost_kwargs = dict(
        bbox_gate=bbox_gate,
        frame_gap_scale=frame_gap_scale,
        w_app=w_app,
        w_dist=w_dist,
        w_iou=w_iou,
        cost_threshold=cost_threshold,
    )

    INF = float("inf")
    heap: list[tuple[float, int, int, int]] = []
    counter = 0

    for i in range(n):
        for j in range(i + 1, n):
            c = _group_cost(groups[i], groups[j], **cost_kwargs)
            if c < INF:
                heapq.heappush(heap, (c, counter, i, j))
                counter += 1

    num_groups = n
    pbar = tqdm(total=max(0, num_groups - max_tracks), desc="Merging tracks")

    while num_groups > max_tracks and heap:
        c, _, i, j = heapq.heappop(heap)
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if ri not in groups or rj not in groups:
            continue

        actual_cost = _group_cost(groups[ri], groups[rj], **cost_kwargs)
        if actual_cost >= INF:
            continue
        if actual_cost > c * 1.3 + 0.05:
            heapq.heappush(heap, (actual_cost, counter, ri, rj))
            counter += 1
            continue

        groups[ri].absorb(groups[rj])
        parent[rj] = ri
        del groups[rj]
        num_groups -= 1
        pbar.update(1)

        for k in list(groups.keys()):
            if k == ri:
                continue
            new_c = _group_cost(groups[ri], groups[k], **cost_kwargs)
            if new_c < INF:
                heapq.heappush(heap, (new_c, counter, ri, k))
                counter += 1

    pbar.close()

    if num_groups > max_tracks:
        print(
            f"Warning: could not reduce to {max_tracks} tracks "
            f"(stopped at {num_groups}, cost threshold {cost_threshold})"
        )
    print(f"tracks left: {num_groups}")

    result: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        result.setdefault(r, []).append(i)
    return list(result.values())


def merge_tracks_into(
    tracks: list,
    partition: list[list[int]],
    survivor_id_map: dict[int, int],
) -> list:
    """
    Merge tracks according to *partition*. Each cluster becomes one track.

    *survivor_id_map* is filled with ``old_track_id -> surviving_track_id``.
    """
    from .track import Track, TrackState

    merged: list[Track] = []
    for cluster_indices in partition:
        if not cluster_indices:
            continue
        base = tracks[cluster_indices[0]]
        base_id = base.track_id
        all_fids = list(base.frame_ids)
        all_hist = list(base.history)
        all_gallery = list(base.gallery)
        cluster_tracks = [tracks[i] for i in cluster_indices]

        for idx in cluster_indices[1:]:
            t = tracks[idx]
            survivor_id_map[t.track_id] = base_id
            all_fids.extend(t.frame_ids)
            all_hist.extend(t.history)
            all_gallery.extend(t.gallery)

        order = np.argsort(all_fids)
        base.frame_ids = [all_fids[i] for i in order]
        base.history = [all_hist[i] for i in order]
        base.gallery = deque(all_gallery, maxlen=getattr(base.gallery, "maxlen", 30))
        base.state = TrackState.CONFIRMED

        last_fid, first_fid = max(all_fids), min(all_fids)
        for t in cluster_tracks:
            fids = getattr(t, "frame_ids", [])
            if fids and max(fids) == last_fid:
                last_b = np.asarray(getattr(t, "_last_bbox", t.bbox), dtype=float)
                base.bbox = last_b.copy()
                base._last_bbox = last_b.copy()
                break
        for t in cluster_tracks:
            fids = getattr(t, "frame_ids", [])
            if fids and min(fids) == first_fid:
                base._first_bbox = np.asarray(getattr(t, "_first_bbox", t.bbox), dtype=float).copy()
                break

        merged.append(base)

    return merged
