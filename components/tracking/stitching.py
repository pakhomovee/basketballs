"""Post-tracking tracklet stitching via hierarchical agglomerative clustering."""

from __future__ import annotations

import heapq
import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from common.classes.player import Player, PlayersDetections
from common.distances import bbox_iou
from config import AppConfig, StitcherConfig

logger = logging.getLogger(__name__)

# Sentinel for infeasible assignment entries.
_INF: float = 1e5


@dataclass
class _TrackletSummary:
    """Per-tracklet statistics extracted from PlayersDetections."""

    track_id: int
    start_frame: int
    end_frame: int
    frame_count: int

    reid_gallery: list[np.ndarray] = field(default_factory=list)
    color_gallery: list[np.ndarray] = field(default_factory=list)

    positions: list[tuple[int, float, float]] = field(default_factory=list)

    jersey_counter: Counter = field(default_factory=Counter)

    exit_pos: tuple[float, float] | None = None
    entry_pos: tuple[float, float] | None = None
    exit_vel: tuple[float, float] | None = None  # (vx, vy) m/frame
    entry_vel: tuple[float, float] | None = None  # (vx, vy) m/frame

    min_jersey_count: int = 2

    @property
    def jersey_num(self) -> int | None:
        if not self.jersey_counter:
            return None
        num, count = self.jersey_counter.most_common(1)[0]
        return num if count >= self.min_jersey_count else None


class _Chain:
    """A linked sequence of tracklets representing one player identity."""

    __slots__ = (
        "track_ids",
        "intervals",
        "reid_gallery",
        "color_gallery",
        "jersey_counter",
        "frame_coverage",
        "last_exit_pos",
        "last_exit_vel",
        "last_exit_frame",
        "_max_gallery",
        "_min_jersey_count",
    )

    def __init__(self, t: _TrackletSummary, cfg: StitcherConfig) -> None:
        self._max_gallery: int = cfg.max_gallery
        self._min_jersey_count: int = cfg.min_jersey_count
        self.track_ids: list[int] = [t.track_id]
        self.intervals: list[tuple[int, int]] = [(t.start_frame, t.end_frame)]
        self.reid_gallery: list[np.ndarray] = list(t.reid_gallery)
        self.color_gallery: list[np.ndarray] = list(t.color_gallery)
        self.jersey_counter: Counter = Counter(t.jersey_counter)
        self.frame_coverage: int = t.frame_count
        self.last_exit_pos = t.exit_pos
        self.last_exit_vel = t.exit_vel
        self.last_exit_frame = t.end_frame

    @property
    def jersey_num(self) -> int | None:
        if not self.jersey_counter:
            return None
        num, count = self.jersey_counter.most_common(1)[0]
        return num if count >= self._min_jersey_count else None

    def append(self, t: _TrackletSummary) -> None:
        """Link tracklet *t* to the end of this chain."""
        self.track_ids.append(t.track_id)
        self.intervals.append((t.start_frame, t.end_frame))
        self.reid_gallery.extend(t.reid_gallery)
        if len(self.reid_gallery) > self._max_gallery:
            self.reid_gallery = self.reid_gallery[-self._max_gallery :]
        self.color_gallery.extend(t.color_gallery)
        if len(self.color_gallery) > self._max_gallery:
            self.color_gallery = self.color_gallery[-self._max_gallery :]
        self.jersey_counter += t.jersey_counter
        self.frame_coverage += t.frame_count
        self.last_exit_pos = t.exit_pos
        self.last_exit_vel = t.exit_vel
        self.last_exit_frame = t.end_frame

    def merge_from(self, other: _Chain) -> None:
        """Absorb *other* chain into this one (galleries, intervals, etc.)."""
        self.track_ids.extend(other.track_ids)
        self.intervals.extend(other.intervals)
        self.intervals.sort()
        self.reid_gallery.extend(other.reid_gallery)
        if len(self.reid_gallery) > self._max_gallery:
            self.reid_gallery = self.reid_gallery[-self._max_gallery :]
        self.color_gallery.extend(other.color_gallery)
        if len(self.color_gallery) > self._max_gallery:
            self.color_gallery = self.color_gallery[-self._max_gallery :]
        self.jersey_counter += other.jersey_counter
        self.frame_coverage += other.frame_coverage
        if other.last_exit_frame > self.last_exit_frame:
            self.last_exit_frame = other.last_exit_frame
            self.last_exit_pos = other.last_exit_pos
            self.last_exit_vel = other.last_exit_vel


# helpers


def _estimate_velocity(
    positions: list[tuple[int, float, float]],
    window: int,
    from_end: bool,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Return ``(position, velocity)`` from the first or last *window* observations."""
    if len(positions) < 2:
        if positions:
            return (positions[-1 if from_end else 0][1], positions[-1 if from_end else 0][2]), None
        return None, None

    if from_end:
        pts = positions[-window:]
    else:
        pts = positions[:window]

    pos = (pts[-1][1], pts[-1][2]) if from_end else (pts[0][1], pts[0][2])

    dt = pts[-1][0] - pts[0][0]
    if dt <= 0:
        return pos, None
    vx = (pts[-1][1] - pts[0][1]) / dt
    vy = (pts[-1][2] - pts[0][2]) / dt
    return pos, (vx, vy)


def _gallery_pairwise_dists(
    ga: list[np.ndarray],
    gb: list[np.ndarray],
) -> np.ndarray:
    """All pairwise cosine distances between two galleries.  Shape (Na*Nb,)."""
    A = np.stack(ga)  # (Na, D)
    B = np.stack(gb)  # (Nb, D)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    sims = A @ B.T  # (Na, Nb)
    return (1.0 - sims).ravel()


def _gallery_gallery_min(
    ga: list[np.ndarray],
    gb: list[np.ndarray],
) -> float:
    """Minimum cosine distance between any pair from two galleries."""
    if not ga or not gb:
        return 1.0
    return float(_gallery_pairwise_dists(ga, gb).min())


def _gallery_gallery_robust(
    ga: list[np.ndarray],
    gb: list[np.ndarray],
    percentile: float = 10.0,
) -> float:
    """Robust gallery-to-gallery distance using a low percentile."""
    if not ga or not gb:
        return 1.0
    dists = _gallery_pairwise_dists(ga, gb)
    return float(np.percentile(dists, percentile))


def _intervals_overlap(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
) -> bool:
    """Check whether any interval in *a* overlaps any in *b* (both sorted)."""
    i, j = 0, 0
    while i < len(a) and j < len(b):
        a_s, a_e = a[i]
        b_s, b_e = b[j]
        if a_s <= b_e and b_s <= a_e:
            return True
        if a_e < b_e:
            i += 1
        else:
            j += 1
    return False


# pairwise tracklet cost


def _tracklet_pair_cost(
    a: _TrackletSummary,
    b: _TrackletSummary,
    fps: float,
    cfg: StitcherConfig,
) -> float:
    """Appearance + spatial cost between two non-overlapping tracklets."""
    if a.end_frame < b.start_frame:
        earlier, later = a, b
    elif b.end_frame < a.start_frame:
        earlier, later = b, a
    else:
        return _INF

    gap_frames = later.start_frame - earlier.end_frame
    gap_sec = max(gap_frames, 1) / fps

    # spatial
    spatial_fade = max(0.0, 1.0 - gap_sec / cfg.spatial_fade_sec)

    if spatial_fade == 0.0 or earlier.exit_pos is None or later.entry_pos is None:
        spatial_cost = 0.5  # neutral
    else:
        if earlier.exit_vel is not None:
            vx, vy = earlier.exit_vel
            pred_x = earlier.exit_pos[0] + vx * gap_frames
            pred_y = earlier.exit_pos[1] + vy * gap_frames
        else:
            pred_x, pred_y = earlier.exit_pos

        dist = float(np.hypot(pred_x - later.entry_pos[0], pred_y - later.entry_pos[1]))
        max_dist = cfg.max_speed * gap_sec * cfg.speed_safety
        spatial_cost = min(dist / max(max_dist, 1e-6), 1.0)

    # combine
    w_pos_eff = cfg.w_position * spatial_fade
    total_cost = w_pos_eff * spatial_cost
    total_weight = w_pos_eff

    if earlier.reid_gallery and later.reid_gallery:
        reid_cost = _gallery_gallery_min(earlier.reid_gallery, later.reid_gallery)
        total_cost += cfg.w_app * reid_cost
        total_weight += cfg.w_app

    if earlier.color_gallery and later.color_gallery:
        color_cost = _gallery_gallery_min(earlier.color_gallery, later.color_gallery)
        total_cost += cfg.w_color * color_cost
        total_weight += cfg.w_color

    cost = total_cost / total_weight if total_weight > 0 else 0.5

    # Jersey match bonus
    j_a = a.jersey_num
    j_b = b.jersey_num
    if j_a is not None and j_b is not None and j_a == j_b:
        cost *= 0.3

    return cost


def _can_merge_groups(ga: _Chain, gb: _Chain, cfg: StitcherConfig) -> bool:
    """Group-level merge feasibility: temporal overlap, jersey, colour."""
    if _intervals_overlap(ga.intervals, gb.intervals):
        return False
    ja, jb = ga.jersey_num, gb.jersey_num
    if ja is not None and jb is not None and ja != jb:
        return False
    if (
        len(ga.color_gallery) >= cfg.color_conflict_min_samples
        and len(gb.color_gallery) >= cfg.color_conflict_min_samples
    ):
        if _gallery_gallery_min(ga.color_gallery, gb.color_gallery) > cfg.color_conflict_gate:
            return False
    return True


# pre-stitching splitting


def _split_inconsistent_tracklets(detections: PlayersDetections, cfg: StitcherConfig) -> int:
    """Detect and split tracklets that contain an ID switch."""
    tracklet_frames: dict[int, list[int]] = {}  # pid → sorted frame ids
    tracklet_embeds: dict[int, dict[int, np.ndarray]] = {}  # pid → {frame: embed}

    for frame_id in sorted(detections):
        players = detections[frame_id]
        tracked = [(i, p) for i, p in enumerate(players) if p.player_id >= 1]

        occluded_indices: set[int] = set()
        for ai, (idx_a, pa) in enumerate(tracked):
            if not pa.bbox or len(pa.bbox) < 4:
                continue
            for bi in range(ai + 1, len(tracked)):
                idx_b, pb = tracked[bi]
                if not pb.bbox or len(pb.bbox) < 4:
                    continue
                if bbox_iou(pa.bbox, pb.bbox) > cfg.occlusion_iou_thresh:
                    occluded_indices.add(idx_a)
                    occluded_indices.add(idx_b)

        for idx, player in enumerate(players):
            pid = player.player_id
            if pid < 1 or player.reid_embedding is None:
                continue
            if idx in occluded_indices:
                continue
            tracklet_frames.setdefault(pid, []).append(frame_id)
            tracklet_embeds.setdefault(pid, {})[frame_id] = player.reid_embedding

    max_pid = max(
        (p.player_id for ps in detections.values() for p in ps),
        default=0,
    )
    next_pid = max_pid + 1
    total_splits = 0

    for pid, frames in tracklet_frames.items():
        embeds = [tracklet_embeds[pid][f] for f in frames]
        n = len(embeds)
        if n < 2 * cfg.split_min_side:
            continue

        E = np.stack(embeds)  # (n, D)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

        best_dist = 0.0
        best_k = -1
        for k in range(cfg.split_min_side, n - cfg.split_min_side + 1):
            left_mean = E[:k].mean(axis=0)
            right_mean = E[k:].mean(axis=0)
            left_mean /= np.linalg.norm(left_mean) + 1e-8
            right_mean /= np.linalg.norm(right_mean) + 1e-8
            dist = 1.0 - float(left_mean @ right_mean)
            if dist > best_dist:
                best_dist = dist
                best_k = k

        if best_dist < cfg.split_threshold or best_k < 0:
            continue

        # Split: frames[best_k:] → new player_id.
        split_frame = frames[best_k]
        new_pid = next_pid
        next_pid += 1
        total_splits += 1
        logger.info(
            "Splitting tracklet %d at frame %d (dist=%.3f) → new id %d",
            pid,
            split_frame,
            best_dist,
            new_pid,
        )

        for frame_id in sorted(detections):
            if frame_id < split_frame:
                continue
            for player in detections[frame_id]:
                if player.player_id == pid:
                    player.player_id = new_pid

    return total_splits


# extract tracklets


def _extract_tracklets(detections: PlayersDetections, cfg: StitcherConfig) -> list[_TrackletSummary]:
    """Build one ``_TrackletSummary`` per unique ``player_id >= 1``."""
    tracks: dict[int, _TrackletSummary] = {}

    for frame_id in sorted(detections):
        players = detections[frame_id]
        tracked = [(i, p) for i, p in enumerate(players) if p.player_id >= 1]

        # Identify players whose bbox is occluded by another tracked player.
        occluded_indices: set[int] = set()
        for ai, (idx_a, pa) in enumerate(tracked):
            if not pa.bbox or len(pa.bbox) < 4:
                continue
            for bi in range(ai + 1, len(tracked)):
                idx_b, pb = tracked[bi]
                if not pb.bbox or len(pb.bbox) < 4:
                    continue
                if bbox_iou(pa.bbox, pb.bbox) > cfg.occlusion_iou_thresh:
                    occluded_indices.add(idx_a)
                    occluded_indices.add(idx_b)

        for idx, player in enumerate(players):
            pid = player.player_id
            if pid < 1:
                continue

            if pid not in tracks:
                tracks[pid] = _TrackletSummary(
                    track_id=pid,
                    start_frame=frame_id,
                    end_frame=frame_id,
                    frame_count=0,
                    min_jersey_count=cfg.min_jersey_count,
                )

            t = tracks[pid]
            t.end_frame = frame_id
            t.frame_count += 1

            is_clean = idx not in occluded_indices
            if is_clean and player.reid_embedding is not None:
                if len(t.reid_gallery) < cfg.max_gallery:
                    t.reid_gallery.append(player.reid_embedding)
            if is_clean and player.embedding is not None:
                if len(t.color_gallery) < cfg.max_gallery:
                    t.color_gallery.append(player.embedding)

            if player.court_position is not None:
                t.positions.append(
                    (
                        frame_id,
                        player.court_position[0],
                        player.court_position[1],
                    )
                )

            if player.number is not None and player.number.num is not None:
                t.jersey_counter[player.number.num] += 1

    for t in tracks.values():
        t.exit_pos, t.exit_vel = _estimate_velocity(
            t.positions,
            cfg.vel_window,
            from_end=True,
        )
        t.entry_pos, t.entry_vel = _estimate_velocity(
            t.positions,
            cfg.vel_window,
            from_end=False,
        )

    return list(tracks.values())


# stitching algorithm


def stitch_tracklets(
    detections: PlayersDetections,
    *,
    fps: float = 30.0,
    num_players: int = 10,
    cfg: AppConfig | None = None,
) -> None:
    """Merge Hungarian-tracker tracklets into *num_players* global IDs, in-place."""
    scfg = cfg.stitcher if cfg is not None else StitcherConfig()
    n_splits = _split_inconsistent_tracklets(detections, scfg)
    if n_splits:
        logger.info("Pre-stitching: split %d inconsistent tracklets", n_splits)

    summaries = _extract_tracklets(detections, scfg)
    if not summaries:
        logger.warning("stitch_tracklets: no tracklets found")
        return

    n = len(summaries)
    logger.info("Stitching %d tracklets → %d identities", n, num_players)
    summaries.sort(key=lambda t: (t.start_frame, t.end_frame))

    pair_cost = np.full((n, n), _INF, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            c = _tracklet_pair_cost(summaries[i], summaries[j], fps, scfg)
            pair_cost[i, j] = c
            pair_cost[j, i] = c

    groups: list[_Chain | None] = [_Chain(s, scfg) for s in summaries]
    group_cost = pair_cost.copy()
    active: set[int] = set(range(n))

    heap: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if pair_cost[i, j] < _INF:
                heapq.heappush(heap, (pair_cost[i, j], i, j))

    # HAC loop
    while len(active) > num_players and heap:
        cost_val, i, j = heapq.heappop(heap)

        if i not in active or j not in active:
            continue
        if abs(group_cost[i, j] - cost_val) > 1e-9:
            continue

        if not _can_merge_groups(groups[i], groups[j], scfg):
            continue

        # Merge j into i
        groups[i].merge_from(groups[j])
        active.discard(j)
        groups[j] = None

        # Update group costs using single-linkage rule:
        #   new_cost(i, k) = min(old_cost(i, k), old_cost(j, k))
        for k in active:
            if k == i:
                continue
            ci = group_cost[i, k]
            cj = group_cost[j, k]
            if ci >= _INF or cj >= _INF:
                new_c = _INF
            else:
                new_c = min(ci, cj)
            group_cost[i, k] = new_c
            group_cost[k, i] = new_c
            if new_c < _INF:
                heapq.heappush(heap, (new_c, min(i, k), max(i, k)))

    final_groups = [groups[i] for i in sorted(active)]
    logger.info("After HAC: %d groups", len(final_groups))

    # Jersey post-merge
    merged_any = True
    while merged_any:
        merged_any = False
        jersey_map: dict[int, list[int]] = {}
        for gi, g in enumerate(final_groups):
            if g is None:
                continue
            jn = g.jersey_num
            if jn is not None:
                jersey_map.setdefault(jn, []).append(gi)
        for jn, indices in jersey_map.items():
            if len(indices) < 2:
                continue
            indices.sort(key=lambda gi: final_groups[gi].frame_coverage, reverse=True)
            base_idx = indices[0]
            for other_idx in indices[1:]:
                base = final_groups[base_idx]
                other = final_groups[other_idx]
                if other is None:
                    continue
                if not _intervals_overlap(base.intervals, other.intervals):
                    base.merge_from(other)
                    final_groups[other_idx] = None
                    merged_any = True
        final_groups = [g for g in final_groups if g is not None]

    logger.info("After jersey post-merge: %d groups", len(final_groups))

    # Overflow absorption
    final_groups.sort(key=lambda g: g.frame_coverage, reverse=True)
    player_groups = final_groups[:num_players]
    overflow = final_groups[num_players:]

    for ochain in overflow:
        best_gi = -1
        best_cost = scfg.absorb_gate
        for gi, pg in enumerate(player_groups):
            if _intervals_overlap(pg.intervals, ochain.intervals):
                continue
            if (
                len(pg.color_gallery) >= scfg.color_conflict_min_samples
                and len(ochain.color_gallery) >= scfg.color_conflict_min_samples
            ):
                if _gallery_gallery_min(pg.color_gallery, ochain.color_gallery) > scfg.color_conflict_gate:
                    continue
            if pg.reid_gallery and ochain.reid_gallery:
                cost = _gallery_gallery_min(pg.reid_gallery, ochain.reid_gallery)
                if cost < best_cost:
                    best_cost = cost
                    best_gi = gi
        if best_gi >= 0:
            player_groups[best_gi].track_ids.extend(ochain.track_ids)
            player_groups[best_gi].intervals.extend(ochain.intervals)
            player_groups[best_gi].intervals.sort()
            player_groups[best_gi].frame_coverage += ochain.frame_coverage
        else:
            player_groups.append(ochain)

    # Write back
    id_map: dict[int, int] = {}
    for global_id, g in enumerate(player_groups, start=1):
        for tid in g.track_ids:
            id_map[tid] = global_id

    reassigned = 0
    for players in detections.values():
        for player in players:
            old = player.player_id
            if old < 1:
                continue
            player.raw_player_id = old
            new = id_map.get(old, -1)
            if new != old:
                reassigned += 1
            player.player_id = new

    logger.info(
        "Stitching complete: %d global IDs, %d reassigned",
        len(player_groups),
        reassigned,
    )
