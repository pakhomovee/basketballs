"""CP-SAT global tracklet-to-identity assignment.

Replaces greedy gallery stitching with an optimal global solver.

Formulation
-----------
Given T tracklets and K=num_players identity slots, the solver finds an
assignment that **minimises total cost** subject to:

1. **Exactly-one**: every tracklet is assigned to exactly one slot.
2. **No-overlap**: two tracklets that overlap in time cannot share a slot.
3. **Jersey hard-block**: a tracklet whose jersey vote *strongly*
   contradicts a slot's jersey is forbidden from that slot.

Cost terms (all scaled to integer centimillis for CP-SAT):

* **Unary – appearance**: cosine distance between tracklet mean ReID and
  slot gallery ReID (bootstrapped from high-confidence jersey detections,
  then refined iteratively).
* **Unary – jersey**: bonus when jersey matches slot, penalty when it
  conflicts.
* **Pairwise – transition**: for every pair of tracklets (u, v) assigned
  to the same slot where v is the temporal successor: spatial plausibility
  cost (bbox/court distance vs predicted constant-velocity continuation)
  plus local ReID continuity cost (u tail vs v head embeddings).
* **Pairwise – anti-teleport**: large penalty if the same-slot transition
  would require implausible speed.

The solver runs for a configurable wall-clock budget and returns the best
feasible solution found (often optimal for ≤ a few hundred tracklets).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

from common.distances import bbox_bottom_mid_distance, cosine_dist

logger = logging.getLogger(__name__)

# Scale factor: float costs → integer centimilli-units for CP-SAT.
_SCALE = 1000


def _is_valid_tracked_player(player) -> bool:
    return getattr(player, "player_id", -1) >= 0


# =====================================================================
# Tracklet representation
# =====================================================================

@dataclass
class Tracklet:
    tid: int
    raw_id: int
    frame_start: int
    frame_end: int
    detections: list  # [(frame_id, Player), ...]
    n_dets: int

    # Summaries
    jersey_number: int | None = None
    jersey_confidence: float = 0.0
    mean_reid: np.ndarray | None = None
    mean_color: np.ndarray | None = None

    # Boundary embeddings for local transition cost
    head_reid: np.ndarray | None = None  # mean of first few detections
    tail_reid: np.ndarray | None = None  # mean of last few detections
    head_color: np.ndarray | None = None
    tail_color: np.ndarray | None = None

    # Boundary positions
    start_bbox: list | None = None
    end_bbox: list | None = None
    start_court: tuple[float, float] | None = None
    end_court: tuple[float, float] | None = None

    # OOF edge flags: "left" / "right" / None
    exits_edge: str | None = None   # last detection near this edge
    enters_edge: str | None = None  # first detection near this edge

    def overlaps(self, other: Tracklet) -> bool:
        return self.frame_start <= other.frame_end and other.frame_start <= self.frame_end


# =====================================================================
# Tracklet cutting
# =====================================================================

def _detect_frame_width(detections: dict) -> float:
    """Infer frame width from the max x2 coordinate across all detections."""
    w = 0.0
    for fid in detections:
        for p in detections[fid]:
            if not _is_valid_tracked_player(p):
                continue
            if p.bbox and len(p.bbox) >= 4:
                w = max(w, float(p.bbox[2]))
    return w


def _near_edge(bbox, frame_width: float, margin: float) -> str | None:
    """Return 'left'/'right' if bbox is near that edge, else None."""
    if not bbox or len(bbox) < 4 or frame_width <= 0:
        return None
    x1, x2 = float(bbox[0]), float(bbox[2])
    if x1 <= margin:
        return "left"
    if x2 >= frame_width - margin:
        return "right"
    return None


def _cut_tracklets(
    detections: dict,
    gap: int,
    max_len: int,
    reid_split: float,
    frame_width: float = 0.0,
    edge_margin: float = 15.0,
) -> list[Tracklet]:
    """Split raw tracks into clean tracklets.

    Splits on: temporal gap, max length, and REID appearance shift.
    """
    per_raw: dict[int, list[tuple[int, object]]] = defaultdict(list)
    for fid in sorted(detections):
        for p in detections[fid]:
            if _is_valid_tracked_player(p):
                per_raw[p.player_id].append((fid, p))

    tracklets: list[Tracklet] = []
    tid = 0
    for raw_id, entries in per_raw.items():
        segs: list[list[tuple[int, object]]] = [[entries[0]]]
        anchors: list[list[np.ndarray]] = [[]]
        if entries[0][1].reid_embedding is not None:
            anchors[0].append(entries[0][1].reid_embedding)

        for i in range(1, len(entries)):
            fid, player = entries[i]
            fgap = fid - entries[i - 1][0]

            shift = False
            if (
                reid_split < 1.0
                and player.reid_embedding is not None
                and len(anchors[-1]) >= 3
            ):
                anchor = np.mean(anchors[-1], axis=0)
                if cosine_dist(player.reid_embedding, anchor) > reid_split:
                    shift = True

            if fgap > gap or len(segs[-1]) >= max_len or shift:
                segs.append([])
                anchors.append([])

            segs[-1].append((fid, player))
            if player.reid_embedding is not None and len(anchors[-1]) < 10:
                anchors[-1].append(player.reid_embedding)

        for seg in segs:
            tracklets.append(_make_tracklet(tid, raw_id, seg, frame_width, edge_margin))
            tid += 1

    return tracklets


def _make_tracklet(tid: int, raw_id: int, entries: list, frame_width: float = 0.0, edge_margin: float = 15.0) -> Tracklet:
    frames = [f for f, _ in entries]

    # Jersey vote
    votes: dict[int, float] = defaultdict(float)
    for _, p in entries:
        if p.number is not None and p.number.num is not None:
            votes[p.number.num] += (p.number.confidence or 0.5)
    jersey_number = max(votes, key=votes.__getitem__) if votes else None
    jersey_confidence = votes[jersey_number] if jersey_number is not None else 0.0

    # Embeddings
    reid_all = [p.reid_embedding for _, p in entries if p.reid_embedding is not None]
    color_all = [p.embedding for _, p in entries if p.embedding is not None]

    # Boundary embeddings (first/last 5)
    boundary_k = min(5, max(1, len(reid_all)))
    head_reid = np.mean(reid_all[:boundary_k], axis=0) if reid_all else None
    tail_reid = np.mean(reid_all[-boundary_k:], axis=0) if reid_all else None
    color_boundary_k = min(5, max(1, len(color_all)))
    head_color = np.mean(color_all[:color_boundary_k], axis=0) if color_all else None
    tail_color = np.mean(color_all[-color_boundary_k:], axis=0) if color_all else None

    # Boundary positions
    start_bbox = None
    end_bbox = None
    start_court = None
    end_court = None
    for _, p in entries:
        if p.bbox and len(p.bbox) >= 4:
            start_bbox = list(p.bbox)
            start_court = p.court_position
            break
    for _, p in reversed(entries):
        if p.bbox and len(p.bbox) >= 4:
            end_bbox = list(p.bbox)
            end_court = p.court_position
            break

    # Detect OOF edges
    exits_edge = _near_edge(end_bbox, frame_width, edge_margin) if end_bbox else None
    enters_edge = _near_edge(start_bbox, frame_width, edge_margin) if start_bbox else None

    return Tracklet(
        tid=tid,
        raw_id=raw_id,
        frame_start=min(frames),
        frame_end=max(frames),
        detections=entries,
        n_dets=len(entries),
        jersey_number=jersey_number,
        jersey_confidence=jersey_confidence,
        mean_reid=np.mean(reid_all, axis=0) if reid_all else None,
        mean_color=np.mean(color_all, axis=0) if color_all else None,
        head_reid=head_reid,
        tail_reid=tail_reid,
        head_color=head_color,
        tail_color=tail_color,
        start_bbox=start_bbox,
        end_bbox=end_bbox,
        start_court=start_court,
        end_court=end_court,
        exits_edge=exits_edge,
        enters_edge=enters_edge,
    )


# =====================================================================
# Gallery builder
# =====================================================================

@dataclass
class _Gallery:
    jersey_number: int
    reid: np.ndarray | None
    color: np.ndarray | None
    n_evidence: int


def _build_gallery(
    detections: dict,
    min_conf: float,
    min_count: int,
) -> list[_Gallery]:
    per_reid: dict[int, list[np.ndarray]] = defaultdict(list)
    per_color: dict[int, list[np.ndarray]] = defaultdict(list)

    for fid in detections:
        for p in detections[fid]:
            if not _is_valid_tracked_player(p):
                continue
            if p.number is None or p.number.num is None:
                continue
            if (p.number.confidence or 0.0) < min_conf:
                continue
            if p.reid_embedding is not None:
                per_reid[p.number.num].append(p.reid_embedding)
            if p.embedding is not None:
                per_color[p.number.num].append(p.embedding)

    gallery: list[_Gallery] = []
    for j in set(per_reid) | set(per_color):
        n = max(len(per_reid.get(j, [])), len(per_color.get(j, [])))
        if n < min_count:
            continue
        gallery.append(_Gallery(
            jersey_number=j,
            reid=np.mean(per_reid[j], axis=0) if per_reid.get(j) else None,
            color=np.mean(per_color[j], axis=0) if per_color.get(j) else None,
            n_evidence=n,
        ))
    gallery.sort(key=lambda g: g.n_evidence, reverse=True)
    return gallery


# =====================================================================
# Cost computation helpers
# =====================================================================

def _reid_dist(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.5
    return cosine_dist(a, b)


def _unary_cost(
    t: Tracklet,
    slot_reid: np.ndarray | None,
    slot_color: np.ndarray | None,
    slot_jersey: int | None,
    *,
    w_reid: float,
    w_color: float,
    jersey_bonus: float,
    jersey_penalty: float,
) -> float:
    """Cost of assigning tracklet *t* to a slot with given gallery."""
    c = 0.0

    # Appearance
    c += w_reid * _reid_dist(t.mean_reid, slot_reid)
    if t.mean_color is not None and slot_color is not None:
        c += w_color * cosine_dist(t.mean_color, slot_color)

    # Jersey
    if slot_jersey is not None and t.jersey_number is not None:
        strength = min(t.jersey_confidence, 3.0)
        if t.jersey_number == slot_jersey:
            c += jersey_bonus * strength  # negative = bonus
        else:
            c += jersey_penalty * strength  # positive = penalty

    return c


def _transition_cost(
    u: Tracklet,
    v: Tracklet,
    *,
    w_transition_reid: float,
    w_transition_spatial: float,
    max_speed_px: float,
    teleport_penalty: float,
    oof_return_bonus: float,
    oof_wrong_side_penalty: float,
) -> float:
    """Pairwise cost for v being the temporal successor of u in the same slot."""
    gap = v.frame_start - u.frame_end
    if gap <= 0:
        return 0.0  # overlapping — should not happen for same-slot

    c = 0.0

    # Local ReID continuity: u's tail should look like v's head
    c += w_transition_reid * _reid_dist(u.tail_reid, v.head_reid)

    # --- OOF-aware spatial plausibility ---
    oof_transition = False
    if u.exits_edge is not None and v.enters_edge is not None:
        if u.exits_edge == v.enters_edge:
            # Player went OOF at one edge and came back on the same edge.
            # Spatial distance is meaningless — give a bonus instead.
            c += oof_return_bonus
            oof_transition = True
        else:
            # Exit left, enter right (or vice versa) — unusual but possible
            # for a broadcast camera pan; still relax spatial somewhat.
            oof_transition = True

    if u.exits_edge is not None and v.enters_edge is None:
        # Player exited at edge but the next tracklet starts mid-frame.
        # This is NOT a valid OOF continuation — add a moderate penalty
        # because the identity slot should be "occupied" off-screen.
        c += oof_wrong_side_penalty

    if not oof_transition:
        # Standard (non-OOF) spatial plausibility
        if u.end_bbox is not None and v.start_bbox is not None:
            px_dist = bbox_bottom_mid_distance(u.end_bbox, v.start_bbox)
            speed = px_dist / max(gap, 1)
            if speed > max_speed_px:
                c += teleport_penalty
            else:
                c += w_transition_spatial * min(px_dist / (max_speed_px * gap), 2.0)

    # Court-space plausibility (skip for OOF transitions — position unknown off screen)
    if not oof_transition and u.end_court is not None and v.start_court is not None:
        court_dist = float(np.hypot(
            u.end_court[0] - v.start_court[0],
            u.end_court[1] - v.start_court[1],
        ))
        court_speed = court_dist / max(gap / 30.0, 0.033)
        if court_speed > 12.0:
            c += teleport_penalty

    return c


def _is_same_edge_oof_return(u: Tracklet, v: Tracklet) -> bool:
    return (
        u.exits_edge is not None
        and v.enters_edge is not None
        and u.exits_edge == v.enters_edge
        and v.frame_start > u.frame_end
    )


def _transition_hard_incompatible(
    u: Tracklet,
    v: Tracklet,
    *,
    max_speed_px: float,
    hard_speed_multiplier: float,
    max_court_speed: float,
    hard_transition_gap: int,
    oof_max_gap: int,
    hard_color_threshold: float,
) -> bool:
    """True when two nearby tracklets cannot plausibly be the same player.

    This is intentionally conservative and only blocks short-gap transitions.
    Long-gap identity reuse remains allowed.
    """
    gap = v.frame_start - u.frame_end
    if gap <= 0:
        return False

    if _is_same_edge_oof_return(u, v):
        return gap > oof_max_gap

    if gap > hard_transition_gap:
        return False

    if u.exits_edge is not None and v.enters_edge is None:
        return True

    if (
        gap <= hard_transition_gap
        and hard_color_threshold > 0
        and u.tail_color is not None
        and v.head_color is not None
    ):
        color_dist = cosine_dist(u.tail_color, v.head_color)
        if color_dist > hard_color_threshold:
            return True

    if u.end_bbox is not None and v.start_bbox is not None:
        px_dist = bbox_bottom_mid_distance(u.end_bbox, v.start_bbox)
        speed = px_dist / max(gap, 1)
        if speed > max_speed_px * hard_speed_multiplier:
            return True

    if u.end_court is not None and v.start_court is not None:
        court_dist = float(np.hypot(
            u.end_court[0] - v.start_court[0],
            u.end_court[1] - v.start_court[1],
        ))
        court_speed = court_dist / max(gap / 30.0, 0.033)
        if court_speed > max_court_speed:
            return True

    return False


def _jersey_hard_conflict(
    t: Tracklet,
    slot_jersey: int | None,
    hard_jersey_conf: float,
) -> bool:
    """True if tracklet is strongly certain of a jersey that differs from slot."""
    if slot_jersey is None or t.jersey_number is None:
        return False
    if t.jersey_number == slot_jersey:
        return False
    return t.jersey_confidence >= hard_jersey_conf


def _max_concurrent_tracklets(tracklets: list[Tracklet]) -> int:
    """Maximum number of simultaneously active tracklets."""
    events: list[tuple[int, int]] = []
    for tracklet in tracklets:
        events.append((tracklet.frame_start, 1))
        events.append((tracklet.frame_end + 1, -1))
    events.sort()

    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


# =====================================================================
# CP-SAT solver
# =====================================================================

def _solve_cpsat(
    tracklets: list[Tracklet],
    slot_reids: list[np.ndarray | None],
    slot_colors: list[np.ndarray | None],
    slot_jerseys: list[int | None],
    *,
    w_reid: float,
    w_color: float,
    jersey_bonus: float,
    jersey_penalty: float,
    w_transition_reid: float,
    w_transition_spatial: float,
    max_speed_px: float,
    teleport_penalty: float,
    hard_jersey_conf: float,
    same_raw_bonus: float,
    oof_return_bonus: float,
    oof_wrong_side_penalty: float,
    oof_shadow_frames: int,
    oof_shadow_steal_penalty: float,
    hard_speed_multiplier: float,
    max_court_speed: float,
    hard_transition_gap: int,
    oof_max_gap: int,
    hard_color_threshold: float,
    solver_time: float,
) -> tuple[list[int], int]:
    """Solve global assignment. Returns slot index per tracklet and status."""

    T = len(tracklets)
    K = len(slot_reids)

    model = cp_model.CpModel()

    # --- Variables: x[t, k] ---
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for t in range(T):
        for k in range(K):
            if _jersey_hard_conflict(tracklets[t], slot_jerseys[k], hard_jersey_conf):
                continue
            x[t, k] = model.new_bool_var(f"x_{t}_{k}")

    # --- Constraint: each tracklet assigned to exactly one slot ---
    for t in range(T):
        vars_t = [x[t, k] for k in range(K) if (t, k) in x]
        if not vars_t:
            # Hard jersey conflicts can make the model infeasible if every slot
            # is anchored. Fall back to allowing all slots for this tracklet and
            # let the objective penalise mismatches.
            for k in range(K):
                x[t, k] = model.new_bool_var(f"x_{t}_{k}")
            vars_t = [x[t, k] for k in range(K) if (t, k) in x]
        model.add_exactly_one(vars_t)

    # --- Constraint: no overlap within same slot ---
    # Precompute overlap pairs
    overlap_pairs: list[tuple[int, int]] = []
    for i in range(T):
        for j in range(i + 1, T):
            if tracklets[i].overlaps(tracklets[j]):
                overlap_pairs.append((i, j))

    for i, j in overlap_pairs:
        for k in range(K):
            if (i, k) in x and (j, k) in x:
                model.add(x[i, k] + x[j, k] <= 1)

    # --- Constraint: impossible short-gap transitions cannot share a slot ---
    hard_incompatible_pairs: list[tuple[int, int]] = []
    sorted_idx = sorted(range(T), key=lambda i: tracklets[i].frame_start)
    for si in range(T):
        u = sorted_idx[si]
        for sj in range(si + 1, T):
            v = sorted_idx[sj]
            if tracklets[v].frame_start <= tracklets[u].frame_end:
                continue
            gap = tracklets[v].frame_start - tracklets[u].frame_end
            if gap > max(hard_transition_gap, oof_max_gap):
                break
            if _transition_hard_incompatible(
                tracklets[u],
                tracklets[v],
                max_speed_px=max_speed_px,
                hard_speed_multiplier=hard_speed_multiplier,
                max_court_speed=max_court_speed,
                hard_transition_gap=hard_transition_gap,
                oof_max_gap=oof_max_gap,
                hard_color_threshold=hard_color_threshold,
            ):
                hard_incompatible_pairs.append((u, v))

    for i, j in hard_incompatible_pairs:
        for k in range(K):
            if (i, k) in x and (j, k) in x:
                model.add(x[i, k] + x[j, k] <= 1)

    # --- Objective ---
    obj_terms: list[tuple[cp_model.IntVar, int]] = []

    # Unary costs
    for t in range(T):
        for k in range(K):
            if (t, k) not in x:
                continue
            c = _unary_cost(
                tracklets[t],
                slot_reids[k],
                slot_colors[k],
                slot_jerseys[k],
                w_reid=w_reid,
                w_color=w_color,
                jersey_bonus=jersey_bonus,
                jersey_penalty=jersey_penalty,
            )
            # Same-raw-track bonus: if this tracklet's raw_id corresponds
            # to a slot that was previously associated with it, give a
            # small continuity bonus.  (Not applicable at this level.)

            ic = int(round(c * _SCALE))
            if ic != 0:
                obj_terms.append((x[t, k], ic))

    # Pairwise transition costs.
    # For each slot k, for each pair of non-overlapping tracklets that could
    # be consecutive in that slot, add a transition cost.
    # We use auxiliary "succession" variables: s[u,v,k] = 1 iff both u and v
    # are in slot k AND v is u's immediate successor (no other tracklet between
    # them in that slot).
    #
    # Exact succession modeling is expensive.  Instead we use a relaxation:
    # for each ordered pair (u, v) where v starts after u ends, add the
    # transition cost scaled by AND(x[u,k], x[v,k]).  We only consider
    # "nearby" pairs (gap < max_gap) to keep the model tractable.

    MAX_TRANSITION_GAP = 150  # frames

    # Sort tracklets by frame_start for efficient neighbor search

    for si in range(T):
        u = sorted_idx[si]
        u_end = tracklets[u].frame_end
        for sj in range(si + 1, T):
            v = sorted_idx[sj]
            if tracklets[v].frame_start <= u_end:
                continue  # overlapping
            gap = tracklets[v].frame_start - u_end
            if gap > MAX_TRANSITION_GAP:
                break  # sorted, so all further are even further

            tc = _transition_cost(
                tracklets[u],
                tracklets[v],
                w_transition_reid=w_transition_reid,
                w_transition_spatial=w_transition_spatial,
                max_speed_px=max_speed_px,
                teleport_penalty=teleport_penalty,
                oof_return_bonus=oof_return_bonus,
                oof_wrong_side_penalty=oof_wrong_side_penalty,
            )

            # Same-raw-track bonus reduces transition cost
            if (
                tracklets[u].raw_id == tracklets[v].raw_id
                and not _transition_hard_incompatible(
                    tracklets[u],
                    tracklets[v],
                    max_speed_px=max_speed_px,
                    hard_speed_multiplier=hard_speed_multiplier,
                    max_court_speed=max_court_speed,
                    hard_transition_gap=hard_transition_gap,
                    oof_max_gap=oof_max_gap,
                    hard_color_threshold=hard_color_threshold,
                )
            ):
                tc += same_raw_bonus  # negative = bonus

            itc = int(round(tc * _SCALE))
            if itc == 0:
                continue

            for k in range(K):
                if (u, k) not in x or (v, k) not in x:
                    continue
                # AND variable: both[u,v,k] = x[u,k] AND x[v,k]
                both = model.new_bool_var(f"b_{u}_{v}_{k}")
                model.add(both <= x[u, k])
                model.add(both <= x[v, k])
                model.add(both >= x[u, k] + x[v, k] - 1)
                obj_terms.append((both, itc))

    # --- OOF shadow anti-steal penalty ---
    # When tracklet u exits at a frame edge, its identity slot should
    # remain "reserved" for oof_shadow_frames.  If a *different* tracklet
    # v starts during the shadow window and is assigned to the same slot,
    # it is likely a slot-steal that causes the returning player to be
    # re-assigned.  Add a penalty for this.
    if oof_shadow_frames > 0 and oof_shadow_steal_penalty > 0:
        ipenalty = int(round(oof_shadow_steal_penalty * _SCALE))
        for si in range(T):
            u = sorted_idx[si]
            if tracklets[u].exits_edge is None:
                continue  # u doesn't exit near an edge
            shadow_end = tracklets[u].frame_end + oof_shadow_frames
            for sj in range(si + 1, T):
                v = sorted_idx[sj]
                if tracklets[v].frame_start > shadow_end:
                    break
                if tracklets[v].frame_start <= tracklets[u].frame_end:
                    continue  # actually overlapping, handled by overlap constraints
                if tracklets[u].raw_id == tracklets[v].raw_id:
                    continue  # same raw track — not a steal
                # v enters during u's OOF shadow.  If v enters at the
                # same edge as u exited, this looks like the same player
                # returning — no penalty.
                if (tracklets[v].enters_edge is not None
                        and tracklets[v].enters_edge == tracklets[u].exits_edge):
                    continue
                # Penalise assigning both u and v to the same slot
                for k in range(K):
                    if (u, k) not in x or (v, k) not in x:
                        continue
                    both = model.new_bool_var(f"oof_{u}_{v}_{k}")
                    model.add(both <= x[u, k])
                    model.add(both <= x[v, k])
                    model.add(both >= x[u, k] + x[v, k] - 1)
                    obj_terms.append((both, ipenalty))

    # Build objective
    model.minimize(
        sum(var * coeff for var, coeff in obj_terms)
    )

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_time
    solver.parameters.num_workers = 4
    solver.parameters.log_search_progress = False

    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning("CP-SAT: no feasible solution found (status=%s)", status)
        return [-1] * T, status

    logger.info(
        "CP-SAT: status=%s obj=%.1f wall=%.1fs",
        solver.status_name(status),
        solver.objective_value / _SCALE,
        solver.wall_time,
    )

    result = [-1] * T
    for t in range(T):
        for k in range(K):
            if (t, k) in x and solver.value(x[t, k]):
                result[t] = k
                break
    return result, status


def _solve_cpsat_with_relaxation(
    tracklets: list[Tracklet],
    slot_reids: list[np.ndarray | None],
    slot_colors: list[np.ndarray | None],
    slot_jerseys: list[int | None],
    *,
    w_reid: float,
    w_color: float,
    jersey_bonus: float,
    jersey_penalty: float,
    w_transition_reid: float,
    w_transition_spatial: float,
    max_speed_px: float,
    teleport_penalty: float,
    hard_jersey_conf: float,
    same_raw_bonus: float,
    oof_return_bonus: float,
    oof_wrong_side_penalty: float,
    oof_shadow_frames: int,
    oof_shadow_steal_penalty: float,
    hard_speed_multiplier: float,
    max_court_speed: float,
    hard_transition_gap: int,
    oof_max_gap: int,
    hard_color_threshold: float,
    solver_time: float,
) -> list[int]:
    """Solve with a conservative fallback cascade instead of failing hard."""
    attempts = [
        {
            "name": "strict",
            "hard_speed_multiplier": hard_speed_multiplier,
            "max_court_speed": max_court_speed,
            "hard_transition_gap": hard_transition_gap,
            "oof_max_gap": oof_max_gap,
            "hard_color_threshold": hard_color_threshold,
        },
        {
            "name": "relaxed-hard-gates",
            "hard_speed_multiplier": hard_speed_multiplier * 2.0,
            "max_court_speed": max_court_speed * 2.0,
            "hard_transition_gap": max(0, min(hard_transition_gap, 20)),
            "oof_max_gap": max(oof_max_gap, 360),
            "hard_color_threshold": 0.0,
        },
        {
            "name": "soft-only",
            "hard_speed_multiplier": 999.0,
            "max_court_speed": 999.0,
            "hard_transition_gap": 0,
            "oof_max_gap": 1000000,
            "hard_color_threshold": 0.0,
        },
    ]

    for attempt in attempts:
        logger.info("CP-SAT solve mode: %s", attempt["name"])
        assignment, status = _solve_cpsat(
            tracklets,
            slot_reids,
            slot_colors,
            slot_jerseys,
            w_reid=w_reid,
            w_color=w_color,
            jersey_bonus=jersey_bonus,
            jersey_penalty=jersey_penalty,
            w_transition_reid=w_transition_reid,
            w_transition_spatial=w_transition_spatial,
            max_speed_px=max_speed_px,
            teleport_penalty=teleport_penalty,
            hard_jersey_conf=hard_jersey_conf,
            same_raw_bonus=same_raw_bonus,
            oof_return_bonus=oof_return_bonus,
            oof_wrong_side_penalty=oof_wrong_side_penalty,
            oof_shadow_frames=oof_shadow_frames,
            oof_shadow_steal_penalty=oof_shadow_steal_penalty,
            hard_speed_multiplier=attempt["hard_speed_multiplier"],
            max_court_speed=attempt["max_court_speed"],
            hard_transition_gap=attempt["hard_transition_gap"],
            oof_max_gap=attempt["oof_max_gap"],
            hard_color_threshold=attempt["hard_color_threshold"],
            solver_time=solver_time,
        )
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return assignment

    return [-1] * len(tracklets)


# =====================================================================
# Iterative solve with gallery refinement
# =====================================================================

def _refine_galleries(
    tracklets: list[Tracklet],
    assignment: list[int],
    n_slots: int,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Recompute slot galleries from assigned tracklets."""
    slot_reids: list[list[np.ndarray]] = [[] for _ in range(n_slots)]
    slot_colors: list[list[np.ndarray]] = [[] for _ in range(n_slots)]

    for t, k in enumerate(assignment):
        if k < 0:
            continue
        if tracklets[t].mean_reid is not None:
            slot_reids[k].append(tracklets[t].mean_reid)
        if tracklets[t].mean_color is not None:
            slot_colors[k].append(tracklets[t].mean_color)

    reid_out = [np.mean(v, axis=0) if v else None for v in slot_reids]
    color_out = [np.mean(v, axis=0) if v else None for v in slot_colors]
    return reid_out, color_out


# =====================================================================
# Apply assignment
# =====================================================================

def _apply(detections: dict, tracklets: list[Tracklet], assignment: list[int], slot_ids: list[int]) -> None:
    """Write new player_id values. slot_ids maps slot index → player_id."""
    lookup: dict[tuple[int, int], int] = {}
    for t, k in enumerate(assignment):
        if k < 0:
            continue
        pid = slot_ids[k]
        for fid, _ in tracklets[t].detections:
            lookup[(tracklets[t].raw_id, fid)] = pid

    for fid in detections:
        for p in detections[fid]:
            new_id = lookup.get((p.player_id, fid))
            if new_id is not None:
                p.player_id = new_id


# =====================================================================
# Public API
# =====================================================================

def stitch_tracks(
    detections: dict,
    *,
    num_players: int = 10,
    gap_threshold: int = 5,
    max_tracklet_len: int = 60,
    reid_split: float = 0.40,
    gallery_min_conf: float = 0.5,
    gallery_min_count: int = 3,
    w_reid: float = 1.5,
    w_color: float = 0.3,
    jersey_bonus: float = -3.0,
    jersey_penalty: float = 4.0,
    w_transition_reid: float = 1.0,
    w_transition_spatial: float = 0.5,
    max_speed_px: float = 80.0,
    teleport_penalty: float = 5.0,
    hard_jersey_conf: float = 3.0,
    same_raw_bonus: float = -0.3,
    oof_return_bonus: float = -1.0,
    oof_wrong_side_penalty: float = 2.0,
    oof_shadow_frames: int = 90,
    oof_shadow_steal_penalty: float = 3.0,
    hard_speed_multiplier: float = 1.35,
    max_court_speed: float = 12.0,
    hard_transition_gap: int = 45,
    oof_max_gap: int = 180,
    hard_color_threshold: float = 0.22,
    edge_margin: float = 15.0,
    solver_time: float = 300.0,
    n_refine: int = 2,
) -> None:
    """CP-SAT global tracklet-to-identity stitcher.

    Modifies ``player.player_id`` in-place.

    Parameters
    ----------
    detections : dict[int, list[Player]]
        Frame-indexed detections with raw ``player_id`` from FlowTracker.
    num_players : int
        Exactly how many identity slots to create.
    gap_threshold : int
        Frame gap that forces a tracklet split.
    max_tracklet_len : int
        Maximum detections per tracklet before forced split.
    reid_split : float
        Cosine distance threshold for mid-tracklet REID-based splitting.
    gallery_min_conf, gallery_min_count : float, int
        Minimum jersey OCR confidence and detection count for gallery entry.
    w_reid, w_color : float
        Weights for unary ReID and colour costs.
    jersey_bonus, jersey_penalty : float
        Jersey match bonus (negative) and mismatch penalty (positive).
    w_transition_reid, w_transition_spatial : float
        Weights for pairwise ReID continuity and spatial plausibility.
    max_speed_px : float
        Maximum plausible speed in pixels/frame (beyond → teleport penalty).
    teleport_penalty : float
        Large penalty for implausible spatial transitions.
    hard_jersey_conf : float
        Jersey confidence threshold for hard-blocking a tracklet from a
        conflicting-jersey slot.
    same_raw_bonus : float
        Small bonus (negative) for keeping consecutive tracklets from the
        same raw FlowTracker track in the same slot.
    oof_return_bonus : float
        Bonus (negative) for same-edge OOF exit→entry transitions.
    oof_wrong_side_penalty : float
        Penalty when a tracklet exits at an edge but the proposed successor
        starts mid-frame (not a valid OOF return).
    oof_shadow_frames : int
        Number of frames after an OOF exit during which the identity slot
        is considered "reserved".  Other tracklets starting during this
        window and assigned to the same slot receive a steal penalty.
    oof_shadow_steal_penalty : float
        Cost penalty for assigning a non-returning tracklet to a slot
        whose previous occupant exited at a frame edge within the shadow.
    hard_speed_multiplier : float
        Short-gap hard spatial gate in units of ``max_speed_px``.
        Any faster transition is forbidden, not just penalized.
    max_court_speed : float
        Hard court-space speed gate in metres/second for short-gap links.
    hard_transition_gap : int
        Apply hard teleportation gating only to transitions with gap up to
        this many frames. Longer gaps stay soft-constrained.
    oof_max_gap : int
        Maximum allowed same-edge OOF return gap before the transition is
        forbidden from using the same slot.
    hard_color_threshold : float
        Hard short-gap color-embedding gate. If two nearby tracklets have
        color cosine distance above this value, they cannot share a slot.
    edge_margin : float
        Pixel margin for detecting whether a bbox is near a frame edge.
    solver_time : float
        Wall-clock seconds budget per CP-SAT solve call.
    n_refine : int
        Number of solve→refine iterations.
    """
    # 1. Cut tracklets
    n_invalid = sum(
        1
        for frame_players in detections.values()
        for player in frame_players
        if not _is_valid_tracked_player(player)
    )
    frame_width = _detect_frame_width(detections)
    tracklets = _cut_tracklets(
        detections, gap_threshold, max_tracklet_len, reid_split,
        frame_width=frame_width, edge_margin=edge_margin,
    )
    if not tracklets:
        return

    n_oof_exits = sum(1 for t in tracklets if t.exits_edge is not None)
    n_oof_entries = sum(1 for t in tracklets if t.enters_edge is not None)
    peak_overlap = _max_concurrent_tracklets(tracklets)

    # 2. Build initial gallery from jersey evidence
    gallery = _build_gallery(detections, gallery_min_conf, gallery_min_count)

    logger.info(
        "stitch_tracks (CP-SAT): %d tracklets (%d OOF-exit, %d OOF-enter), "
        "%d gallery jerseys=%s, frame_width=%.0f, ignored_invalid=%d",
        len(tracklets),
        n_oof_exits,
        n_oof_entries,
        len(gallery),
        [g.jersey_number for g in gallery],
        frame_width,
        n_invalid,
    )

    # 3. Create slot representations
    # Jersey-anchored slots first, then anonymous slots
    slot_reids: list[np.ndarray | None] = []
    slot_colors: list[np.ndarray | None] = []
    slot_jerseys: list[int | None] = []
    slot_ids: list[int] = []

    for i, g in enumerate(gallery[:num_players]):
        slot_reids.append(g.reid)
        slot_colors.append(g.color)
        slot_jerseys.append(g.jersey_number)
        slot_ids.append(i + 1)

    target_slots = max(num_players, peak_overlap)
    n_anon = max(0, target_slots - len(gallery))
    for i in range(n_anon):
        slot_reids.append(None)
        slot_colors.append(None)
        slot_jerseys.append(None)
        slot_ids.append(len(gallery) + i + 1)

    K = len(slot_reids)
    if K > num_players:
        logger.warning(
            "CP-SAT: expanding slot count from %d to %d to cover peak overlap",
            num_players,
            K,
        )

    # 4. Iterative: solve → refine galleries → re-solve
    assignment = list(range(len(tracklets)))  # dummy init
    for iteration in range(n_refine):
        logger.info("CP-SAT iteration %d/%d", iteration + 1, n_refine)

        assignment = _solve_cpsat_with_relaxation(
            tracklets,
            slot_reids,
            slot_colors,
            slot_jerseys,
            w_reid=w_reid,
            w_color=w_color,
            jersey_bonus=jersey_bonus,
            jersey_penalty=jersey_penalty,
            w_transition_reid=w_transition_reid,
            w_transition_spatial=w_transition_spatial,
            max_speed_px=max_speed_px,
            teleport_penalty=teleport_penalty,
            hard_jersey_conf=hard_jersey_conf,
            same_raw_bonus=same_raw_bonus,
            oof_return_bonus=oof_return_bonus,
            oof_wrong_side_penalty=oof_wrong_side_penalty,
            oof_shadow_frames=oof_shadow_frames,
            oof_shadow_steal_penalty=oof_shadow_steal_penalty,
            hard_speed_multiplier=hard_speed_multiplier,
            max_court_speed=max_court_speed,
            hard_transition_gap=hard_transition_gap,
            oof_max_gap=oof_max_gap,
            hard_color_threshold=hard_color_threshold,
            solver_time=solver_time,
        )

        if all(a < 0 for a in assignment):
            logger.warning("CP-SAT produced no assignment — keeping raw IDs")
            return

        # Refine galleries
        new_reids, new_colors = _refine_galleries(tracklets, assignment, K)
        # Only update anonymous or weakly-anchored slots
        for k in range(K):
            if new_reids[k] is not None:
                slot_reids[k] = new_reids[k]
            if new_colors[k] is not None:
                slot_colors[k] = new_colors[k]

    # 5. Apply
    _apply(detections, tracklets, assignment, slot_ids)

    # Log assignment summary
    assigned_count = sum(1 for a in assignment if a >= 0)
    logger.info(
        "stitch_tracks: assigned %d / %d tracklets to %d slots",
        assigned_count, len(tracklets), K,
    )
