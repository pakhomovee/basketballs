"""Offline graph-based tracker using min-cost max-flow.

Each detection becomes a pair of nodes (d_in, d_out) in a flow network.
Edges encode the likelihood that two detections belong to the same track.
The MCMF solver finds the globally optimal assignment of detections to
a fixed number of tracks (default 10 for basketball).

Combines DeepSORT-style costs (spatial, appearance, IoU) with a global
optimization that eliminates the need for post-hoc track merging.

    Supports multi-pass refinement: after the first pass produces initial
    tracks, subsequent passes enrich link costs with past/future mean
    tracklet embeddings from the previous solution, extending temporal consistency.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from common.distances import bbox_bottom_mid_distance, bbox_iou, cosine_dist
from tracking.min_cost_flow import MinCostFlow

logger = logging.getLogger(__name__)
S, T = 0, 1


# Node layout: S=0, T=1, d_i_in=2+2i, d_i_out=2+2i+1, oof_i=2+2*N+i, start_out, end_out
def _det_in(i: int, n_det: int) -> int:
    return 2 + 2 * i


def _det_out(i: int, n_det: int) -> int:
    return 2 + 2 * i + 1


def _oof_node(i: int, n_det: int) -> int:
    """Node representing 'out-of-frame' state starting after detection i."""
    return 2 + 2 * n_det + i


def _is_detection_node(v: int, n_det: int) -> bool:
    return 2 <= v < 2 + 2 * n_det


def _det_idx_from_node(v: int, n_det: int) -> int | None:
    """Detection index if v is d_in or d_out, else None."""
    if not _is_detection_node(v, n_det):
        return None
    return (v - 2) // 2


def _start_out(n_det: int) -> int:
    return 2 + 3 * n_det


def _end_out(n_det: int) -> int:
    return 2 + 3 * n_det + 1


class FlowTracker:
    """Offline multi-object tracker using min-cost max-flow.

    Parameters
    ----------
    num_tracks : int
        Number of tracks to find (= number of flow units). 10 for basketball.
    max_skip : int
        Maximum frame gap for linking detections. Detections further apart
        in time cannot be directly linked (track would need to end/restart).
    max_speed : float
        Maximum player speed in meters/second. Used for field-coordinate
        gating — links exceeding this speed are pruned.
    bbox_gate : float
        Maximum pixel distance (bbox bottom-center) for linking when field
        coordinates are unavailable. Scaled by sqrt(frame_gap).
    w_spatial, w_app, w_iou : float
        Weights for spatial, appearance, and IoU cost components.
    w_extended : float
        Weight for the extended appearance cost from mean tracklet embeddings.
        Used in passes 2+ to penalise links whose tracklet neighbourhoods
        look different. Set to 0 to effectively disable multi-pass enrichment.
    oof_entry_cost : float
        Additional cost to use transitions that come from out-of-frame
        state (``start_out -> d_in`` and ``oof_i -> d_in_j``). Helps reduce
        overuse of OOF routes when regular in-frame links are available.
    skip_penalty : float
        Per-skipped-frame additive penalty on link cost.
    detection_reward : float
        Reward (negative cost) on each detection edge. Incentivises the
        solver to include as many detections as possible in tracks.
        A track through N detections gets N × detection_reward savings.
    enter_cost, exit_cost : float
        Cost for a track to start / end. Higher values encourage longer,
        unbroken tracks.
    fps : float
        Video frame rate (for speed ↔ distance conversion).
    frame_width : float | None
        Frame width in pixels. If None, inferred from max bbox x2 across detections.
        Used for out-of-frame handling (detections near left/right edge).
    edge_margin : float
        Pixel distance from frame edge for "near edge" (can go out-of-frame).
    k_warmup_frames : int
        Tracks can start only in the first k_warmup frames or from start_out (entering from edge).
    last_frames : int
        Tracks can end only in the last N frames or in end_out (exiting to edge).
    n_passes : int
        Number of MCMF passes.  Pass 1 uses standard costs; passes 2+
        enrich link costs with mean tracklet embeddings computed from the
        previous solution, extending temporal consistency.
    lookback : int
        Number of track positions to look back/forward when building the
        past/future mean tracklet embeddings for each detection during
        multi-pass refinement.
    """

    def __init__(
        self,
        num_tracks: int = 10,
        max_skip: int = 15,
        max_speed: float = 10.0,
        bbox_gate: float = 20.0,
        w_spatial: float = 0.2,
        w_app: float = 0.6,
        w_iou: float = 0.2,
        w_extended: float = 1.5,
        skip_penalty: float = 1.0,
        detection_reward: float = 1.0,
        enter_cost: float = 2.0,
        exit_cost: float = 2.0,
        fps: float = 30.0,
        frame_width: float | None = None,
        edge_margin: float = 5.0,
        k_warmup_frames: int = 10,
        last_frames: int = 10,
        n_passes: int = 5,
        lookback: int = 5,
        oof_entry_cost: float = 0.5,
    ):
        self.num_tracks = num_tracks
        self.max_skip = max_skip
        self.max_speed = max_speed
        self.bbox_gate = bbox_gate
        self.w_spatial = w_spatial
        self.w_app = w_app
        self.w_iou = w_iou
        self.w_extended = w_extended
        self.skip_penalty = skip_penalty
        self.detection_reward = detection_reward
        self.enter_cost = enter_cost
        self.exit_cost = exit_cost
        self.fps = fps
        self.frame_width = frame_width
        self.edge_margin = edge_margin
        self.k_warmup_frames = k_warmup_frames
        self.last_frames = last_frames
        self.n_passes = n_passes
        self.lookback = lookback
        self.oof_entry_cost = oof_entry_cost

    def track(self, detections: dict) -> None:
        """Assign track IDs to all players using min-cost flow.

        Runs ``n_passes`` iterations.  After the first pass, each subsequent
        pass enriches link costs with mean tracklet embeddings computed from
        the previous solution, improving long-range temporal consistency.

        Modifies ``player.player_id`` in-place. Unassigned detections
        keep ``player_id = -1``.
        """
        for players in detections.values():
            for p in players:
                p.player_id = -1

        det_list: list[tuple[int, object]] = []
        frame_to_dets: dict[int, list[int]] = defaultdict(list)

        for frame_id in sorted(detections):
            for player in detections[frame_id]:
                idx = len(det_list)
                det_list.append((frame_id, player))
                frame_to_dets[frame_id].append(idx)

        n_det = len(det_list)
        if n_det == 0:
            return
        frame_width = self.frame_width
        if frame_width is None:
            inferred_width = 0.0
            for _, player in det_list:
                bbox = player.bbox
                if bbox is not None and len(bbox) >= 3:
                    inferred_width = max(inferred_width, float(bbox[2]))
            frame_width = inferred_width
        sorted_frames = sorted(frame_to_dets.keys())

        first_frames = set(sorted_frames[: self.k_warmup_frames])
        last_frames_set = set(sorted_frames[-self.last_frames :]) if sorted_frames else set()

        tracks: list[list[int]] | None = None
        lookback = self.lookback
        for pass_idx in range(self.n_passes):
            past_embs: dict[int, np.ndarray] | None = None
            future_embs: dict[int, np.ndarray] | None = None
            if tracks is not None:
                past_embs, future_embs = self._compute_tracklet_embeddings(
                    tracks,
                    det_list,
                    lookback,
                )
                lookback *= 2

            tracks = self._solve_pass(
                det_list,
                frame_to_dets,
                sorted_frames,
                first_frames,
                last_frames_set,
                frame_width,
                n_det,
                past_embs,
                future_embs,
                pass_idx,
            )

        assert tracks is not None
        for track_id, path in enumerate(tracks, start=1):
            for det_idx in path:
                det_list[det_idx][1].player_id = track_id

        logger.info(
            "Assigned %d tracks covering %d / %d detections",
            len(tracks),
            sum(len(p) for p in tracks),
            n_det,
        )

    # ------------------------------------------------------------------
    # Multi-pass refinement helpers
    # ------------------------------------------------------------------

    def _compute_tracklet_embeddings(
        self,
        tracks: list[list[int]],
        det_list: list[tuple[int, object]],
        lookback: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Directional mean embeddings for each tracked detection.

        Parameters
        ----------
        lookback : int
            Window size (in track positions) for the past / future
            averaging.  Doubled after every refinement pass by the caller.

        Returns
        -------
        tuple[dict[int, np.ndarray], dict[int, np.ndarray]]
            ``(past_embs, future_embs)`` where:
            - ``past_embs[det_idx]`` is the mean embedding over the lookback
              positions ending at (and including) the detection in its track.
            - ``future_embs[det_idx]`` is the mean embedding over the lookback
              positions starting at (and including) the detection in its track.

        Only detections that belong to a track from the previous pass receive
        entries.  Untracked detections are left out so ``_extended_cost``
        falls back gracefully (returns None) for them.
        """
        past_embs: dict[int, np.ndarray] = {}
        future_embs: dict[int, np.ndarray] = {}

        for track in tracks:
            for pos, det_idx in enumerate(track):
                # Past: positions [max(0, pos - lookback) .. pos] inclusive
                past_start = max(0, pos - lookback)
                past_vecs: list[np.ndarray] = []
                for s_idx in track[past_start : pos + 1]:
                    player = det_list[s_idx][1]
                    emb = player.reid_embedding if player.reid_embedding is not None else player.embedding
                    if emb is not None:
                        past_vecs.append(emb)
                if past_vecs:
                    past_embs[det_idx] = np.mean(past_vecs, axis=0)

                # Future: positions [pos .. min(len(track)-1, pos + lookback)] inclusive
                future_end = min(len(track), pos + lookback + 1)
                future_vecs: list[np.ndarray] = []
                for s_idx in track[pos:future_end]:
                    player = det_list[s_idx][1]
                    emb = player.reid_embedding if player.reid_embedding is not None else player.embedding
                    if emb is not None:
                        future_vecs.append(emb)
                if future_vecs:
                    future_embs[det_idx] = np.mean(future_vecs, axis=0)

        return past_embs, future_embs

    def _extended_cost(
        self,
        i: int,
        j: int,
        past_embs: dict[int, np.ndarray],
        future_embs: dict[int, np.ndarray],
    ) -> float | None:
        """Squared cosine distance between the *past* mean embedding of
        source ``i`` and the *future* mean embedding of target ``j``.

        Squaring makes the penalty super-linear: same-person links (low
        cosine dist ~0.2 → 0.04) are barely affected, while cross-identity
        links (high cosine dist ~0.7 → 0.49) are hit much harder.  This
        closes the gap with ``enter_cost`` and prevents the solver from
        preferring cheap ID swaps over creating new tracks.

        Returns None when either embedding is unavailable.
        """
        emb_i = past_embs.get(i)
        emb_j = future_embs.get(j)
        if emb_i is None or emb_j is None:
            return None
        d = cosine_dist(emb_i, emb_j)
        return d * d

    # ------------------------------------------------------------------
    # Single MCMF pass (graph build + solve + extract)
    # ------------------------------------------------------------------

    def _solve_pass(
        self,
        det_list: list[tuple[int, object]],
        frame_to_dets: dict[int, list[int]],
        sorted_frames: list[int],
        first_frames: set[int],
        last_frames_set: set[int],
        frame_width: float,
        n_det: int,
        past_embs: dict[int, np.ndarray] | None,
        future_embs: dict[int, np.ndarray] | None,
        pass_idx: int,
    ) -> list[list[int]]:
        """Build the flow graph, solve MCMF, and return extracted tracks."""

        n_nodes = 2 + 3 * n_det + 2
        mcf = MinCostFlow(n_nodes)

        start_out = _start_out(n_det)
        end_out = _end_out(n_det)

        mcf.add_edge(S, start_out, self.num_tracks, 0.0)
        mcf.add_edge(end_out, T, self.num_tracks, 0.0)

        for i in range(n_det):
            frame_id = det_list[i][0]
            d_in = _det_in(i, n_det)
            d_out = _det_out(i, n_det)

            mcf.add_edge(d_in, d_out, 1, -self.detection_reward)

            # Entry conditions
            if frame_id in first_frames:
                mcf.add_edge(S, d_in, 1, self.enter_cost)
            near_edge = frame_width > 0 and (
                self._near_edge(det_list[i][1], frame_width, self.edge_margin, "left")
                or self._near_edge(det_list[i][1], frame_width, self.edge_margin, "right")
            )
            if near_edge:
                mcf.add_edge(start_out, d_in, 1, self.enter_cost + self.oof_entry_cost)

            # Exit conditions
            if frame_id in last_frames_set:
                mcf.add_edge(d_out, T, 1, self.exit_cost)
            mcf.add_edge(d_out, end_out, 1, 0.0 if near_edge else self.exit_cost)

            # OOF Node: d_out_i -> oof_i
            oof_node = _oof_node(i, n_det)
            mcf.add_edge(d_out, oof_node, 1, 0.0)

            mcf.add_edge(oof_node, T, 1, self.exit_cost)
            if near_edge:
                mcf.add_edge(oof_node, end_out, 1, 0.0)
            else:
                mcf.add_edge(oof_node, end_out, 1, self.exit_cost)

        n_links = 0
        n_oof_links = 0

        pass_label = f"Pass {pass_idx + 1}/{self.n_passes}: building flow graph"
        for fi_idx in tqdm(range(len(sorted_frames)), desc=pass_label):
            frame_i = sorted_frames[fi_idx]
            for fj_idx in range(fi_idx + 1, len(sorted_frames)):
                frame_j = sorted_frames[fj_idx]
                gap = frame_j - frame_i

                # -- Standard links (short term) --
                if gap <= self.max_skip:
                    for i in frame_to_dets[frame_i]:
                        player_i = det_list[i][1]
                        for j in frame_to_dets[frame_j]:
                            player_j = det_list[j][1]
                            cost = self._link_cost(player_i, player_j, gap)
                            if cost is not None:
                                if past_embs is not None and future_embs is not None:
                                    ext = self._extended_cost(i, j, past_embs, future_embs)
                                    if ext is not None:
                                        cost += self.w_extended * ext
                                mcf.add_edge(_det_out(i, n_det), _det_in(j, n_det), 1, cost)
                                n_links += 1

                # -- Out-of-frame (OOF) links via OOF nodes --
                if frame_width > 0:
                    margin_exit = self.edge_margin
                    margin_entry = self.edge_margin

                    for i in frame_to_dets[frame_i]:
                        player_i = det_list[i][1]
                        is_left_exit = self._near_edge(player_i, frame_width, margin_exit, "left")
                        is_right_exit = self._near_edge(player_i, frame_width, margin_exit, "right")

                        if not (is_left_exit or is_right_exit):
                            continue

                        oof_node_i = _oof_node(i, n_det)

                        for j in frame_to_dets[frame_j]:
                            player_j = det_list[j][1]

                            is_left_entry = self._near_edge(player_j, frame_width, margin_entry, "left")
                            is_right_entry = self._near_edge(player_j, frame_width, margin_entry, "right")

                            if (is_left_exit and is_left_entry) or (is_right_exit and is_right_entry):
                                cost = self._link_cost_oof(player_i, player_j, gap)
                                if cost is not None:
                                    cost += self.oof_entry_cost
                                    if past_embs is not None and future_embs is not None:
                                        ext = self._extended_cost(i, j, past_embs, future_embs)
                                        if ext is not None:
                                            cost += self.w_extended * ext
                                    mcf.add_edge(oof_node_i, _det_in(j, n_det), 1, cost)
                                    n_oof_links += 1

        logger.info(
            "Pass %d: %d detections, %d link edges, %d oof edges, %d nodes",
            pass_idx + 1,
            n_det,
            n_links,
            n_oof_links,
            n_nodes,
        )

        logger.info("Pass %d: solving MCMF for %d tracks …", pass_idx + 1, self.num_tracks)
        flow, total_cost = mcf.solve(S, T, self.num_tracks)
        logger.info("Pass %d: flow=%d, cost=%.2f", pass_idx + 1, flow, total_cost)

        return self._extract_tracks(mcf, n_det)

    # ------------------------------------------------------------------
    # Out-of-frame helpers
    # ------------------------------------------------------------------

    def _near_edge(self, player, frame_width: float, margin: float, side: str) -> bool:
        """True if bbox left/right side is within margin of the frame edge."""
        bbox = player.bbox
        if not bbox or len(bbox) < 4:
            return False
        b = np.asarray(bbox, dtype=float).ravel()
        x1, x2 = float(b[0]), float(b[2])
        if side == "left":
            return x1 <= margin
        if side == "right":
            return x2 >= frame_width - margin
        return False

    def _link_cost_oof(self, player_i, player_j, frame_gap: int) -> float | None:
        """Link cost for out-of-frame path: appearance only (no spatial, IoU, or skip penalty)."""
        _ = frame_gap  # kept for API symmetry with _link_cost
        return self._appearance_cost(player_i, player_j)

    # ------------------------------------------------------------------
    # Link cost (DeepSORT-style: spatial + appearance + IoU)
    # ------------------------------------------------------------------

    def _appearance_cost(self, player_i, player_j) -> float | None:
        """Appearance-only cosine distance (None when either embedding is missing)."""
        emb_i = player_i.reid_embedding if player_i.reid_embedding is not None else player_i.embedding
        emb_j = player_j.reid_embedding if player_j.reid_embedding is not None else player_j.embedding
        if emb_i is None or emb_j is None:
            return None
        return cosine_dist(emb_i, emb_j)

    def _link_cost(self, player_i, player_j, frame_gap: int, *, skip_penalty: bool = True) -> float | None:
        """Cost of linking two detections. Returns None if gated out."""
        # --- Spatial ---
        spatial_cost = 0.5
        bbox_i = player_i.bbox
        bbox_j = player_j.bbox

        if bbox_i and bbox_j and len(bbox_i) >= 4 and len(bbox_j) >= 4:
            px_dist = bbox_bottom_mid_distance(bbox_i, bbox_j)
            max_px = self.bbox_gate * (frame_gap**0.5)
            if px_dist > max_px:
                return None
            spatial_cost = px_dist / max(max_px, 1e-6)

        # --- Appearance ---
        app_cost = self._appearance_cost(player_i, player_j)
        if app_cost is None:
            app_cost = 1.0

        # --- IoU ---
        iou_cost = 0.0
        iou_fade = 0.0

        if frame_gap <= 5 and bbox_i and bbox_j and len(bbox_i) >= 4 and len(bbox_j) >= 4:
            iou_cost = 1.0 - bbox_iou(bbox_i, bbox_j)
            iou_fade = max(0.0, 1.0 - (frame_gap - 1) / 5.0)

        # Redistribute faded IoU weight to spatial + appearance
        w_iou_eff = self.w_iou * iou_fade
        extra = self.w_iou - w_iou_eff
        w_sp = self.w_spatial + extra * 0.5
        w_ap = self.w_app + extra * 0.5

        cost = w_sp * spatial_cost + w_ap * app_cost + w_iou_eff * iou_cost
        if skip_penalty:
            cost += max(0, frame_gap - 1) * self.skip_penalty

        return cost

    # ------------------------------------------------------------------
    # Track extraction from solved flow network
    # ------------------------------------------------------------------

    def _extract_tracks(self, mcf: MinCostFlow, n_det: int) -> list[list[int]]:
        """Trace detection paths through the solved flow network."""
        start_out = _start_out(n_det)

        successors: dict[int, int | None] = {}
        track_starts: list[int] = []

        for i in range(n_det):
            d_in = _det_in(i, n_det)
            d_out = _det_out(i, n_det)

            det_edge = mcf.graph[d_in][0]
            if det_edge[1] > 0:
                continue  # no flow through this detection

            # Track starts if flow arrived from S or start_out
            for edge in mcf.graph[d_in]:
                if edge[0] in (S, start_out) and edge[1] > 0:
                    track_starts.append(i)
                    break

            next_det = None
            for k in range(1, len(mcf.graph[d_out])):
                edge = mcf.graph[d_out][k]
                if edge[1] == 0:  # forward edge fully used
                    v = edge[0]
                    if _is_detection_node(v, n_det) and v % 2 == 0:
                        next_det = _det_idx_from_node(v, n_det)
                        break
                    # Check if flow went to an OOF node
                    if 2 + 2 * n_det <= v < 2 + 3 * n_det:
                        # Follow flow from OOF node to next d_in
                        for oof_edge in mcf.graph[v]:
                            # Look for used forward edge to a d_in
                            v_next, cap_next, _, _ = oof_edge
                            if cap_next == 0 and _is_detection_node(v_next, n_det) and v_next % 2 == 0:
                                next_det = _det_idx_from_node(v_next, n_det)
                                break
                        if next_det is not None:
                            break
            successors[i] = next_det

        tracks: list[list[int]] = []
        for start in track_starts:
            path: list[int] = []
            current: int | None = start
            visited: set[int] = set()
            while current is not None and current not in visited:
                visited.add(current)
                path.append(current)
                current = successors.get(current)
            tracks.append(path)

        return tracks
