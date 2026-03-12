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

import bisect
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from common.distances import bbox_bottom_mid_distance, bbox_iou, court_position_distance, cosine_dist
from tracking.min_cost_flow import MinCostFlow

logger = logging.getLogger(__name__)
S, T = 0, 1


def _kalman_smooth_positions(
    positions: list[tuple[float, float]],
    process_noise: float = 0.5,
    measurement_noise: float = 1.0,
) -> list[tuple[float, float]]:
    """Smooth a sequence of (x, y) court positions using a 2D constant-velocity Kalman filter.

    Returns the filtered (smoothed) position at each time step.
    Uses state [x, y, vx, vy] with dt=1 frame.
    """
    if not positions:
        return []
    n = len(positions)
    pos_arr = np.array(positions, dtype=float)
    dt = 1.0

    # State: [x, y, vx, vy]
    F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    Q = np.eye(4) * process_noise
    Q[2:, 2:] *= 0.5  # velocity process noise
    R = np.eye(2) * measurement_noise

    x = np.zeros(4)
    x[:2] = pos_arr[0]
    P = np.eye(4) * 10.0

    smoothed: list[tuple[float, float]] = []
    for k in range(n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update
        z = pos_arr[k]
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        smoothed.append((float(x[0]), float(x[1])))
    return smoothed


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
    bbox_scale : float
        Scale (pixels) for exponential spatial cost. Cost = exp(px_dist / scale) - 1,
        with scale = bbox_scale * sqrt(frame_gap).  No hard gate — links are never
        rejected by distance, but teleportation is exponentially discouraged.
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
    w_color : float
        Weight for continuous color-embedding cosine distance in link costs.
        Provides soft team-aware association without binary clustering.
        Also used in extended cost (mean tracklet color embeddings).
    max_occlusion : int
        Maximum frame gap bridgeable by an occlusion edge.  An occluded
        player can be linked across at most this many frames by riding
        along an existing track.
    occlusion_gate : float
        Maximum pixel distance (bbox bottom-center) between a detection
        and an existing track's detection for the pair to be considered
        an occlusion entry/exit point.
    occlusion_penalty : float
        Per-frame additive cost on occlusion edges.  Kept small because
        the skipped frames already carry no detection reward.
    occlusion_start_pass : int
        0-indexed pass at which occlusion edges are first added.
    """

    def __init__(
        self,
        num_tracks: int = 10,
        max_skip: int = 15,
        max_speed: float = 10.0,
        bbox_scale: float = 20.0,
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
        w_color: float = 0.3,
        max_occlusion: int = 45,
        occlusion_gate: float = 50.0,
        occlusion_penalty: float = 0.1,
        occlusion_start_pass: int = 2,
    ):
        self.num_tracks = num_tracks
        self.max_skip = max_skip
        self.max_speed = max_speed
        self.bbox_scale = bbox_scale
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
        self.w_color = w_color
        self.max_occlusion = max_occlusion
        self.occlusion_gate = occlusion_gate
        self.occlusion_penalty = occlusion_penalty
        self.occlusion_start_pass = occlusion_start_pass

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
        past_positions: dict[int, tuple[float, float]] | None = None
        future_positions: dict[int, tuple[float, float]] | None = None
        for pass_idx in range(self.n_passes):
            past_embs: dict[int, np.ndarray] | None = None
            future_embs: dict[int, np.ndarray] | None = None
            past_color_embs: dict[int, np.ndarray] | None = None
            future_color_embs: dict[int, np.ndarray] | None = None
            if tracks is not None:
                (
                    past_embs,
                    future_embs,
                    past_color_embs,
                    future_color_embs,
                    past_positions,
                    future_positions,
                ) = self._compute_tracklet_embeddings(tracks, det_list, lookback)
                lookback *= 2
                self.w_app *= 0.8

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
                past_color_embs,
                future_color_embs,
                pass_idx,
                prev_tracks=tracks,
                past_positions=past_positions,
                future_positions=future_positions,
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
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, tuple[float, float]],
        dict[int, tuple[float, float]],
    ]:
        """Directional mean embeddings (reid + color) and Kalman-smoothed positions for each tracked detection.

        Returns
        -------
        (past_embs, future_embs, past_color_embs, future_color_embs, past_positions, future_positions)
            Reid and color embeddings averaged over a lookback window in each direction.
            past_positions / future_positions: Kalman-smoothed court (x, y) for spatial cost in later passes.
        """
        past_embs: dict[int, np.ndarray] = {}
        future_embs: dict[int, np.ndarray] = {}
        past_color_embs: dict[int, np.ndarray] = {}
        future_color_embs: dict[int, np.ndarray] = {}
        past_positions: dict[int, tuple[float, float]] = {}
        future_positions: dict[int, tuple[float, float]] = {}

        for track in tracks:
            for pos, det_idx in enumerate(track):
                past_start = max(0, pos - lookback)
                past_reid: list[np.ndarray] = []
                past_color: list[np.ndarray] = []
                past_xy: list[tuple[float, float]] = []
                for s_idx in track[past_start : pos + 1]:
                    player = det_list[s_idx][1]
                    emb = player.reid_embedding if player.reid_embedding is not None else player.embedding
                    if emb is not None:
                        past_reid.append(emb)
                    if player.embedding is not None:
                        past_color.append(player.embedding)
                    if player.court_position is not None:
                        past_xy.append(player.court_position)
                if past_reid:
                    past_embs[det_idx] = np.mean(past_reid, axis=0)
                if past_color:
                    past_color_embs[det_idx] = np.mean(past_color, axis=0)
                if past_xy:
                    smoothed = _kalman_smooth_positions(past_xy)
                    past_positions[det_idx] = smoothed[-1]

                future_end = min(len(track), pos + lookback + 1)
                future_reid: list[np.ndarray] = []
                future_color: list[np.ndarray] = []
                future_xy: list[tuple[float, float]] = []
                for s_idx in track[pos:future_end]:
                    player = det_list[s_idx][1]
                    emb = player.reid_embedding if player.reid_embedding is not None else player.embedding
                    if emb is not None:
                        future_reid.append(emb)
                    if player.embedding is not None:
                        future_color.append(player.embedding)
                    if player.court_position is not None:
                        future_xy.append(player.court_position)
                if future_reid:
                    future_embs[det_idx] = np.mean(future_reid, axis=0)
                if future_color:
                    future_color_embs[det_idx] = np.mean(future_color, axis=0)
                if future_xy:
                    smoothed = _kalman_smooth_positions(future_xy)
                    future_positions[det_idx] = smoothed[0]

        return (
            past_embs,
            future_embs,
            past_color_embs,
            future_color_embs,
            past_positions,
            future_positions,
        )

    @staticmethod
    def _color_cost(player_i, player_j) -> float:
        """Cosine distance between color (histogram) embeddings.  0 if unavailable."""
        if player_i.embedding is None or player_j.embedding is None:
            return 0.0
        return cosine_dist(player_i.embedding, player_j.embedding)

    def _extended_cost(
        self,
        i: int,
        j: int,
        past_embs: dict[int, np.ndarray],
        future_embs: dict[int, np.ndarray],
        past_color_embs: dict[int, np.ndarray] | None,
        future_color_embs: dict[int, np.ndarray] | None,
    ) -> float | None:
        """Combined reid + color extended cost between mean tracklet embeddings.

        Reid component: squared cosine distance (super-linear penalty).
        Color component: cosine distance weighted by ``w_color``.

        Returns None when reid embeddings are unavailable for either side.
        """
        emb_i = past_embs.get(i)
        emb_j = future_embs.get(j)
        if emb_i is None or emb_j is None:
            return None
        d = cosine_dist(emb_i, emb_j)
        cost = d * d

        if past_color_embs is not None and future_color_embs is not None:
            c_i = past_color_embs.get(i)
            c_j = future_color_embs.get(j)
            if c_i is not None and c_j is not None:
                cost += self.w_color * cosine_dist(c_i, c_j)

        return cost

    # ------------------------------------------------------------------
    # Occlusion handling
    # ------------------------------------------------------------------

    def _build_occlusion_edges(
        self,
        mcf: MinCostFlow,
        det_list: list[tuple[int, object]],
        frame_to_dets: dict[int, list[int]],
        n_det: int,
        prev_tracks: list[list[int]],
        past_embs: dict[int, np.ndarray] | None,
        future_embs: dict[int, np.ndarray] | None,
        past_color_embs: dict[int, np.ndarray] | None,
        future_color_embs: dict[int, np.ndarray] | None,
    ) -> int:
        """Add long-range occlusion edges validated by existing tracks.

        When player A is occluded by player B, A's detection disappears
        and coincides with B's track.  For each detection *i* that is
        spatially close to some track *T* (but not on *T*), we scan *T*
        forward and add edges ``d_out(i) → d_in(j)`` for every detection
        *j* (also close to *T*) in the gap range
        ``(max_skip, max_occlusion]``.  The cost is appearance-based
        plus a per-frame penalty — no detection reward is collected for
        the skipped frames.
        """
        det_to_track: dict[int, int] = {}
        track_at_frame: list[dict[int, int]] = []
        track_sorted_frames: list[list[int]] = []
        for t_idx, track in enumerate(prev_tracks):
            fm: dict[int, int] = {}
            for d in track:
                det_to_track[d] = t_idx
                fm[det_list[d][0]] = d
            track_at_frame.append(fm)
            track_sorted_frames.append(sorted(fm.keys()))

        n_occ = 0
        gate = self.occlusion_gate
        min_gap = self.max_skip + 1
        max_gap = self.max_occlusion

        for i in range(n_det):
            frame_i = det_list[i][0]
            player_i = det_list[i][1]
            bbox_i = player_i.bbox
            if not bbox_i or len(bbox_i) < 4:
                continue

            i_track = det_to_track.get(i)

            for t_idx in range(len(prev_tracks)):
                if t_idx == i_track:
                    continue

                fm = track_at_frame[t_idx]
                if frame_i not in fm:
                    continue

                t_det_i = fm[frame_i]
                t_bbox_i = det_list[t_det_i][1].bbox
                if not t_bbox_i or len(t_bbox_i) < 4:
                    continue

                if bbox_bottom_mid_distance(bbox_i, t_bbox_i) > gate:
                    continue

                # Track t_idx is close to detection i — scan forward
                # along the track for valid exit points.
                frames = track_sorted_frames[t_idx]
                lo = bisect.bisect_left(frames, frame_i + min_gap)
                for fi in range(lo, len(frames)):
                    frame_j = frames[fi]
                    gap = frame_j - frame_i
                    if gap > max_gap:
                        break

                    t_det_j = fm[frame_j]
                    t_bbox_j = det_list[t_det_j][1].bbox
                    if not t_bbox_j or len(t_bbox_j) < 4:
                        continue

                    if frame_j not in frame_to_dets:
                        continue

                    for j in frame_to_dets[frame_j]:
                        j_track = det_to_track.get(j)
                        if j_track == t_idx:
                            continue
                        if i_track is not None and j_track == i_track:
                            continue

                        player_j = det_list[j][1]
                        bbox_j = player_j.bbox
                        if not bbox_j or len(bbox_j) < 4:
                            continue

                        if bbox_bottom_mid_distance(bbox_j, t_bbox_j) > gate:
                            continue

                        app_cost = self._appearance_cost(player_i, player_j)
                        if app_cost is None:
                            continue

                        color = self._color_cost(player_i, player_j)
                        cost = app_cost + self.w_color * color + gap * self.occlusion_penalty
                        if past_embs is not None and future_embs is not None:
                            ext = self._extended_cost(
                                i,
                                j,
                                past_embs,
                                future_embs,
                                past_color_embs,
                                future_color_embs,
                            )
                            if ext is not None:
                                cost += self.w_extended * ext

                        mcf.add_edge(
                            _det_out(i, n_det),
                            _det_in(j, n_det),
                            1,
                            cost,
                        )
                        n_occ += 1

        return n_occ

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
        past_color_embs: dict[int, np.ndarray] | None,
        future_color_embs: dict[int, np.ndarray] | None,
        pass_idx: int,
        prev_tracks: list[list[int]] | None = None,
        past_positions: dict[int, tuple[float, float]] | None = None,
        future_positions: dict[int, tuple[float, float]] | None = None,
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
                            cost = self._link_cost(
                                player_i,
                                player_j,
                                gap,
                                past_positions=past_positions,
                                future_positions=future_positions,
                                det_i=i,
                                det_j=j,
                            )
                            if cost is not None:
                                if past_embs is not None and future_embs is not None:
                                    ext = self._extended_cost(
                                        i,
                                        j,
                                        past_embs,
                                        future_embs,
                                        past_color_embs,
                                        future_color_embs,
                                    )
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
                                        ext = self._extended_cost(
                                            i,
                                            j,
                                            past_embs,
                                            future_embs,
                                            past_color_embs,
                                            future_color_embs,
                                        )
                                        if ext is not None:
                                            cost += self.w_extended * ext
                                    mcf.add_edge(oof_node_i, _det_in(j, n_det), 1, cost)
                                    n_oof_links += 1

        n_occ_links = 0
        if prev_tracks is not None and pass_idx >= self.occlusion_start_pass:
            n_occ_links = self._build_occlusion_edges(
                mcf,
                det_list,
                frame_to_dets,
                n_det,
                prev_tracks,
                past_embs,
                future_embs,
                past_color_embs,
                future_color_embs,
            )

        logger.info(
            "Pass %d: %d detections, %d link edges, %d oof edges, %d occ edges, %d nodes",
            pass_idx + 1,
            n_det,
            n_links,
            n_oof_links,
            n_occ_links,
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
        """OOF link cost: appearance + color (no spatial / IoU / skip)."""
        _ = frame_gap
        app = self._appearance_cost(player_i, player_j)
        if app is None:
            return None
        return app + self.w_color * self._color_cost(player_i, player_j)

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

    def _link_cost(
        self,
        player_i,
        player_j,
        frame_gap: int,
        *,
        skip_penalty: bool = True,
        past_positions: dict[int, tuple[float, float]] | None = None,
        future_positions: dict[int, tuple[float, float]] | None = None,
        det_i: int | None = None,
        det_j: int | None = None,
    ) -> float | None:
        """Cost of linking two detections. Exponential spatial cost discourages teleportation.

        In later passes, uses Kalman-smoothed court positions when available for more stable spatial cost.
        """
        # --- Spatial (exponential: no gate, but teleportation is heavily penalised) ---
        # Prefer court coordinates (meters) when available; fallback to bbox (pixels)
        # In later passes, use Kalman-smoothed positions for stability
        spatial_cost = 0.5
        pos_i = None
        pos_j = None
        if past_positions is not None and future_positions is not None and det_i is not None and det_j is not None:
            pos_i = past_positions.get(det_i)
            pos_j = future_positions.get(det_j)
        if pos_i is None:
            pos_i = player_i.court_position
        if pos_j is None:
            pos_j = player_j.court_position
        bbox_i = player_i.bbox
        bbox_j = player_j.bbox

        if pos_i is not None and pos_j is not None:
            m_dist = court_position_distance(pos_i, pos_j)
            scale = (self.max_speed / max(self.fps, 1)) * (max(1, frame_gap) ** 0.5)
            x = min(m_dist / max(scale, 1e-6), 15.0)  # clip to avoid overflow
            spatial_cost = np.exp(x) - 1.0
        elif bbox_i and bbox_j and len(bbox_i) >= 4 and len(bbox_j) >= 4:
            px_dist = bbox_bottom_mid_distance(bbox_i, bbox_j)
            scale = self.bbox_scale * (max(1, frame_gap) ** 0.5)
            x = min(px_dist / max(scale, 1e-6), 15.0)  # clip to avoid overflow
            spatial_cost = np.exp(x) - 1.0

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

        color = self._color_cost(player_i, player_j)

        cost = w_sp * spatial_cost + w_ap * app_cost + w_iou_eff * iou_cost + self.w_color * color
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
