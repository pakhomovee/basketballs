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
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from common.distances import bbox_bottom_mid_distance, bbox_iou, cosine_dist
from config import load_default_config
from tracking.min_cost_flow import MinCostFlow

if TYPE_CHECKING:
    from config import AppConfig

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

    Almost all hyperparameters live in the application config under
    ``cfg.tracker`` (:class:`~config.TrackerConfig`). If ``cfg`` is omitted,
    :func:`config.load_default_config` loads the default YAML.

    Only ``fps`` and ``frame_width`` are set on the constructor; everything
    else below is read from ``cfg.tracker``.

    Constructor
    -----------
    cfg : AppConfig | None, optional
        Full app config; tracker fields are ``cfg.tracker.*``.
    frame_width : float | None, optional
        Frame width in pixels. If ``None``, inferred from the maximum bbox
        ``x2`` over all detections. Used for out-of-frame handling (players
        near the left/right edge).
    fps : float, optional
        Video frame rate. Not part of ``cfg.tracker``; supply from the input
        video (e.g. computed in the pipeline ``main``).

    Fields in ``cfg.tracker``
    -------------------------
    num_tracks : int
        Number of concurrent tracks (= flow units). E.g. 10 for basketball.
    max_skip : int
        Maximum frame gap for a direct link between two detections. Larger
        gaps cannot be bridged by a single edge (the track must end/restart).
    bbox_scale : float
        Pixel scale for exponential spatial cost:
        ``cost ≈ exp(px_dist / (bbox_scale * sqrt(frame_gap))) - 1``. There is
        no hard distance gate; very long jumps are discouraged exponentially.
    w_spatial, w_app, w_iou : float
        Weights for spatial, appearance (ReID / embedding), and IoU link costs.
    w_extended : float
        Weight for the extra term built from mean tracklet embeddings in
        passes 2+. Penalises links whose local tracklet neighbourhoods disagree.
        Set to ``0`` to turn off that enrichment.
    w_color : float
        Weight for colour-embedding cosine distance on links (soft team cue),
        also folded into the extended / tracklet-neighbourhood cost.
    skip_penalty : float
        Additive cost per skipped frame along an inter-detection edge.
    detection_reward : float
        Negative cost (reward) per taken detection node — encourages covering
        detections; a track of length ``N`` gains ``N × detection_reward``.
    enter_cost, exit_cost : float
        Costs to start or end a track; higher values favour longer continuous tracks.
    oof_entry_cost : float
        Extra cost for edges from out-of-frame state (``start_out -> d_in``,
        ``oof_i -> d_in``) to avoid abusing OOF when a normal in-frame link exists.
    edge_margin : float
        Pixel margin from the frame border for treating a bbox as “near edge”
        (candidate for entering/leaving via out-of-frame nodes).
    k_warmup_frames : int
        New tracks may start only in the first ``k_warmup`` frames or from
        ``start_out`` (entering from the side).
    last_frames : int
        Tracks may finish only in the last ``last_frames`` frames or via
        ``end_out`` (exit to the side).
    n_passes : int
        Number of min-cost flow passes. Pass 1 uses base costs; later passes
        add tracklet embedding context from the previous solution.
    lookback : int
        How many past/future steps per track enter the mean embedding used in
        multi-pass refinement.
    max_occlusion : int
        Largest frame gap that an occlusion edge may span (riding along an
        existing track while the player has no own detection).
    occlusion_gate : float
        Maximum pixel distance (bbox bottom-centre) between a detection and a
        candidate occlusion partner on another track.
    occlusion_penalty : float
        Per-frame cost on occlusion edges (usually small; skipped frames forgo
        detection reward already).
    occlusion_start_pass : int
        Pass index (0-based) at which occlusion edges are first enabled.
    """

    def __init__(
        self,
        cfg: "AppConfig | None" = None,
        frame_width: float | None = None,
        fps: float = 30.0,
    ):
        if cfg is None:
            cfg = load_default_config()
        tracker_cfg = cfg.tracker

        self.num_tracks = tracker_cfg.num_tracks
        self.max_skip = tracker_cfg.max_skip
        self.bbox_scale = tracker_cfg.bbox_scale
        self.w_spatial = tracker_cfg.w_spatial
        self.start_w_app = tracker_cfg.w_app
        self.w_app = self.start_w_app
        self.w_iou = tracker_cfg.w_iou
        self.w_extended = tracker_cfg.w_extended
        self.skip_penalty = tracker_cfg.skip_penalty
        self.detection_reward = tracker_cfg.detection_reward
        self.enter_cost = tracker_cfg.enter_cost
        self.exit_cost = tracker_cfg.exit_cost
        self.fps = fps
        self.frame_width = frame_width
        self.edge_margin = tracker_cfg.edge_margin
        self.k_warmup_frames = tracker_cfg.k_warmup_frames
        self.last_frames = tracker_cfg.last_frames
        self.n_passes = tracker_cfg.n_passes
        self.lookback = tracker_cfg.lookback
        self.oof_entry_cost = tracker_cfg.oof_entry_cost
        self.w_color = tracker_cfg.w_color
        self.max_occlusion = tracker_cfg.max_occlusion
        self.occlusion_gate = tracker_cfg.occlusion_gate
        self.occlusion_penalty = tracker_cfg.occlusion_penalty
        self.occlusion_start_pass = tracker_cfg.occlusion_start_pass

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
        self.w_app = self.start_w_app
        for pass_idx in range(self.n_passes):
            past_embs: dict[int, np.ndarray] | None = None
            future_embs: dict[int, np.ndarray] | None = None
            past_color_embs: dict[int, np.ndarray] | None = None
            future_color_embs: dict[int, np.ndarray] | None = None
            if tracks is not None:
                past_embs, future_embs, past_color_embs, future_color_embs = self._compute_tracklet_embeddings(
                    tracks,
                    det_list,
                    lookback,
                )
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
    ]:
        """Directional mean embeddings (reid + color) for each tracked detection.

        Returns
        -------
        (past_embs, future_embs, past_color_embs, future_color_embs)
            Reid and color embeddings averaged over a lookback window in
            each direction along the track.
        """
        past_embs: dict[int, np.ndarray] = {}
        future_embs: dict[int, np.ndarray] = {}
        past_color_embs: dict[int, np.ndarray] = {}
        future_color_embs: dict[int, np.ndarray] = {}

        for track in tracks:
            for pos, det_idx in enumerate(track):
                past_start = max(0, pos - lookback)
                past_reid: list[np.ndarray] = []
                past_color: list[np.ndarray] = []
                for s_idx in track[past_start : pos + 1]:
                    player = det_list[s_idx][1]
                    emb = player.reid_embedding if player.reid_embedding is not None else player.embedding
                    if emb is not None:
                        past_reid.append(emb)
                    if player.embedding is not None:
                        past_color.append(player.embedding)
                if past_reid:
                    past_embs[det_idx] = np.mean(past_reid, axis=0)
                if past_color:
                    past_color_embs[det_idx] = np.mean(past_color, axis=0)

                future_end = min(len(track), pos + lookback + 1)
                future_reid: list[np.ndarray] = []
                future_color: list[np.ndarray] = []
                for s_idx in track[pos:future_end]:
                    player = det_list[s_idx][1]
                    emb = player.reid_embedding if player.reid_embedding is not None else player.embedding
                    if emb is not None:
                        future_reid.append(emb)
                    if player.embedding is not None:
                        future_color.append(player.embedding)
                if future_reid:
                    future_embs[det_idx] = np.mean(future_reid, axis=0)
                if future_color:
                    future_color_embs[det_idx] = np.mean(future_color, axis=0)

        return past_embs, future_embs, past_color_embs, future_color_embs

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

    def _link_cost(self, player_i, player_j, frame_gap: int, *, skip_penalty: bool = True) -> float | None:
        """Cost of linking two detections. Exponential spatial cost discourages teleportation."""
        # --- Spatial (exponential: no gate, but teleportation is heavily penalised) ---
        spatial_cost = 0.5
        bbox_i = player_i.bbox
        bbox_j = player_j.bbox

        if bbox_i and bbox_j and len(bbox_i) >= 4 and len(bbox_j) >= 4:
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
