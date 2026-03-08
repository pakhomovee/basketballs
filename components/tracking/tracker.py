"""
Basketball player tracker — Kalman filter + IoU + appearance re-id.

- filterpy KalmanFilter for motion prediction (state lives on each Track)
- Bbox bottom-middle (feet) distance + IoU + appearance for matching
- Hard physics gate: max pixel displacement between bbox feet per frame
- Tentative tracks require n_init consecutive matches before confirmation
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .assignment import run_hungarian
from common.distances import bbox_bottom_mid_distance, bbox_iou, gallery_distance
from .merge import greedy_merge, merge_tracks_into
from .track import Track


class PlayerTracker:
    """
    Multi-object tracker for basketball.

    Parameters
    ----------
    dt : float
        Frame interval in seconds (default 1/30 for 30 fps).
    max_age : int
        Maximum time_since_update before a track is excluded from matching.
    n_init : int
        Hits required before a tentative track is promoted to confirmed.
    field_gate : float
        Maximum field distance for gating in benchmark (dict) path.
    bbox_gate : float
        Maximum pixel distance between bbox bottom-centers (feet) for gating.
    measurement_noise : float
        Diagonal of the Kalman measurement noise matrix **R**.
    """

    def __init__(
        self,
        dt: float = 1 / 30,
        max_age: int = 15,
        n_init: int = 3,
        field_gate: float = 4.0,
        bbox_gate: float = 10.0,
        measurement_noise: float = 2.0,
        max_track_length: int | None = None,
    ):
        self.dt = dt
        self.max_age = max_age
        self.max_track_length = max_track_length
        self.n_init = n_init
        self.field_gate = field_gate
        self.bbox_gate = bbox_gate
        self.measurement_noise = measurement_noise

        self.tracks: list[Track] = []
        self._next_id = 1

    # ── public API ──────────────────────────────────────────────────────

    def track(self, detections):
        """
        Track players in-place. Assigns persistent track IDs to player_id.

        Parameters
        ----------
        detections : dict[int, list[Player]]
            Per-frame player detections. Matching uses bbox feet distance + IoU + appearance.
            Expects player.embedding to be set (e.g. by extract_player_embeddings).
        """
        for frame_id in tqdm(sorted(detections.keys()), desc="Tracking"):
            self._process_players_frame(frame_id, detections[frame_id])

    def process_frame(self, frame_id: int, measurements: list[dict]):
        """Run one tracking step for *frame_id* (benchmark/CSV path)."""
        for m in measurements:
            m["_frame_id"] = frame_id

        for t in self.tracks:
            t.predict()

        all_idx = list(range(len(self.tracks)))
        matches, unmatched_tracks, unmatched_dets = self._match_dicts(all_idx, measurements)

        for ti, di in matches:
            self.tracks[ti].update(measurements[di])
            if (ref := measurements[di].get("_player_ref")) is not None:
                ref.player_id = self.tracks[ti].track_id

        for di in unmatched_dets:
            if (ref := measurements[di].get("_player_ref")) is not None:
                ref.player_id = self._next_id
            self._create_track(measurements[di])

    def run_tracking(self, data: list[dict]):
        """Track over all frames in *data* (flat list of detection dicts)."""
        from .data import get_measurements

        frame_ids = sorted({d["frame_id"] for d in data})
        for fid in frame_ids:
            meas = get_measurements(fid, data)
            self.process_frame(fid, meas)
        self.merge_tracks(max_tracks=10)

    # ── per-frame processing (Player path) ──────────────────────────────

    def _process_players_frame(self, frame_id: int, players):
        candidates = list(players)
        embeddings = [getattr(p, "embedding", None) for p in candidates]

        for t in self.tracks:
            t.predict()

        all_idx = list(range(len(self.tracks)))
        matches, unmatched_tracks, unmatched_players = self._match_players(all_idx, candidates, embeddings)

        for ti, pi in matches:
            track = self.tracks[ti]
            player = candidates[pi]
            track.update_from_player(player, frame_id, embedding=embeddings[pi])
            player.player_id = track.track_id

        for pi in unmatched_players:
            player = candidates[pi]
            player.player_id = self._next_id
            self._create_track_from_player(player, frame_id, embedding=embeddings[pi])

    # ── matching ─────────────────────────────────────────────────────────

    def _match_players(self, track_indices, players, embeddings):
        """
        Cascaded matching: recently-updated tracks get first pick of
        detections (prevents stale tracks from hijacking IDs).
        Unmatched detections then go through an IoU-only fallback stage.
        """
        active_indices = [
            i
            for i in track_indices
            if self.tracks[i].time_since_update <= self.max_age
            and (self.max_track_length is None or len(self.tracks[i].frame_ids) < self.max_track_length)
        ]

        by_age: dict[int, list[int]] = defaultdict(list)
        for i in active_indices:
            by_age[self.tracks[i].time_since_update].append(i)

        all_matches: list[tuple[int, int]] = []
        unmatched_tracks: list[int] = []
        remaining_det = list(range(len(players)))

        for age in sorted(by_age.keys()):
            age_indices = by_age[age]
            if not age_indices or not remaining_det:
                unmatched_tracks.extend(age_indices)
                continue

            sub_players = [players[j] for j in remaining_det]
            sub_embs = [embeddings[j] for j in remaining_det]

            matches, um_t, um_d = self._match_subset(
                age_indices,
                sub_players,
                sub_embs,
            )
            all_matches.extend((ti, remaining_det[di]) for ti, di in matches)
            unmatched_tracks.extend(um_t)
            remaining_det = [remaining_det[j] for j in um_d]

        if unmatched_tracks and remaining_det:
            sub_players = [players[j] for j in remaining_det]
            iou_matches, um_t2, um_d2 = self._match_iou_only(
                unmatched_tracks,
                sub_players,
            )
            all_matches.extend((ti, remaining_det[di]) for ti, di in iou_matches)
            unmatched_tracks = um_t2
            remaining_det = [remaining_det[j] for j in um_d2]

        return all_matches, unmatched_tracks, remaining_det

    def _match_subset(self, track_indices, players, embeddings):
        """Full-cost matching (distance + IoU + appearance) on a subset."""
        tracks = [self.tracks[i] for i in track_indices]
        nt, np_ = len(tracks), len(players)
        INF = 1e5
        cost = np.full((nt, np_), INF)

        for i, t in enumerate(tracks):
            for j, p in enumerate(players):
                iou_cost = 1.0 - bbox_iou(t.bbox, np.asarray(p.bbox, dtype=float)) if p.bbox else 0.5
                app_cost = gallery_distance(list(t.gallery), embeddings[j])

                if p.bbox and len(np.asarray(p.bbox).ravel()) >= 4:
                    px_dist = bbox_bottom_mid_distance(t.bbox, p.bbox)
                    effective_gate = self.bbox_gate * (1 + 0.5 * t.time_since_update)
                    if px_dist > effective_gate:
                        continue
                    px_cost = px_dist / effective_gate
                    cost[i, j] = 0.3 * px_cost + 0.4 * iou_cost + 0.3 * app_cost
                else:
                    cost[i, j] = 0.5 * iou_cost + 0.5 * app_cost

        matches, um_t, um_p = run_hungarian(cost, INF)
        matches_out = [(track_indices[ti], pj) for ti, pj in matches]
        um_tracks_out = [track_indices[i] for i in um_t]
        return matches_out, um_tracks_out, um_p

    def _match_iou_only(self, track_indices, players, iou_threshold: float = 0.3):
        """IoU-only fallback for detections that failed appearance matching."""
        tracks = [self.tracks[i] for i in track_indices]
        nt, np_ = len(tracks), len(players)
        INF = 1e5
        cost = np.full((nt, np_), INF)

        for i, t in enumerate(tracks):
            for j, p in enumerate(players):
                if not p.bbox or len(np.asarray(p.bbox).ravel()) < 4:
                    continue
                iou_val = bbox_iou(t.bbox, np.asarray(p.bbox, dtype=float))
                if iou_val >= iou_threshold:
                    cost[i, j] = 1.0 - iou_val

        matches, um_t, um_p = run_hungarian(cost, INF)
        matches_out = [(track_indices[ti], pj) for ti, pj in matches]
        um_tracks_out = [track_indices[i] for i in um_t]
        return matches_out, um_tracks_out, um_p

    def _match_dicts(self, track_indices, detections):
        """Match tracks to measurement dicts (benchmark path)."""
        active_indices = [i for i in track_indices if self.tracks[i].time_since_update <= self.max_age]
        tracks = [self.tracks[i] for i in active_indices]
        nt, nd = len(tracks), len(detections)
        INF = 1e5
        cost = np.full((nt, nd), INF)

        for i, t in enumerate(tracks):
            for j, det in enumerate(detections):
                z = np.asarray(det["field_coords"], dtype=float).ravel()
                fdist = np.linalg.norm(t.field_pos - z)

                if fdist > self.field_gate:
                    continue

                iou_val = bbox_iou(t.bbox, det["bbox"])
                field_cost = fdist / self.field_gate
                iou_cost = 1.0 - iou_val
                cost[i, j] = 0.6 * field_cost + 0.4 * iou_cost

        matches, um_t, um_d = run_hungarian(cost, INF)
        unmatched_tracks = [active_indices[i] for i in um_t]
        unmatched_dets = um_d
        matches_pairs = [(active_indices[ti], di) for ti, di in matches]
        return matches_pairs, unmatched_tracks, unmatched_dets

    # ── helpers ──────────────────────────────────────────────────────────

    def merge_tracks(
        self,
        max_tracks: int = 10,
        detections=None,
        track_id_to_team: dict[int, int] | None = None,
        cost_threshold: float = 0.7,
    ):
        """
        Reduce to at most max_tracks by merging similar non-overlapping tracks.

        Uses greedy merging to minimize cost (distance + IoU + appearance).
        Tracks that appear in the same frame cannot be merged.
        When track_id_to_team is provided, never merges tracks from different teams.

        If detections is provided (dict[frame_id, list[Player]]), updates player_ids
        to reflect the merge (e.g. after track() in the main pipeline).

        cost_threshold prevents merging dissimilar tracks even if max_tracks
        hasn't been reached — wrong merges are worse than extra tracks.
        """
        if len(self.tracks) <= max_tracks:
            old_to_compact: dict[int, int] = {}
            for new_id, t in enumerate(self.tracks, start=1):
                old_to_compact[t.track_id] = new_id
            for t in self.tracks:
                t.track_id = old_to_compact[t.track_id]
            if detections is not None:
                self._apply_merge_to_detections(detections, old_to_compact)
            return

        partition = greedy_merge(
            self.tracks,
            max_tracks,
            bbox_gate=self.bbox_gate * 2,
            w_app=0.4,
            w_dist=0.3,
            w_iou=0.3,
            frame_gap_scale=0.5,
            track_id_to_team=track_id_to_team,
            cost_threshold=cost_threshold,
        )

        kept_indices = set()
        for grp in partition:
            kept_indices.update(grp)
        kept_track_ids = {self.tracks[i].track_id for i in kept_indices}

        survivor_id_map: dict[int, int] = {}
        merged = merge_tracks_into(self.tracks, partition, survivor_id_map)

        all_track_ids = {t.track_id for t in self.tracks}
        dropped_ids = all_track_ids - kept_track_ids
        for did in dropped_ids:
            survivor_id_map[did] = -1

        self.tracks = merged

        old_to_compact: dict[int, int] = {}
        for new_id, t in enumerate(self.tracks, start=1):
            old_to_compact[t.track_id] = new_id
        for old_id, survivor_id in survivor_id_map.items():
            if survivor_id == -1:
                old_to_compact[old_id] = -1
                continue
            root = survivor_id
            while root in survivor_id_map and survivor_id_map[root] != -1:
                root = survivor_id_map[root]
            old_to_compact[old_id] = old_to_compact.get(root, -1)

        for t in self.tracks:
            t.track_id = old_to_compact[t.track_id]

        if detections is not None:
            self._apply_merge_to_detections(detections, old_to_compact)

    def _apply_merge_to_detections(self, detections, old_to_compact: dict[int, int]):
        """Remap player_ids in detections and drop players with player_id=-1."""
        for frame_id in detections:
            players = detections[frame_id]
            for p in players:
                if p.player_id in old_to_compact:
                    p.player_id = old_to_compact[p.player_id]
            unique_by_id: dict[int, object] = {}
            for p in players:
                pid = p.player_id
                if pid < 0:
                    continue
                if pid not in unique_by_id:
                    unique_by_id[pid] = p
                elif (not unique_by_id[pid].bbox) and p.bbox:
                    unique_by_id[pid] = p
            detections[frame_id] = list(unique_by_id.values())

    def _create_track(self, meas):
        t = Track(self._next_id, meas, dt=self.dt, measurement_noise=self.measurement_noise, n_init=self.n_init)
        self._next_id += 1
        self.tracks.append(t)

    def _create_track_from_player(self, player, frame_id: int, embedding: np.ndarray | None = None):
        t = Track.from_player(
            self._next_id,
            player,
            frame_id,
            dt=self.dt,
            measurement_noise=self.measurement_noise,
            n_init=self.n_init,
            embedding=embedding,
        )
        self._next_id += 1
        self.tracks.append(t)
