"""Appearance + spatial tracker using Hungarian assignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.classes.player import PlayersDetections
from common.distances import cosine_dist

log = logging.getLogger(__name__)

_INF_COST = 1e9


@dataclass
class _Track:
    track_id: int
    last_frame: int
    bbox: list[int]
    reid_embedding: np.ndarray | None
    embedding: np.ndarray | None
    age: int = 0

    _ema_alpha: float = field(default=0.3, repr=False)

    def update(self, player, frame_id: int) -> None:
        self.last_frame = frame_id
        self.bbox = list(player.bbox)
        self.age = 0
        self._update_embedding("reid_embedding", player.reid_embedding)
        self._update_embedding("embedding", player.embedding)

    def _update_embedding(self, attr: str, new: np.ndarray | None) -> None:
        if new is None:
            return
        current = getattr(self, attr)
        if current is None:
            setattr(self, attr, new.copy())
        else:
            setattr(self, attr, self._ema_alpha * new + (1 - self._ema_alpha) * current)


def _bbox_centre(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


class SimpleAppearanceTracker:
    """Frame-to-frame tracker using Hungarian assignment on ReID + spatial costs."""

    def __init__(
        self,
        num_tracks: int = 10,
        max_age: int = 30,
        max_pixel_dist: float = 300.0,
        w_reid: float = 0.7,
        w_spatial: float = 0.3,
        ema_alpha: float = 0.3,
    ) -> None:
        self.num_tracks = num_tracks
        self.max_age = max_age
        self.max_pixel_dist = max_pixel_dist
        self.w_reid = w_reid
        self.w_spatial = w_spatial
        self.ema_alpha = ema_alpha

        self._tracks: list[_Track] = []
        self._next_id: int = 1

    def _spatial_cost(self, track: _Track, player) -> float:
        """Pixel-distance cost normalised to [0, 1] using max_pixel_dist."""
        cx1, cy1 = _bbox_centre(track.bbox)
        cx2, cy2 = _bbox_centre(list(player.bbox))
        dist = float(np.hypot(cx2 - cx1, cy2 - cy1))
        if self.max_pixel_dist > 0 and dist > self.max_pixel_dist:
            return _INF_COST
        scale = self.max_pixel_dist if self.max_pixel_dist > 0 else 1000.0
        return min(dist / scale, 1.0)

    def _appearance_cost(self, track: _Track, player) -> float:
        """Cosine distance using reid_embedding when available, else embedding."""
        track_emb = track.reid_embedding if track.reid_embedding is not None else track.embedding
        det_emb = player.reid_embedding if player.reid_embedding is not None else player.embedding
        if track_emb is None or det_emb is None:
            return 0.5  # neutral fallback when no embeddings exist
        return cosine_dist(track_emb, det_emb)

    def _link_cost(self, track: _Track, player) -> float:
        spatial = self._spatial_cost(track, player)
        if spatial >= _INF_COST:
            return _INF_COST
        appearance = self._appearance_cost(track, player)
        return self.w_reid * appearance + self.w_spatial * spatial

    def _match(self, players: list, frame_id: int) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian matching of active tracks against current-frame detections."""
        if not self._tracks or not players:
            return [], list(range(len(self._tracks))), list(range(len(players)))

        n_tracks = len(self._tracks)
        n_dets = len(players)
        cost_matrix = np.full((n_tracks, n_dets), _INF_COST, dtype=np.float64)

        for ti, track in enumerate(self._tracks):
            for pi, player in enumerate(players):
                cost_matrix[ti, pi] = self._link_cost(track, player)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(n_tracks))
        unmatched_dets = list(range(n_dets))

        for ti, pi in zip(row_ind, col_ind):
            if cost_matrix[ti, pi] < _INF_COST:
                matches.append((ti, pi))
                unmatched_tracks.remove(ti)
                unmatched_dets.remove(pi)

        return matches, unmatched_tracks, unmatched_dets

    def _process_frame(self, players: list, frame_id: int) -> None:
        matches, unmatched_tracks, unmatched_dets = self._match(players, frame_id)

        # matched detections
        for ti, pi in matches:
            track = self._tracks[ti]
            players[pi].player_id = track.track_id
            track.update(players[pi], frame_id)

        # unmatched tracks: age out
        for ti in unmatched_tracks:
            self._tracks[ti].age += 1

        # unmatched detections: spawn new
        active_ids = {t.track_id for t in self._tracks if t.age == 0}
        for pi in unmatched_dets:
            if len(active_ids) >= self.num_tracks:
                break
            new_id = self._next_id
            self._next_id += 1
            new_track = _Track(
                track_id=new_id,
                last_frame=frame_id,
                bbox=list(players[pi].bbox),
                reid_embedding=players[pi].reid_embedding.copy() if players[pi].reid_embedding is not None else None,
                embedding=players[pi].embedding.copy() if players[pi].embedding is not None else None,
                _ema_alpha=self.ema_alpha,
            )
            players[pi].player_id = new_id
            active_ids.add(new_id)
            self._tracks.append(new_track)

        # prune
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

    def track(self, detections: PlayersDetections) -> None:
        """Assign ``player_id`` in-place for every player in *detections*."""
        for frame_id in sorted(detections):
            self._process_frame(detections[frame_id], frame_id)

        n_assigned = sum(1 for dets in detections.values() for p in dets if p.player_id >= 0)
        log.debug("SimpleAppearanceTracker: assigned %d detections across %d frames", n_assigned, len(detections))
