"""Online Hungarian-matching tracker for dynamic-camera basketball footage."""

from __future__ import annotations

from collections import deque
from enum import Enum, auto

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.classes.player import Player, PlayersDetections
from common.distances import cosine_dist
from config import AppConfig, TrackerConfig

# Sentinel for infeasible assignment entries.
_INF = 1e5


# Kalman for coordinates


class _KalmanPositionFilter:
    """4-state constant-velocity Kalman filter over 2-D **court position**.

    State vector ``x = [x, y, vx, vy]``  (metres + metres/frame).
    Measurement  ``z = [x, y]``  from homography-mapped court coordinates.
    """

    _std_meas: float = 0.3
    _std_acc: float = 0.5

    def __init__(self, measurement: tuple[float, float]) -> None:
        self._F = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self._H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        self.mean = np.array([float(measurement[0]), float(measurement[1]), 0.0, 0.0])
        self.covariance = np.diag([1.0, 1.0, 5.0, 5.0])

    def _process_noise(self) -> np.ndarray:
        q = self._std_acc**2
        return np.diag([q * 0.25, q * 0.25, q, q])

    def _measurement_noise(self) -> np.ndarray:
        s = self._std_meas**2
        return np.diag([s, s])

    def predict(self) -> None:
        Q = self._process_noise()
        self.mean = self._F @ self.mean
        self.covariance = self._F @ self.covariance @ self._F.T + Q

    def update(self, measurement: np.ndarray) -> None:
        R = self._measurement_noise()
        H = self._H
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ (measurement - H @ self.mean)
        self.covariance = (np.eye(4) - K @ H) @ self.covariance

    def mahalanobis(self, measurement: np.ndarray) -> float:
        """Squared Mahalanobis distance in observation (position) space."""
        H = self._H
        d = measurement - H @ self.mean
        S = H @ self.covariance @ H.T + self._measurement_noise()
        try:
            L = np.linalg.cholesky(S)
            z = np.linalg.solve(L, d)
            return float(z @ z)
        except np.linalg.LinAlgError:
            return float(d @ np.linalg.pinv(S) @ d)

    @property
    def predicted_pos(self) -> tuple[float, float]:
        """Current (possibly predicted) 2-D court position in metres."""
        return (float(self.mean[0]), float(self.mean[1]))


# TrackState & Track


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()


class Track:
    """Single-target track with shape Kalman, position memory and embedding buffers."""

    def __init__(
        self,
        track_id: int,
        detection: Player,
        *,
        cfg: TrackerConfig,
    ) -> None:
        self.track_id = track_id
        self.state = TrackState.TENTATIVE
        self.hits: int = 1
        self.time_since_update: int = 0

        self._cfg = cfg
        self._n_init = cfg.n_init
        self._max_age = cfg.max_skip

        assert detection.court_position is not None
        self.kf = _KalmanPositionFilter(detection.court_position)

        self.last_bbox_bottom: float | None = float(detection.bbox[3]) if detection.bbox else None
        self.last_bbox_height: float | None = float(detection.bbox[3] - detection.bbox[1]) if detection.bbox else None

        self._reid_buf: deque[np.ndarray] = deque(maxlen=cfg.lookback)
        self._color_buf: deque[np.ndarray] = deque(maxlen=cfg.lookback)
        self._store_embeddings(detection)

    # gallery

    def _store_embeddings(self, det: Player) -> None:
        if det.reid_embedding is not None:
            self._reid_buf.append(det.reid_embedding.copy())
        if det.embedding is not None:
            self._color_buf.append(det.embedding.copy())

    @staticmethod
    def _l2_mean(buf: deque[np.ndarray]) -> np.ndarray | None:
        if not buf:
            return None
        m = np.mean(list(buf), axis=0)
        n = np.linalg.norm(m)
        return m / n if n > 1e-8 else m

    @property
    def mean_reid(self) -> np.ndarray | None:
        return self._l2_mean(self._reid_buf)

    @property
    def mean_color(self) -> np.ndarray | None:
        return self._l2_mean(self._color_buf)

    # updates

    def predict(self) -> None:
        if self.time_since_update >= self._cfg.kf_velocity_reset_age:
            self.kf.mean[2] = 0.0
            self.kf.mean[3] = 0.0
        self.kf.predict()
        self.time_since_update += 1

    def update(self, det: Player) -> None:
        # Detect partial-crop frames.  Two independent signals:
        #   1. Bbox bottom jumps upward by > bbox_bottom_jump_frac * height
        #      (player cropped at the bottom — foot point wrong for homography).
        #   2. Bbox height shrinks by > bbox_height_shrink_frac relative to
        #      the track's last observed height (partial occlusion / cut-off).
        bbox_unreliable = False
        if det.bbox:
            curr_bottom = float(det.bbox[3])
            curr_height = max(float(det.bbox[3] - det.bbox[1]), 1.0)
            if self.last_bbox_bottom is not None:
                if abs(curr_bottom - self.last_bbox_bottom) > self._cfg.bbox_bottom_jump_frac * curr_height:
                    bbox_unreliable = True
            if self.last_bbox_height is not None:
                if curr_height < (1.0 - self._cfg.bbox_height_shrink_frac) * self.last_bbox_height:
                    bbox_unreliable = True
            self.last_bbox_bottom = curr_bottom
            self.last_bbox_height = curr_height

        if det.court_position is not None:
            meas = np.array([det.court_position[0], det.court_position[1]])
            if not bbox_unreliable and self.kf.mahalanobis(meas) <= self._cfg.kf_update_gate:
                self.kf.update(meas)
        if not bbox_unreliable:
            self._store_embeddings(det)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.LOST
        elif self.time_since_update > self._max_age:
            self.state = TrackState.LOST

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.LOST


class HungarianTracker:
    """Online multi-object tracker using cascaded Hungarian matching."""

    def __init__(
        self,
        cfg: AppConfig | None = None,
        frame_width: float | None = None,
        fps: float = 30.0,
    ) -> None:
        tc = cfg.tracker if cfg else TrackerConfig()
        self._tc = tc

        # Cost weights
        self.w_spatial: float = tc.w_spatial
        self.w_app: float = tc.w_app
        self.w_color: float = tc.w_color

        self._tracks: list[Track] = []
        self._next_id: int = 1

    # interface

    def track(self, detections: PlayersDetections) -> None:
        """Assign ``player_id`` for every detection in *detections*, in-place."""
        self._tracks = []
        self._next_id = 1
        for players in detections.values():
            for player in players:
                player.player_id = -1

        for i, frame_id in enumerate(sorted(detections)):
            self._step(detections[frame_id])

    # step

    def _step(self, players: list[Player]) -> None:
        # 0. Mask-based false-positive filter.
        valid = self._valid_detection_indices(players)
        median_h = self._median_confirmed_bbox_height()
        if median_h is not None:
            valid = [
                i
                for i in valid
                if not (
                    players[i].bbox
                    and float(players[i].bbox[3] - players[i].bbox[1])
                    < (1.0 - self._tc.bbox_height_shrink_frac) * median_h
                )
            ]

        # 1. Predict all existing tracks forward one frame.
        for t in self._tracks:
            t.predict()

        # 2. Cascade matching
        unmatched_det: set[int] = set(valid)
        matched_pairs: list[tuple[Track, int]] = []

        # Group all active (non-lost) tracks by freshness level.
        levels: dict[int, list[Track]] = {}
        for t in self._tracks:
            if t.state == TrackState.LOST:
                continue
            levels.setdefault(t.time_since_update, []).append(t)
        # Sort within each level: confirmed before tentative
        for lvl in levels:
            levels[lvl].sort(key=lambda t: t.state != TrackState.CONFIRMED)

        matched_track_ids: set[int] = set()
        for level in sorted(levels):
            if not unmatched_det:
                break
            level_tracks = [t for t in levels[level] if t.track_id not in matched_track_ids]
            m_t, m_d = self._hungarian_match(
                level_tracks,
                players,
                sorted(unmatched_det),
                gate=True,
            )
            for t, d in zip(m_t, m_d):
                matched_pairs.append((t, d))
                matched_track_ids.add(t.track_id)
                unmatched_det.discard(d)

        # 3. Apply updates.
        for t, d_idx in matched_pairs:
            t.update(players[d_idx])
            if t.state == TrackState.CONFIRMED:
                players[d_idx].player_id = t.track_id

        # 4. Mark unmatched tracks.
        for t in self._tracks:
            if t.track_id not in matched_track_ids:
                t.mark_missed()

        # 5. Spawn hypothesis tracks for unmatched valid detections.
        median_h = self._median_confirmed_bbox_height()
        for d_idx in sorted(unmatched_det):
            det_spawn = players[d_idx]
            if median_h is not None and det_spawn.bbox:
                det_h = float(det_spawn.bbox[3] - det_spawn.bbox[1])
                if det_h < (1.0 - self._tc.bbox_height_shrink_frac) * median_h:
                    continue  # looks like a partial crop
            t = Track(
                self._next_id,
                det_spawn,
                cfg=self._tc,
            )
            self._next_id += 1
            self._tracks.append(t)

        # 6. Remove deleted tracks.
        self._tracks = [t for t in self._tracks if not t.is_deleted]

    # helpers

    def _median_confirmed_bbox_height(self) -> float | None:
        """Median bbox height (px) across all currently confirmed tracks."""
        heights = [
            t.last_bbox_height
            for t in self._tracks
            if t.state == TrackState.CONFIRMED and t.last_bbox_height is not None
        ]
        return float(np.median(heights)) if heights else None

    def _on_court(self, court_pos: tuple[float, float] | None) -> bool:
        """Return True when *court_pos* is inside the playing court boundaries."""
        if court_pos is None:
            return False
        x, y = court_pos
        return abs(x) <= self._tc.court_half_x and abs(y) <= self._tc.court_half_y

    def _valid_detection_indices(self, players: list[Player]) -> list[int]:
        """Return indices of detections eligible for matching."""
        has_any_mask = any(p.mask_polygon is not None and len(p.mask_polygon) > 0 for p in players)
        if not has_any_mask:
            return [i for i, p in enumerate(players) if self._on_court(p.court_position)]
        return [
            i
            for i, p in enumerate(players)
            if p.mask_polygon is not None and len(p.mask_polygon) > 0 and self._on_court(p.court_position)
        ]

    def _hungarian_match(
        self,
        tracks: list[Track],
        players: list[Player],
        det_indices: list[int],
        *,
        gate: bool,
    ) -> tuple[list[Track], list[int]]:
        if not tracks or not det_indices:
            return [], []

        cost = np.full((len(tracks), len(det_indices)), _INF)
        for i, t in enumerate(tracks):
            for j, d_idx in enumerate(det_indices):
                c = self._combined_cost(t, players[d_idx], gate=gate)
                if c is not None:
                    cost[i, j] = c

        row, col = linear_sum_assignment(cost)
        matched_t: list[Track] = []
        matched_d: list[int] = []
        for r, c in zip(row, col):
            if cost[r, c] < _INF:
                matched_t.append(tracks[r])
                matched_d.append(det_indices[c])
        return matched_t, matched_d

    # cost

    def _combined_cost(self, track: Track, det: Player, *, gate: bool) -> float | None:
        """Weighted combination of spatial and appearance cost."""
        court_pos_b = det.court_position
        if court_pos_b is None:
            return None

        # position gate (hard)
        px, py = track.kf.predicted_pos
        court_dist = float(np.hypot(px - court_pos_b[0], py - court_pos_b[1]))
        gate_m = min(
            self._tc.court_gate_base_m + self._tc.court_gate_per_frame_m * track.time_since_update,
            self._tc.court_gate_max_m,
        )
        if gate and court_dist > gate_m:
            return None

        # Mahalanobis spatial cost (soft)
        bbox_unreliable = False
        if det.bbox:
            curr_bottom = float(det.bbox[3])
            curr_height = max(float(det.bbox[3] - det.bbox[1]), 1.0)
            if track.last_bbox_bottom is not None:
                if abs(curr_bottom - track.last_bbox_bottom) > self._tc.bbox_bottom_jump_frac * curr_height:
                    bbox_unreliable = True
            if track.last_bbox_height is not None:
                if curr_height < (1.0 - self._tc.bbox_height_shrink_frac) * track.last_bbox_height:
                    bbox_unreliable = True

        meas = np.array([court_pos_b[0], court_pos_b[1]])
        maha = track.kf.mahalanobis(meas)
        if bbox_unreliable or maha > self._tc.kf_update_gate:
            spatial_cost = 0.5  # position unreliable — let appearance decide
        else:
            spatial_cost = min(maha / self._tc.chi2_95_2dof, 1.0)

        total_cost = self.w_spatial * spatial_cost
        total_weight = self.w_spatial

        # ReID — only when both track gallery and detection embedding exist.
        mean_reid = track.mean_reid
        if mean_reid is not None and det.reid_embedding is not None:
            reid_cost = cosine_dist(mean_reid, det.reid_embedding)
            total_cost += self.w_app * reid_cost
            total_weight += self.w_app

        # Colour — hard gate (team conflict) then soft cost.
        mean_color = track.mean_color
        if mean_color is not None and det.embedding is not None:
            color_cost = cosine_dist(mean_color, det.embedding)
            if len(track._color_buf) >= self._tc.color_min_samples and color_cost > self._tc.color_team_gate:
                return None  # clearly different team jersey colour
            total_cost += self.w_color * color_cost
            total_weight += self.w_color

        return total_cost / total_weight
