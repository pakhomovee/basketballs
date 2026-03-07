"""
Track state and Track class for multi-object tracking.

Each Track owns a filterpy KalmanFilter (state=[x, y, vx, vy], measurement=[x, y])
so tracks are fully self-contained — no matrices need to be passed in from outside.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from filterpy.kalman import KalmanFilter


class TrackState:
    """Track lifecycle states."""
    TENTATIVE = 0
    CONFIRMED = 1


class Track:
    """Single object track with Kalman state, bbox, and appearance gallery."""

    _GALLERY_MAX = 30

    def __init__(self, track_id: int, meas: dict, *,
                 dt: float = 1 / 30,
                 measurement_noise: float = 2.0,
                 n_init: int = 3):
        self.track_id = track_id
        self.state = TrackState.TENTATIVE
        self._n_init = n_init

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.kf.Q = np.diag([0.5, 0.5, 5.0, 5.0])
        self.kf.R = np.eye(2) * measurement_noise
        self.kf.P = np.diag([1.0, 1.0, 50.0, 50.0])

        fc = np.asarray(meas["field_coords"], dtype=float).ravel()
        self.kf.x = np.array([[fc[0]], [fc[1]], [0.0], [0.0]])

        self.bbox = np.asarray(meas["bbox"], dtype=float)
        self._first_bbox = self.bbox.copy()
        self._last_bbox = self.bbox.copy()
        self._bbox_vel = np.zeros(4)

        self.gallery: deque[np.ndarray] = deque(maxlen=self._GALLERY_MAX)
        emb = meas.get("embedding")
        if emb is not None:
            self.gallery.append(emb)

        self.history: list[list[float]] = [fc.tolist()]
        self.frame_ids: list[int] = [meas.get("_frame_id", 0)]
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    @classmethod
    def from_player(cls, track_id: int, player, frame_id: int, *,
                    dt: float = 1 / 30, measurement_noise: float = 2.0,
                    n_init: int = 3, embedding: np.ndarray | None = None):
        """Build track directly from a Player object."""
        fc = list(player.court_position) if player.court_position else [0.0, 0.0]
        meas = {
            "field_coords": fc,
            "bbox": player.bbox,
            "_frame_id": frame_id,
        }
        if embedding is not None:
            meas["embedding"] = embedding
        return cls(track_id, meas, dt=dt, measurement_noise=measurement_noise,
                   n_init=n_init)

    def predict(self):
        """Predict state and bbox one step ahead."""
        self.kf.predict()
        self.bbox = self.bbox + self._bbox_vel
        self.age += 1
        self.time_since_update += 1

    def update(self, meas: dict):
        """
        Update with measurement dict.

        Keys: bbox (required), field_coords (optional — skips Kalman if absent),
        embedding (optional), _frame_id (optional, default 0).
        """
        has_field = "field_coords" in meas and meas["field_coords"] is not None

        if has_field:
            z = np.asarray(meas["field_coords"], dtype=float).reshape(2, 1)
            self.kf.update(z)
            new_bbox = np.asarray(meas["bbox"], dtype=float)
            self._bbox_vel = new_bbox - self.bbox
            self.bbox = new_bbox
            self._last_bbox = new_bbox.copy()
        else:
            self.bbox = np.asarray(meas["bbox"], dtype=float)
            self._last_bbox = self.bbox.copy()

        emb = meas.get("embedding")
        if emb is not None:
            self.gallery.append(emb)

        self.history.append(self.field_pos.tolist())
        self.frame_ids.append(meas.get("_frame_id", 0))

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def update_from_player(self, player, frame_id: int,
                           embedding: np.ndarray | None = None):
        """Update directly from a Player object."""
        meas: dict = {"bbox": player.bbox, "_frame_id": frame_id}
        if player.court_position is not None:
            meas["field_coords"] = list(player.court_position)
        if embedding is not None:
            meas["embedding"] = embedding
        self.update(meas)

    @property
    def field_pos(self):
        return self.kf.x[:2, 0]
