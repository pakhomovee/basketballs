"""Trajectory extraction and observation building for coordinate smoothing."""

import numpy as np

from common.classes.player import FrameDetections


def extract_player_trajectories_with_gaps(
    detections: FrameDetections,
) -> dict[int, list[tuple[int, float | None, float | None]]]:
    """Extract per-player (frame_id, x_or_None, y_or_None) including frames with missing court_position."""
    trajectories: dict[int, list[tuple[int, float | None, float | None]]] = {}
    for frame_id in sorted(detections.keys()):
        for player in detections[frame_id]:
            pid = player.player_id
            if pid not in trajectories:
                trajectories[pid] = []
            pos = player.court_position
            if pos is not None:
                trajectories[pid].append((frame_id, pos[0], pos[1]))
            else:
                trajectories[pid].append((frame_id, None, None))
    return trajectories


def initial_state_4d(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute initial mean [x0, y0, vx0, vy0] and 4x4 cov from (2, T) observations."""
    x0, y0 = float(xs[0, 0]), float(xs[1, 0])
    if xs.shape[1] >= 2:
        dt = 1.0
        vx0 = (xs[0, 1] - xs[0, 0]) / dt
        vy0 = (xs[1, 1] - xs[1, 0]) / dt
    else:
        vx0, vy0 = 0.0, 0.0
    start_mean = np.array([x0, y0, vx0, vy0], dtype=float)
    start_cov = np.diag([0.1, 0.1, 0.5, 0.5])
    return start_mean, start_cov
