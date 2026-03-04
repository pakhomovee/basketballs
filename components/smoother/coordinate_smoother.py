"""Kalman smoothing for player court coordinates."""

import numpy as np

from common.classes.player import FrameDetections
from kalmanlib import KalmanFilter

from .filter import filter_signal_with_missing
from .models import A_4STATE, B_4STATE
from .trajectory import extract_player_trajectories_with_gaps, initial_state_4d


def smooth_detection_coordinates(
    detections: FrameDetections,
    *,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Rx: np.ndarray | None = None,
    Ry: np.ndarray | None = None,
    min_points: int = 4,
) -> None:
    """
    Apply Kalman smoothing to court coordinates in-place.

    Uses 4-state model (x, y, vx, vy). Imputes missing positions
    when homography fails.

    Args:
        detections: FrameDetections to smooth (modified in-place).
        A: Dynamics matrix (4,4). Default: constant-velocity model.
        B: Observation matrix (2,4). Default: observe [x,y] from state.
        Rx: Observation noise covariance (2,2). Default: 0.1*eye(2).
        Ry: Process noise covariance (4,4). Default: diag(0.01,0.01,0.1,0.1).
        min_points: Minimum observations per player to apply smoothing.
    """
    if A is None:
        A = A_4STATE.copy()
    if B is None:
        B = B_4STATE.copy()
    if Rx is None:
        Rx = 0.1 * np.eye(2)
    if Ry is None:
        Ry = np.diag([0.01, 0.01, 0.1, 0.1])

    trajectories_full = extract_player_trajectories_with_gaps(detections)
    frame_to_smoothed: dict[int, dict[int, tuple[float, float]]] = {}

    for player_id, traj in trajectories_full.items():
        frame_ids = [p[0] for p in traj]
        observed_mask = np.array([p[1] is not None for p in traj])
        n_observed = int(observed_mask.sum())

        if n_observed == 0:
            continue
        if n_observed < min_points and not np.any(~observed_mask):
            continue

        xs = np.zeros((2, len(traj)))
        for t, p in enumerate(traj):
            if p[1] is not None:
                xs[0, t], xs[1, t] = p[1], p[2]

        xs_obs = xs[:, observed_mask]
        start_mean, start_cov = initial_state_4d(xs_obs)

        kf = KalmanFilter(A=A.copy(), B=B.copy(), Ry=Ry.copy(), Rx=Rx.copy(), startMean=start_mean, startCov=start_cov)

        sig, errs, apr_sig, apr_errs = filter_signal_with_missing(kf, xs, observed_mask)
        smoothed_sig, _ = kf.smoothSignal(sig, errs, apr_sig, apr_errs)

        for t, frame_id in enumerate(frame_ids):
            x_smooth = float(smoothed_sig[0, t])
            y_smooth = float(smoothed_sig[1, t])
            if frame_id not in frame_to_smoothed:
                frame_to_smoothed[frame_id] = {}
            frame_to_smoothed[frame_id][player_id] = (x_smooth, y_smooth)

    for frame_id, players in detections.items():
        if frame_id not in frame_to_smoothed:
            continue
        smoothed = frame_to_smoothed[frame_id]
        for player in players:
            if player.player_id in smoothed:
                player.court_position = smoothed[player.player_id]
