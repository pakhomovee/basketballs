"""Kalman smoothing for player court coordinates."""

from typing import Optional

import numpy as np

from common.classes.player import PlayersDetections
from kalmanlib import KalmanFilter

from .filter import filter_signal_with_missing
from .models import A_4STATE, B_4STATE
from .trajectory import extract_player_trajectories_with_gaps, initial_state_4d


def _build_kalman_chunks(
    traj: list[tuple[int, float | None, float | None]],
    max_gap: int,
) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
    """Split a player trajectory into independent Kalman segments.

    Returns:
        List of (frame_ids, xs, observed_mask) per segment.
        xs: (2, T) positions (zeros where unobserved).
        observed_mask: (T,) bool.
    """
    if not traj:
        return []

    segments: list[list[tuple[int, float | None, float | None]]] = []
    cur: list[tuple[int, float | None, float | None]] = [traj[0]]
    for i in range(1, len(traj)):
        if traj[i][0] - traj[i - 1][0] > max_gap:
            segments.append(cur)
            cur = [traj[i]]
        else:
            cur.append(traj[i])
    segments.append(cur)

    chunks = []
    for seg in segments:
        f_start, f_end = seg[0][0], seg[-1][0]
        frame_pos: dict[int, tuple[float, float] | None] = {f: ((x, y) if x is not None else None) for f, x, y in seg}
        frame_ids = list(range(f_start, f_end + 1))
        T = len(frame_ids)
        observed_mask = np.zeros(T, dtype=bool)
        xs = np.zeros((2, T))
        for i, f in enumerate(frame_ids):
            pos = frame_pos.get(f)
            if pos is not None:
                observed_mask[i] = True
                xs[0, i], xs[1, i] = pos
        chunks.append((frame_ids, xs, observed_mask))

    return chunks


def _clean_outliers(
    traj: list[tuple[int, float | None, float | None]],
    player_id: int,
    detections: PlayersDetections,
    *,
    max_y2_dev: float = 30.0,
    window: int = 15,
) -> list[tuple[int, float | None, float | None]]:
    """Null out court_position for frames where the player's bbox bottom is cut off.

    Args:
        traj: Per-player (frame_id, x_or_None, y_or_None) list.
        player_id: Used to look up bboxes in detections.
        detections: Full PlayersDetections for bbox access.
        max_y2_dev: Maximum allowed deviation (px) of bbox y2 from the local
            rolling median before the frame is marked as cut.
        window: Half-width (in frames) of the rolling window.

    Returns:
        Copy of traj with (frame_id, None, None) for cut frames.
    """
    y2_by_frame: dict[int, float] = {}
    for frame_id, players in detections.items():
        for player in players:
            if player.player_id == player_id and len(player.bbox) == 4:
                y2_by_frame[frame_id] = float(player.bbox[3])

    if not y2_by_frame:
        return traj

    sorted_frames = sorted(y2_by_frame)
    y2_arr = np.array([y2_by_frame[f] for f in sorted_frames])
    n = len(sorted_frames)

    local_median: dict[int, float] = {}
    for idx, frame_id in enumerate(sorted_frames):
        lo = max(0, idx - window)
        hi = min(n, idx + window + 1)
        local_median[frame_id] = float(np.median(y2_arr[lo:hi]))

    cut_frames: set[int] = {f for f, y2 in y2_by_frame.items() if abs(y2 - local_median.get(f, y2)) > max_y2_dev}

    return [(f, None, None) if f in cut_frames else (f, x, y) for f, x, y in traj]


def smooth_detection_coordinates(
    detections: PlayersDetections,
    *,
    cfg=None,
    A: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Rx: np.ndarray | None = None,
    Ry: np.ndarray | None = None,
    min_points: int = 4,
) -> None:
    """Apply Kalman RTS smoothing in-place, split by presence segments.

    Args:
        detections: PlayersDetections to smooth (modified in-place).
        cfg: AppConfig (uses cfg.smoother params when provided).
        frame_width: Unused; kept for call-site compatibility.
        A, B, Rx, Ry: Kalman matrices; derived from cfg/defaults when None.
    """
    if A is None:
        A = A_4STATE.copy()
    if B is None:
        B = B_4STATE.copy()
    if Rx is None:
        obs = cfg.smoother.obs_noise if cfg is not None else 0.1
        Rx = obs * np.eye(2)
    if Ry is None:
        pnp = cfg.smoother.process_noise_pos if cfg is not None else 0.01
        pnv = cfg.smoother.process_noise_vel if cfg is not None else 0.1
        Ry = np.diag([pnp, pnp, pnv, pnv])

    max_gap = cfg.smoother.max_gap_frames if cfg is not None else 45

    trajectories_full = extract_player_trajectories_with_gaps(detections)
    frame_to_smoothed: dict[int, dict[int, tuple[float, float]]] = {}

    for player_id, traj in trajectories_full.items():
        if not traj:
            continue
        traj = _clean_outliers(traj, player_id, detections)
        chunks = _build_kalman_chunks(traj, max_gap=max_gap)
        for frame_ids, xs, observed_mask in chunks:
            n_observed = int(observed_mask.sum())
            if n_observed == 0:
                continue
            if n_observed < min_points:
                for i, frame_id in enumerate(frame_ids):
                    if observed_mask[i]:
                        frame_to_smoothed.setdefault(frame_id, {})[player_id] = (float(xs[0, i]), float(xs[1, i]))
                continue

            xs_obs = xs[:, observed_mask]
            start_mean, start_cov = initial_state_4d(xs_obs)
            kf = KalmanFilter(
                A=A.copy(),
                B=B.copy(),
                Ry=Ry.copy(),
                Rx=Rx.copy(),
                startMean=start_mean,
                startCov=start_cov,
            )
            sig, errs, apr_sig, apr_errs = filter_signal_with_missing(kf, xs, observed_mask)
            smoothed_sig, _ = kf.smoothSignal(sig, errs, apr_sig, apr_errs)

            for i, frame_id in enumerate(frame_ids):
                frame_to_smoothed.setdefault(frame_id, {})[player_id] = (
                    float(smoothed_sig[0, i]),
                    float(smoothed_sig[1, i]),
                )

    for frame_id, players in detections.items():
        if frame_id not in frame_to_smoothed:
            continue
        smoothed = frame_to_smoothed[frame_id]
        for player in players:
            if player.player_id in smoothed:
                player.court_position = smoothed[player.player_id]
