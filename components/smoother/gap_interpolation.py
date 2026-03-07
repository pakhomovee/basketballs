"""
Gap interpolation for track positions.

Fills gaps in detection (detection → no detection for some frames → detection)
using linear interpolation, but only when the two bounding detections are close.
"""

from __future__ import annotations

from collections import defaultdict

from common.classes.player import Player, PlayersDetections


def interpolate_track_gaps(
    detections: PlayersDetections,
    *,
    max_distance: float = 4.0,
) -> None:
    """
    Add interpolated Player entries for gaps within tracks.

    For each track, finds consecutive detection pairs (f1, pos1) and (f2, pos2).
    If the gap f1+1..f2-1 is non-empty and ||pos2 - pos1|| <= max_distance,
    linearly interpolates positions for the missing frames and appends
    synthetic Player objects to detections.

    Modifies detections in place.
    """
    # Build per-player trajectory: (frame_id, court_position) for detections with valid pos
    trajectories: dict[int, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    for frame_id in sorted(detections.keys()):
        for player in detections[frame_id]:
            if player.player_id < 0:
                continue
            pos = player.court_position
            if pos is None:
                continue
            trajectories[player.player_id].append((frame_id, pos))

    for pid, traj in trajectories.items():
        if len(traj) < 2:
            continue
        traj.sort(key=lambda x: x[0])

        for i in range(len(traj) - 1):
            f1, pos1 = traj[i]
            f2, pos2 = traj[i + 1]
            gap_start, gap_end = f1 + 1, f2 - 1
            if gap_start > gap_end:
                continue

            x1, y1 = pos1
            x2, y2 = pos2
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if dist > max_distance:
                continue

            for f in range(gap_start, gap_end + 1):
                t = (f - f1) / (f2 - f1)
                x = (1 - t) * x1 + t * x2
                y = (1 - t) * y1 + t * y2
                synthetic = Player(
                    player_id=pid,
                    bbox=[],
                    court_position=(float(x), float(y)),
                )
                if f not in detections:
                    detections[f] = []
                detections[f].append(synthetic)
