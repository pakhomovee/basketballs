from __future__ import annotations

from typing import Sequence

from common.classes.pass_event import PassEvent
from common.classes.player import PlayersDetections
from common.classes.possession_segment import PossessionSegment


def _team_for_player(
    players_detections: PlayersDetections,
    player_id: int,
    frame_id: int,
    *,
    max_frame_offset: int = 5,
) -> int | None:
    """Resolve `team_id` for `player_id`, searching `frame_id` ± offset if missing."""
    search_order = [frame_id]
    for d in range(1, max_frame_offset + 1):
        search_order.extend([frame_id - d, frame_id + d])
    for fid in search_order:
        if fid < 0:
            continue
        for p in players_detections.get(fid, []):
            if p.player_id == player_id and p.team_id is not None:
                return p.team_id
    return None


def find_team_passes(
    possession_segments: Sequence[PossessionSegment],
    players_detections: PlayersDetections,
    *,
    max_gap_frames: int = 45,
) -> list[PassEvent]:
    """
    Detect passes: possession moves from one player to a teammate within a short time gap.

    Uses consecutive `PossessionSegment` entries (sorted by `start_frame`). A pass is recorded
    when the owner changes, both players belong to the same non-null `team_id`, and the
    gap between segments (end of first to start of second) is at most `max_gap_frames`.

    Args:
        possession_segments: Ordered possession intervals (e.g. from greedy segmentation).
        players_detections: Per-frame player list (must include `team_id` after team clustering).
        max_gap_frames: Maximum idle frames allowed between the two possession segments.

    Returns:
        List of `PassEvent`; `frame` is the start frame of the receiving segment.
    """
    segs = sorted(possession_segments, key=lambda s: (s.start_frame, s.end_frame))
    out: list[PassEvent] = []

    for i in range(len(segs) - 1):
        a, b = segs[i], segs[i + 1]
        if a.owner_player_id == b.owner_player_id:
            continue

        gap = b.start_frame - a.end_frame - 1
        if gap < 0 or gap > max_gap_frames:
            continue

        team_from = _team_for_player(players_detections, a.owner_player_id, a.end_frame)
        team_to = _team_for_player(players_detections, b.owner_player_id, b.start_frame)
        if team_from is None or team_to is None or team_from != team_to:
            continue

        out.append(
            PassEvent(
                frame=b.start_frame,
                from_frame=a.end_frame,
                from_player_id=a.owner_player_id,
                to_player_id=b.owner_player_id,
                team_id=team_from,
            )
        )

    return out
