from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PassEvent:
    """A teammate-to-teammate possession handoff over a short gap in time."""

    frame_start: int
    """Frame index where the passing player's possession segment ends."""

    frame_end: int
    """Frame index where the receiving player's possession segment starts."""

    from_player_id: int
    to_player_id: int
    team_id: int
