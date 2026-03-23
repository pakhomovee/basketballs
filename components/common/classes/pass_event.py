from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PassEvent:
    """A teammate-to-teammate possession handoff over a short gap in time."""

    frame: int
    """Frame index where the receiving player's possession segment starts (pass moment)."""

    from_player_id: int
    to_player_id: int
    team_id: int
