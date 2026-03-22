from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PossessionSegment:
    """A contiguous frame range where one player is considered the ball owner."""

    start_frame: int
    end_frame: int
    owner_player_id: int
