from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShotEvent:
    frame_start: int
    frame_end: int
    """Start end end of the shot event."""

    is_make: bool
    """Is the shot a make (i. e. successful shot)"""

    make_start: int | None
    make_end: int | None
    """Start and end of the make event."""
