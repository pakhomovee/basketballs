from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Referee:
    """
    Representation of a detected referee.
    Populated by detector.
    """

    bbox: list[int] = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float | None = None


# frame_id -> list of Referee detections for that frame
RefereesDetections = dict[int, list[Referee]]
