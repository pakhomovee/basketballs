from dataclasses import dataclass, field


@dataclass
class Player:
    """
    Shared representation of a detected player, progressively enriched
    by pipeline stages.

    Populated by:
        - **Detector / Tracker**: ``player_id``, ``bbox``, ``speed``
        - **Team clustering**:   ``team_id``
        - **Court detector**:    ``court_position``
    """

    player_id: int
    bbox: list[int] = field(default_factory=list)  # [x1, y1, x2, y2]

    # Enriched by tracker
    speed: float | None = None

    # Enriched by team clustering
    team_id: int | None = None

    # Enriched by court detector
    court_position: tuple[float, float] | None = None


# frame_id -> list of Player detections for that frame
FrameDetections = dict[int, list[Player]]
