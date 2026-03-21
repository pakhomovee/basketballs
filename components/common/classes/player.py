from __future__ import annotations

from dataclasses import dataclass, field
from common.classes.number import Number
from common.classes.skeleton import Skeleton

import numpy as np


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

    player_id: int = -1
    bbox: list[int] = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float | None = None

    number: Number | None = None
    skeleton: Skeleton | None = None
    has_ball: bool = False
    has_ball_raw: bool = False
    has_ball_post: bool = False
    possession_conf_raw: float = 0.0
    possession_conf_post: float = 0.0
    is_dribble: bool = False

    # Enriched by embedding extraction (mask-based color histograms, for team clustering)
    embedding: np.ndarray | None = None

    # Enriched by ReID model (learned identity features, for tracker appearance matching)
    reid_embedding: np.ndarray | None = None

    # Enriched by tracker
    speed: float | None = None

    # Enriched by team clustering
    team_id: int | None = None

    # Enriched by court detector
    court_position: tuple[float, float] | None = None


# frame_id -> list of Player detections for that frame
PlayersDetections = dict[int, list[Player]]
