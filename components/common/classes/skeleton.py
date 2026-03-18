from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Skeleton:
    """
    2D pose/skeleton for a person detection.

    keypoints: array of shape (K, 3) with (x, y, confidence) per keypoint in image pixels.
    """

    keypoints: np.ndarray | None = None
