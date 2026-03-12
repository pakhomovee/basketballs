"""
Bbox and embedding distance utilities for detection, tracking, and merging.

Shared by detector (NMS), tracker (matching), and merge (cost computation).
"""

from __future__ import annotations

import numpy as np


def bbox_bottom_mid(bbox) -> tuple[float, float]:
    """Bottom-center of bbox [x1, y1, x2, y2] (feet location)."""
    a = np.asarray(bbox, dtype=float).ravel()
    if len(a) < 4:
        return (0.0, 0.0)
    return ((a[0] + a[2]) / 2, a[3])


def bbox_bottom_mid_distance(a, b) -> float:
    """Euclidean distance between bottom-centers of two bboxes (pixels)."""
    pa = bbox_bottom_mid(a)
    pb = bbox_bottom_mid(b)
    return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))


def court_position_distance(pos_a: tuple[float, float], pos_b: tuple[float, float]) -> float:
    """Euclidean distance between two court positions (meters)."""
    return float(np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]))


def bbox_iou(a, b) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    a, b = np.asarray(a, dtype=float).ravel()[:4], np.asarray(b, dtype=float).ravel()[:4]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_overlap_ratio(a, b) -> float:
    """Intersection / min_area. 1.0 when one bbox is fully inside the other."""
    a, b = np.asarray(a, dtype=float).ravel()[:4], np.asarray(b, dtype=float).ravel()[:4]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0 else 0.0


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]; returns 1.0 for degenerate inputs."""
    if a is None or b is None:
        return 1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def gallery_distance(gallery: list[np.ndarray], query: np.ndarray) -> float:
    """Min cosine distance between *query* and any vector in *gallery*."""
    if not gallery or query is None:
        return 1.0
    return min(cosine_dist(g, query) for g in gallery)


def bbox_size_ratio(a, b) -> float:
    """Ratio of bbox areas (smaller/larger), in [0, 1]. 1 = same size, 0 = degenerate."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) < 4 or len(b) < 4:
        return 1.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)
