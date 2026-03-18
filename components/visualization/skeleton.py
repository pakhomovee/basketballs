from __future__ import annotations

import cv2
import numpy as np

from common.classes.skeleton import Skeleton


def draw_skeleton(
    frame: np.ndarray,
    skeleton: Skeleton | None,
    *,
    keypoint_color: tuple[int, int, int] = (0, 255, 0),
    limb_color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
    thickness: int = 2,
    conf_threshold: float = 0.1,
) -> None:
    """
    Draw keypoints and a simple skeleton on the frame (in-place).

    Expects skeleton.keypoints of shape (K, 3): (x, y, confidence) in image pixels.
    Uses a COCO-style subset of limbs (works for common 17-kp layouts).
    """
    if skeleton is None or skeleton.keypoints is None:
        return
    keypoints = skeleton.keypoints
    if keypoints.size == 0:
        return

    num_kp = keypoints.shape[0]

    # Joints
    for idx in range(num_kp):
        x, y, c = keypoints[idx]
        if c < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, keypoint_color, -1, lineType=cv2.LINE_AA)

    # Limbs
    limbs = [
        (5, 6),  # shoulders
        (5, 7),
        (7, 9),  # left arm
        (6, 8),
        (8, 10),  # right arm
        (11, 12),  # hips
        (11, 13),
        (13, 15),  # left leg
        (12, 14),
        (14, 16),  # right leg
    ]
    for i, j in limbs:
        if i >= num_kp or j >= num_kp:
            continue
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 < conf_threshold or c2 < conf_threshold:
            continue
        cv2.line(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            limb_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
