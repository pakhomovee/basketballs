from common.classes.ball import Ball
from math import exp
import numpy as np


def remove_bad_ball_detections(
    ball_detections: dict[int, list[Ball]],
    frame_size: tuple[int, int],
    max_skip: int = 50,
    remove_cost=0.5,
    confedence_coeff=0.01,
    eps=0.03,
    homographies=None,
    out_of_court_penalty=0.02,
    teleportation_cost=3.0,
    conf_threshold=0.3,
) -> dict[int, Ball]:
    inf = float("inf")
    ball_detections_list = [detection for detection in ball_detections.items() if len(detection[1]) > 0]
    ball_detections_list = [
        [frame_id, [ball for ball in balls if ball.confidence >= conf_threshold]]
        for frame_id, balls in ball_detections_list
    ]
    ball_detections_list = [detection for detection in ball_detections_list if len(detection[1]) > 0]
    ball_detections_list.sort(key=lambda x: x[0])
    n = len(ball_detections_list)
    if n == 0:
        return {}

    def _project_norm_to_court(x_norm: float, y_norm: float, H) -> tuple[float, float] | None:
        """Project (x_norm,y_norm,1) using homography H: [0,1]^2 -> [-0.5,0.5]^2."""
        if H is None:
            return None
        # homography is expected to be a numpy 3x3
        import numpy as _np

        p = H @ _np.array([x_norm, y_norm, 1.0], dtype=_np.float64)
        if abs(float(p[2])) < 1e-12:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    dist_to_court = [[0.0 for _ in range(len(ball_detections_list[i][1]))] for i in range(n)]
    if homographies is not None:
        for i in range(n):
            frame_id = ball_detections_list[i][0]
            if frame_id < 0 or frame_id >= len(homographies):
                continue
            H = homographies[frame_id]
            for j, ball in enumerate(ball_detections_list[i][1]):
                cx = (ball.bbox[0] + ball.bbox[2]) / 2
                cy = (ball.bbox[1] + ball.bbox[3]) / 2
                x_norm = cx / frame_size[0]
                y_norm = cy / frame_size[1]
                proj = _project_norm_to_court(x_norm, y_norm, H)
                if proj is None:
                    continue
                x_c, y_c = proj
                dist_to_court[i][j] = max(0.0, x_c - 0.5, -0.5 - x_c, y_c - 0.5, -0.5 - y_c)
            # print("Frame ", frame_id)
            # print(f"Dist to court: {dist_to_court[i]}")

    dp = [[(inf, -1, -1) for j in range(len(ball_detections_list[i][1]))] for i in range(n)]
    dp = [[(0, -1, -1)]] + dp + [[(inf, -1, -1)]]

    def distance(ball1: Ball, ball2: Ball) -> float:
        x1, y1 = (ball1.bbox[0] + ball1.bbox[2]) / 2, (ball1.bbox[1] + ball1.bbox[3]) / 2
        x2, y2 = (ball2.bbox[0] + ball2.bbox[2]) / 2, (ball2.bbox[1] + ball2.bbox[3]) / 2
        x1, x2 = x1 / frame_size[0], x2 / frame_size[0]
        y1, y2 = y1 / frame_size[1], y2 / frame_size[1]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def cost(i: int, j: int, i_prev: int, j_prev: int) -> float:
        cost = dp[i_prev][j_prev][0] + (abs(i - i_prev) - 1) * remove_cost
        i -= 1
        i_prev -= 1
        if 0 <= i < n:
            cost -= confedence_coeff * ball_detections_list[i][1][j].confidence
            cost += out_of_court_penalty * dist_to_court[i][j]
        if i < 0 or i >= n or i_prev < 0 or i_prev >= n:
            return cost
        cost += distance(ball_detections_list[i_prev][1][j_prev], ball_detections_list[i][1][j]) * teleportation_cost
        return cost

    for i in range(1, n + 2):
        for j in range(len(dp[i])):
            for i_prev in range(max(0, i - max_skip), i):
                for j_prev in range(len(dp[i_prev])):
                    dp[i][j] = min(dp[i][j], (cost(i, j, i_prev, j_prev), i_prev, j_prev))
    new_ball_detections = {}
    i = n + 1
    j = 0
    while i >= 1:
        if i <= n:
            ball = ball_detections_list[i - 1][1][j]
            pos = ball_detections_list[i - 1][0]
            new_ball_detections[pos] = ball
        i, j = dp[i][j][1], dp[i][j][2]
    print("Removed bad ball detections: ", len(ball_detections_list) - len(new_ball_detections))
    return new_ball_detections
