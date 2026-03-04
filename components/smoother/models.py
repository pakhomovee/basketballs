"""4-state constant-velocity Kalman model for position smoothing."""

import numpy as np

# State: [x, y, vx, vy]
# x(t+1)=x(t)+vx(t), y(t+1)=y(t)+vy(t), vx(t+1)=vx(t), vy(t+1)=vy(t)
A_4STATE = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)

# We observe position only: [x, y] from state [x, y, vx, vy]
B_4STATE = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ],
    dtype=float,
)
