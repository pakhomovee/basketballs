"""
Bipartite assignment via Hungarian algorithm with cost gating.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def run_hungarian(cost, gate):
    """
    Solve linear sum assignment and filter by cost gate.

    Returns
    -------
    matches : list of (row, col) pairs
    unmatched_rows : list of row indices
    unmatched_cols : list of col indices
    """
    nr, nc = cost.shape
    if nr == 0 or nc == 0:
        return [], list(range(nr)), list(range(nc))
    rows, cols = linear_sum_assignment(cost)
    matches = []
    um_r, um_c = set(range(nr)), set(range(nc))
    for r, c in zip(rows, cols):
        if cost[r, c] < gate:
            matches.append((r, c))
            um_r.discard(r)
            um_c.discard(c)
    return matches, sorted(um_r), sorted(um_c)
