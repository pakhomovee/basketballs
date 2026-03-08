"""Min-cost max-flow solver — SPFA for initial potentials, then Dijkstra.

Handles negative edge costs via Johnson's potential reweighting:
1. First iteration: SPFA (Bellman-Ford variant with SLF) finds shortest
   paths even with negative edges, producing initial potentials.
2. Subsequent iterations: Dijkstra with reduced costs (guaranteed ≥ 0).

Complexity: O(V·E) for the SPFA pass, then O(F · (V+E) log V) for
F Dijkstra passes. F = flow value (typically 10 for basketball).
"""

from __future__ import annotations

import heapq
from collections import deque


class MinCostFlow:
    """Sparse min-cost max-flow on an explicit adjacency list."""

    def __init__(self, n: int):
        self.n = n
        # Each edge: [to, remaining_cap, cost, rev_idx]
        self.graph: list[list[list]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: float) -> None:
        """Add a directed edge u→v. Cost may be negative for forward edges."""
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

    def solve(self, s: int, t: int, max_flow: int) -> tuple[int, float]:
        """Find min-cost flow of value ≤ max_flow from s to t.

        Returns (flow_value, total_cost).
        """
        n = self.n
        INF = float("inf")
        h = [0.0] * n  # Johnson potentials
        total_flow = 0
        total_cost = 0.0
        first = True

        while total_flow < max_flow:
            if first:
                dist, prev_v, prev_e = self._spfa(s)
                first = False
            else:
                dist, prev_v, prev_e = self._dijkstra(s, h)

            if dist[t] >= INF:
                break

            for i in range(n):
                if dist[i] < INF:
                    h[i] += dist[i]

            # Bottleneck capacity along shortest path
            f = max_flow - total_flow
            v = t
            while v != s:
                f = min(f, self.graph[prev_v[v]][prev_e[v]][1])
                v = prev_v[v]

            # Augment flow and accumulate real cost
            path_cost = 0.0
            v = t
            while v != s:
                e = self.graph[prev_v[v]][prev_e[v]]
                path_cost += e[2]
                e[1] -= f
                self.graph[v][e[3]][1] += f
                v = prev_v[v]

            total_flow += f
            total_cost += f * path_cost

        return total_flow, total_cost

    # ------------------------------------------------------------------

    def _spfa(self, s: int) -> tuple[list[float], list[int], list[int]]:
        """SPFA with SLF (Small Label First). Handles negative edge costs."""
        INF = float("inf")
        eps = 1e-9
        n = self.n
        dist = [INF] * n
        in_queue = [False] * n
        prev_v = [-1] * n
        prev_e = [-1] * n
        dist[s] = 0.0
        q: deque[int] = deque([s])
        in_queue[s] = True

        while q:
            u = q.popleft()
            in_queue[u] = False
            for i, (v, cap, cost, _) in enumerate(self.graph[u]):
                if cap > 0:
                    nd = dist[u] + cost
                    if nd < dist[v] - eps:
                        dist[v] = nd
                        prev_v[v] = u
                        prev_e[v] = i
                        if not in_queue[v]:
                            if q and nd < dist[q[0]] - eps:
                                q.appendleft(v)
                            else:
                                q.append(v)
                            in_queue[v] = True

        return dist, prev_v, prev_e

    def _dijkstra(self, s: int, h: list[float]) -> tuple[list[float], list[int], list[int]]:
        """Dijkstra with Johnson's potentials (reduced costs ≥ 0)."""
        INF = float("inf")
        eps = 1e-9
        dist = [INF] * self.n
        prev_v = [-1] * self.n
        prev_e = [-1] * self.n
        dist[s] = 0.0
        pq = [(0.0, s)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + eps:
                continue
            for i, (v, cap, cost, _) in enumerate(self.graph[u]):
                if cap > 0:
                    nd = d + cost + h[u] - h[v]
                    if nd < dist[v] - eps:
                        dist[v] = nd
                        prev_v[v] = u
                        prev_e[v] = i
                        heapq.heappush(pq, (nd, v))

        return dist, prev_v, prev_e
