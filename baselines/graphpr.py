from __future__ import annotations

from typing import Dict, Hashable, List
import heapq

Graph = Dict[Hashable, Dict[Hashable, float]]

def graph_pr(graph: Graph, src: Hashable, dst: Hashable) -> List[Hashable]:
    """Compute the shortest path between two nodes using Dijkstra's algorithm.

    The graph is represented as a nested mapping where ``graph[u][v]`` is the
    weight of the edge from ``u`` to ``v``.  The function returns the sequence of
    nodes on the minimum-cost path from ``src`` to ``dst``.
    """
    pq = [(0.0, src, [])]
    visited = set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == dst:
            return path
        for nbr, w in graph.get(node, {}).items():
            if nbr not in visited:
                heapq.heappush(pq, (cost + w, nbr, path))
    raise ValueError(f"No path from {src} to {dst}")
