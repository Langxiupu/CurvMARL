from typing import Dict, Tuple, Iterable, Set

__all__ = ['Graph']

class Graph:
    """A minimal undirected graph structure for simulation."""
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.positions: Dict[int, Tuple[float, float, float]] = {}
        self.E_physical: Set[Tuple[int, int]] = set()
        self.dist: Dict[Tuple[int, int], float] = {}
        self.tprop: Dict[Tuple[int, int], float] = {}
        self.cap: Dict[Tuple[int, int], float] = {}

    # Position handling
    def load_positions(self, pos: Dict[int, Tuple[float, float, float]]) -> None:
        self.positions = pos

    # Edge handling
    def add_edges_physical(self, edges: Iterable[Tuple[int, int]]) -> None:
        for u, v in edges:
            if u == v:
                continue
            self.E_physical.add(tuple(sorted((u, v))))
