"""Graph rewiring utilities based on Ollivier-Ricci curvature.

This module implements the :class:`CurvRewirer` class which augments a
physical network topology with additional *virtual* edges.  The virtual
edges are selected according to the negative curvature of physical edges and
only serve as auxiliary connections for the GNN/MARL agent – they are never
used for actual packet forwarding.

The implementation follows the specification provided in the project
blueprint and mirrors the pseudo code shipped with this challenge.  The
core ideas are:

* Compute weighted, directed OR curvature for all physical edges.
* Select a budgeted subset of the most negative edges.
* From the corresponding optimal transport plan pick candidate end points
  and add virtual edges between them.

The resulting graph contains both the original physical edges (``virt_flag``
set to ``0``) and the new virtual edges (``virt_flag`` set to ``1``).  For
undirected satellite topologies the virtual edges are inserted in both
directions to mimic a symmetric shortcut.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

try:  # POT – Python Optimal Transport
    import ot
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("CurvRewirer requires the 'pot' package") from exc

__all__ = ["CurvRewirer"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _emd2(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    """Exact Wasserstein-1 distance using POT's emd2."""

    return float(ot.emd2(a, b, C))


def _emd_plan(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return optimal transport plan for distributions ``a`` and ``b``."""

    return ot.emd(a, b, C)


def _sinkhorn2(a: np.ndarray, b: np.ndarray, C: np.ndarray, reg: float = 1e-1) -> float:
    """Sinkhorn approximation of the Wasserstein-1 distance."""

    return float(ot.sinkhorn2(a, b, C, reg))


def topk_pairs_from_plan(plan: np.ndarray, P: List[int], Q: List[int], k: int) -> List[Tuple[int, int]]:
    """Return ``k`` pairs with largest mass in the transport ``plan``."""

    if k <= 0:
        return []

    flat = plan.reshape(-1)
    idx = np.argsort(flat)[::-1]  # largest first
    pairs: List[Tuple[int, int]] = []
    nP, nQ = plan.shape
    for id_ in idx:
        if flat[id_] <= 0:
            break
        i = id_ // nQ
        j = id_ % nQ
        pairs.append((P[i], Q[j]))
        if len(pairs) >= k:
            break
    return pairs


def pick_by_budget(sorted_edges: List[Tuple[float, int, int, Tuple]], *, mode: str, budget: float) -> List[Tuple]:
    """Select edges according to the budget strategy.

    ``sorted_edges`` must be sorted by curvature in ascending order.
    """

    if not sorted_edges:
        return []

    if mode == "fraction":
        K = max(0, int(math.ceil(len(sorted_edges) * float(budget))))
    elif mode == "count":
        K = max(0, int(budget))
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown budget_mode: {mode}")

    K = min(K, len(sorted_edges))
    return sorted_edges[:K]


def approx_line_or_geo_dist(x: int, y: int, G: nx.DiGraph) -> float:
    """Approximate physical distance between ``x`` and ``y``.

    We use the existing ``dist_km`` edge attribute if available.  If no path
    exists the distance defaults to ``0``.
    """

    try:
        return float(nx.shortest_path_length(G, x, y, weight="dist_km"))
    except nx.NetworkXNoPath:
        return 0.0


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def get_mu_out(u: int, G: nx.DiGraph, w_len: Dict[Tuple[int, int], float], cache: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    """Return normalised outgoing neighbour distribution for ``u``."""

    if u in cache:
        return cache[u]

    nbrs = list(G.successors(u))
    if not nbrs:
        cache[u] = {u: 1.0}
        return cache[u]

    ws = np.array([w_len[(u, v)] for v in nbrs], dtype=float)
    s = float(ws.sum())
    if s <= 0:
        mu = {u: 1.0}
    else:
        mu = {v: float(ws[i] / s) for i, v in enumerate(nbrs)}
    cache[u] = mu
    return mu


def get_mu_in(v: int, G: nx.DiGraph, w_len: Dict[Tuple[int, int], float], cache: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    """Return normalised incoming neighbour distribution for ``v``."""

    if v in cache:
        return cache[v]

    nbrs = list(G.predecessors(v))
    if not nbrs:
        cache[v] = {v: 1.0}
        return cache[v]

    ws = np.array([w_len[(u, v)] for u in nbrs], dtype=float)
    s = float(ws.sum())
    if s <= 0:
        mu = {v: 1.0}
    else:
        mu = {u: float(ws[i] / s) for i, u in enumerate(nbrs)}
    cache[v] = mu
    return mu


# ---------------------------------------------------------------------------
# Shortest path helper
# ---------------------------------------------------------------------------

class ShortestPathHelper:
    """Convenience wrapper for shortest path computations with caching."""

    def __init__(self, G: nx.DiGraph, w_len: Dict[Tuple[int, int], float], cache: bool = True):
        self.G = G
        self.w_len = w_len
        self.cache = cache
        self._cache: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    def _weight(self, u: int, v: int, data: Dict[str, float]) -> float:
        return float(self.w_len[(u, v)])

    # ------------------------------------------------------------------
    def distance(self, s: int, t: int) -> float:
        if self.cache and (s, t) in self._cache:
            return self._cache[(s, t)]
        try:
            d = float(nx.shortest_path_length(self.G, s, t, weight=self._weight))
        except nx.NetworkXNoPath:
            d = float("inf")
        if self.cache:
            self._cache[(s, t)] = d
        return d

    # ------------------------------------------------------------------
    def cost_matrix(self, P: List[int], Q: List[int]) -> np.ndarray:
        return np.array([[self.distance(p, q) for q in Q] for p in P], dtype=float)


# ---------------------------------------------------------------------------
# Main rewiring class
# ---------------------------------------------------------------------------


class CurvRewirer:
    """Compute curvature based virtual edges and augment the topology."""

    def __init__(
        self,
        beta: float = 0.5,
        alpha: float = 0.3,
        *,
        budget_mode: str = "fraction",
        budget: float = 0.25,
        topk_per_edge: int = 2,
        max_virtual_degree: int = 8,
        use_sinkhorn_when: int = 40,
        cache_shortest: bool = True,
    ):
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.budget_mode = budget_mode
        self.budget = budget
        self.topk_per_edge = int(topk_per_edge)
        self.max_virtual_degree = int(max_virtual_degree)
        self.use_sinkhorn_when = int(use_sinkhorn_when)
        self.cache_shortest = bool(cache_shortest)
        # simulation step duration used for simple transmission delay estimate
        self._dt = 1.0

    # ------------------------------------------------------------------
    def build_logic_topology(self, G_phys: nx.DiGraph) -> nx.DiGraph:
        """Return logical topology ``G_phys`` augmented with virtual edges."""

        # 1) edge weights and lengths -------------------------------------------------
        w_len: Dict[Tuple[int, int], float] = {}
        for u, v, d in G_phys.edges(data=True):
            if d.get("virt_flag", 0) == 1:  # ignore existing virtual edges
                continue
            C = float(d.get("cap_bps", 0.0))
            R = max(float(d.get("R_tot_bps", 0.0)), 0.0)
            L = min(R / max(C, 1e-9), 0.999999)
            tau_prop = float(d.get("tprop_s", 0.0))
            tau_tran = min(1.0, L) * self._dt
            gamma_bar = tau_prop + tau_tran
            w = gamma_bar * (1.0 + self.beta * L / (1.0 - L + 1e-6))
            w_len[(u, v)] = max(w, 1e-9)

        sp = ShortestPathHelper(G_phys, w_len, cache=self.cache_shortest)

        # 3) curvature for physical edges -------------------------------------------
        edge_scores: List[Tuple[float, int, int, Tuple]] = []
        mu_cache_out: Dict[int, Dict[int, float]] = {}
        mu_cache_in: Dict[int, Dict[int, float]] = {}
        for u, v in G_phys.edges():
            if G_phys[u][v].get("virt_flag", 0) == 1:
                continue
            mu_u = get_mu_out(u, G_phys, w_len, mu_cache_out)
            mu_v = get_mu_in(v, G_phys, w_len, mu_cache_in)
            P = list(mu_u.keys())
            Q = list(mu_v.keys())
            Cmat = sp.cost_matrix(P, Q)
            x = np.array([mu_u[p] for p in P], dtype=float)
            y = np.array([mu_v[q] for q in Q], dtype=float)
            if len(P) * len(Q) > self.use_sinkhorn_when:
                W = _sinkhorn2(x, y, Cmat)
            else:
                W = _emd2(x, y, Cmat)
            d_uv = sp.distance(u, v)
            if not math.isfinite(d_uv) or d_uv <= 0:
                kappa = 0.0
            else:
                kappa = 1.0 - W / d_uv
            edge_scores.append((kappa, u, v, (P, Q, Cmat, x, y)))

        # 4) pick most negative edges ------------------------------------------------
        neg_sorted = sorted(edge_scores, key=lambda t: t[0])
        chosen = pick_by_budget(neg_sorted, mode=self.budget_mode, budget=self.budget)

        # 5) add virtual edges -------------------------------------------------------
        G_logic = G_phys.copy()
        virtual_degree: Dict[int, int] = defaultdict(int)
        for _, u, v, (P, Q, Cmat, x, y) in chosen:
            plan = _emd_plan(x, y, Cmat)
            cand_pairs = topk_pairs_from_plan(plan, P, Q, k=self.topk_per_edge)
            for xnode, ynode in cand_pairs:
                # ensure the pair is not already connected in either direction
                if G_logic.has_edge(xnode, ynode) or G_logic.has_edge(ynode, xnode):
                    continue
                if (
                    virtual_degree[xnode] >= self.max_virtual_degree
                    or virtual_degree[ynode] >= self.max_virtual_degree
                ):
                    continue
                d_xy = sp.distance(xnode, ynode)
                if not math.isfinite(d_xy):
                    continue
                wv = max(self.alpha * d_xy, 1e-9)
                tprop = nx.shortest_path_length(
                    G_phys, xnode, ynode, weight="tprop_s"
                )
                dist = approx_line_or_geo_dist(xnode, ynode, G_phys)
                attr = {
                    "virt_flag": 1,
                    "cap_bps": 0.0,
                    "R_tot_bps": 0.0,
                    "p_loss": 0.0,
                    "tprop_s": float(tprop),
                    "dist_km": float(dist),
                    "w_virtual": wv,
                }
                # add symmetric virtual edges to mimic undirected connection
                G_logic.add_edge(xnode, ynode, **attr)
                G_logic.add_edge(ynode, xnode, **attr)
                virtual_degree[xnode] += 1
                virtual_degree[ynode] += 1

        return G_logic
