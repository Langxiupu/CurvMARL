"""Graph attention network utilities.

This module implements a small, self contained GAT backbone with support
for edge features together with helper classes used throughout the
project.  The code is intentionally lightweight and does not depend on
external graph libraries such as PyG; instead it operates on simple
NetworkX-like graphs.

The main components are:

* :class:`ObservationBuilder` – extracts ``L``-hop ego networks and
  produces normalised node/edge feature tensors.
* :class:`GATLayer` and :class:`GATBackbone` – multi–head attention
  message passing with edge features.
* :class:`PolicyHead` – computes per–neighbour logits for action
  selection.

The implementation follows the design sketched in the project blueprint
and is written using only PyTorch primitives so it can easily be reused
or extended in future experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

__all__ = [
    "ObservationBuilder",
    "GATLayer",
    "GATBackbone",
    "PolicyHead",
    "scatter_softmax",
    "gat_infer_actions",
]


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------


class ObservationBuilder:
    """Construct ``L``-hop sub-graphs and feature tensors.

    The class assumes a NetworkX-like graph ``G`` where each edge stores a
    dictionary of attributes.  Only a small subset of attributes is
    accessed – see the code below for the exact names.  Features are
    standardised on a per-batch basis which keeps the implementation
    self-contained.
    """

    def __init__(self, L: int = 3, use_dst: bool = False):
        self.L = L
        self.use_dst = use_dst

    # ------------------------------------------------------------------
    def node_features(self, G, nodes: Iterable[int]) -> torch.Tensor:
        """Return a ``(N,d_x)`` tensor of node features.

        Parameters
        ----------
        G:
            Graph-like object with ``neighbors`` method and edge
            attributes accessible via ``G[u][v]``.
        nodes:
            Iterable of node identifiers.
        """

        feats: List[List[float]] = []
        for i in nodes:
            nbrs = list(G.neighbors(i))  # logical neighbours
            deg_phys = sum(1 for j in nbrs if G[i][j].get("virt_flag", 0) == 0)
            deg_logic = len(nbrs)
            qin = sum(G[i][j].get("R_tot_bps", 0.0) for j in nbrs)
            cap = sum(G[i][j].get("cap_bps", 0.0) for j in nbrs)
            phi = sum(G[i][j].get("phi_pkts", 0.0) for j in nbrs)
            tprop = (
                float(np.mean([G[i][j].get("tprop_s", 0.0) for j in nbrs]))
                if nbrs
                else 0.0
            )
            feats.append(
                [
                    deg_phys,
                    deg_logic,
                    qin / 1e9,
                    cap / 1e9,
                    phi / 640.0,
                    tprop * 1000.0,
                ]
            )

        X = torch.tensor(feats, dtype=torch.float32)
        # simple standardisation
        X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
        return X

    # ------------------------------------------------------------------
    def edge_arrays(self, G, nodes: Iterable[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return edge index and edge attribute arrays for the subgraph."""

        idx_map = {n: i for i, n in enumerate(nodes)}
        src: List[int] = []
        dst: List[int] = []
        attrs: List[List[float]] = []
        for i in nodes:
            for j in G.neighbors(i):
                if j not in idx_map:
                    continue
                e = G[i][j]
                edge_feat = [
                    e.get("cap_bps", 0.0) / 1e9,
                    e.get("R_tot_bps", 0.0)
                    / max(e.get("cap_bps", 1e-6), 1e-6),
                    e.get("p_loss", 0.0),
                    e.get("tprop_s", 0.0) * 1000.0,
                    1.0 if e.get("virt_flag", 0) == 1 else 0.0,
                    e.get("dist_km", 0.0) / 10000.0,
                ]
                src.append(idx_map[i])
                dst.append(idx_map[j])
                attrs.append(edge_feat)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        if attrs:
            edge_attr = torch.tensor(attrs, dtype=torch.float32)
            edge_attr = (edge_attr - edge_attr.mean(0)) / (edge_attr.std(0) + 1e-6)
        else:
            edge_attr = torch.zeros((0, 6), dtype=torch.float32)
        return edge_index, edge_attr

    # ------------------------------------------------------------------
    def ego_nodes(self, G, center: int) -> List[int]:
        """Return a list of nodes in the ``L``-hop ego network of ``center``."""

        frontier = {center}
        visited = {center}
        for _ in range(self.L):
            nxt = set()
            for u in frontier:
                nxt.update(G.neighbors(u))
            frontier = {v for v in nxt if v not in visited}
            visited.update(frontier)
        return list(visited)


# ---------------------------------------------------------------------------
# GAT layers
# ---------------------------------------------------------------------------


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, N: int) -> torch.Tensor:
    """Group-wise softmax implemented with :func:`index_add_`.

    Parameters
    ----------
    scores: ``(E,)`` tensor of raw scores.
    index: ``(E,)`` tensor mapping each score to a destination node.
    N: number of nodes.
    """

    exp = torch.exp(scores)
    denom = torch.zeros(N, device=scores.device)
    denom.index_add_(0, index, exp)
    return exp / (denom[index] + 1e-9)


class GATLayer(nn.Module):
    """A single multi-head graph attention layer with edge features."""

    def __init__(
        self,
        d_in: int,
        d_edge: int,
        d_out: int,
        heads: int = 4,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(d_in, d_out * heads, bias=False)
        self.V = nn.Linear(d_in, d_out * heads, bias=False)
        self.U = nn.Linear(d_edge, d_out * heads, bias=False)
        self.a = nn.Parameter(torch.randn(heads, 3 * d_out))
        self.leaky = nn.LeakyReLU(negative_slope)

    # ------------------------------------------------------------------
    def forward(
        self, X: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        N = X.size(0)
        H = self.heads
        d = self.a.size(-1) // 3

        Wh = self.W(X).view(N, H, d)
        Vh = self.V(X).view(N, H, d)
        Uh = self.U(edge_attr).view(-1, H, d)

        src, dst = edge_index
        e_i = Wh[src]
        e_j = Vh[dst]
        e_ij = Uh
        cat = torch.cat([e_i, e_j, e_ij], dim=-1)
        logits = self.leaky((cat * self.a).sum(-1))  # (E,H)

        alpha = torch.zeros_like(logits)
        for h in range(H):
            alpha[:, h] = scatter_softmax(logits[:, h], dst, N)

        msg = e_j * alpha.unsqueeze(-1)
        out = torch.zeros(N, H, d, device=X.device)
        out.index_add_(0, src, msg)
        return out.reshape(N, H * d)


class GATBackbone(nn.Module):
    """Stacked GAT layers with ELU activations."""

    def __init__(
        self,
        d_x: int,
        d_e: int,
        d_h: int = 32,
        heads: int = 4,
        L: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
        [
            GATLayer(d_x if l == 0 else d_h * heads, d_e, d_h, heads)
            for l in range(L)
        ]
        )
        self.act = nn.ELU()

    # ------------------------------------------------------------------
    def forward(
        self, X: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = X
        for layer in self.layers:
            h = self.act(layer(h, edge_index, edge_attr))
        return h


# ---------------------------------------------------------------------------
# Action head
# ---------------------------------------------------------------------------


class PolicyHead(nn.Module):
    """Simple MLP producing logits for each available neighbour."""

    def __init__(self, d_in: int, d_e: int, d_mid: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_in + d_e, d_mid),
            nn.ReLU(),
            nn.Linear(d_mid, 1),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        emb: torch.Tensor,
        center_idx: List[int],
        act_neighbors: List[List[int]],
        edge_feats: Dict[Tuple[int, int], torch.Tensor],
    ) -> List[torch.Tensor]:
        logits_all: List[torch.Tensor] = []
        for i, nbrs in zip(center_idx, act_neighbors):
            if len(nbrs) == 0:
                logits_all.append(torch.empty(0, device=emb.device))
                continue
            cat_vecs = []
            for j in nbrs:
                e_ij = edge_feats[(i, j)]
                cat_vecs.append(torch.cat([emb[i], emb[j], e_ij], dim=-1))
            x = torch.stack(cat_vecs, 0)
            logits = self.mlp(x).squeeze(-1)
            logits_all.append(logits)
        return logits_all


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def gat_infer_actions(
    G_logic, G_phys, centers: Iterable[int], backbone: GATBackbone, policy_head: PolicyHead, L: int = 3
) -> Dict[int, Optional[int]]:
    """Run a forward pass of the GAT and choose an action per center node.

    The helper mirrors the pseudo code from the blueprint and is primarily
    intended for evaluation without learning.  It expects ``G_logic`` to
    contain both physical and virtual edges whereas ``G_phys`` only
    contains physical links.
    """

    ob = ObservationBuilder(L=L)
    center_to_sub: List[Tuple[int, List[int]]] = []
    all_nodes: List[int] = []
    for c in centers:
        nodes = ob.ego_nodes(G_logic, c)
        center_to_sub.append((c, nodes))
        all_nodes.extend(nodes)
    # deduplicate while keeping order
    uniq_nodes: List[int] = list(dict.fromkeys(all_nodes))
    X = ob.node_features(G_logic, uniq_nodes)
    edge_index, edge_attr = ob.edge_arrays(G_logic, uniq_nodes)
    emb = backbone(X, edge_index, edge_attr)

    act_neighbors: List[List[int]] = []
    edge_feats: Dict[Tuple[int, int], torch.Tensor] = {}
    idx_map = {n: i for i, n in enumerate(uniq_nodes)}
    for c, _ in center_to_sub:
        nbrs = [j for j in G_phys.neighbors(c)]
        act_neighbors.append([idx_map[j] for j in nbrs if j in idx_map])
        for j in nbrs:
            if j in idx_map:
                e = G_phys[c][j]
                ef = torch.tensor(
                    [
                        e.get("cap_bps", 0.0) / 1e9,
                        e.get("R_tot_bps", 0.0)
                        / max(e.get("cap_bps", 1e-6), 1e-6),
                        e.get("p_loss", 0.0),
                        e.get("tprop_s", 0.0) * 1000.0,
                        0.0,
                        e.get("dist_km", 0.0) / 10000.0,
                    ],
                    dtype=torch.float32,
                )
                edge_feats[(idx_map[c], idx_map[j])] = ef

    center_idx = [idx_map[c] for c, _ in center_to_sub]
    logits_list = policy_head(emb, center_idx, act_neighbors, edge_feats)

    actions: Dict[int, Optional[int]] = {}
    for (c, _), logits in zip(center_to_sub, logits_list):
        if logits.numel() == 0:
            actions[c] = None
        else:
            probs = torch.softmax(logits, dim=0)
            a_idx = torch.argmax(probs).item()
            actions[c] = list(G_phys.neighbors(c))[a_idx]
    return actions
