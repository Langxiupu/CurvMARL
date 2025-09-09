"""State representation utilities for satellite routing RL."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import networkx as nx


def _directional_neighbors(G: nx.DiGraph, node: int) -> Dict[str, Optional[int]]:
    """Return mapping of directions to neighbour node ids.

    Directions follow the logical grid layout of Walker constellations:
    ``left``/``right`` correspond to neighbours in the same orbital plane
    whereas ``up``/``down`` refer to adjacent planes.  If a neighbour in a
    given direction does not exist, the entry is ``None``.
    """

    plane = G.nodes[node].get("plane", 0)
    slot = G.nodes[node].get("slot", 0)
    S = G.graph.get("slots_per_plane", 1)
    P = G.graph.get("planes", 1)

    dirs: Dict[str, Optional[int]] = {"up": None, "down": None, "left": None, "right": None}
    for nbr in G.neighbors(node):
        p = G.nodes[nbr].get("plane", 0)
        s = G.nodes[nbr].get("slot", 0)
        if p == plane:
            if (s - slot) % S == 1:
                dirs["right"] = nbr
            elif (slot - s) % S == 1:
                dirs["left"] = nbr
        else:
            if (p - plane) % P == 1:
                dirs["up"] = nbr
            elif (plane - p) % P == 1:
                dirs["down"] = nbr
    return dirs


def build_state(
    G: nx.DiGraph,
    sat_id: int,
    dst_id: int,
    hidden: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Construct the reinforcement learning state vector for ``sat_id``.

    Parameters
    ----------
    G:
        Physical graph as produced by :class:`MultiSatEnv`.
    sat_id:
        Identifier of the current satellite.
    dst_id:
        Identifier of the destination satellite for the packet/flow.
    hidden:
        Mapping from node id to hidden state tensor output by the GNN.

    Returns
    -------
    torch.Tensor
        A 1-D tensor comprising the concatenated features described in the
        project discussion: satellite metadata, directional queue lengths,
        neighbouring hidden states, own hidden state and destination info.
    """

    plane = G.nodes[sat_id].get("plane", 0)
    slot = G.nodes[sat_id].get("slot", 0)
    x, y, z = G.nodes[sat_id].get("pos", (0.0, 0.0, 0.0))

    dirs = _directional_neighbors(G, sat_id)
    queues = []
    h_neigh = []
    h_self = hidden.get(sat_id, torch.zeros_like(next(iter(hidden.values()))))
    for d in ("up", "down", "left", "right"):
        nbr = dirs[d]
        if nbr is not None and G.has_edge(sat_id, nbr):
            queues.append(float(G[sat_id][nbr].get("phi_pkts", 0.0)))
            h_neigh.append(hidden.get(nbr, torch.zeros_like(h_self)))
        else:
            queues.append(0.0)
            h_neigh.append(torch.zeros_like(h_self))

    d_plane = G.nodes[dst_id].get("plane", 0)
    d_slot = G.nodes[dst_id].get("slot", 0)
    dx, dy, dz = G.nodes[dst_id].get("pos", (0.0, 0.0, 0.0))

    feat_list = [
        float(plane),
        float(slot),
        float(x),
        float(y),
        float(z),
        *queues,
        float(d_plane),
        float(d_slot),
        float(dx),
        float(dy),
        float(dz),
    ]

    feat_tensor = torch.tensor(feat_list, dtype=torch.float32)
    neigh_stack = torch.stack(h_neigh + [h_self], 0).view(-1)
    return torch.cat([feat_tensor, neigh_stack], dim=0)

__all__ = ["build_state"]

