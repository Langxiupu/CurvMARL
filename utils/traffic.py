from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable

import networkx as nx


__all__ = ["Flow", "update_loss_and_queue", "aggregate_metrics"]


@dataclass
class Flow:
    """Minimal flow representation used by ``update_loss_and_queue``."""

    id: int
    src: int
    dst: int
    rate_bps: float
    path_edges: List[Tuple[int, int]]
    edge_in_bps: Dict[Tuple[int, int], float] = field(default_factory=dict)
    edge_out_bps: Dict[Tuple[int, int], float] = field(default_factory=dict)
    hop_latency_s: Dict[Tuple[int, int], float] = field(default_factory=dict)
    q_list: List[float] = field(default_factory=list)


def update_loss_and_queue(
    G: nx.DiGraph,
    flows: Iterable[Flow],
    dt_s: float,
    S_bytes: int,
    K_pkts: int,
    Nmax: int,
) -> List[Dict[str, float]]:
    """Update per-edge and per-flow metrics.

    Parameters
    ----------
    G: networkx.DiGraph
        Directed graph whose edges carry attributes ``cap_bps`` and
        ``tprop_s``.  The function populates several step-specific attributes
        such as ``R_tot_bps``, ``p_loss`` and queue statistics.
    flows: Iterable[Flow]
        Active flows with predefined paths and rates.
    dt_s: float
        Simulation step length in seconds.
    S_bytes: int
        Packet size in bytes.
    K_pkts: int
        Queue capacity per edge, in packets.
    Nmax: int
        Maximum number of retransmissions for goodput estimation.

    Returns
    -------
    list of dict
        For each flow, a dictionary with ``flow_id``, ``goodput_bps``,
        ``latency_s`` and ``q`` (end-to-end success probability).
    """

    # Reset per-edge step stats
    for u, v, data in G.edges(data=True):
        data.setdefault("phi_pkts", 0.0)
        data["R_tot_bps"] = 0.0

    # STEP 1: accumulate arrivals
    for f in flows:
        r = f.rate_bps
        for (u, v) in f.path_edges:
            G[u][v]["R_tot_bps"] += r
            f.edge_in_bps[(u, v)] = r

    # STEP 2: congestion-based loss probability
    eps = 1e-9
    for u, v, data in G.edges(data=True):
        C = data.get("cap_bps", 0.0)
        R = data["R_tot_bps"]
        data["rho"] = R / max(C, eps)
        data["p_loss"] = max(0.0, 1.0 - C / max(R, eps)) if R > 0 else 0.0

    # STEP 3: per-flow edge_out and q_list
    for f in flows:
        f.q_list = []
        for (u, v) in f.path_edges:
            p = G[u][v]["p_loss"]
            r_in = f.edge_in_bps[(u, v)]
            r_out = r_in * (1.0 - p)
            f.edge_out_bps[(u, v)] = r_out
            f.q_list.append(1.0 - p)

    # STEP 4: queue update
    S_bits = 8 * S_bytes
    for u, v, data in G.edges(data=True):
        C = data.get("cap_bps", 0.0)
        R = data["R_tot_bps"]
        phi = float(data.get("phi_pkts", 0.0))

        A_pkts = (R * dt_s) / S_bits
        mu_pkts = (C * dt_s) / S_bits

        pre = phi + A_pkts
        served = min(mu_pkts, pre)
        post = pre - served
        overflow = max(0.0, post - K_pkts)
        phi_next = min(K_pkts, post)

        data["A_pkts"] = A_pkts
        data["S_pkts"] = served
        data["D_pkts"] = overflow
        data["phi_pkts"] = phi_next

    # STEP 5: per-flow latency and goodput
    results: List[Dict[str, float]] = []
    for f in flows:
        q_f = 1.0
        e2e_latency = 0.0
        for (u, v) in f.path_edges:
            d = G[u][v]
            p = d["p_loss"]
            tau_prop = d.get("tprop_s", 0.0)
            rho = d.get("rho", 0.0)
            tau_tran = dt_s * min(1.0, rho)
            phi = d.get("phi_pkts", 0.0)
            C = d.get("cap_bps", 0.0)
            tau_q = (phi * S_bits) / max(C, eps)

            Omega = tau_prop + tau_tran + tau_q
            Gamma = Omega / max(1.0 - p, 1e-6)

            f.hop_latency_s[(u, v)] = Gamma
            e2e_latency += Gamma
            q_f *= (1.0 - p)

        G_f = f.rate_bps * (1.0 - (1.0 - q_f) ** (Nmax + 1))
        results.append(
            {
                "flow_id": f.id,
                "goodput_bps": G_f,
                "latency_s": e2e_latency,
                "q": q_f,
            }
        )

    return results


def aggregate_metrics(flows: Iterable[Flow], results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate system-level metrics from per-flow results."""
    total_R = sum(f.rate_bps for f in flows)
    total_G = sum(r["goodput_bps"] for r in results)
    plr = (total_R - total_G) / total_R if total_R > 0 else 0.0
    latencies = [r["latency_s"] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "packet_loss_rate": plr,
        "avg_delivery_time_s": avg_latency,
        "system_throughput_bps": total_G,
    }
