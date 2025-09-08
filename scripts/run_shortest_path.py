from __future__ import annotations

"""Run a simple shortest-path routing baseline.

This script builds a small satellite network, routes a set of random flows
using weighted shortest paths and reports basic metrics such as latency,
packet loss rate and throughput. It bypasses any learning components and
serves as a sanity check for the simulation infrastructure.
"""

import argparse
import random
from typing import List

import networkx as nx

from env.topology import ConstellationConfig, TopologyBuilder
from utils.traffic import Flow, update_loss_and_queue, aggregate_metrics


def graph_to_nx(G) -> nx.DiGraph:
    """Convert internal Graph to a networkx.DiGraph."""
    H = nx.DiGraph()
    for u in range(G.num_nodes):
        H.add_node(u)
    for (u, v) in G.E_physical:
        cap = G.cap[(u, v)]
        tprop = G.tprop[(u, v)]
        dist_km = G.dist[(u, v)] / 1000.0
        attr = {
            "cap_bps": cap,
            "tprop_s": tprop,
            "dist_km": dist_km,
            "phi_pkts": 0.0,
        }
        H.add_edge(u, v, **attr)
        H.add_edge(v, u, **attr)
    return H


def random_flows(G: nx.DiGraph, num_flows: int, rate_bps: float) -> List[Flow]:
    nodes = list(G.nodes)
    flows: List[Flow] = []
    for fid in range(num_flows):
        src, dst = random.sample(nodes, 2)
        flows.append(Flow(id=fid, src=src, dst=dst, rate_bps=rate_bps, path_edges=[]))
    return flows


def compute_weights(G: nx.DiGraph, beta: float, dt_s: float) -> None:
    """Compute link weights according to the load-aware formula."""
    for u, v, data in G.edges(data=True):
        C = data.get("cap_bps", 0.0)
        R = data.get("R_tot_bps", 0.0)
        L = min(R / max(C, 1e-9), 0.999999)
        tau_prop = data.get("tprop_s", 0.0)
        tau_tran = min(1.0, L) * dt_s
        gamma_bar = tau_prop + tau_tran
        w = gamma_bar * (1.0 + beta * L / (1.0 - L + 1e-6))
        data["weight_w"] = w


def route_flows(G: nx.DiGraph, flows: List[Flow]) -> None:
    for f in flows:
        try:
            nodes = nx.shortest_path(G, f.src, f.dst, weight="weight_w")
            f.path_edges = list(zip(nodes[:-1], nodes[1:]))
        except nx.NetworkXNoPath:
            f.path_edges = []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    cfg = ConstellationConfig(
        altitude_km=550.0,
        inclination_deg=53.0,
        num_sats=172,
        step_seconds=60,
        num_steps=args.steps,
        epoch_iso="2025-08-08T04:00:00",
    )
    builder = TopologyBuilder(cfg)
    G0 = builder.build_G_t(0)
    G = graph_to_nx(G0)
    flows = random_flows(G, num_flows=4, rate_bps=1.6e6)

    dt_s = float(cfg.step_seconds)
    for step in range(args.steps):
        compute_weights(G, args.beta, dt_s)
        route_flows(G, flows)
        results = update_loss_and_queue(
            G,
            flows,
            dt_s=dt_s,
            S_bytes=1500,
            K_pkts=640,
            Nmax=3,
        )
        metrics = aggregate_metrics(flows, results)
        print(f"step {step}: {metrics}")


if __name__ == "__main__":
    main()
