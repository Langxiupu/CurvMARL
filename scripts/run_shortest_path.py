from __future__ import annotations

"""Run a simple shortest-path routing baseline.

This script builds a small satellite network, routes a set of random flows
using weighted shortest paths and reports basic metrics such as latency,
packet loss rate and throughput. It bypasses any learning components and
serves as a sanity check for the simulation infrastructure.
"""

import argparse
import math
from typing import Dict, List

import networkx as nx

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.topology import ConstellationConfig, TopologyBuilder
from utils.traffic import (
    Flow,
    update_loss_and_queue,
    aggregate_metrics,
    GroundStationPoissonTraffic,
    GROUND_STATIONS,
)


# Total ground-to-satellite transmission delay for a packet travelling
# from one ground station to another (two hops).
GROUND_GROUND_DELAY_S = 0.040


def link_capacity_jsac(dist_km: float, bandwidth_mhz: float) -> float:
    """Return link capacity following the JSAC'24 model.

    C(d) = 0.5 * B * log2(1 + 5.85e5 * exp(-3.12e-5 * d))
    where ``d`` is the inter-satellite distance in km and ``B`` is the
    bandwidth in MHz.
    """

    return 0.5 * bandwidth_mhz * 1e6 * math.log2(
        1.0 + 5.85e5 * math.exp(-3.12e-5 * dist_km)
    )


def graph_to_nx(G, bandwidth_mhz: float) -> nx.DiGraph:
    """Convert internal Graph to a networkx.DiGraph with capacities."""

    H = nx.DiGraph()
    for u in range(G.num_nodes):
        H.add_node(u)
    for (u, v) in G.E_physical:
        dist_km = G.dist[(u, v)] / 1000.0
        cap = link_capacity_jsac(dist_km, bandwidth_mhz)
        tprop = G.tprop[(u, v)]
        attr = {
            "cap_bps": cap,
            "tprop_s": tprop,
            "dist_km": dist_km,
            "phi_pkts": 0.0,
        }
        H.add_edge(u, v, **attr)
        H.add_edge(v, u, **attr)
    return H


def geodetic_to_ecef(lat_deg: float, lon_deg: float, Re_km: float = 6378.137):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x = Re_km * math.cos(lat) * math.cos(lon)
    y = Re_km * math.cos(lat) * math.sin(lon)
    z = Re_km * math.sin(lat)
    return x, y, z


def associate_ground_stations(pos: Dict[int, tuple]) -> Dict[int, int]:
    """Map each ground station to its nearest visible satellite."""
    mapping: Dict[int, int] = {}
    for gs in GROUND_STATIONS:
        gs_vec = geodetic_to_ecef(gs.lat_deg, gs.lon_deg)
        best, best_dist = None, 1e12
        for sid, svec in pos.items():
            rel = (svec[0] - gs_vec[0], svec[1] - gs_vec[1], svec[2] - gs_vec[2])
            dot_val = gs_vec[0] * rel[0] + gs_vec[1] * rel[1] + gs_vec[2] * rel[2]
            if dot_val <= 0:
                continue
            dist = math.sqrt(rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2)
            if dist < best_dist:
                best, best_dist = sid, dist
        if best is not None:
            mapping[gs.id] = best
    return mapping


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
    # Generate random Poisson arrivals of Pareto-sized flows at each ground station.
    traffic = GroundStationPoissonTraffic(
        rate_bps=1.6e6,
        pareto_shape=1.5,
        pareto_scale_bytes=640 * 1024,
        mean_flows_per_min=60.0,
    )

    dt_s = float(cfg.step_seconds)
    throughputs: List[float] = []
    plrs: List[float] = []
    hop_counts: List[float] = []
    flow_delays_ms: Dict[int, float] = {}
    for step in range(args.steps):
        G_t = builder.build_G_t(step)
        H = graph_to_nx(G_t, cfg.bandwidth_mhz)
        gs_map = associate_ground_stations(G_t.positions)
        gs_flows = traffic.step(dt_s)
        flows: List[Flow] = []
        for f in gs_flows:
            src_sat = gs_map.get(f.src)
            dst_sat = gs_map.get(f.dst)
            if src_sat is None or dst_sat is None:
                continue
            flows.append(Flow(id=f.id, src=src_sat, dst=dst_sat, rate_bps=f.rate_bps, path_edges=[]))


        compute_weights(H, args.beta, dt_s)
        route_flows(H, flows)
        results = update_loss_and_queue(
            H,
            flows,
            dt_s=dt_s,
            S_bytes=1500,
            K_pkts=640,
            Nmax=3,
        )
        metrics = aggregate_metrics(flows, results)
        metrics.pop("avg_delivery_time_s", None)
        throughputs.append(metrics["system_throughput_Mbps"])
        plrs.append(metrics["packet_loss_rate"])
        hop_counts.append(metrics["avg_hop_count"])
        for r in results:
            fid = r.get("flow_id")
            if fid is not None and fid not in flow_delays_ms:
                delay_ms = (r.get("avg_packet_delay_s", 0.0) + GROUND_GROUND_DELAY_S) * 1000
                flow_delays_ms[fid] = delay_ms
        print(f"step {step}: {metrics}")

    avg_plr = sum(plrs) / len(plrs) if plrs else 0.0
    avg_thr = sum(throughputs) / len(throughputs) if throughputs else 0.0
    avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0.0
    avg_pkt_delay_ms = sum(flow_delays_ms.values()) / len(flow_delays_ms) if flow_delays_ms else 0.0
    print(f"Average packet loss rate over {args.steps} steps: {avg_plr:.2f}%")
    print(f"Average system throughput over {args.steps} steps: {avg_thr:.3f} Mbps")
    print(f"Average hop count over {args.steps} steps: {avg_hops:.2f}")
    print(
        f"Average packet transmission delay over {args.steps} steps: {avg_pkt_delay_ms:.3f} ms",
    )


if __name__ == "__main__":
    main()
