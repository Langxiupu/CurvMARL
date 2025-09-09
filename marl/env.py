import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import networkx as nx

from env.topology import ConstellationConfig, TopologyBuilder
from utils.traffic import Flow, update_loss_and_queue, aggregate_metrics


@dataclass
class EnvConfig:
    """Configuration parameters for :class:`MultiSatEnv`."""

    num_sats: int
    num_steps: int
    step_seconds: int = 60
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    max_range_km: float = 6000.0
    polar_cutoff_lat_deg: float = 75.0
    hysteresis_seconds: int = 30
    same_plane_neighbors: int = 1
    adjacent_plane_delta: int = 1
    bandwidth_mhz: float = 25.0

    num_flows: int = 4
    flow_rate_bps: float = 1.6e6
    packet_size_bytes: int = 1500
    queue_cap_pkts: int = 640
    retransmissions: int = 3
    w1: float = 0.5
    w2: float = 0.5
    max_hops: int = 32


class MultiSatEnv:
    """Simplified multi-satellite environment used for MAPPO training.

    The environment maintains a physical topology that evolves over time and
    a fixed set of random end-to-end flows.  Agents select next-hop neighbours
    for packets originating from their node.  The actual queuing dynamics and
    metrics are handled by :func:`update_loss_and_queue` from ``utils.traffic``.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.builder = TopologyBuilder(
            ConstellationConfig(
                altitude_km=cfg.altitude_km,
                inclination_deg=cfg.inclination_deg,
                num_sats=cfg.num_sats,
                step_seconds=cfg.step_seconds,
                num_steps=cfg.num_steps,
                epoch_iso="2020-01-01T00:00:00",
                max_range_km=cfg.max_range_km,
                polar_cutoff_lat_deg=cfg.polar_cutoff_lat_deg,
                hysteresis_seconds=cfg.hysteresis_seconds,
                same_plane_neighbors=cfg.same_plane_neighbors,
                adjacent_plane_delta=cfg.adjacent_plane_delta,
                bandwidth_mhz=cfg.bandwidth_mhz,
            )
        )
        self.step_idx: int = 0
        self.G_phys: nx.DiGraph = nx.DiGraph()
        self.flows: List[Flow] = []

    # ------------------------------------------------------------------
    def _graph_to_nx(self, G) -> nx.DiGraph:
        H = nx.DiGraph()
        # carry over basic node metadata for state construction
        for u in range(G.num_nodes):
            p, s = self.builder._inv(u)
            pos = G.positions.get(u, (0.0, 0.0, 0.0))
            H.add_node(u, plane=p, slot=s, pos=pos)

        H.graph["slots_per_plane"] = getattr(self.builder, "S", 1)
        H.graph["planes"] = getattr(self.builder, "P", 1)

        for (u, v) in G.E_physical:
            cap = G.cap[(u, v)]
            tprop = G.tprop[(u, v)]
            dist_km = G.dist[(u, v)] / 1000.0
            attr = {
                "cap_bps": cap,
                "tprop_s": tprop,
                "dist_km": dist_km,
                "virt_flag": 0,
                "phi_pkts": 0.0,
            }
            H.add_edge(u, v, **attr)
            H.add_edge(v, u, **attr)
        return H

    # ------------------------------------------------------------------
    def _random_flows(self) -> List[Flow]:
        flows: List[Flow] = []
        nodes = list(self.G_phys.nodes)
        for fid in range(self.cfg.num_flows):
            src, dst = random.sample(nodes, 2)
            flows.append(Flow(id=fid, src=src, dst=dst, rate_bps=self.cfg.flow_rate_bps, path_edges=[]))
        return flows

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.step_idx = 0
        G0 = self.builder.build_G_t(0)
        self.G_phys = self._graph_to_nx(G0)
        self.flows = self._random_flows()
        return self.G_phys

    # ------------------------------------------------------------------
    def _derive_paths(self, actions: Dict[int, Optional[int]]) -> Tuple[List[Flow], float]:
        """Update each flow's path according to ``actions``.

        Returns the fraction of flows that failed to reach their destination
        which serves as a loop penalty.
        """

        loops = 0
        for f in self.flows:
            path: List[Tuple[int, int]] = []
            cur = f.src
            visited = {cur}
            success = False
            for _ in range(self.cfg.max_hops):
                nxt = actions.get(cur)
                if nxt is None or not self.G_phys.has_edge(cur, nxt):
                    break
                path.append((cur, nxt))
                if nxt == f.dst:
                    success = True
                    break
                if nxt in visited:
                    break
                visited.add(nxt)
                cur = nxt
            if not success:
                loops += 1
            f.path_edges = path
        loop_penalty = loops / max(1, len(self.flows))
        return self.flows, loop_penalty

    # ------------------------------------------------------------------
    def step(self, action_dict: Dict[int, Optional[int]]):
        flows, loop_penalty = self._derive_paths(action_dict)
        # Update traffic metrics on current graph
        results = update_loss_and_queue(
            self.G_phys,
            flows,
            dt_s=float(self.cfg.step_seconds),
            S_bytes=self.cfg.packet_size_bytes,
            K_pkts=self.cfg.queue_cap_pkts,
            Nmax=self.cfg.retransmissions,
        )
        metrics = aggregate_metrics(flows, results)
        overflow = sum(self.G_phys[u][v].get("D_pkts", 0.0) for u, v in self.G_phys.edges)

        reward = (
            self.cfg.w1 * metrics["system_throughput_bps"] / 1e9
            - self.cfg.w2 * metrics["avg_delivery_time_s"]
            - 0.01 * overflow
            - 1.0 * loop_penalty
        )
        info = {
            "metrics": metrics,
            "overflow": overflow,
            "loop_penalty": loop_penalty,
        }

        self.step_idx += 1
        done = self.step_idx >= self.cfg.num_steps
        if not done:
            G_next = self.builder.build_G_t(self.step_idx)
            self.G_phys = self._graph_to_nx(G_next)
        return self.G_phys, reward, done, info
