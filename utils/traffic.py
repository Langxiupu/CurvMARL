from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, TYPE_CHECKING

import random
import math

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    import networkx as nx


__all__ = [
    "Flow",
    "update_loss_and_queue",
    "aggregate_metrics",
    "GroundStation",
    "GROUND_STATIONS",
    "GroundStationTraffic",
    "GroundStationPoissonTraffic",
]


@dataclass(frozen=True)
class GroundStation:
    """Simple ground station description."""

    id: int
    name: str
    lat_deg: float
    lon_deg: float


# fmt: off
GROUND_STATIONS: List[GroundStation] = [
    GroundStation(0, "GS0", 43.0, 7.0),     # Southern France / Italy border
    GroundStation(1, "GS1", 45.0, 12.0),    # Northern Italy
    GroundStation(2, "GS2", 52.0, 4.0),     # UK / Belgium region
    GroundStation(3, "GS3", -34.0, 151.0),  # Australia (Sydney/Melbourne)
    GroundStation(4, "GS4", -37.0, 175.0),  # New Zealand North Island
    GroundStation(5, "GS5", 46.0, -123.0),  # US Pacific Northwest
    GroundStation(6, "GS6", 20.0, -156.0),  # Hawaii
    GroundStation(7, "GS7", 34.0, -118.0),  # US Southwest / Los Angeles
    GroundStation(8, "GS8", 29.0, -98.0),   # Gulf of Mexico region
    GroundStation(9, "GS9", 28.0, -82.0),   # US Southeast / Florida
    GroundStation(10, "GS10", 41.0, -88.0), # US Midwest / Great Lakes
    GroundStation(11, "GS11", 38.0, -77.0), # US East Coast / Virginia
]
# fmt: on


@dataclass
class _FlowState:
    """Internal state for an active ground-station flow."""

    id: int
    dst: int
    remaining_bits: float


class GroundStationTraffic:
    """Generate persistent Pareto-sized flows between ground stations.

    Each ground station maintains a single active flow.  When a flow finishes
    transmitting, a new destination is chosen uniformly from the remaining
    stations and a new flow size is sampled from a Pareto distribution.
    """

    def __init__(
        self,
        rate_bps: float,
        pareto_shape: float,
        pareto_scale_bytes: float,
        stations: Sequence[GroundStation] = GROUND_STATIONS,
        seed: Optional[int] = None,
    ) -> None:
        self.rate_bps = rate_bps
        self.pareto_shape = pareto_shape
        self.pareto_scale_bytes = pareto_scale_bytes
        self.stations = list(stations)
        self.rng = random.Random(seed)
        self._next_id = 0
        self._active: Dict[int, _FlowState] = {}

    # ------------------------------------------------------------------
    def _sample_size_bits(self) -> float:
        scale = self.pareto_scale_bytes * 8.0
        return scale * self.rng.paretovariate(self.pareto_shape)

    def _pick_dst(self, src_id: int) -> int:
        choices = [gs.id for gs in self.stations if gs.id != src_id]
        return self.rng.choice(choices)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all state, forcing new flow generation on next step."""

        self._active.clear()

    # ------------------------------------------------------------------
    def step(self, dt_s: float) -> List[Flow]:
        """Advance time and return current active flows.

        Parameters
        ----------
        dt_s: float
            Duration of the simulated step in seconds.
        """

        flows: List[Flow] = []
        bits_per_step = self.rate_bps * dt_s
        for gs in self.stations:
            st = self._active.get(gs.id)
            if st is None:
                dst = self._pick_dst(gs.id)
                st = _FlowState(id=self._next_id, dst=dst, remaining_bits=self._sample_size_bits())
                self._next_id += 1
                self._active[gs.id] = st
            else:
                st.remaining_bits -= bits_per_step
                if st.remaining_bits <= 0:
                    dst = self._pick_dst(gs.id)
                    st.id = self._next_id
                    st.dst = dst
                    st.remaining_bits = self._sample_size_bits()
                    self._next_id += 1

            flows.append(Flow(id=st.id, src=gs.id, dst=st.dst, rate_bps=self.rate_bps, path_edges=[]))
        return flows


class GroundStationPoissonTraffic:
    """Generate Poisson arrivals of Pareto-sized flows at ground stations."""

    def __init__(
        self,
        rate_bps: float,
        pareto_shape: float,
        pareto_scale_bytes: float = 512 * 1024,
        mean_flows_per_min: float = 5.0,
        stations: Sequence[GroundStation] = GROUND_STATIONS,
        seed: Optional[int] = None,
    ) -> None:
        self.rate_bps = rate_bps
        self.pareto_shape = pareto_shape
        self.pareto_scale_bytes = pareto_scale_bytes
        self.mean_flows_per_min = mean_flows_per_min
        self.stations = list(stations)
        self.rng = random.Random(seed)
        self._next_id = 0
        self._active: Dict[int, List[_FlowState]] = {gs.id: [] for gs in self.stations}

    # ------------------------------------------------------------------
    def _sample_size_bits(self) -> float:
        scale = self.pareto_scale_bytes * 8.0
        return scale * self.rng.paretovariate(self.pareto_shape)

    def _pick_dst(self, src_id: int) -> int:
        choices = [gs.id for gs in self.stations if gs.id != src_id]
        return self.rng.choice(choices)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        for lst in self._active.values():
            lst.clear()
        self._next_id = 0

    # ------------------------------------------------------------------
    def step(self, dt_s: float) -> List[Flow]:
        flows: List[Flow] = []
        bits_per_step = self.rate_bps * dt_s
        lam = self.mean_flows_per_min * (dt_s / 60.0)
        for gs in self.stations:
            active_list = self._active[gs.id]

            # Generate new flows for this step
            num_new = self._poisson(lam)
            for _ in range(num_new):
                dst = self._pick_dst(gs.id)
                st = _FlowState(id=self._next_id, dst=dst, remaining_bits=self._sample_size_bits())
                self._next_id += 1
                active_list.append(st)

            # Emit active flows and update remaining sizes
            for st in active_list[:]:
                flows.append(Flow(id=st.id, src=gs.id, dst=st.dst, rate_bps=self.rate_bps, path_edges=[]))
                st.remaining_bits -= bits_per_step
                if st.remaining_bits <= 0:
                    active_list.remove(st)
        return flows

    # ------------------------------------------------------------------
    def _poisson(self, lam: float) -> int:
        # Knuth's algorithm
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1


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
    G: 'nx.DiGraph',
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
        ``latency_s``, ``q`` (end-to-end success probability),
        ``packet_loss_rate`` and ``avg_packet_delay_s`` which
        estimates the average transmission time for a single packet as
        the flow completion time divided by the number of packets.
    """

    import networkx as nx

    # Reset per-edge step stats and flow counters
    for u, v, data in G.edges(data=True):
        data.setdefault("phi_pkts", 0.0)
        data["R_tot_bps"] = 0.0
        data["R_eff_bps"] = 0.0
        data["flow_count"] = 0

    # STEP 1: count flows per edge and offered load
    for f in flows:
        r = f.rate_bps
        for (u, v) in f.path_edges:
            edge = G[u][v]
            edge["R_tot_bps"] += r
            edge["flow_count"] += 1

    # STEP 2: compute equal bandwidth share per edge and per-link loss rate
    for u, v, data in G.edges(data=True):
        C = data.get("cap_bps", 0.0)
        n = data.get("flow_count", 0)
        data["share_bps"] = C / n if n > 0 else C
        R_tot = data.get("R_tot_bps", 0.0)
        # Single-link loss probability p_k = max(0, 1 - C_k / R_tot_k)
        data["p_loss"] = max(0.0, 1.0 - C / R_tot) if R_tot > 0 else 0.0

    # STEP 3: propagate per-flow rates using the per-edge share
    S_bits = 8 * S_bytes
    results: List[Dict[str, float]] = []
    for f in flows:
        r_in = f.rate_bps
        e2e_latency = 0.0
        q_f = 1.0
        for (u, v) in f.path_edges:
            data = G[u][v]
            share = data.get("share_bps", 0.0)
            r_out = min(r_in, share)
            f.edge_in_bps[(u, v)] = r_in
            f.edge_out_bps[(u, v)] = r_out
            p_k = data.get("p_loss", 0.0)
            q_f *= (1.0 - p_k)
            # One-hop latency including retransmissions: Omega * 1/(1-p_k)
            hop_base = data.get("tprop_s", 0.0) + (S_bits / r_out if r_out > 0 else 0.0)
            hop_delay = hop_base / max(1.0 - p_k, 1e-9)
            f.hop_latency_s[(u, v)] = hop_delay
            e2e_latency += hop_delay
            data["R_eff_bps"] += r_out
            r_in = r_out

        # Success probability with up to Nmax transmission attempts
        success_prob = 1.0 - (1.0 - q_f) ** max(Nmax, 1)
        goodput = r_in * success_prob if f.path_edges else 0.0
        pkt_delay = S_bits / goodput if goodput > 0 else 0.0
        plr_f = 1.0 - success_prob
        results.append(
            {
                "flow_id": f.id,
                "goodput_bps": goodput,
                "latency_s": e2e_latency,
                "q": success_prob,
                "success_prob": success_prob,
                "packet_loss_rate": plr_f,
                "avg_packet_delay_s": pkt_delay,
            }
        )

    # Update edge utilization stats
    eps = 1e-9
    for u, v, data in G.edges(data=True):
        C = data.get("cap_bps", 0.0)
        R = data.get("R_eff_bps", 0.0)
        R_tot = data.get("R_tot_bps", 0.0)
        data["p_loss"] = data.get("p_loss", 0.0)
        data["rho_eff"] = R / max(C, eps)
        data["A_pkts"] = 0.0
        data["S_pkts"] = 0.0
        data["D_pkts"] = 0.0

    return results


def aggregate_metrics(flows: Iterable[Flow], results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate system-level metrics from per-flow results.

    Returns a dictionary containing the average packet loss rate, average
    end-to-end delivery time, system throughput (both in bps and Mbps) and the
    average number of hops traversed by successful flows.
    """
    total_G = sum(r["goodput_bps"] for r in results)
    plrs = [r.get("packet_loss_rate", 0.0) for r in results]
    avg_plr = sum(plrs) / len(plrs) if plrs else 0.0
    latencies = [r["latency_s"] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    hop_counts = [len(f.path_edges) for f in flows if f.path_edges]
    avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0.0
    return {
        # return percentage value
        "packet_loss_rate": avg_plr * 100.0,
        "avg_delivery_time_s": avg_latency,
        "system_throughput_Mbps": total_G / 1e6,
        "system_throughput_bps": total_G,
        "avg_hop_count": avg_hops,
    }
