"""Topology generation for Walker satellite constellations.

This module implements a simplified version of the design discussed in the
project blueprint.  It is self-contained and relies only on the Python
standard library.  The goal is to generate, for each simulation step, the
physical inter-satellite link (ISL) graph together with basic link metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Set, Optional, NamedTuple

from utils.angles import deg2rad, ring_angle_distance, eci_to_geodetic_lat_deg
from utils.time_utils import parse_iso_utc
from utils.graph_utils import Graph


class ConstellationConfig(NamedTuple):
    altitude_km: float
    inclination_deg: float
    num_sats: int
    step_seconds: int
    num_steps: int
    epoch_iso: str
    max_range_km: float = 6000.0
    polar_cutoff_lat_deg: float = 75.0
    hysteresis_seconds: int = 30
    same_plane_neighbors: int = 1
    adjacent_plane_delta: int = 1
    bandwidth_mhz: float = 25.0


class EdgeHysteresis:
    """Simple edge hysteresis controller.

    An edge changes state only after "window" consecutive disagreement steps.
    """

    def __init__(self, hysteresis_seconds: int, step_seconds: int) -> None:
        self.window = max(1, hysteresis_seconds // step_seconds)
        self.state: Dict[Tuple[str, Tuple[int, int]], Tuple[bool, int]] = {}

    def permit(self, typ: str, a: Tuple[int, int], b: Tuple[int, int], step: int, allow: bool) -> bool:
        key = (typ, tuple(sorted((a, b))))
        active, count = self.state.get(key, (allow, self.window))
        if allow == active:
            self.state[key] = (active, self.window)
            return active
        count -= 1
        if count <= 0:
            self.state[key] = (allow, self.window)
            return allow
        self.state[key] = (active, count)
        return active


class TopologyBuilder:
    def __init__(self, cfg: ConstellationConfig):
        self.cfg = cfg
        self._derive_walker_parameters()
        self._hysteresis = EdgeHysteresis(cfg.hysteresis_seconds, cfg.step_seconds)
        self.n_rad_s = math.sqrt(self.mu_km3_s2 / (self.a_km ** 3))

    # ------------------------------------------------------------------
    # Walker parameters
    def _derive_walker_parameters(self) -> None:
        N = self.cfg.num_sats
        P = self._nearest_factor_to_sqrt(N)
        S = int(round(N / P))
        if P * S < N:
            # Last plane will have fewer satellites
            pass
        F = self._choose_coprime_phase(S)

        self.P, self.S, self.F = P, S, F
        self.T = P * S

        Re = 6378.137
        self.a_km = Re + self.cfg.altitude_km
        self.e = 1e-4
        self.i_rad = deg2rad(self.cfg.inclination_deg)
        self.raan0 = 0.0
        self.omega = 0.0
        self.M0 = 0.0

        self.Omega = [self.raan0 + 2 * math.pi * p / P for p in range(P)]
        self.M_init = [
            [
                (self.M0 + 2 * math.pi * s / S + 2 * math.pi * F * p / (P * S)) % (2 * math.pi)
                for s in range(S)
            ]
            for p in range(P)
        ]

        # Gravitational parameter mu (km^3/s^2)
        self.mu_km3_s2 = 398600.4418

    @staticmethod
    def _nearest_factor_to_sqrt(N: int) -> int:
        root = int(round(math.sqrt(N)))
        for delta in range(0, root + 1):
            for candidate in (root - delta, root + delta):
                if candidate <= 0:
                    continue
                if N % candidate == 0:
                    return candidate
        return max(1, root)

    @staticmethod
    def _choose_coprime_phase(S: int) -> int:
        candidate = S // 2
        if math.gcd(candidate, S) == 1:
            return candidate
        for k in range(1, S):
            if math.gcd(k, S) == 1:
                return k
        return 1

    # ------------------------------------------------------------------
    def _sat_exists(self, p: int, s: int) -> bool:
        return (p * self.S + s) < self.cfg.num_sats

    def _id(self, ps: Tuple[int, int]) -> int:
        return ps[0] * self.S + ps[1]

    def _inv(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.S)

    # ------------------------------------------------------------------
    def build_G_t(self, step_idx: int) -> Graph:
        pos_eci = self._positions_at(step_idx)
        edges = self._grid4_edges(pos_eci, step_idx)
        G = Graph(self.cfg.num_sats)
        id_pos = {self._id(k): v for k, v in pos_eci.items()}
        G.load_positions(id_pos)
        G.add_edges_physical(edges)
        self._annotate_link_metrics(G, id_pos)
        return G

    # ------------------------------------------------------------------
    def _positions_at(self, step_idx: int) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
        tsec = step_idx * self.cfg.step_seconds
        pos: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        for p in range(self.P):
            for s in range(self.S):
                if not self._sat_exists(p, s):
                    continue
                M = (self.M_init[p][s] + self.n_rad_s * tsec) % (2 * math.pi)
                u = self.omega + M
                r = self.a_km
                cosu, sinu = math.cos(u), math.sin(u)
                cosO, sinO = math.cos(self.Omega[p]), math.sin(self.Omega[p])
                cosi, sini = math.cos(self.i_rad), math.sin(self.i_rad)
                x = r * (cosO * cosu - sinO * sinu * cosi)
                y = r * (sinO * cosu + cosO * sinu * cosi)
                z = r * (sinu * sini)
                pos[(p, s)] = (x, y, z)
        return pos

    # ------------------------------------------------------------------
    def _grid4_edges(
        self, pos_eci: Dict[Tuple[int, int], Tuple[float, float, float]], step_idx: int
    ) -> Set[Tuple[int, int]]:
        E: Set[Tuple[int, int]] = set()
        # same-plane neighbors
        for p in range(self.P):
            for s in range(self.S):
                if not self._sat_exists(p, s):
                    continue
                for ds in (-1, 1):
                    s2 = (s + ds) % self.S
                    if not self._sat_exists(p, s2):
                        continue
                    self._try_add_edge(E, (p, s), (p, s2), pos_eci, step_idx, False)
        # adjacent planes
        for p in range(self.P):
            for s in range(self.S):
                if not self._sat_exists(p, s):
                    continue
                for dp in (-1, 1):
                    q = (p + dp) % self.P
                    s_prime = self._nearest_slot_by_u((p, s), q, step_idx)
                    if s_prime is None:
                        continue
                    self._try_add_edge(E, (p, s), (q, s_prime), pos_eci, step_idx, True)
        return E

    def _nearest_slot_by_u(self, ps: Tuple[int, int], q: int, step_idx: int) -> Optional[int]:
        tsec = step_idx * self.cfg.step_seconds
        u_ps = (self.omega + (self.M_init[ps[0]][ps[1]] + self.n_rad_s * tsec)) % (2 * math.pi)
        best, best_gap = None, 1e9
        for s2 in range(self.S):
            if not self._sat_exists(q, s2):
                continue
            u_qs = (self.omega + (self.M_init[q][s2] + self.n_rad_s * tsec)) % (2 * math.pi)
            gap = ring_angle_distance(u_ps, u_qs)
            if gap < best_gap:
                best, best_gap = s2, gap
        return best

    def _try_add_edge(
        self,
        E: Set[Tuple[int, int]],
        a: Tuple[int, int],
        b: Tuple[int, int],
        pos: Dict[Tuple[int, int], Tuple[float, float, float]],
        step_idx: int,
        cross_plane: bool,
    ) -> None:
        ra = pos[a]
        rb = pos[b]
        d_km = self._distance_km(ra, rb)
        if d_km > self.cfg.max_range_km:
            return
        if cross_plane:
            lat_a = eci_to_geodetic_lat_deg(ra)
            lat_b = eci_to_geodetic_lat_deg(rb)
            if max(abs(lat_a), abs(lat_b)) >= self.cfg.polar_cutoff_lat_deg:
                if not self._hysteresis.permit('xplane', a, b, step_idx, allow=False):
                    return
            else:
                if not self._hysteresis.permit('xplane', a, b, step_idx, allow=True):
                    return
        E.add(tuple(sorted((self._id(a), self._id(b)))))

    @staticmethod
    def _distance_km(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    # ------------------------------------------------------------------
    def _annotate_link_metrics(self, G: Graph, pos: Dict[int, Tuple[float, float, float]]):
        B = self.cfg.bandwidth_mhz
        for (u, v) in G.E_physical:
            ru = pos[u]
            rv = pos[v]
            d_km = self._distance_km(ru, rv)
            dist_m = d_km * 1000.0
            G.dist[(u, v)] = dist_m
            G.tprop[(u, v)] = dist_m / 299792458.0
            C = 0.5 * B * 1e6 * math.log2(1.0 + 5.85e5 * math.exp(-3.12e-5 * d_km))
            G.cap[(u, v)] = max(C, 0.0)
