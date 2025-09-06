"""Generate satellite topology graphs using Skyfield and NetworkX.

The script propagates satellites defined by TLEs and builds a graph for each
minute of the simulation. Every satellite is connected to four neighbours:

* Up and down: the preceding and following satellites in the same orbital plane.
* Left and right: the nearest satellites in the adjacent left and right planes.

Each edge includes a ``length`` attribute in meters.  Graphs are saved as pickle
files that can later be loaded with :func:`networkx.read_gpickle`.

The input TLE file must group satellites by orbital plane.  Planes are separated
by blank lines, and each satellite uses three lines: a name followed by the two
standard TLE lines.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import List

import numpy as np
import networkx as nx
from skyfield.api import EarthSatellite, Loader


def load_planes(tle_path: str) -> List[List[EarthSatellite]]:
    """Load satellites grouped by orbital plane from a TLE file.

    Parameters
    ----------
    tle_path: str
        Path to a text file containing satellite TLEs.  Satellites belonging to
        the same orbital plane are grouped together, and planes are separated by
        blank lines.  Each satellite is described by three lines: a name and the
        two TLE lines.

    Returns
    -------
    List[List[EarthSatellite]]
        Satellites grouped by plane, where the satellites within a plane are in
        along-track order.
    """

    planes: List[List[EarthSatellite]] = []
    loader = Loader(".")
    with open(tle_path, "r", encoding="utf-8") as f:
        group: List[str] = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                if group:
                    planes.append(_parse_plane(group))
                    group = []
                continue
            group.append(stripped)
        if group:
            planes.append(_parse_plane(group))
    return planes


def _parse_plane(lines: List[str]) -> List[EarthSatellite]:
    satellites: List[EarthSatellite] = []
    for i in range(0, len(lines), 3):
        name = lines[i]
        l1 = lines[i + 1]
        l2 = lines[i + 2]
        satellites.append(EarthSatellite(l1, l2, name))
    return satellites


def generate_graphs(
    planes: List[List[EarthSatellite]],
    start: datetime,
    duration_minutes: int = 200,
) -> List[nx.Graph]:
    """Propagate satellites and build graphs for each minute.

    Parameters
    ----------
    planes: list
        Satellites grouped by orbital plane.
    start: datetime
        Simulation start time.
    duration_minutes: int
        Number of minutes to simulate.

    Returns
    -------
    list
        A list of NetworkX graphs, one for each minute.
    """

    loader = Loader(".")
    ts = loader.timescale()
    graphs: List[nx.Graph] = []
    num_planes = len(planes)

    for minute in range(duration_minutes + 1):
        current_time = start + timedelta(minutes=minute)
        t = ts.utc(
            current_time.year,
            current_time.month,
            current_time.day,
            current_time.hour,
            current_time.minute,
            current_time.second,
        )

        # Determine positions for all satellites in meters.
        positions: List[List[np.ndarray]] = []
        for plane in planes:
            plane_positions = []
            for sat in plane:
                geocentric = sat.at(t)
                plane_positions.append(geocentric.position.km * 1_000.0)
            positions.append(plane_positions)

        # Build graph for this time step.
        G = nx.Graph(time=current_time.isoformat())

        for p_index, plane in enumerate(planes):
            for s_index, sat in enumerate(plane):
                node_id = f"p{p_index}_s{s_index}"
                G.add_node(node_id, name=sat.name, plane=p_index, index=s_index)

        # Connect along-track neighbours (up/down) within each plane.
        for p_index, plane in enumerate(planes):
            n = len(plane)
            for s_index in range(n):
                node = f"p{p_index}_s{s_index}"
                up = (s_index + 1) % n
                down = (s_index - 1) % n
                for neighbour in [up, down]:
                    other = f"p{p_index}_s{neighbour}"
                    if not G.has_edge(node, other):
                        d = np.linalg.norm(
                            positions[p_index][s_index] - positions[p_index][neighbour]
                        )
                        G.add_edge(node, other, length=float(d))

        # Connect to nearest satellites in adjacent planes.
        for p_index, plane in enumerate(planes):
            left_index = (p_index - 1) % num_planes
            right_index = (p_index + 1) % num_planes
            for s_index, _sat in enumerate(plane):
                node = f"p{p_index}_s{s_index}"

                left_node, left_dist = _nearest_neighbour(
                    positions[p_index][s_index], positions[left_index]
                )
                left_id = f"p{left_index}_s{left_node}"
                if not G.has_edge(node, left_id):
                    G.add_edge(node, left_id, length=float(left_dist))

                right_node, right_dist = _nearest_neighbour(
                    positions[p_index][s_index], positions[right_index]
                )
                right_id = f"p{right_index}_s{right_node}"
                if not G.has_edge(node, right_id):
                    G.add_edge(node, right_id, length=float(right_dist))

        graphs.append(G)

    return graphs


def _nearest_neighbour(
    pos: np.ndarray, plane_positions: List[np.ndarray]
) -> tuple[int, float]:
    distances = [np.linalg.norm(pos - other) for other in plane_positions]
    index = int(np.argmin(distances))
    return index, distances[index]


def save_graphs(graphs: List[nx.Graph], prefix: str = "topology") -> None:
    """Save graphs to disk using ``networkx.write_gpickle``.

    Parameters
    ----------
    graphs: list
        Graphs to save.
    prefix: str
        File name prefix.  Files are saved as ``"{prefix}_{i:04d}.gpickle"``.
    """

    for i, graph in enumerate(graphs):
        nx.write_gpickle(graph, f"{prefix}_{i:04d}.gpickle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tle_file", help="Path to TLE file with satellite data")
    parser.add_argument(
        "--start",
        required=True,
        help="Simulation start time in ISO format, e.g. 2025-08-08T04:00:00",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=200,
        help="Duration of the simulation in minutes",
    )
    parser.add_argument(
        "--output",
        default="topology",
        help="Prefix for output graph files (default: topology)",
    )
    args = parser.parse_args()

    start_time = datetime.fromisoformat(args.start)
    planes = load_planes(args.tle_file)
    graphs = generate_graphs(planes, start_time, args.duration)
    save_graphs(graphs, args.output)


if __name__ == "__main__":
    main()

