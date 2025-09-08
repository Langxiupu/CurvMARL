# CurvMARL

## Satellite topology generation

This repository includes a script for building dynamic satellite network graphs.
`scripts/satellite_topology.py` uses the [Skyfield](https://rhodesmill.org/skyfield/)
library to propagate satellites from TLE data and produces a NetworkX graph for
each minute of the simulation. Every satellite connects to four neighbours: the
two along-track neighbours in the same orbital plane and the nearest satellites
in the adjacent left and right planes. Each edge stores its length in meters.

Example usage:

```bash
pip install skyfield networkx numpy
python scripts/satellite_topology.py path/to/constellation.tle --start "2025-08-08T04:00:00" --duration 200
```

Graphs are saved as pickle files (`.gpickle`) with names like `topology_0000.gpickle`.

## Shortest-path routing baseline

`scripts/run_shortest_path.py` constructs a small constellation and routes random flows using weighted shortest paths. Link weights follow:

$w_{ij}^t = \overline{\Gamma}_{ij}^t \left( 1 + \beta \frac{L_{ij}^t}{1 - L_{ij}^t} \right)$

The script prints per-step packet loss rate, average latency and system throughput.

Example:

```bash
python scripts/run_shortest_path.py --steps 3 --beta 0.5
```

## Algorithm entry points

`scripts/run_graphpr.py` demonstrates the `graph_pr` shortest-path routine on a JSON edge list:

```bash
python scripts/run_graphpr.py --graph example_graph.json --src A --dst C
```

`scripts/run_curvmarl.py` launches the MAPPO-based CurvMARL training using a configuration file:

```bash
python scripts/run_curvmarl.py --config configs/exp1_no_rewire.json
```
