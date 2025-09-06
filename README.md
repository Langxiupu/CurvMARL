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

