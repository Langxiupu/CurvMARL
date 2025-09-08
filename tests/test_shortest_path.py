import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baselines.graphpr import graph_pr


def test_graphpr_shortest_path():
    graph = {
        'A': {'B': 1, 'C': 5},
        'B': {'A': 1, 'C': 2},
        'C': {'A': 5, 'B': 2},
    }
    assert graph_pr(graph, 'A', 'C') == ['A', 'B', 'C']
