import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from baselines.graphpr import graph_pr


def load_graph(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    graph = {}
    for u, v, w in data:
        graph.setdefault(u, {})[v] = w
        graph.setdefault(v, {})[u] = w
    return graph


def main():
    parser = argparse.ArgumentParser(description='Run GraphPR shortest path algorithm')
    parser.add_argument('--graph', required=True, help='Path to JSON edge list')
    parser.add_argument('--src', required=True, help='Source node')
    parser.add_argument('--dst', required=True, help='Destination node')
    args = parser.parse_args()

    graph = load_graph(args.graph)
    path = graph_pr(graph, args.src, args.dst)
    print(' -> '.join(map(str, path)))


if __name__ == '__main__':
    main()
