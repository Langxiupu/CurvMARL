import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import Flow, aggregate_metrics


def test_aggregate_metrics_avg_hops():
    flows = [
        Flow(id=0, src=0, dst=2, rate_bps=0.0, path_edges=[(0, 1), (1, 2)]),
        Flow(id=1, src=3, dst=4, rate_bps=0.0, path_edges=[(3, 5), (5, 4)]),
    ]
    results = [
        {"goodput_bps": 1.0, "latency_s": 0.0},
        {"goodput_bps": 1.0, "latency_s": 0.0},
    ]
    metrics = aggregate_metrics(flows, results)
    assert metrics["avg_hop_count"] == 2.0
