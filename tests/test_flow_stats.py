import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import GroundStationPoissonTraffic


def test_average_flow_stats():
    traffic = GroundStationPoissonTraffic(rate_bps=1.6e6, pareto_shape=1.5)
    assert traffic.average_rate_bps() == pytest.approx(1.6e6)
    assert traffic.average_flow_size_bytes() == pytest.approx(1_966_080.0)
