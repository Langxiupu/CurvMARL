"""Print average flow statistics for the default traffic model."""

from utils import GroundStationPoissonTraffic


def main() -> None:
    traffic = GroundStationPoissonTraffic(
        pareto_shape=1.5,
        pareto_scale_bytes=640 * 1024,
        mean_flows_per_min=60.0,
    )
    avg_size = traffic.average_flow_size_bytes()
    avg_rate = traffic.average_rate_bps()
    print(f"Average flow size: {avg_size:.0f} bytes")
    print(f"Average end-to-end transmission rate: {avg_rate/1e6:.1f} Mbps")


if __name__ == "__main__":
    main()
