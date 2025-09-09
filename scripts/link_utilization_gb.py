"""Compute offered load and link utilization using GB-scale flows.

This script assumes:
- Flow sizes follow a Pareto distribution with the scale expressed in gigabytes.
- Each simulation step represents one minute, producing a Poisson-distributed
  number of flows per minute.
- Link bandwidth is expressed in Mbps (megabits per second).

It reports the mean flow size, the average offered load, link utilization
relative to a 500 Mbps link, and the time required to transmit the mean flow.
"""


def main() -> None:
    shape = 1.5  # Pareto shape parameter
    scale_GB = 5.0  # Pareto scale (minimum) expressed in gigabytes
    flows_per_min = 60.0  # average number of flows per minute
    link_capacity_Mbps = 500.0  # link bandwidth

    # --- Unit conversions ---
    scale_bytes = scale_GB * 1024 ** 3  # GB -> bytes
    mean_flow_size_bytes = shape * scale_bytes / (shape - 1.0)

    flow_rate_per_sec = flows_per_min / 60.0
    offered_load_bps = flow_rate_per_sec * mean_flow_size_bytes * 8.0

    link_capacity_bps = link_capacity_Mbps * 1e6
    utilization = offered_load_bps / link_capacity_bps

    # Time to transmit a mean-sized flow at link capacity
    mean_flow_tx_time_s = mean_flow_size_bytes * 8.0 / link_capacity_bps

    print(f"Mean flow size: {mean_flow_size_bytes / (1024 ** 3):.2f} GB")
    print(f"Offered load: {offered_load_bps / 1e6:.2f} Mbps")
    print(f"Link capacity: {link_capacity_bps / 1e6:.2f} Mbps")
    print(f"Utilization: {utilization * 100:.2f}%")
    print(f"Time to transmit mean flow: {mean_flow_tx_time_s:.2f} s")


if __name__ == "__main__":
    main()
