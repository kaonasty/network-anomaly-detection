"""
generate_data.py — Synthetic Telecom Network Metrics Generator

Generates realistic network telemetry data with injected anomalies for
training anomaly detection models.

Metrics generated per node per minute:
  - latency_ms        : Network round-trip latency
  - packet_loss_pct   : Percentage of dropped packets
  - cpu_utilization   : Node CPU usage (0-100%)
  - bandwidth_mbps    : Throughput in Mbps
  - error_rate        : Error count per minute

Anomaly types injected:
  - spike             : Sudden sharp increase in one or more metrics
  - degradation       : Gradual metric worsening over a window
  - correlated        : Multiple metrics degrade simultaneously
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
# Node Profiles — each node has a different baseline
# ─────────────────────────────────────────────
NODE_PROFILES = {
    "node-core-01":   {"latency": 5,   "loss": 0.01, "cpu": 35, "bw": 950, "err": 2},
    "node-core-02":   {"latency": 6,   "loss": 0.02, "cpu": 40, "bw": 920, "err": 3},
    "node-edge-01":   {"latency": 15,  "loss": 0.05, "cpu": 55, "bw": 500, "err": 5},
    "node-edge-02":   {"latency": 18,  "loss": 0.06, "cpu": 50, "bw": 480, "err": 4},
    "node-edge-03":   {"latency": 20,  "loss": 0.08, "cpu": 60, "bw": 450, "err": 6},
    "node-access-01": {"latency": 30,  "loss": 0.10, "cpu": 45, "bw": 200, "err": 8},
    "node-access-02": {"latency": 35,  "loss": 0.12, "cpu": 50, "bw": 180, "err": 10},
    "node-access-03": {"latency": 25,  "loss": 0.07, "cpu": 42, "bw": 220, "err": 7},
    "node-transit-01": {"latency": 8,  "loss": 0.03, "cpu": 30, "bw": 1000,"err": 1},
    "node-transit-02": {"latency": 10, "loss": 0.04, "cpu": 32, "bw": 980, "err": 2},
}


def generate_timestamps(days: int) -> pd.DatetimeIndex:
    """Create 1-minute interval timestamps for the given number of days."""
    start = pd.Timestamp("2026-01-01 00:00:00")
    end = start + pd.Timedelta(days=days)
    return pd.date_range(start=start, end=end, freq="1min", inclusive="left")


def add_seasonality(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a seasonality signal with daily + weekly patterns.
    Returns a multiplier array centered around 1.0.
    """
    t = np.arange(n)
    # Daily cycle: peak at 10am-2pm, trough at 3am-5am
    daily = 0.15 * np.sin(2 * np.pi * (t - 6 * 60) / (24 * 60))
    # Weekly cycle: slightly higher on weekdays
    weekly = 0.05 * np.sin(2 * np.pi * t / (7 * 24 * 60))
    # Small random walk for natural drift
    walk = np.cumsum(rng.normal(0, 0.001, n))
    walk = walk - walk.mean()  # center around 0

    return 1.0 + daily + weekly + walk


def generate_normal_metrics(
    timestamps: pd.DatetimeIndex,
    profile: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate normal (non-anomalous) metrics for a single node."""
    n = len(timestamps)
    season = add_seasonality(n, rng)

    # Each metric: baseline × seasonality + gaussian noise
    latency = np.maximum(
        0.5,
        profile["latency"] * season + rng.normal(0, profile["latency"] * 0.1, n),
    )
    packet_loss = np.clip(
        profile["loss"] * season + rng.normal(0, profile["loss"] * 0.15, n),
        0, 100,
    )
    cpu = np.clip(
        profile["cpu"] * season + rng.normal(0, profile["cpu"] * 0.08, n),
        0, 100,
    )
    bandwidth = np.maximum(
        0,
        profile["bw"] * season + rng.normal(0, profile["bw"] * 0.05, n),
    )
    error_rate = np.maximum(
        0,
        profile["err"] * season + rng.normal(0, profile["err"] * 0.2, n),
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "latency_ms": np.round(latency, 2),
        "packet_loss_pct": np.round(packet_loss, 4),
        "cpu_utilization": np.round(cpu, 2),
        "bandwidth_mbps": np.round(bandwidth, 2),
        "error_rate": np.round(error_rate, 2),
    })


def inject_spike(
    df: pd.DataFrame,
    idx: int,
    profile: dict,
    rng: np.random.Generator,
) -> None:
    """Inject a sudden spike anomaly at the given index (1-5 minute duration)."""
    duration = rng.integers(1, 6)
    end_idx = min(idx + duration, len(df))

    # Pick 1-3 metrics to spike
    metrics_to_spike = rng.choice(
        ["latency_ms", "packet_loss_pct", "cpu_utilization", "error_rate"],
        size=rng.integers(1, 4),
        replace=False,
    )

    for metric in metrics_to_spike:
        if metric == "latency_ms":
            df.loc[idx:end_idx - 1, metric] = profile["latency"] * rng.uniform(5, 15)
        elif metric == "packet_loss_pct":
            df.loc[idx:end_idx - 1, metric] = rng.uniform(5, 30)
        elif metric == "cpu_utilization":
            df.loc[idx:end_idx - 1, metric] = rng.uniform(90, 100)
        elif metric == "error_rate":
            df.loc[idx:end_idx - 1, metric] = profile["err"] * rng.uniform(10, 50)


def inject_degradation(
    df: pd.DataFrame,
    idx: int,
    profile: dict,
    rng: np.random.Generator,
) -> None:
    """Inject a gradual degradation over 15-60 minutes."""
    duration = rng.integers(15, 61)
    end_idx = min(idx + duration, len(df))
    ramp = np.linspace(1.0, rng.uniform(3, 8), end_idx - idx)

    # Gradual latency increase + packet loss increase
    df.loc[idx:end_idx - 1, "latency_ms"] = (
        profile["latency"] * ramp + rng.normal(0, 1, end_idx - idx)
    )
    df.loc[idx:end_idx - 1, "packet_loss_pct"] = np.clip(
        profile["loss"] * ramp * 2, 0, 50
    )


def inject_correlated(
    df: pd.DataFrame,
    idx: int,
    profile: dict,
    rng: np.random.Generator,
) -> None:
    """Inject a correlated failure: all metrics degrade together."""
    duration = rng.integers(5, 30)
    end_idx = min(idx + duration, len(df))
    severity = rng.uniform(3, 10)

    df.loc[idx:end_idx - 1, "latency_ms"] = profile["latency"] * severity
    df.loc[idx:end_idx - 1, "packet_loss_pct"] = np.clip(
        profile["loss"] * severity * 5, 0, 50
    )
    df.loc[idx:end_idx - 1, "cpu_utilization"] = np.clip(
        profile["cpu"] * rng.uniform(1.5, 2.5), 0, 100
    )
    df.loc[idx:end_idx - 1, "bandwidth_mbps"] = profile["bw"] * rng.uniform(0.1, 0.4)
    df.loc[idx:end_idx - 1, "error_rate"] = profile["err"] * severity * 2


def inject_anomalies(
    df: pd.DataFrame,
    profile: dict,
    anomaly_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject anomalies into the dataframe.
    Returns (is_anomaly, anomaly_type) arrays.
    """
    n = len(df)
    is_anomaly = np.zeros(n, dtype=int)
    anomaly_type = np.array(["none"] * n, dtype=object)

    # Determine anomaly positions — spread them out
    num_anomaly_events = int(n * anomaly_rate / 15)  # average 15 rows per event
    anomaly_starts = rng.choice(
        range(100, n - 100),  # avoid edges
        size=min(num_anomaly_events, n // 100),
        replace=False,
    )

    injection_funcs = {
        "spike": inject_spike,
        "degradation": inject_degradation,
        "correlated": inject_correlated,
    }

    for start_idx in sorted(anomaly_starts):
        atype = rng.choice(["spike", "degradation", "correlated"], p=[0.4, 0.35, 0.25])
        injection_funcs[atype](df, start_idx, profile, rng)

        # Mark affected rows
        if atype == "spike":
            duration = min(rng.integers(1, 6), n - start_idx)
        elif atype == "degradation":
            duration = min(rng.integers(15, 61), n - start_idx)
        else:
            duration = min(rng.integers(5, 30), n - start_idx)

        end_idx = min(start_idx + duration, n)
        is_anomaly[start_idx:end_idx] = 1
        anomaly_type[start_idx:end_idx] = atype

    return is_anomaly, anomaly_type


def generate_node_data(
    node_id: str,
    profile: dict,
    timestamps: pd.DatetimeIndex,
    anomaly_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Generate complete data for a single node."""
    rng = np.random.default_rng(seed)

    # Generate baseline metrics
    df = generate_normal_metrics(timestamps, profile, rng)
    df.insert(0, "node_id", node_id)

    # Inject anomalies
    is_anomaly, anomaly_type = inject_anomalies(df, profile, anomaly_rate, rng)
    df["is_anomaly"] = is_anomaly
    df["anomaly_type"] = anomaly_type

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic telco network data")
    parser.add_argument("--num_nodes", type=int, default=10)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--anomaly_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output directory relative to project root
    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select nodes
    node_items = list(NODE_PROFILES.items())[:args.num_nodes]

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Telecom Network Data Generator                 ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Nodes     : {args.num_nodes:<36}║")
    print(f"║  Days      : {args.days:<36}║")
    print(f"║  Anomaly % : {args.anomaly_rate * 100:.1f}%{' ' * 32}║")
    print(f"║  Seed      : {args.seed:<36}║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    timestamps = generate_timestamps(args.days)
    print(f"  Timestamps : {len(timestamps):,} per node ({args.days} days × 1440 min/day)")
    print()

    all_dfs = []
    for i, (node_id, profile) in enumerate(node_items):
        node_seed = args.seed + i
        df = generate_node_data(node_id, profile, timestamps, args.anomaly_rate, node_seed)
        anomaly_count = df["is_anomaly"].sum()
        anomaly_pct = anomaly_count / len(df) * 100
        print(f"  [{i+1:>2}/{args.num_nodes}] {node_id:<20} "
              f"{len(df):>7,} rows  |  {anomaly_count:>5,} anomalies ({anomaly_pct:.1f}%)")
        all_dfs.append(df)

    # Combine and save
    full_df = pd.concat(all_dfs, ignore_index=True)
    output_path = output_dir / "network_metrics.csv"
    full_df.to_csv(output_path, index=False)

    # Print summary
    total_anomalies = full_df["is_anomaly"].sum()
    total_rows = len(full_df)
    print()
    print(f"  ───────────────────────────────────────────────")
    print(f"  Total rows     : {total_rows:>10,}")
    print(f"  Total anomalies: {total_anomalies:>10,} ({total_anomalies/total_rows*100:.2f}%)")
    print()

    # Anomaly type breakdown
    type_counts = full_df[full_df["is_anomaly"] == 1]["anomaly_type"].value_counts()
    print(f"  Anomaly breakdown:")
    for atype, count in type_counts.items():
        print(f"    {atype:<15} {count:>6,} ({count/total_anomalies*100:.1f}%)")

    print(f"\n  ✅ Saved to: {output_path}")
    print()

    return full_df


if __name__ == "__main__":
    main()
