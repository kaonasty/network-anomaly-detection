"""
stream_features.py — Simulated Streaming Feature Pipeline

Simulates a real-time streaming pipeline by reading raw metrics one row at a
time and computing rolling features in-memory using a sliding window buffer.

This demonstrates how features would be computed in a production streaming
system (e.g., Kafka Streams, Flink) but uses a simple Python implementation.

Usage:
    python features/stream_features.py                    # Process all data
    python features/stream_features.py --limit 100        # Process first 100 rows
    python features/stream_features.py --node node-core-01  # Single node
"""
import argparse
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field


METRIC_COLS = ["latency_ms", "packet_loss_pct", "cpu_utilization", "bandwidth_mbps", "error_rate"]


@dataclass
class StreamingWindow:
    """Sliding window buffer for real-time feature computation."""
    window_size: int = 60  # 1 hour (60 minutes)
    buffer: dict = field(default_factory=dict)

    def __post_init__(self):
        for metric in METRIC_COLS:
            self.buffer[metric] = deque(maxlen=self.window_size)

    def update(self, row: dict) -> dict:
        """
        Push a new data point and compute real-time features.
        Returns a feature dictionary for this single record.
        """
        # Add new values to buffers
        for metric in METRIC_COLS:
            self.buffer[metric].append(row[metric])

        features = {}

        for metric in METRIC_COLS:
            values = list(self.buffer[metric])
            n = len(values)

            # Raw value
            features[metric] = row[metric]

            # Rolling stats (use available window)
            features[f"{metric}_roll_mean"] = np.mean(values)
            features[f"{metric}_roll_std"] = np.std(values) if n > 1 else 0
            features[f"{metric}_roll_max"] = np.max(values)

            # Z-score
            std = features[f"{metric}_roll_std"]
            if std > 0:
                features[f"{metric}_zscore"] = (
                    (row[metric] - features[f"{metric}_roll_mean"]) / std
                )
            else:
                features[f"{metric}_zscore"] = 0

            # Delta (1-step change)
            if n >= 2:
                features[f"{metric}_delta"] = values[-1] - values[-2]
            else:
                features[f"{metric}_delta"] = 0

        # Cross-metric features
        features["latency_x_loss"] = row["latency_ms"] * row["packet_loss_pct"]
        features["cpu_bw_ratio"] = (
            row["cpu_utilization"] / row["bandwidth_mbps"]
            if row["bandwidth_mbps"] > 0 else 0
        )

        return features


def simulate_stream(input_path: str, node_filter: str = None, limit: int = None):
    """Simulate processing a real-time data stream row by row."""
    df = pd.read_csv(input_path)

    if node_filter:
        df = df[df["node_id"] == node_filter]

    if limit:
        df = df.head(limit)

    # One window per node
    windows: dict[str, StreamingWindow] = {}

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Streaming Feature Pipeline (Simulated)         ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"  Records to process: {len(df):,}")
    print(f"  Nodes: {df['node_id'].nunique()}")
    print()

    processed = 0
    anomalies_detected = 0
    start_time = time.time()

    for _, row in df.iterrows():
        node_id = row["node_id"]

        # Initialize window for new nodes
        if node_id not in windows:
            windows[node_id] = StreamingWindow()

        # Compute streaming features
        features = windows[node_id].update(row.to_dict())

        # Simple real-time anomaly check (z-score based)
        max_zscore = max(
            abs(features.get(f"{m}_zscore", 0)) for m in METRIC_COLS
        )
        if max_zscore > 3.0:
            anomalies_detected += 1

        processed += 1

        # Progress update every 10,000 records
        if processed % 10000 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            print(f"  Processed {processed:>8,} records | "
                  f"{rate:.0f} rec/s | "
                  f"z-score anomalies: {anomalies_detected:,}")

    elapsed = time.time() - start_time
    print()
    print(f"  ───────────────────────────────────────────────")
    print(f"  Total processed  : {processed:>10,}")
    print(f"  Runtime          : {elapsed:>10.2f}s")
    print(f"  Throughput       : {processed/elapsed:>10.0f} rec/s")
    print(f"  Z-score anomalies: {anomalies_detected:>10,}")
    print(f"\n  ✅ Streaming simulation complete")
    print()


def main():
    parser = argparse.ArgumentParser(description="Simulate streaming feature pipeline")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--node", type=str, default=None, help="Filter to single node")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = args.input or str(project_root / "data" / "raw" / "network_metrics.csv")

    if not Path(input_path).exists():
        print(f"  ❌ Data not found at {input_path}")
        print(f"  Run 'python data/generate_data.py' first.")
        sys.exit(1)

    simulate_stream(input_path, node_filter=args.node, limit=args.limit)


if __name__ == "__main__":
    main()
