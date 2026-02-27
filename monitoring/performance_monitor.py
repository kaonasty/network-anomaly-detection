"""
performance_monitor.py — Model Performance & Latency Monitoring

Tracks prediction performance metrics over time:
  - Prediction latency (p50, p95, p99)
  - Anomaly rate trends
  - Model accuracy (when ground truth is available)
  - Request throughput

Logs all metrics to MLflow for visualization.

Usage:
    python monitoring/performance_monitor.py
    python monitoring/performance_monitor.py --endpoint http://localhost:8080
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


def simulate_performance_monitoring(features_path: str, num_requests: int = 500):
    """
    Simulate monitoring by sending test data through the prediction pipeline.
    Measures latency and tracks anomaly rate.
    """
    # Load test data
    df = pd.read_parquet(features_path)
    sample = df.sample(n=min(num_requests, len(df)), random_state=42)

    latencies = []
    anomaly_count = 0
    error_count = 0

    print(f"  Simulating {num_requests} prediction requests...")
    print()

    # Import fallback detector for simulation (no server needed)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from serving.fallback import RuleBasedDetector
    detector = RuleBasedDetector()

    for i, (_, row) in enumerate(sample.iterrows()):
        start = time.perf_counter()

        try:
            metrics = {
                "latency_ms": row.get("latency_ms", 0),
                "packet_loss_pct": row.get("packet_loss_pct", 0),
                "cpu_utilization": row.get("cpu_utilization", 0),
                "bandwidth_mbps": row.get("bandwidth_mbps", 0),
                "error_rate": row.get("error_rate", 0),
            }
            result = detector.predict(metrics)
            if result["is_anomaly"]:
                anomaly_count += 1
        except Exception:
            error_count += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{num_requests} requests completed...")

    return latencies, anomaly_count, error_count, num_requests


def compute_stats(latencies: list) -> dict:
    """Compute latency percentile statistics."""
    arr = np.array(latencies)
    return {
        "p50_ms": round(np.percentile(arr, 50), 3),
        "p95_ms": round(np.percentile(arr, 95), 3),
        "p99_ms": round(np.percentile(arr, 99), 3),
        "mean_ms": round(np.mean(arr), 3),
        "max_ms": round(np.max(arr), 3),
        "min_ms": round(np.min(arr), 3),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor model performance")
    parser.add_argument("--features_path", type=str, default=None)
    parser.add_argument("--num_requests", type=int, default=500)
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Live endpoint URL (if not set, runs simulation)")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    features_path = args.features_path or str(project_root / "data" / "features" / "features.parquet")

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Performance Monitor                            ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    if args.endpoint and HTTPX_AVAILABLE:
        print(f"  Mode: Live endpoint monitoring ({args.endpoint})")
        # TODO: implement live endpoint testing with httpx
        print(f"  ⚠️  Live monitoring not yet implemented. Using simulation.")
        print()

    if not os.path.exists(features_path):
        print(f"❌ Features not found at {features_path}")
        sys.exit(1)

    # Run simulation
    latencies, anomalies, errors, total = simulate_performance_monitoring(
        features_path, args.num_requests
    )

    stats = compute_stats(latencies)
    anomaly_rate = anomalies / total if total > 0 else 0
    error_rate = errors / total if total > 0 else 0

    # Display results
    print()
    print(f"  ───────────────────────────────────────────────")
    print(f"  Latency Statistics:")
    print(f"    P50  : {stats['p50_ms']:>8.3f} ms")
    print(f"    P95  : {stats['p95_ms']:>8.3f} ms")
    print(f"    P99  : {stats['p99_ms']:>8.3f} ms")
    print(f"    Mean : {stats['mean_ms']:>8.3f} ms")
    print(f"    Max  : {stats['max_ms']:>8.3f} ms")
    print()
    print(f"  Throughput:")
    print(f"    Total requests: {total:>8,}")
    print(f"    Anomaly rate  : {anomaly_rate*100:>7.2f}%")
    print(f"    Error rate    : {error_rate*100:>7.2f}%")
    print()

    # Health check
    sla_p99 = 100  # ms
    if stats["p99_ms"] > sla_p99:
        print(f"  🔴 SLA VIOLATION: P99 latency ({stats['p99_ms']:.1f}ms) > {sla_p99}ms target")
    else:
        print(f"  🟢 SLA OK: P99 latency ({stats['p99_ms']:.1f}ms) within {sla_p99}ms target")
    print()

    # Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("network-anomaly-monitoring")

    with mlflow.start_run(run_name=f"perf-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        mlflow.log_metric("latency_p50_ms", stats["p50_ms"])
        mlflow.log_metric("latency_p95_ms", stats["p95_ms"])
        mlflow.log_metric("latency_p99_ms", stats["p99_ms"])
        mlflow.log_metric("latency_mean_ms", stats["mean_ms"])
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.log_metric("error_rate", error_rate)
        mlflow.log_metric("total_requests", total)

    print(f"  ✅ Metrics logged to MLflow experiment 'network-anomaly-monitoring'")
    print()


if __name__ == "__main__":
    main()
