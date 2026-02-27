"""
feature_engineering.py — Batch Feature Computation Pipeline

Reads raw network metrics and computes engineered features:
  1. Rolling window statistics (5min, 15min, 1hr)
  2. Rate-of-change features (first derivative)
  3. Cross-metric features (correlations, ratios)
  4. Node-vs-fleet comparison features

Outputs feature-enriched dataset to data/features/ as Parquet.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────
# Rolling Window Features
# ─────────────────────────────────────────────
METRIC_COLS = ["latency_ms", "packet_loss_pct", "cpu_utilization", "bandwidth_mbps", "error_rate"]
WINDOWS = {"5min": 5, "15min": 15, "1hr": 60}


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling mean, std, max for each metric across multiple windows."""
    result = df.copy()

    for metric in METRIC_COLS:
        for window_name, window_size in WINDOWS.items():
            col = df[metric]
            result[f"{metric}_roll_mean_{window_name}"] = (
                col.rolling(window=window_size, min_periods=1).mean()
            )
            result[f"{metric}_roll_std_{window_name}"] = (
                col.rolling(window=window_size, min_periods=1).std().fillna(0)
            )
            result[f"{metric}_roll_max_{window_name}"] = (
                col.rolling(window=window_size, min_periods=1).max()
            )

    return result


# ─────────────────────────────────────────────
# Rate of Change Features
# ─────────────────────────────────────────────

def compute_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first derivative (rate of change) for key metrics."""
    result = df.copy()

    for metric in METRIC_COLS:
        # 1-minute change
        result[f"{metric}_delta_1m"] = df[metric].diff(1).fillna(0)
        # 5-minute change
        result[f"{metric}_delta_5m"] = df[metric].diff(5).fillna(0)
        # Absolute rate of change
        result[f"{metric}_abs_delta_1m"] = result[f"{metric}_delta_1m"].abs()

    return result


# ─────────────────────────────────────────────
# Cross-Metric Features
# ─────────────────────────────────────────────

def compute_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features that capture relationships between metrics."""
    result = df.copy()

    # Latency × packet loss interaction
    result["latency_x_loss"] = df["latency_ms"] * df["packet_loss_pct"]

    # CPU-to-bandwidth ratio (high CPU + low BW may indicate issue)
    result["cpu_bw_ratio"] = np.where(
        df["bandwidth_mbps"] > 0,
        df["cpu_utilization"] / df["bandwidth_mbps"],
        0,
    )

    # Error rate normalized by CPU
    result["error_per_cpu"] = np.where(
        df["cpu_utilization"] > 0,
        df["error_rate"] / df["cpu_utilization"] * 100,
        0,
    )

    # Rolling correlation between latency and CPU (30-minute window)
    result["latency_cpu_corr_30m"] = (
        df["latency_ms"]
        .rolling(window=30, min_periods=10)
        .corr(df["cpu_utilization"])
        .fillna(0)
    )

    return result


# ─────────────────────────────────────────────
# Z-Score Features (how far from node's own mean)
# ─────────────────────────────────────────────

def compute_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute z-score for each metric using 1-hour rolling statistics."""
    result = df.copy()

    for metric in METRIC_COLS:
        roll_mean = df[metric].rolling(window=60, min_periods=10).mean()
        roll_std = df[metric].rolling(window=60, min_periods=10).std()
        result[f"{metric}_zscore"] = np.where(
            roll_std > 0,
            (df[metric] - roll_mean) / roll_std,
            0,
        )

    return result


# ─────────────────────────────────────────────
# Time-based Features
# ─────────────────────────────────────────────

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    result = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek  # 0=Monday
    result["is_business_hours"] = ((ts.dt.hour >= 8) & (ts.dt.hour <= 18)).astype(int)
    result["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    return result


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def run_feature_pipeline(input_path: str, output_dir: str) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    print("  Loading raw data...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows from {input_path}")

    # Sort by node and time for correct rolling computations
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    all_features = []

    nodes = df["node_id"].unique()
    for i, node_id in enumerate(nodes):
        node_df = df[df["node_id"] == node_id].copy().reset_index(drop=True)
        print(f"  [{i+1}/{len(nodes)}] Processing {node_id}...", end="")

        # Apply feature transformations per node
        node_df = compute_rolling_features(node_df)
        node_df = compute_rate_of_change(node_df)
        node_df = compute_cross_features(node_df)
        node_df = compute_zscore_features(node_df)
        node_df = compute_time_features(node_df)

        all_features.append(node_df)
        print(f" {len(node_df.columns)} features")

    # Concatenate all nodes
    features_df = pd.concat(all_features, ignore_index=True)

    # Save as Parquet
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "features.parquet")
    features_df.to_parquet(output_path, index=False, engine="pyarrow")

    # Also save feature names for reference
    feature_cols = [c for c in features_df.columns
                    if c not in ["timestamp", "node_id", "is_anomaly", "anomaly_type"]]
    feature_list_path = os.path.join(output_dir, "feature_columns.txt")
    with open(feature_list_path, "w") as f:
        f.write("\n".join(feature_cols))

    return features_df


def parse_args():
    parser = argparse.ArgumentParser(description="Compute features from raw network metrics")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = args.input or str(project_root / "data" / "raw" / "network_metrics.csv")
    output_dir = args.output_dir or str(project_root / "data" / "features")

    if not os.path.exists(input_path):
        print(f"  ❌ Raw data not found at {input_path}")
        print(f"  Run 'python data/generate_data.py' first.")
        sys.exit(1)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Batch Feature Engineering Pipeline             ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    features_df = run_feature_pipeline(input_path, output_dir)

    # Summary statistics
    feature_cols = [c for c in features_df.columns
                    if c not in ["timestamp", "node_id", "is_anomaly", "anomaly_type"]]

    print()
    print(f"  ───────────────────────────────────────────────")
    print(f"  Total rows    : {len(features_df):>10,}")
    print(f"  Total features: {len(feature_cols):>10}")
    print(f"  Output        : {output_dir}")
    print()
    print(f"  Feature categories:")
    print(f"    Raw metrics     : {len(METRIC_COLS)}")
    rolling_count = sum(1 for c in feature_cols if "roll_" in c)
    delta_count = sum(1 for c in feature_cols if "delta_" in c)
    cross_count = sum(1 for c in feature_cols if any(x in c for x in ["_x_", "_ratio", "_per_", "_corr_"]))
    zscore_count = sum(1 for c in feature_cols if "zscore" in c)
    time_count = sum(1 for c in feature_cols if c in ["hour", "day_of_week", "is_business_hours", "is_weekend"])
    print(f"    Rolling window  : {rolling_count}")
    print(f"    Rate of change  : {delta_count}")
    print(f"    Cross-metric    : {cross_count}")
    print(f"    Z-score         : {zscore_count}")
    print(f"    Time-based      : {time_count}")
    print(f"\n  ✅ Features saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
