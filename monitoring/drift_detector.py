"""
drift_detector.py — Data Drift Detection with Evidently

Compares the distribution of incoming (production) data against
the training data to detect feature drift and concept drift.

Generates:
  - Drift report (HTML) for visual inspection
  - Drift summary (JSON) for programmatic use
  - MLflow logged metrics for tracking drift over time

Usage:
    python monitoring/drift_detector.py
    python monitoring/drift_detector.py --reference data/features/features.parquet --current data/incoming/latest.parquet
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime

try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.metrics import DataDriftTable, DatasetDriftMetric
    EVIDENTLY_AVAILABLE = True
except (ImportError, Exception) as e:
    EVIDENTLY_AVAILABLE = False
    print(f"⚠️  Evidently not available: {e}")

METRIC_COLS = ["latency_ms", "packet_loss_pct", "cpu_utilization", "bandwidth_mbps", "error_rate"]

def simulate_drifted_data(reference_df: pd.DataFrame, drift_level: str = "moderate") -> pd.DataFrame:
    """
    Simulate drifted production data for testing.
    Applies realistic drift patterns to the reference data.
    """
    rng = np.random.default_rng(99)
    drifted = reference_df.copy()
    n = len(drifted)

    if drift_level == "none":
        # No drift — just add small noise
        for col in METRIC_COLS:
            drifted[col] += rng.normal(0, drifted[col].std() * 0.05, n)

    elif drift_level == "moderate":
        # Moderate drift — shift some distributions
        drifted["latency_ms"] *= 1.5  # Latency increased by 50%
        drifted["packet_loss_pct"] += rng.exponential(0.5, n)  # More packet loss
        drifted["cpu_utilization"] += 10  # CPU shifted up
        drifted["cpu_utilization"] = drifted["cpu_utilization"].clip(0, 100)

    elif drift_level == "severe":
        # Severe drift — all distributions shift dramatically
        drifted["latency_ms"] *= 3
        drifted["packet_loss_pct"] += rng.exponential(2, n)
        drifted["cpu_utilization"] = rng.normal(80, 10, n).clip(0, 100)
        drifted["bandwidth_mbps"] *= 0.3
        drifted["error_rate"] *= 5

    return drifted


def run_drift_detection(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_dir: str) -> dict:
    """
    Run Evidently drift detection and generate report.
    Returns drift summary dictionary.
    """
    if not EVIDENTLY_AVAILABLE:
        return _manual_drift_detection(reference_df, current_df)

    os.makedirs(output_dir, exist_ok=True)

    # Select only numeric feature columns for drift detection
    feature_cols = [c for c in reference_df.columns
                    if c in METRIC_COLS or "roll_" in c or "zscore" in c or "delta" in c]

    # Limit to manageable number of features for the report
    analysis_cols = METRIC_COLS + [c for c in feature_cols if "zscore" in c]
    analysis_cols = [c for c in analysis_cols if c in reference_df.columns and c in current_df.columns]

    ref = reference_df[analysis_cols].copy()
    curr = current_df[analysis_cols].copy()

    # Build Evidently report
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])
    report.run(reference_data=ref, current_data=curr)

    # Save HTML report
    report_path = os.path.join(output_dir, "drift_report.html")
    report.save_html(report_path)

    # Extract drift results
    report_dict = report.as_dict()
    results = report_dict.get("metrics", [])

    dataset_drift = False
    drifted_features = []
    drift_share = 0.0

    for metric_result in results:
        metric_id = metric_result.get("metric", "")
        result = metric_result.get("result", {})

        if "DatasetDriftMetric" in metric_id:
            dataset_drift = result.get("dataset_drift", False)
            drift_share = result.get("share_of_drifted_columns", 0)

        if "DataDriftTable" in metric_id:
            drift_by_columns = result.get("drift_by_columns", {})
            for col_name, col_data in drift_by_columns.items():
                if col_data.get("column_drifted", False):
                    drifted_features.append(col_name)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_drift_detected": dataset_drift,
        "drift_share": round(drift_share, 4),
        "drifted_features": drifted_features,
        "num_drifted_features": len(drifted_features),
        "total_features_analyzed": len(analysis_cols),
        "reference_size": len(ref),
        "current_size": len(curr),
        "report_path": report_path,
    }

    # Save JSON summary
    summary_path = os.path.join(output_dir, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _manual_drift_detection(ref_df: pd.DataFrame, curr_df: pd.DataFrame) -> dict:
    """
    Simple statistical drift detection without Evidently.
    Uses KS-test and mean shift for each metric.
    """
    from scipy import stats

    drifted_features = []
    drift_details = {}

    for col in METRIC_COLS:
        if col not in ref_df.columns or col not in curr_df.columns:
            continue

        ref_vals = ref_df[col].dropna()
        curr_vals = curr_df[col].dropna()

        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, curr_vals)

        # Mean shift
        mean_shift = abs(curr_vals.mean() - ref_vals.mean()) / (ref_vals.std() + 1e-10)

        is_drifted = ks_pvalue < 0.05 or mean_shift > 1.0

        if is_drifted:
            drifted_features.append(col)

        drift_details[col] = {
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 6),
            "mean_shift_std": round(mean_shift, 4),
            "drifted": is_drifted,
            "ref_mean": round(ref_vals.mean(), 4),
            "curr_mean": round(curr_vals.mean(), 4),
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset_drift_detected": len(drifted_features) > len(METRIC_COLS) * 0.5,
        "drift_share": round(len(drifted_features) / len(METRIC_COLS), 4),
        "drifted_features": drifted_features,
        "num_drifted_features": len(drifted_features),
        "total_features_analyzed": len(METRIC_COLS),
        "drift_details": drift_details,
        "method": "manual-ks-test",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Detect data drift")
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--current", type=str, default=None)
    parser.add_argument("--drift_level", type=str, default="moderate",
                        choices=["none", "moderate", "severe"],
                        help="Simulated drift level (when --current not provided)")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    ref_path = args.reference or str(project_root / "data" / "features" / "features.parquet")
    output_dir = args.output_dir or str(project_root / "outputs" / "drift_reports")

    if not os.path.exists(ref_path):
        print(f"❌ Reference data not found at {ref_path}")
        sys.exit(1)

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Data Drift Detection                           ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # Load reference data
    ref_df = pd.read_parquet(ref_path)
    print(f"  Reference data: {len(ref_df):,} rows")

    # Load or simulate current data
    if args.current and os.path.exists(args.current):
        curr_df = pd.read_parquet(args.current)
        print(f"  Current data  : {len(curr_df):,} rows (from file)")
    else:
        # Simulate drifted data for demonstration
        sample = ref_df.sample(n=min(50000, len(ref_df)), random_state=42)
        curr_df = simulate_drifted_data(sample, args.drift_level)
        print(f"  Current data  : {len(curr_df):,} rows (simulated, drift={args.drift_level})")

    print()

    # Run detection
    summary = run_drift_detection(ref_df, curr_df, output_dir)

    # Display results
    drift_status = "🔴 DRIFT DETECTED" if summary["dataset_drift_detected"] else "🟢 NO DRIFT"
    print(f"  Result: {drift_status}")
    print(f"  Drift share: {summary['drift_share']*100:.1f}% of features drifted")
    print()

    if summary["drifted_features"]:
        print(f"  Drifted features:")
        for feat in summary["drifted_features"]:
            print(f"    ⚠️  {feat}")
        print()

    if "drift_details" in summary:
        print(f"  Detailed stats:")
        for col, details in summary["drift_details"].items():
            status = "⚠️ " if details["drifted"] else "✅"
            print(f"    {status} {col:<25} KS={details['ks_statistic']:.4f}  "
                  f"p={details['ks_pvalue']:.4f}  shift={details['mean_shift_std']:.2f}σ")
        print()

    # Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("network-anomaly-monitoring")

    with mlflow.start_run(run_name=f"drift-check-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        mlflow.log_metric("drift_detected", int(summary["dataset_drift_detected"]))
        mlflow.log_metric("drift_share", summary["drift_share"])
        mlflow.log_metric("drifted_features_count", summary["num_drifted_features"])

        if "report_path" in summary and os.path.exists(summary["report_path"]):
            mlflow.log_artifact(summary["report_path"])

    print(f"  ✅ Results logged to MLflow experiment 'network-anomaly-monitoring'")

    if summary.get("report_path"):
        print(f"  📊 HTML report: {summary['report_path']}")
    print()

    # Return drift status for use by rollback script
    return summary["dataset_drift_detected"]


if __name__ == "__main__":
    main()
