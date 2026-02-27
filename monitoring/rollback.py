"""
rollback.py — Automated Model Rollback

Monitors drift detection and model performance. If drift is detected
or performance degrades below thresholds, automatically rolls back
the production model to the previous version.

Usage:
    python monitoring/rollback.py                    # Check and rollback if needed
    python monitoring/rollback.py --force             # Force rollback regardless
    python monitoring/rollback.py --dry-run           # Check without actually rolling back
"""
import argparse
import json
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from datetime import datetime


def get_production_model(client: MlflowClient, model_name: str):
    """Get the current production model version info."""
    versions = client.get_latest_versions(model_name, stages=["Production"])
    return versions[0] if versions else None


def get_previous_production(client: MlflowClient, model_name: str):
    """Get the most recently archived model that was in Production."""
    versions = client.search_model_versions(f"name='{model_name}'")

    # Find archived versions, sorted by version number descending
    archived = [v for v in versions if v.current_stage == "Archived"]
    if archived:
        archived.sort(key=lambda v: int(v.version), reverse=True)
        return archived[0]

    # If no archived, try Staging
    staging = [v for v in versions if v.current_stage == "Staging"]
    if staging:
        staging.sort(key=lambda v: int(v.version), reverse=True)
        return staging[0]

    return None


def check_drift_status(project_root: Path) -> dict:
    """Read the latest drift detection result."""
    drift_summary_path = project_root / "outputs" / "drift_reports" / "drift_summary.json"

    if drift_summary_path.exists():
        with open(drift_summary_path) as f:
            return json.load(f)

    return {"dataset_drift_detected": False, "drift_share": 0}


def check_performance_thresholds(client: MlflowClient) -> dict:
    """Check latest performance metrics against thresholds."""
    thresholds = {
        "latency_p99_max_ms": 100,
        "anomaly_rate_max": 0.50,  # Flag if >50% predictions are anomaly
        "error_rate_max": 0.05,
    }

    try:
        experiment = client.get_experiment_by_name("network-anomaly-monitoring")
        if experiment is None:
            return {"violation": False, "reason": "No monitoring data"}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
            filter_string="tags.mlflow.runName LIKE 'perf-%'",
        )

        if not runs:
            return {"violation": False, "reason": "No performance runs"}

        latest = runs[0]
        metrics = latest.data.metrics

        violations = []
        if metrics.get("latency_p99_ms", 0) > thresholds["latency_p99_max_ms"]:
            violations.append(f"P99 latency {metrics['latency_p99_ms']:.1f}ms > {thresholds['latency_p99_max_ms']}ms")
        if metrics.get("anomaly_rate", 0) > thresholds["anomaly_rate_max"]:
            violations.append(f"Anomaly rate {metrics['anomaly_rate']*100:.1f}% > {thresholds['anomaly_rate_max']*100}%")
        if metrics.get("error_rate", 0) > thresholds["error_rate_max"]:
            violations.append(f"Error rate {metrics['error_rate']*100:.1f}% > {thresholds['error_rate_max']*100}%")

        return {
            "violation": len(violations) > 0,
            "violations": violations,
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
        }

    except Exception as e:
        return {"violation": False, "reason": f"Error checking: {e}"}


def rollback(client: MlflowClient, model_name: str, target_version, dry_run: bool = False):
    """
    Rollback production model to the specified version.
    """
    current_prod = get_production_model(client, model_name)

    print(f"  🔄 Rolling back:")
    print(f"     From: v{current_prod.version} (current Production)")
    print(f"     To  : v{target_version.version}")

    if dry_run:
        print(f"     ⚡ DRY RUN — no changes made")
        return True

    # Demote current production
    client.transition_model_version_stage(
        name=model_name,
        version=current_prod.version,
        stage="Archived",
        archive_existing_versions=False,
    )

    # Promote target to production
    client.transition_model_version_stage(
        name=model_name,
        version=target_version.version,
        stage="Production",
        archive_existing_versions=False,
    )

    # Tag the rollback
    client.set_model_version_tag(
        model_name, target_version.version,
        "rollback_from", str(current_prod.version)
    )
    client.set_model_version_tag(
        model_name, target_version.version,
        "rollback_date", datetime.now().isoformat()
    )

    print(f"     ✅ Rollback complete!")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Automated model rollback")
    parser.add_argument("--force", action="store_true", help="Force rollback")
    parser.add_argument("--dry-run", action="store_true", help="Check without rolling back")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    MODEL_NAME = "network-anomaly-detector"

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Automated Rollback Monitor                     ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # Check current production model
    current_prod = get_production_model(client, MODEL_NAME)
    if not current_prod:
        print(f"  ❌ No Production model found. Nothing to rollback.")
        return

    run = client.get_run(current_prod.run_id)
    model_type = run.data.params.get("model_type", "?")
    f1 = run.data.metrics.get("f1_score", 0)
    print(f"  Current Production: v{current_prod.version} ({model_type}, F1={f1:.4f})")
    print()

    should_rollback = False
    reasons = []

    # ── Check 1: Data Drift ──
    drift = check_drift_status(project_root)
    if drift.get("dataset_drift_detected"):
        should_rollback = True
        reasons.append(f"Data drift detected ({drift['drift_share']*100:.1f}% features drifted)")
        print(f"  🔴 Drift: DETECTED — {drift['drift_share']*100:.1f}% features drifted")
    else:
        print(f"  🟢 Drift: None detected")

    # ── Check 2: Performance ──
    perf = check_performance_thresholds(client)
    if perf.get("violation"):
        should_rollback = True
        for v in perf["violations"]:
            reasons.append(v)
        print(f"  🔴 Performance: SLA violations detected")
        for v in perf["violations"]:
            print(f"       ⚠️  {v}")
    else:
        reason = perf.get("reason", "All within thresholds")
        print(f"  🟢 Performance: {reason}")

    print()

    # ── Decision ──
    if args.force:
        should_rollback = True
        reasons.append("Forced by --force flag")

    if not should_rollback:
        print(f"  ✅ No rollback needed. Model is healthy.")
        return

    # Find rollback target
    target = get_previous_production(client, MODEL_NAME)
    if not target:
        print(f"  ❌ No previous model version available for rollback.")
        return

    target_run = client.get_run(target.run_id)
    target_type = target_run.data.params.get("model_type", "?")
    target_f1 = target_run.data.metrics.get("f1_score", 0)

    print(f"  Rollback reasons:")
    for r in reasons:
        print(f"    ⚠️  {r}")
    print()
    print(f"  Target: v{target.version} ({target_type}, F1={target_f1:.4f})")

    # Execute rollback
    rollback(client, MODEL_NAME, target, dry_run=args.dry_run)

    # Log to MLflow
    mlflow.set_experiment("network-anomaly-monitoring")
    with mlflow.start_run(run_name=f"rollback-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        mlflow.log_param("action", "rollback" if not args.dry_run else "dry-run")
        mlflow.log_param("from_version", current_prod.version)
        mlflow.log_param("to_version", target.version)
        mlflow.log_param("reasons", "; ".join(reasons))

    print()


if __name__ == "__main__":
    main()
