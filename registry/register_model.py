"""
register_model.py — Register Models with Comprehensive Metadata

Registers trained models in the MLflow Model Registry with rich tags
and annotations for tracking model lineage, performance, and ownership.

Usage:
    python registry/register_model.py
"""
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    MODEL_NAME = "network-anomaly-detector"

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Model Registry — Metadata Tagging              ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # ── List all versions ──
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        print(f"  ❌ Model '{MODEL_NAME}' not found. Train models first.")
        return

    if not versions:
        print(f"  ❌ No versions found for '{MODEL_NAME}'. Train models first.")
        return

    print(f"  Found {len(versions)} version(s) of '{MODEL_NAME}':\n")

    for v in versions:
        run = client.get_run(v.run_id)
        model_type = run.data.params.get("model_type", "unknown")
        f1 = run.data.metrics.get("f1_score", 0)
        roc_auc = run.data.metrics.get("roc_auc", 0)

        print(f"  Version {v.version}")
        print(f"    Run ID     : {v.run_id[:12]}...")
        print(f"    Model Type : {model_type}")
        print(f"    Stage      : {v.current_stage}")
        print(f"    F1 Score   : {f1:.4f}")
        print(f"    ROC AUC    : {roc_auc:.4f}")

        # ── Tag the model version ──
        tags = {
            "model_type": model_type,
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "author": "ml-team",
            "project": "network-anomaly-detection",
            "data_version": "v1.0",
            "anomaly_types_detected": "spike,degradation,correlated",
            "f1_score": str(round(f1, 4)),
            "roc_auc": str(round(roc_auc, 4)),
            "framework": _detect_framework(model_type),
        }

        for tag_key, tag_value in tags.items():
            client.set_model_version_tag(MODEL_NAME, v.version, tag_key, tag_value)

        print(f"    ✅ Tagged with {len(tags)} metadata fields")
        print()

    # ── Update model description ──
    description = (
        "Network anomaly detection model for telecom infrastructure. "
        "Detects spike, degradation, and correlated anomalies in network "
        "metrics (latency, packet loss, CPU, bandwidth, error rate). "
        f"Total versions: {len(versions)}."
    )

    try:
        client.update_registered_model(MODEL_NAME, description=description)
        print(f"  ✅ Updated model description")
    except Exception as e:
        print(f"  ⚠️  Could not update description: {e}")

    print()


def _detect_framework(model_type: str) -> str:
    """Map model type to framework name."""
    mapping = {
        "IsolationForest": "scikit-learn",
        "XGBoost": "xgboost",
        "Autoencoder": "pytorch",
    }
    return mapping.get(model_type, "unknown")


if __name__ == "__main__":
    main()
