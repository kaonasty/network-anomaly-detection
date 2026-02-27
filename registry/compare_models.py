"""
compare_models.py — Compare Model Versions via Registry API

Queries the MLflow Model Registry to compare all versions of the
anomaly detector, ranking them by F1 score and recommending which
version to promote.

Usage:
    python registry/compare_models.py
"""
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    MODEL_NAME = "network-anomaly-detector"

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Model Comparison — Registry API                ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # ── Fetch all versions ──
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        print(f"  ❌ Model '{MODEL_NAME}' not found.")
        return

    if not versions:
        print(f"  ❌ No versions found.")
        return

    # ── Build comparison table ──
    rows = []
    for v in versions:
        run = client.get_run(v.run_id)
        params = run.data.params
        metrics = run.data.metrics

        rows.append({
            "Version": v.version,
            "Stage": v.current_stage,
            "Model Type": params.get("model_type", "?"),
            "F1": metrics.get("f1_score", 0),
            "ROC AUC": metrics.get("roc_auc", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "Run ID": v.run_id[:12],
        })

    df = pd.DataFrame(rows).sort_values("F1", ascending=False)

    # ── Display comparison ──
    print("  Model Version Comparison (ranked by F1 score):\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ── Recommendation ──
    best = df.iloc[0]
    print(f"  ───────────────────────────────────────────────")
    print(f"  🏆 Recommended for Production:")
    print(f"     Version    : {best['Version']}")
    print(f"     Model Type : {best['Model Type']}")
    print(f"     F1 Score   : {best['F1']:.4f}")
    print(f"     ROC AUC    : {best['ROC AUC']:.4f}")
    print()

    # ── Check current production ──
    production = [v for v in versions if v.current_stage == "Production"]
    if production:
        prod_run = client.get_run(production[0].run_id)
        prod_f1 = prod_run.data.metrics.get("f1_score", 0)
        if best["F1"] > prod_f1:
            improvement = (best["F1"] - prod_f1) / prod_f1 * 100 if prod_f1 > 0 else float("inf")
            print(f"  📈 This is {improvement:.1f}% better than current production (v{production[0].version})")
        else:
            print(f"  ✅ Current production model (v{production[0].version}) is already the best")
    else:
        print(f"  ⚠️  No model currently in Production stage")
        print(f"     Run 'python registry/promote_model.py' to promote")

    print()


if __name__ == "__main__":
    main()
