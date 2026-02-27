"""
promote_model.py — Model Promotion Pipeline

Promotes the best-performing model version through stages:
  None → Staging → Production

Supports:
  - Automatic best-model selection (by F1 score)
  - Manual version specification
  - Archiving previous production versions

Usage:
    python registry/promote_model.py                        # Auto-select best
    python registry/promote_model.py --version 3 --stage Production  # Manual
"""
import argparse
import mlflow
from mlflow.tracking import MlflowClient


def parse_args():
    parser = argparse.ArgumentParser(description="Promote model versions")
    parser.add_argument("--version", type=str, default=None,
                        help="Specific version to promote (default: auto-select best)")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["Staging", "Production"],
                        help="Target stage (default: auto-advance)")
    parser.add_argument("--archive", action="store_true", default=True,
                        help="Archive existing versions in target stage")
    return parser.parse_args()


def main():
    args = parse_args()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    MODEL_NAME = "network-anomaly-detector"

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Model Promotion Pipeline                       ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # ── Fetch all versions ──
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        print(f"  ❌ Model '{MODEL_NAME}' not found.")
        return

    # Show current state
    print(f"  Current state of '{MODEL_NAME}':")
    for v in versions:
        run = client.get_run(v.run_id)
        model_type = run.data.params.get("model_type", "?")
        f1 = run.data.metrics.get("f1_score", 0)
        print(f"    v{v.version} [{v.current_stage:<12}] {model_type:<20} F1={f1:.4f}")
    print()

    # ── Determine which version to promote ──
    if args.version:
        target_version = args.version
    else:
        # Auto-select: find version with highest F1 that's not yet in Production
        best_f1 = -1
        target_version = None
        for v in versions:
            if v.current_stage == "Production":
                continue
            run = client.get_run(v.run_id)
            f1 = run.data.metrics.get("f1_score", 0)
            if f1 > best_f1:
                best_f1 = f1
                target_version = v.version

        if target_version is None:
            print("  ✅ All models are already in Production or no models available")
            return

    # ── Determine target stage ──
    current_version_info = None
    for v in versions:
        if str(v.version) == str(target_version):
            current_version_info = v
            break

    if current_version_info is None:
        print(f"  ❌ Version {target_version} not found")
        return

    if args.stage:
        target_stage = args.stage
    else:
        # Auto-advance: None → Staging → Production
        stage_map = {"None": "Staging", "Staging": "Production"}
        current_stage = current_version_info.current_stage
        target_stage = stage_map.get(current_stage, "Staging")

    # ── Promote ──
    print(f"  Promoting v{target_version}: {current_version_info.current_stage} → {target_stage}")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=target_version,
        stage=target_stage,
        archive_existing_versions=args.archive,
    )

    print(f"  ✅ Version {target_version} promoted to {target_stage}")

    if args.archive:
        print(f"  📦 Previous {target_stage} versions archived")

    # ── Show final state ──
    print(f"\n  Final state:")
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        run = client.get_run(v.run_id)
        model_type = run.data.params.get("model_type", "?")
        f1 = run.data.metrics.get("f1_score", 0)
        marker = " ← PROMOTED" if str(v.version) == str(target_version) else ""
        print(f"    v{v.version} [{v.current_stage:<12}] {model_type:<20} F1={f1:.4f}{marker}")
    print()


if __name__ == "__main__":
    main()
