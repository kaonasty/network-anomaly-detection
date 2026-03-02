"""
dvc_versioning_demo.py — Data Versioning with DVC

Demonstrates DVC (Data Version Control) for data versioning:
  1. Show current data version (DVC hash)
  2. Regenerate data with different params (different anomaly rate)
  3. Track the new version with DVC
  4. Show version history and how to switch between versions
  5. Log data version in MLflow for lineage tracking

This script covers the learning objectives:
  - "Able to use data versioning tools for reproducibility"
  - "Understanding data versioning and lineage tracking"

Prerequisites:
    - DVC initialized: dvc init
    - Data tracked: dvc add data/raw/network_metrics.csv
    - Git repo initialized (DVC works alongside Git)

Usage:
    python features/dvc_versioning_demo.py
    python features/dvc_versioning_demo.py --new-version --anomaly-rate 0.10
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DVC_EXE = str(Path(sys.executable).parent / "dvc.exe")


def get_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file (same method DVC uses)."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def read_dvc_file(dvc_path: str) -> dict:
    """Read a .dvc file and extract the tracked file info."""
    import yaml
    if not os.path.exists(dvc_path):
        return {}
    with open(dvc_path) as f:
        return yaml.safe_load(f)


def show_current_version():
    """Step 1: Show current data version."""
    print("  Step 1: Current Data Versions\n")

    dvc_files = [
        ("Raw data", "data/raw/network_metrics.csv.dvc", "data/raw/network_metrics.csv"),
        ("Features", "data/features/features.parquet.dvc", "data/features/features.parquet"),
    ]

    for label, dvc_path, data_path in dvc_files:
        full_dvc = PROJECT_ROOT / dvc_path
        full_data = PROJECT_ROOT / data_path

        if full_dvc.exists():
            dvc_info = read_dvc_file(str(full_dvc))
            outs = dvc_info.get("outs", [{}])
            md5 = outs[0].get("md5", "unknown") if outs else "unknown"
            size = outs[0].get("size", 0) if outs else 0
            size_mb = size / (1024 * 1024) if size else 0

            print(f"    {label}:")
            print(f"      DVC file  : {dvc_path}")
            print(f"      MD5 hash  : {md5}")
            print(f"      Size      : {size_mb:.1f} MB")
        else:
            print(f"    {label}: ❌ Not tracked by DVC yet")
            print(f"      Run: dvc add {data_path}")

        print()


def create_new_version(anomaly_rate: float):
    """Step 2: Generate a new data version with different parameters."""
    print(f"  Step 2: Creating new data version (anomaly_rate={anomaly_rate})\n")

    python_exe = sys.executable

    # Generate new data
    print("    Regenerating data...")
    result = subprocess.run(
        [python_exe, "data/generate_data.py", "--anomaly_rate", str(anomaly_rate)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        print(f"    ❌ Data generation failed: {result.stderr}")
        return False
    print("    ✅ Data regenerated")

    # Recompute features
    print("    Recomputing features...")
    result = subprocess.run(
        [python_exe, "features/feature_engineering.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if result.returncode != 0:
        print(f"    ❌ Feature computation failed: {result.stderr}")
        return False
    print("    ✅ Features recomputed")

    # Track with DVC
    print("    Updating DVC tracking...")
    for data_path in ["data/raw/network_metrics.csv", "data/features/features.parquet"]:
        result = subprocess.run(
            [DVC_EXE, "add", data_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"    ✅ DVC updated: {data_path}")
        else:
            print(f"    ⚠️  DVC update: {result.stderr.strip()}")

    print()
    print("    To commit this version to Git:")
    print("      git add data/raw/network_metrics.csv.dvc data/features/features.parquet.dvc")
    print(f'      git commit -m "data: anomaly_rate={anomaly_rate}"')
    print()

    return True


def show_dvc_status():
    """Step 3: Show DVC status."""
    print("  Step 3: DVC Status\n")

    result = subprocess.run(
        [DVC_EXE, "status"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True,
    )

    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    else:
        print("    ✅ All data files are up to date with DVC")

    print()


def show_switching_versions():
    """Step 4: Show how to switch between data versions."""
    print("  Step 4: Switching Between Data Versions\n")
    print("    DVC works alongside Git. To switch data versions:\n")
    print("    # See all data versions (Git commits that changed .dvc files)")
    print("    git log --oneline -- data/raw/network_metrics.csv.dvc\n")
    print("    # Switch to a previous data version")
    print("    git checkout <commit-hash> -- data/raw/network_metrics.csv.dvc")
    print("    dvc checkout\n")
    print("    # Switch back to latest")
    print("    git checkout main -- data/raw/network_metrics.csv.dvc")
    print("    dvc checkout\n")
    print("    This ensures reproducibility: you can always go back to")
    print("    the exact data that was used for any training run.\n")


def show_lineage_tracking():
    """Step 5: Show how to log data version in MLflow for lineage."""
    print("  Step 5: Data Lineage with MLflow\n")
    print("    To track which data version was used for each model:\n")
    print('    ```python')
    print('    import mlflow')
    print('    ')
    print('    # Read DVC hash as the data version identifier')
    print('    with open("data/raw/network_metrics.csv.dvc") as f:')
    print('        import yaml')
    print('        dvc_info = yaml.safe_load(f)')
    print('        data_version = dvc_info["outs"][0]["md5"]')
    print('    ')
    print('    with mlflow.start_run():')
    print('        mlflow.log_param("data_version", data_version)')
    print('        mlflow.set_tag("data.raw_hash", data_version)')
    print('        mlflow.set_tag("data.anomaly_rate", "0.05")')
    print('        # ... train model ...')
    print('    ```')
    print()
    print("    This creates a complete lineage chain:")
    print("    Raw Data (DVC hash) → Features → Model (MLflow run)")
    print("    You can always trace back: which data produced which model.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="DVC Data Versioning Demo")
    parser.add_argument("--new-version", action="store_true",
                        help="Generate a new data version")
    parser.add_argument("--anomaly-rate", type=float, default=0.10,
                        help="Anomaly rate for new version (default: 0.10)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  DVC Data Versioning Demo                       ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # Step 1: Show current versions
    show_current_version()

    # Step 2: Optionally create new version
    if args.new_version:
        create_new_version(args.anomaly_rate)

    # Step 3: DVC status
    show_dvc_status()

    # Step 4: How to switch versions
    show_switching_versions()

    # Step 5: Lineage tracking
    show_lineage_tracking()

    print(f"  ───────────────────────────────────────────────")
    print(f"  DVC concepts demonstrated:")
    print(f"    ✅ Data version tracking (dvc add)")
    print(f"    ✅ Version identification (MD5 hashes)")
    print(f"    ✅ Version switching (git checkout + dvc checkout)")
    print(f"    ✅ Data lineage (DVC hash → MLflow param)")
    print()


if __name__ == "__main__":
    main()
