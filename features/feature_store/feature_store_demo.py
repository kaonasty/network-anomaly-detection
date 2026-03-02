"""
feature_store_demo.py — Feast Feature Store Demo

Demonstrates the full Feast workflow:
  1. Prepare data in Feast-compatible format
  2. Apply feature definitions to the registry
  3. Materialize features to the online store
  4. Retrieve features for training (offline) and serving (online)
  5. Feature discovery — list available features

This script covers the learning objectives:
  - "Able to use the feature store to discover, create, and retrieve features"

Usage:
    cd features/feature_store
    python feature_store_demo.py
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_feast_data():
    """
    Step 1: Convert our data into Feast-compatible Parquet format.
    Feast requires:
      - A timestamp column (datetime type)
      - Entity columns matching the Entity definition
      - Feature columns matching the FeatureView schema
    """
    print("  Step 1: Preparing data for Feast...\n")

    # ── Raw metrics ──
    raw_path = PROJECT_ROOT / "data" / "raw" / "network_metrics.csv"
    if not raw_path.exists():
        print(f"    ❌ Raw data not found at {raw_path}")
        print(f"    Run: python data/generate_data.py")
        return False

    raw_df = pd.read_csv(raw_path)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

    # Save as Parquet for Feast
    raw_parquet = PROJECT_ROOT / "data" / "raw" / "network_metrics.parquet"
    raw_df.to_parquet(raw_parquet, index=False, engine="pyarrow")
    print(f"    ✅ Raw metrics → {raw_parquet}")
    print(f"       {len(raw_df):,} rows, {len(raw_df.columns)} columns")

    # ── Engineered features ──
    features_path = PROJECT_ROOT / "data" / "features" / "features.parquet"
    if not features_path.exists():
        print(f"    ❌ Features not found at {features_path}")
        print(f"    Run: python features/feature_engineering.py")
        return False

    feat_df = pd.read_parquet(features_path)
    feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])

    # Save the Feast-compatible version (subset of columns Feast knows about)
    feast_cols = [
        "timestamp", "node_id",
        # Rolling means
        "latency_ms_roll_mean_5min", "packet_loss_pct_roll_mean_5min",
        "cpu_utilization_roll_mean_5min", "bandwidth_mbps_roll_mean_5min",
        "error_rate_roll_mean_5min",
        # Rolling std
        "latency_ms_roll_std_5min", "packet_loss_pct_roll_std_5min",
        "cpu_utilization_roll_std_5min",
        # Z-scores
        "latency_ms_zscore", "packet_loss_pct_zscore",
        "cpu_utilization_zscore", "bandwidth_mbps_zscore", "error_rate_zscore",
        # Rate of change
        "latency_ms_delta_1m", "packet_loss_pct_delta_1m",
        # Cross-metric
        "latency_x_loss", "cpu_bw_ratio",
        # Time
        "hour", "is_business_hours",
    ]
    feast_cols = [c for c in feast_cols if c in feat_df.columns]
    feast_df = feat_df[feast_cols].copy()

    feast_features_path = PROJECT_ROOT / "data" / "features" / "feast_features.parquet"
    feast_df.to_parquet(feast_features_path, index=False, engine="pyarrow")
    print(f"    ✅ Engineered features → {feast_features_path}")
    print(f"       {len(feast_df):,} rows, {len(feast_cols)} columns")
    print()

    return True


def apply_feature_store():
    """
    Step 2: Apply feature definitions to the Feast registry.
    This registers our entities and feature views.
    """
    from feast import FeatureStore
    from feature_definitions import (
        network_node,
        raw_metrics_view,
        engineered_features_view,
    )

    print("  Step 2: Applying feature store definitions...\n")

    store_path = str(Path(__file__).parent)
    store = FeatureStore(repo_path=store_path)

    # Register entities and feature views
    store.apply([
        network_node,
        raw_metrics_view,
        engineered_features_view,
    ])
    print("    ✅ Feature store definitions applied to registry")
    print(f"    📁 Registry: {store_path}")
    print(f"    📊 Registered: 1 entity, 2 feature views")
    print()

    return store


def discover_features(store):
    """
    Step 3: Feature Discovery — list all available features.
    This is how data scientists discover what features exist.
    """
    print("  Step 3: Feature Discovery\n")

    # List all feature views
    feature_views = store.list_feature_views()
    print(f"    Available Feature Views ({len(feature_views)}):")
    for fv in feature_views:
        print(f"\n    📊 {fv.name}")
        print(f"       Description: {fv.description}")
        print(f"       Entities: {[e.name for e in fv.entity_columns]}")
        print(f"       TTL: {fv.ttl}")
        print(f"       Tags: {fv.tags}")
        print(f"       Features ({len(fv.schema)}):")
        for feature in fv.schema:
            print(f"         - {feature.name} ({feature.dtype})")

    # List entities
    entities = store.list_entities()
    print(f"\n    Entities ({len(entities)}):")
    for entity in entities:
        print(f"      - {entity.name}: {entity.description}")

    print()
    return feature_views


def get_training_features(store):
    """
    Step 4: Retrieve features for TRAINING (offline store).
    Uses get_historical_features() — point-in-time join.
    """
    print("  Step 4: Get Training Features (Offline Store)\n")

    # Create an entity DataFrame — "which nodes at which times do I want features for?"
    entity_df = pd.DataFrame({
        "node_id": ["node-core-01", "node-edge-01", "node-access-01"] * 3,
        "timestamp": pd.to_datetime([
            "2026-01-15 10:00:00", "2026-01-15 10:00:00", "2026-01-15 10:00:00",
            "2026-01-15 14:00:00", "2026-01-15 14:00:00", "2026-01-15 14:00:00",
            "2026-01-20 08:00:00", "2026-01-20 08:00:00", "2026-01-20 08:00:00",
        ]),
    })

    print(f"    Entity DataFrame ({len(entity_df)} rows):")
    print(f"    {entity_df.to_string(index=False)}\n")

    # Get historical features — point-in-time correct!
    try:
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "network_raw_metrics:latency_ms",
                "network_raw_metrics:packet_loss_pct",
                "network_raw_metrics:cpu_utilization",
                "network_engineered_features:latency_ms_zscore",
                "network_engineered_features:cpu_bw_ratio",
            ],
        ).to_df()

        print(f"    ✅ Retrieved {len(training_df)} rows with {len(training_df.columns)} columns")
        print(f"\n    Training features sample:")
        print(f"    {training_df.head().to_string(index=False)}")
    except Exception as e:
        print(f"    ⚠️  Historical feature retrieval: {e}")
        print(f"    (This is expected if data timestamps don't align exactly)")

    print()


def get_serving_features(store):
    """
    Step 5: Retrieve features for SERVING (online store).
    Uses get_online_features() — latest feature values.
    Requires materialization first.
    """
    print("  Step 5: Get Serving Features (Online Store)\n")

    try:
        # Materialize features to the online store
        print("    Materializing features to online store...")
        store.materialize(
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 1, 31),
        )
        print("    ✅ Materialization complete\n")

        # Now retrieve online features
        online_features = store.get_online_features(
            features=[
                "network_raw_metrics:latency_ms",
                "network_raw_metrics:packet_loss_pct",
                "network_raw_metrics:cpu_utilization",
                "network_engineered_features:latency_ms_zscore",
            ],
            entity_rows=[
                {"node_id": "node-core-01"},
                {"node_id": "node-edge-01"},
                {"node_id": "node-access-01"},
            ],
        ).to_dict()

        print("    ✅ Online features retrieved:")
        for key, values in online_features.items():
            print(f"       {key}: {values}")

    except Exception as e:
        print(f"    ⚠️  Online feature retrieval: {e}")
        print(f"    (Materialization may need more time or data)")

    print()


def main():
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Feast Feature Store Demo                       ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # Step 1: Prepare data
    if not prepare_feast_data():
        return

    # Step 2: Apply definitions
    store = apply_feature_store()

    # Step 3: Discover features
    discover_features(store)

    # Step 4: Get training features (offline)
    get_training_features(store)

    # Step 5: Get serving features (online)
    get_serving_features(store)

    print(f"  ───────────────────────────────────────────────")
    print(f"  Feast feature store is now set up!")
    print(f"  Key concepts demonstrated:")
    print(f"    ✅ Feature definitions (entities, views, schemas)")
    print(f"    ✅ Feature discovery (list views / features)")
    print(f"    ✅ Offline retrieval (get_historical_features)")
    print(f"    ✅ Online retrieval (get_online_features)")
    print(f"    ✅ Materialization (offline → online)")
    print()


if __name__ == "__main__":
    main()
