"""
Feast Feature Store Definitions for Network Anomaly Detection

Defines feature views for telco network metrics:
  - network_metrics_hourly    : aggregated hourly features (offline)
  - network_metrics_realtime  : latest raw metrics per node (online)

Usage:
    # Apply definitions
    feast apply

    # Materialize features to online store
    feast materialize 2026-01-01T00:00:00 2026-01-31T00:00:00

    # Or use the Python API (see feature_store_demo.py)
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String


# ─────────────────────────────────────────────
# Entity — the "who" we're tracking
# ─────────────────────────────────────────────
network_node = Entity(
    name="node_id",
    description="Unique identifier for a network node (core, edge, access, transit)",
    join_keys=["node_id"],
)


# ─────────────────────────────────────────────
# Data Sources — where Feast reads feature data from
# ─────────────────────────────────────────────
import os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Raw metrics source
raw_metrics_source = FileSource(
    path=os.path.join(_PROJECT_ROOT, "data", "raw", "network_metrics.parquet"),
    timestamp_field="timestamp",
    description="Raw network telemetry at 1-minute intervals",
)

# Engineered features source
engineered_features_source = FileSource(
    path=os.path.join(_PROJECT_ROOT, "data", "features", "feast_features.parquet"),
    timestamp_field="timestamp",
    description="Engineered features with rolling stats, z-scores, cross-metrics",
)


# ─────────────────────────────────────────────
# Feature Views — what features are available
# ─────────────────────────────────────────────

# Raw metrics — the base telemetry
raw_metrics_view = FeatureView(
    name="network_raw_metrics",
    entities=[network_node],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="latency_ms", dtype=Float64),
        Field(name="packet_loss_pct", dtype=Float64),
        Field(name="cpu_utilization", dtype=Float64),
        Field(name="bandwidth_mbps", dtype=Float64),
        Field(name="error_rate", dtype=Float64),
    ],
    source=raw_metrics_source,
    description="Raw network metrics per node per minute",
    tags={
        "team": "ml-platform",
        "data_tier": "raw",
    },
    online=True,
)

# Engineered features — rolling stats, z-scores, etc.
engineered_features_view = FeatureView(
    name="network_engineered_features",
    entities=[network_node],
    ttl=timedelta(hours=24),
    schema=[
        # Rolling mean (5-minute window)
        Field(name="latency_ms_roll_mean_5min", dtype=Float64),
        Field(name="packet_loss_pct_roll_mean_5min", dtype=Float64),
        Field(name="cpu_utilization_roll_mean_5min", dtype=Float64),
        Field(name="bandwidth_mbps_roll_mean_5min", dtype=Float64),
        Field(name="error_rate_roll_mean_5min", dtype=Float64),
        # Rolling std (5-minute window)
        Field(name="latency_ms_roll_std_5min", dtype=Float64),
        Field(name="packet_loss_pct_roll_std_5min", dtype=Float64),
        Field(name="cpu_utilization_roll_std_5min", dtype=Float64),
        # Z-scores
        Field(name="latency_ms_zscore", dtype=Float64),
        Field(name="packet_loss_pct_zscore", dtype=Float64),
        Field(name="cpu_utilization_zscore", dtype=Float64),
        Field(name="bandwidth_mbps_zscore", dtype=Float64),
        Field(name="error_rate_zscore", dtype=Float64),
        # Rate of change
        Field(name="latency_ms_delta_1m", dtype=Float64),
        Field(name="packet_loss_pct_delta_1m", dtype=Float64),
        # Cross-metric
        Field(name="latency_x_loss", dtype=Float64),
        Field(name="cpu_bw_ratio", dtype=Float64),
        # Time features
        Field(name="hour", dtype=Int64),
        Field(name="is_business_hours", dtype=Int64),
    ],
    source=engineered_features_source,
    description="Engineered features: rolling stats, z-scores, rate-of-change, cross-metric",
    tags={
        "team": "ml-platform",
        "data_tier": "features",
    },
    online=True,
)
