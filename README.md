# Telecom Network Anomaly Detection — Full MLOps Project

An end-to-end MLOps project that detects anomalies in telecom network metrics using machine learning. This project demonstrates the complete ML lifecycle from data generation through model serving and post-deployment monitoring.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Phase 1: Data Generation](#phase-1-data-generation)
- [Phase 2: Data & Feature Infrastructure](#phase-2-data--feature-infrastructure)
  - [Batch Feature Engineering](#batch-feature-engineering)
  - [Streaming Feature Pipeline](#streaming-feature-pipeline)
  - [Feast Feature Store](#feast-feature-store)
  - [DVC Data Versioning](#dvc-data-versioning)
- [Phase 3: Training & Experiment Infrastructure](#phase-3-training--experiment-infrastructure)
  - [Isolation Forest (Unsupervised)](#1-isolation-forest-unsupervised)
  - [XGBoost (Supervised)](#2-xgboost-supervised)
  - [Autoencoder (Deep Learning)](#3-autoencoder-deep-learning)
  - [Hyperparameter Optimization](#4-hyperparameter-optimization-with-optuna)
- [Phase 4: Model Artifact & Registry](#phase-4-model-artifact--registry)
- [Phase 5: Serving & Ops](#phase-5-serving--ops)
- [Phase 6: Post-Deploy Monitoring](#phase-6-post-deploy-monitoring)
- [Tech Stack](#tech-stack)

---

## Overview

This project simulates a real-world scenario where we need to **detect anomalous behavior** in telecom network nodes — things like latency spikes, packet loss surges, CPU saturation, and correlated failures across metrics.

The dataset is **synthetic but realistic**: 10 network nodes generate telemetry at 1-minute intervals over 30 days, with ~5% injected anomalies of three types:

| Anomaly Type | Description | Duration |
|-------------|-------------|----------|
| **Spike** | Sudden sharp increase in 1-3 metrics | 1–5 minutes |
| **Degradation** | Gradual metric worsening over time | 15–60 minutes |
| **Correlated** | Multiple metrics degrade simultaneously | 5–30 minutes |

We train three different model types, compare them in the MLflow Model Registry, serve the best model through a REST API with A/B testing support, and monitor for data drift post-deployment.

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              DATA LAYER                     │
                        │                                             │
  generate_data.py ──►  │  data/raw/           data/features/         │
                        │  network_metrics.csv  features.parquet      │
                        │        │                    ▲                │
                        │        │   feature_engineering.py            │
                        │        └────────────────────┘                │
                        │                                             │
                        │  DVC tracks both files with .dvc hashes     │
                        │  Feast serves features via online store     │
                        └─────────────────────────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │           TRAINING LAYER                    │
                        │                                             │
                        │  train_isolation_forest.py  (unsupervised)  │
                        │  train_xgboost.py           (supervised)    │
                        │  train_autoencoder.py       (deep learning) │
                        │  optimize_hyperparams.py    (Optuna)        │
                        │                                             │
                        │  All runs tracked in MLflow with:           │
                        │  - Parameters, metrics, artifacts           │
                        │  - Model signatures, input examples         │
                        │  - Data version (DVC hash) for lineage      │
                        └─────────────────────────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │          REGISTRY LAYER                     │
                        │                                             │
                        │  register_model.py   → tag with metadata    │
                        │  compare_models.py   → rank by F1 score     │
                        │  promote_model.py    → None→Staging→Prod    │
                        │                                             │
                        │  Model: "network-anomaly-detector"          │
                        │  Stages: None | Staging | Production        │
                        └─────────────────────────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │          SERVING LAYER                      │
                        │                                             │
                        │  serve.py (FastAPI)                         │
                        │    POST /predict       ← single record      │
                        │    POST /predict/batch  ← multiple records   │
                        │    GET  /health         ← liveness probe    │
                        │                                             │
                        │  A/B routing: 80% primary / 20% challenger  │
                        │  Fallback: rule-based detector if ML fails  │
                        └─────────────────────────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │        MONITORING LAYER                     │
                        │                                             │
                        │  drift_detector.py       → Evidently / KS   │
                        │  performance_monitor.py  → latency SLAs     │
                        │  rollback.py             → auto-rollback    │
                        └─────────────────────────────────────────────┘
```

---

## Project Structure

```
network-anomaly-detection/
│
├── data/
│   ├── generate_data.py              # Synthetic data generator
│   ├── raw/
│   │   ├── network_metrics.csv       # Raw telemetry (432K rows, DVC-tracked)
│   │   └── network_metrics.csv.dvc   # DVC tracking file (commit this to Git)
│   └── features/
│       ├── features.parquet          # Engineered features (78 columns, DVC-tracked)
│       ├── features.parquet.dvc      # DVC tracking file
│       ├── feast_features.parquet    # Feast-compatible subset of features
│       └── feature_columns.txt       # List of all feature column names
│
├── features/
│   ├── feature_engineering.py        # Batch feature computation pipeline
│   ├── stream_features.py           # Simulated streaming pipeline
│   ├── dvc_versioning_demo.py        # DVC data versioning demo
│   └── feature_store/
│       ├── feature_store.yaml        # Feast configuration (local SQLite)
│       ├── feature_definitions.py    # Feast entity + feature view definitions
│       └── feature_store_demo.py     # End-to-end Feast workflow demo
│
├── training/
│   ├── train_isolation_forest.py     # Unsupervised anomaly detection
│   ├── train_xgboost.py             # Supervised binary classification
│   ├── train_autoencoder.py          # PyTorch deep learning + checkpointing
│   └── optimize_hyperparams.py       # Optuna hyperparameter search
│
├── registry/
│   ├── register_model.py             # Tag models with rich metadata
│   ├── compare_models.py            # Compare all model versions by metrics
│   └── promote_model.py             # Promote models through stages
│
├── serving/
│   ├── serve.py                     # FastAPI app with A/B routing
│   └── fallback.py                  # Rule-based fallback detector
│
├── monitoring/
│   ├── drift_detector.py            # Data drift detection (Evidently + KS-test)
│   ├── performance_monitor.py       # Prediction latency & SLA monitoring
│   └── rollback.py                  # Automated model rollback
│
├── docker/
│   ├── Dockerfile                   # Production container
│   └── docker-compose.yml           # Standard + A/B deployment profiles
│
├── MLproject                        # MLflow project entry points
├── requirements.txt                 # Python dependencies
├── PLAYGROUND_DEPLOYMENT.md         # Server deployment guide
└── .dvc/                            # DVC internal directory
```

---

## Quick Start

```bash
# 1. Activate the shared virtual environment
C:\Users\tritr\Documents\Tritronik\.venv\Scripts\activate

# 2. Navigate to project
cd C:\Users\tritr\Documents\Tritronik\network-anomaly-detection

# 3. Generate synthetic data (432K rows)
python data/generate_data.py

# 4. Compute features (78 features)
python features/feature_engineering.py

# 5. Train all three models
python training/train_isolation_forest.py
python training/train_xgboost.py
python training/train_autoencoder.py

# 6. Run hyperparameter optimization
python training/optimize_hyperparams.py --n_trials 20

# 7. Register, compare, and promote models
python registry/register_model.py
python registry/compare_models.py
python registry/promote_model.py       # None → Staging
python registry/promote_model.py       # Staging → Production

# 8. Start the API server
uvicorn serving.serve:app --port 8080

# 9. Run monitoring
python monitoring/drift_detector.py
python monitoring/performance_monitor.py

# 10. View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

> **Note**: On Windows PowerShell, prefix commands with `$env:PYTHONIOENCODING='utf-8'` to avoid Unicode encoding errors with the box-drawing characters in the output.

---

## Phase 1: Data Generation

### `data/generate_data.py`

Generates synthetic but realistic telecom network telemetry data.

**How it works:**

1. **Node Profiles** – 10 nodes are defined, each with unique baseline characteristics. Core nodes have low latency (~5ms) and high bandwidth (~950 Mbps), while access nodes have higher latency (~30ms) and lower bandwidth (~200 Mbps). This mirrors real telecom hierarchies:
   - `node-core-01/02` — backbone routers (lowest latency, highest bandwidth)
   - `node-edge-01/02/03` — edge aggregation (medium metrics)
   - `node-access-01/02/03` — last-mile access (highest latency)
   - `node-transit-01/02` — inter-network transit

2. **Seasonality** – Each metric follows a daily cycle (peaks from 10am–2pm, troughs at 3–5am) and a weekly cycle (slightly higher on weekdays). A random walk adds natural drift so the data isn't too "clean."

3. **Anomaly Injection** – About 5% of data points are anomalous, spread across the timeline:
   - **Spikes** (40%) — Sudden jump in 1–3 metrics for 1–5 minutes
   - **Degradation** (35%) — Gradual metric worsening over 15–60 minutes (like a memory leak or link degradation)
   - **Correlated** (25%) — All metrics degrade simultaneously (like a node overload)

**Output columns:**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | 1-minute intervals, 30 days |
| `node_id` | string | Which network node |
| `latency_ms` | float | Round-trip latency in milliseconds |
| `packet_loss_pct` | float | Percentage of dropped packets |
| `cpu_utilization` | float | CPU usage 0–100% |
| `bandwidth_mbps` | float | Throughput in Mbps |
| `error_rate` | float | Errors per minute |
| `is_anomaly` | int | Label: 0=normal, 1=anomaly |
| `anomaly_type` | string | none / spike / degradation / correlated |

**Run:**
```bash
python data/generate_data.py
python data/generate_data.py --num_nodes 5 --days 7 --anomaly_rate 0.10
```

---

## Phase 2: Data & Feature Infrastructure

This phase covers **feature engineering**, **feature stores**, and **data versioning** — the infrastructure that transforms raw telemetry into ML-ready features and keeps everything reproducible.

### Batch Feature Engineering

#### `features/feature_engineering.py`

Transforms raw metrics into 78 engineered features, processed per-node to ensure rolling calculations are correct.

**Feature categories:**

| Category | Count | Examples | Why it helps |
|----------|-------|---------|--------------|
| **Raw metrics** | 5 | `latency_ms`, `cpu_utilization` | Baseline values |
| **Rolling window** | 45 | `latency_ms_roll_mean_5min`, `cpu_roll_std_1hr` | Smooths noise, captures trends. Mean shows general trend, std shows volatility, max catches peaks |
| **Rate of change** | 15 | `latency_ms_delta_1m`, `packet_loss_abs_delta_1m` | First derivative — detects sudden jumps. A spike has a large positive delta |
| **Cross-metric** | 4 | `latency_x_loss`, `cpu_bw_ratio`, `latency_cpu_corr_30m` | Captures relationships between metrics. Correlated anomalies show up here |
| **Z-score** | 5 | `latency_ms_zscore`, `error_rate_zscore` | How many standard deviations from the rolling mean. Values > 3 are very unusual |
| **Time-based** | 4 | `hour`, `day_of_week`, `is_business_hours`, `is_weekend` | Network behavior varies by time of day and day of week |

**How rolling features work:**

```
Raw latency:     [10, 12, 11, 150, 160, 12, 10]
                                ↑     ↑
                            anomaly!
                            
roll_mean_5min:  [10, 11, 11, 45.8, 68.6, 69.0, 68.4]
                                ↑ jumps up — trend detected

roll_std_5min:   [0,  1,  0.8, 60.2, 70.1, 62.3, 62.0]
                                ↑ high std — volatility detected

zscore:          [0,  0.5, 0.2, 8.5, 9.2, 0.1, -0.1]
                                ↑ z=8.5 — extreme outlier!
```

**Run:**
```bash
python features/feature_engineering.py
python features/feature_engineering.py --input path/to/data.csv --output_dir path/to/output
```

### Streaming Feature Pipeline

#### `features/stream_features.py`

Simulates a real-time streaming pipeline — processes data **one record at a time** using an in-memory sliding window buffer (Python `deque`), mimicking what you'd build with Kafka Streams or Apache Flink in production.

**Key concept:** In production ML, you need features computed in real-time for inference. You can't run a batch Pandas pipeline on every incoming request. Instead, you maintain a sliding window buffer and update features incrementally as new data arrives.

```
Incoming data stream:  → [record 1] → [record 2] → [record 3] → ...
                              ↓              ↓              ↓
Sliding window buffer: [    record 1    ]
                       [record 1, record 2]
                       [record 1, record 2, record 3]
                              ↓
Real-time features:    mean, std, max, zscore, delta computed from buffer
```

**Run:**
```bash
python features/stream_features.py
python features/stream_features.py --node node-core-01 --limit 1000
```

### Feast Feature Store

The **Feature Store** is a centralized system for managing ML features. Without it, data scientists end up with duplicate feature logic scattered across notebooks, training scripts, and serving code. Feast solves this by providing:

1. **Feature Definitions** — A single source of truth for feature schemas
2. **Feature Discovery** — Data scientists can browse what features exist
3. **Offline Store** — Get historical features for training (point-in-time correct)
4. **Online Store** — Get latest feature values for real-time serving
5. **Materialization** — Move features from offline → online store

#### `features/feature_store/feature_store.yaml`

Feast configuration — we use the **local provider** with SQLite for both the registry and online store. In production, this would be a PostgreSQL registry + Redis/DynamoDB online store.

#### `features/feature_store/feature_definitions.py`

Defines the Feast schema:

- **Entity**: `network_node` — identifies which node we're getting features for (join key: `node_id`)
- **Feature View `network_raw_metrics`**: Raw telemetry (latency, loss, CPU, bandwidth, error rate), 24h TTL, sourced from the raw Parquet file
- **Feature View `network_engineered_features`**: Computed features (rolling stats, z-scores, cross-metrics), 24h TTL, sourced from the engineered features Parquet file

#### `features/feature_store/feature_store_demo.py`

End-to-end demo showing all five Feast capabilities:

```python
# 1. Apply definitions → register entity + feature views in Feast registry
store.apply([network_node, raw_metrics_view, engineered_features_view])

# 2. Discover features → list all available feature views and their schemas
feature_views = store.list_feature_views()

# 3. Get training features (offline) → point-in-time join
training_df = store.get_historical_features(
    entity_df=entity_df,  # "give me features for these nodes at these times"
    features=["network_raw_metrics:latency_ms", "network_engineered_features:latency_ms_zscore"]
).to_df()

# 4. Materialize → copy latest feature values to the online store
store.materialize(start_date=..., end_date=...)

# 5. Get serving features (online) → latest values for real-time inference
online_features = store.get_online_features(
    features=["network_raw_metrics:latency_ms"],
    entity_rows=[{"node_id": "node-core-01"}]
).to_dict()
```

**Run:**
```bash
python features/feature_store/feature_store_demo.py
```

### DVC Data Versioning

**DVC (Data Version Control)** extends Git to track large data files. Git tracks code; DVC tracks data. Together they give you full reproducibility.

**How it works:**

```
Git tracks                          DVC tracks (in .dvc/cache)
├── data/raw/network_metrics.csv.dvc  ← pointer file (MD5 hash)
│   contains: md5: 0dc80a75...       ← points to actual data
│   contains: size: 30.9 MB
│
├── data/features/features.parquet.dvc
│   contains: md5: 30bc1c7d...
│   contains: size: 143.9 MB
```

The `.dvc` files are tiny pointer files committed to Git. The actual data lives in `.dvc/cache/`. When you do `git checkout <old-commit>` + `dvc checkout`, DVC restores the exact data version from that commit.

**Data lineage** is achieved by logging the DVC hash as an MLflow parameter during training:

```python
# In training scripts, log data version for lineage tracking
import yaml
with open("data/raw/network_metrics.csv.dvc") as f:
    data_version = yaml.safe_load(f)["outs"][0]["md5"]
mlflow.log_param("data_version", data_version)  # e.g. "0dc80a75..."
```

Now every MLflow run knows which **exact dataset** it was trained on. You can always reproduce any experiment.

#### `features/dvc_versioning_demo.py`

Demonstrates:
- Viewing current data versions (MD5 hashes + sizes)
- Creating new data versions (regenerate with different anomaly rate)
- Switching between versions (`git checkout` + `dvc checkout`)
- Lineage tracking (DVC hash → MLflow param)

**Run:**
```bash
python features/dvc_versioning_demo.py
python features/dvc_versioning_demo.py --new-version --anomaly-rate 0.10
```

---

## Phase 3: Training & Experiment Infrastructure

All training scripts follow the same MLflow pattern:
1. Load features from Parquet
2. Split into train/test
3. Start an `mlflow.start_run()` context
4. Log all parameters with `mlflow.log_param()`
5. Train the model
6. Evaluate and log metrics with `mlflow.log_metric()`
7. Save artifacts (plots, reports) with `mlflow.log_artifact()`
8. Register the model with `mlflow.sklearn.log_model()` (or `.xgboost.` / `.pytorch.`)

### 1. Isolation Forest (Unsupervised)

#### `training/train_isolation_forest.py`

**Concept**: Isolation Forest detects anomalies **without using labels**. It works by randomly partitioning data — anomalies are "isolated" in fewer partitions because they're different from the majority. Points that are isolated quickly (short path length) are anomalous.

```
Normal data points:    clustered together → need many splits to isolate
Anomalous points:      far from the cluster → isolated in just 2-3 splits
```

**Why use it**: In real telco networks, you often don't have labeled anomaly data. Isolation Forest can detect anomalies without any labeled examples.

**MLflow logs**: contamination, n_estimators, precision, recall, F1, ROC AUC, confusion matrix

**Typical results**: ROC AUC ~0.87, F1 ~0.36 (recall that unsupervised methods trade off precision for coverage)

**Run:**
```bash
python training/train_isolation_forest.py
python training/train_isolation_forest.py --contamination 0.08 --n_estimators 300
```

### 2. XGBoost (Supervised)

#### `training/train_xgboost.py`

**Concept**: XGBoost is a gradient-boosted tree classifier. It uses labeled data (is_anomaly=0/1) to learn the exact boundary between normal and anomalous. Since anomalies are rare (~5%), we use `scale_pos_weight` to handle **class imbalance** — this tells XGBoost that misclassifying an anomaly is more costly than misclassifying a normal point.

```python
scale_pos_weight = n_negative / n_positive  # e.g. 15.78x weight on anomalies
```

**MLflow logs**: All XGBoost hyperparams, precision, recall, F1, ROC AUC, PR AUC, feature importance plot, ROC curve

**Typical results**: ROC AUC ~0.998, F1 ~0.84 — significantly better than Isolation Forest because it uses labels

**Run:**
```bash
python training/train_xgboost.py
python training/train_xgboost.py --n_estimators 300 --max_depth 8 --learning_rate 0.05
```

### 3. Autoencoder (Deep Learning)

#### `training/train_autoencoder.py`

**Concept**: A PyTorch autoencoder is trained on **normal data only**. It learns to compress and reconstruct normal network patterns. At inference, anomalies have **high reconstruction error** because the model has never seen anomalous patterns.

```
Normal data:    Input → [Encoder] → compressed → [Decoder] → Output ≈ Input  (low error)
Anomaly:        Input → [Encoder] → compressed → [Decoder] → Output ≠ Input  (HIGH error)
```

**Architecture**: `Input(78) → Linear(mid) → ReLU → BN → Dropout → Linear(8) → ReLU → Linear(mid) → ReLU → BN → Dropout → Linear(78)`

The bottleneck dimension (default 8) forces the model to learn a compressed representation of "normal."

**Checkpointing**: Saves model state every N epochs to `outputs/autoencoder/checkpoints/`. If training crashes, you can resume:

```bash
# Initial training
python training/train_autoencoder.py --epochs 50

# Resume from checkpoint
python training/train_autoencoder.py --epochs 100 --resume outputs/autoencoder/checkpoints/checkpoint_epoch_50.pt
```

**Threshold selection**: After training, the optimal anomaly threshold is found automatically by maximizing F1 score on the test set across percentiles of the normal reconstruction error distribution.

**MLflow logs**: Architecture params, loss curves (per epoch), reconstruction error distribution, threshold, precision, recall, F1, ROC AUC, all checkpoints as artifacts

### 4. Hyperparameter Optimization with Optuna

#### `training/optimize_hyperparams.py`

**Concept**: Instead of manually tuning XGBoost hyperparameters, Optuna runs an automated search. Each trial samples a different combination, trains a model, and logs the result. Optuna uses Bayesian optimization so later trials are smarter than random search.

**MLflow integration**: Each trial is a **nested (child) run** under a parent "optuna-study" run:

```
MLflow Experiment: network-anomaly-detection
├── optuna-study (parent run)
│   ├── trial-0  (child run) — F1=0.78
│   ├── trial-1  (child run) — F1=0.81
│   ├── trial-2  (child run) — F1=0.85  ← best
│   └── ...
│   └── trial-29 (child run)
```

**Hyperparameters searched** (9 total): n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda

**Run:**
```bash
python training/optimize_hyperparams.py --n_trials 30
```

---

## Phase 4: Model Artifact & Registry

The **MLflow Model Registry** is a centralized hub for managing model versions. It provides versioning, stage management, and metadata tagging.

### `registry/register_model.py`

Tags each model version with rich metadata:

| Tag | Example | Purpose |
|-----|---------|---------|
| `model_type` | IsolationForest | Which algorithm |
| `training_date` | 2026-03-02 | When it was trained |
| `author` | ml-team | Who trained it |
| `data_version` | v1.0 | Which dataset (links to DVC) |
| `anomaly_types_detected` | spike,degradation,correlated | Capabilities |
| `f1_score` | 0.8449 | Quick performance reference |
| `framework` | xgboost | Underlying ML framework |

### `registry/compare_models.py`

Queries the registry API to compare all model versions in a table:

```
  Version Stage        Model Type     F1     ROC AUC  Precision  Recall  Run ID
  2       None         XGBoost        0.8449 0.9982   0.7360     0.9917  dbc5933d
  1       None         IsolationForest 0.3587 0.8699   0.3897     0.3322  9bad1eaf
```

Recommends which version to promote based on F1 score.

### `registry/promote_model.py`

Manages the model lifecycle through stages:

```
None → Staging → Production
```

- **Auto-select mode** (default): Finds the best non-production model by F1 and advances it one stage
- **Manual mode**: `--version 2 --stage Production`
- **Archiving**: Previous production models are automatically archived

**Run:**
```bash
python registry/promote_model.py                              # Auto-advance best model
python registry/promote_model.py --version 2 --stage Production  # Manual promotion
```

---

## Phase 5: Serving & Ops

### `serving/serve.py`

A **FastAPI** application that loads the production model from MLflow and exposes REST endpoints.

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Score a single network metrics record |
| `POST` | `/predict/batch` | Score multiple records at once |
| `GET` | `/health` | Liveness probe with model + A/B test status |
| `GET` | `/model-info` | Detailed model metadata |
| `POST` | `/reload` | Hot-reload models without restarting |

**A/B Testing:**

When enabled (`AB_ENABLED=true`), traffic is split between two models:
- **Primary model** (Production stage) — gets `AB_SPLIT_RATIO` (default 80%) of traffic
- **Challenger model** (Staging stage) — gets the remaining 20%

This lets you test a new model version on real traffic before fully promoting it. The `/health` endpoint shows A/B stats (how many requests went to each model).

```bash
# Enable A/B testing via environment variables
AB_ENABLED=true AB_SPLIT_RATIO=0.8 CHALLENGER_STAGE=Staging uvicorn serving.serve:app --port 8080
```

**Fallback Strategy:**

If the ML model fails to load or times out, the server falls back to a rule-based detector (`serving/fallback.py`). This ensures the API never returns a 500 error — it degrades gracefully.

### `serving/fallback.py`

A configurable **rule-based anomaly detector** used as:
1. A fallback when ML models are unavailable
2. A baseline to compare ML model performance against

Rules are based on domain knowledge:

```python
# Examples of threshold rules
latency_ms      > 100   → warning
latency_ms      > 500   → critical
packet_loss_pct > 10.0  → critical
cpu_utilization > 95    → critical
```

**Test the API:**
```bash
# Health check
curl http://localhost:8080/health

# Normal data point
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"latency_ms":10, "packet_loss_pct":0.02, "cpu_utilization":40, "bandwidth_mbps":500, "error_rate":3}'

# Anomalous data point
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"latency_ms":800, "packet_loss_pct":15, "cpu_utilization":98, "bandwidth_mbps":50, "error_rate":100}'
```

---

## Phase 6: Post-Deploy Monitoring

After deploying a model to production, its performance can degrade over time. Monitoring detects this before it impacts users.

### `monitoring/drift_detector.py`

**Data drift** happens when the distribution of incoming (production) data differs from the training data. For example, if network traffic patterns change due to new customers or infrastructure upgrades, the model may become unreliable.

**How it works:**

1. Loads the reference (training) data and current (production) data
2. Runs statistical tests on each feature to detect distribution shifts:
   - **Evidently** (if installed) — generates an HTML drift report with KL divergence, Wasserstein distance, etc.
   - **KS-test fallback** — uses the Kolmogorov-Smirnov test (p-value < 0.05 = drift) and mean shift (> 1 standard deviation = drift)
3. Logs drift metrics to MLflow for tracking over time

**Simulated drift** for demo purposes:
```bash
python monitoring/drift_detector.py --drift_level none       # No drift — all features OK
python monitoring/drift_detector.py --drift_level moderate   # Latency +50%, packet loss ↑
python monitoring/drift_detector.py --drift_level severe     # All metrics shifted dramatically
```

### `monitoring/performance_monitor.py`

Tracks prediction performance metrics:
- **Latency percentiles**: P50, P95, P99
- **Anomaly rate**: What fraction of predictions are anomalies (if this spikes, something changed)
- **SLA check**: P99 latency must be < 100ms

All metrics are logged to an MLflow experiment named `network-anomaly-monitoring`.

```bash
python monitoring/performance_monitor.py
```

### `monitoring/rollback.py`

**Automated rollback** — if drift is detected OR performance degrades below thresholds, automatically switches the production model back to the previous version.

**Decision logic:**
1. Check drift status (`drift_summary.json` from drift detector)
2. Check performance thresholds (P99 latency, anomaly rate, error rate from performance monitor)
3. If either triggers, find the most recent archived model version and promote it to Production
4. Tag the rollback event on the new production model

```bash
python monitoring/rollback.py               # Check and rollback if needed
python monitoring/rollback.py --dry-run      # Check without actually rolling back
python monitoring/rollback.py --force        # Force rollback regardless of checks
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Experiment Tracking** | MLflow | Log parameters, metrics, artifacts, models |
| **Model Registry** | MLflow Model Registry | Version models, manage stages |
| **Feature Store** | Feast | Define, discover, retrieve features |
| **Data Versioning** | DVC | Track data file versions, reproducibility |
| **ML Algorithms** | scikit-learn, XGBoost, PyTorch | Model training |
| **Hyperparameter Tuning** | Optuna | Bayesian hyperparameter search |
| **Drift Detection** | Evidently / scipy | Distribution shift detection |
| **Model Serving** | FastAPI + Uvicorn | REST API |
| **Containerization** | Docker | Production deployment |
| **Data Processing** | Pandas, NumPy, PyArrow | Feature engineering |
