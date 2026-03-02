# Telecom Network Anomaly Detection

End-to-end MLOps project for detecting anomalies in telecom network metrics.

## Objectives Covered

| # | Objective |
|---|-----------|
| 1 | Data & Feature Infrastructure |
| 2 | Training & Experiment Infrastructure |
| 3 | Model Artifact & Registry |
| 4 | Serving & Ops |
| 5 | Post-Deploy Monitoring |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data
python data/generate_data.py

# 4. Compute features
python features/feature_engineering.py

# 5. Train models
python training/train_isolation_forest.py
python training/train_xgboost.py
python training/train_autoencoder.py

# 6. Optimize hyperparameters
python training/optimize_hyperparams.py

# 7. Register & promote models
python registry/register_model.py
python registry/compare_models.py
python registry/promote_model.py

# 8. Serve
uvicorn serving.serve:app --port 8080

# 9. Monitor
python monitoring/drift_detector.py
```

## Project Structure

```
├── data/                    # Synthetic data generation
├── features/                # Feature engineering pipelines
├── training/                # Model training scripts
├── registry/                # Model registry management
├── serving/                 # FastAPI model serving
├── monitoring/              # Drift detection & rollback
├── docker/                  # Containerization
├── notebooks/               # Exploration notebooks
└── MLproject                # MLflow project definition
```

## Models

| Model | Type | Use Case |
|-------|------|----------|
| Isolation Forest | Unsupervised | No labels needed, general anomaly detection |
| XGBoost | Supervised | Uses labeled data, best precision |
| Autoencoder | Deep Learning | Learns normal patterns, detects novel anomalies |

## Tech Stack

MLflow · scikit-learn · XGBoost · PyTorch · Optuna · Feast · DVC · Evidently · FastAPI · Docker
