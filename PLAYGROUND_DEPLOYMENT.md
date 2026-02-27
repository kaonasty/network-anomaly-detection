# 🚀 Playground Deployment Guide

Guide for deploying the Network Anomaly Detection project to the AI-ML Playground server.

## Prerequisites

- SSH access to the AI-ML Playground server
- Models trained and registered in MLflow
- Project tested locally

### Server Credentials

| Field | Value |
|-------|-------|
| Hostname | `172.16.254.189` |
| Port | `22` |
| Username | `apps` |
| Password | `apps#2025!` |

---

## 1. Copy Project to Server

From your **local PowerShell**:

```powershell
scp -r -P 22 C:\Users\tritr\Documents\Tritronik\network-anomaly-detection apps@172.16.254.189:~/
```

This uploads the entire project to `/home/apps/network-anomaly-detection`.

---

## 2. SSH into the Server

```powershell
ssh -o ServerAliveInterval=60 -p 22 apps@172.16.254.189
```

---

## 3. Setup Python Environment

```bash
cd ~/network-anomaly-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Run the Full Pipeline

```bash
# Step 1: Generate synthetic data
python data/generate_data.py

# Step 2: Compute features
python features/feature_engineering.py

# Step 3: Train models
python training/train_isolation_forest.py
python training/train_xgboost.py
python training/train_autoencoder.py

# Step 4: Optimize hyperparameters (optional, takes time)
python training/optimize_hyperparams.py --n_trials 20

# Step 5: Register and tag models
python registry/register_model.py

# Step 6: Compare and promote
python registry/compare_models.py
python registry/promote_model.py                    # None → Staging
python registry/promote_model.py                    # Staging → Production

# Step 7: View MLflow UI (from local PC via SSH tunnel)
# On local: ssh -L 5000:localhost:5000 -p 22 apps@172.16.254.189
# On server:
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

---

## 5. Deploy with Docker

### Option A: Direct Docker

```bash
cd ~/network-anomaly-detection

# Build
docker build -f docker/Dockerfile -t anomaly-detection:latest .

# Run standard
docker run -d \
    --name anomaly-api \
    -p 8080:8080 \
    -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
    -e MODEL_NAME=network-anomaly-detector \
    -e MODEL_STAGE=Production \
    -e FALLBACK_ENABLED=true \
    -v $(pwd)/mlflow.db:/app/mlflow.db \
    -v $(pwd)/mlruns:/app/mlruns \
    -v $(pwd)/data:/app/data \
    anomaly-detection:latest

# Run with A/B testing
docker run -d \
    --name anomaly-api-ab \
    -p 8081:8080 \
    -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
    -e MODEL_NAME=network-anomaly-detector \
    -e MODEL_STAGE=Production \
    -e AB_ENABLED=true \
    -e AB_SPLIT_RATIO=0.8 \
    -e CHALLENGER_STAGE=Staging \
    -v $(pwd)/mlflow.db:/app/mlflow.db \
    -v $(pwd)/mlruns:/app/mlruns \
    -v $(pwd)/data:/app/data \
    anomaly-detection:latest
```

### Option B: Docker Compose

```bash
cd ~/network-anomaly-detection/docker

# Standard deployment
docker compose up -d

# With A/B testing
docker compose --profile ab up -d
```

---

## 6. Test the API

```bash
# Health check
curl http://localhost:8080/health

# Single prediction (normal)
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"latency_ms":10,"packet_loss_pct":0.02,"cpu_utilization":40,"bandwidth_mbps":500,"error_rate":3}'

# Single prediction (anomaly)
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"latency_ms":800,"packet_loss_pct":15,"cpu_utilization":98,"bandwidth_mbps":50,"error_rate":100}'

# Batch prediction
curl -X POST http://localhost:8080/predict/batch \
    -H "Content-Type: application/json" \
    -d '{"records":[{"latency_ms":10,"packet_loss_pct":0.02,"cpu_utilization":40,"bandwidth_mbps":500,"error_rate":3},{"latency_ms":800,"packet_loss_pct":15,"cpu_utilization":98,"bandwidth_mbps":50,"error_rate":100}]}'
```

---

## 7. Run Monitoring

```bash
# Drift detection
python monitoring/drift_detector.py --drift_level moderate

# Performance monitoring
python monitoring/performance_monitor.py

# Check if rollback is needed
python monitoring/rollback.py --dry-run

# Force rollback (if needed)
python monitoring/rollback.py --force
```

---

## Container Management

```bash
# View logs
docker logs -f anomaly-api

# Restart (after model update)
docker restart anomaly-api

# Stop and remove
docker stop anomaly-api && docker rm anomaly-api
```

---

## Accessing MLflow UI from Local PC

```powershell
ssh -L 5000:localhost:5000 -p 22 apps@172.16.254.189
```

Then open → **http://localhost:5000**
