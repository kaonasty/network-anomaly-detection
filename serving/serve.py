"""
serve.py — FastAPI Model Serving with A/B Routing

Loads anomaly detection models from MLflow and exposes REST endpoints:
  GET  /health          — liveness probe + model info
  GET  /model-info      — detailed model metadata
  POST /predict         — score a single network record
  POST /predict/batch   — score multiple records

A/B Testing:
  Set AB_ENABLED=true and configure AB_SPLIT_RATIO (default 0.8).
  80% of traffic goes to primary (Production) model,
  20% goes to challenger (Staging) model.

Feature Flags:
  AB_ENABLED      — enable/disable A/B routing
  AB_SPLIT_RATIO  — fraction of traffic to primary model
  FALLBACK_ENABLED — enable rule-based fallback on model failure
"""
import os
import sys
import time
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "network-anomaly-detector")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
ENV_NAME = os.getenv("ENV_NAME", "production")

# A/B testing config
AB_ENABLED = os.getenv("AB_ENABLED", "false").lower() == "true"
AB_SPLIT_RATIO = float(os.getenv("AB_SPLIT_RATIO", "0.8"))  # primary model traffic fraction
CHALLENGER_STAGE = os.getenv("CHALLENGER_STAGE", "Staging")

# Fallback config
FALLBACK_ENABLED = os.getenv("FALLBACK_ENABLED", "true").lower() == "true"

# Feature names (must match training feature columns)
METRIC_NAMES = ["latency_ms", "packet_loss_pct", "cpu_utilization", "bandwidth_mbps", "error_rate"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomaly-serve")


# ─────────────────────────────────────────────
# Global Model State
# ─────────────────────────────────────────────
_primary_model = None
_primary_model_uri = ""
_challenger_model = None
_challenger_model_uri = ""
_loaded_at = 0.0
_feature_columns = None

# A/B tracking counters
_ab_stats = {"primary_count": 0, "challenger_count": 0, "fallback_count": 0}


def load_model(stage: str):
    """Load a model from MLflow registry by stage."""
    try:
        model_uri = f"models:/{MODEL_NAME}/{stage}"
        logger.info(f"Loading model from {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        return model, model_uri
    except Exception as e:
        logger.warning(f"Could not load {stage} model: {e}")
        return None, ""


def load_feature_columns():
    """Load feature column names from saved file."""
    project_root = Path(__file__).resolve().parent.parent
    feature_list = project_root / "data" / "features" / "feature_columns.txt"
    if feature_list.exists():
        with open(feature_list) as f:
            return [line.strip() for line in f if line.strip()]
    return None


def load_all_models():
    """Load primary and optionally challenger models."""
    global _primary_model, _primary_model_uri
    global _challenger_model, _challenger_model_uri
    global _loaded_at, _feature_columns

    _primary_model, _primary_model_uri = load_model(MODEL_STAGE)
    if _primary_model is None:
        logger.warning("Primary model not available")

    if AB_ENABLED:
        _challenger_model, _challenger_model_uri = load_model(CHALLENGER_STAGE)
        if _challenger_model:
            logger.info(f"A/B testing enabled: {AB_SPLIT_RATIO*100:.0f}% primary, "
                        f"{(1-AB_SPLIT_RATIO)*100:.0f}% challenger")
        else:
            logger.warning("Challenger model not available — all traffic to primary")

    _feature_columns = load_feature_columns()
    _loaded_at = time.time()


# ─────────────────────────────────────────────
# Rule-Based Fallback
# ─────────────────────────────────────────────

def fallback_predict(features: dict) -> dict:
    """
    Simple rule-based anomaly detection as fallback.
    Triggers if any metric exceeds predefined thresholds.
    """
    rules = {
        "latency_ms": lambda x: x > 200,
        "packet_loss_pct": lambda x: x > 5.0,
        "cpu_utilization": lambda x: x > 95,
        "error_rate": lambda x: x > 50,
    }

    triggered = []
    for metric, check in rules.items():
        if metric in features and check(features[metric]):
            triggered.append(metric)

    is_anomaly = len(triggered) > 0
    return {
        "is_anomaly": int(is_anomaly),
        "confidence": 1.0 if triggered else 0.0,
        "method": "rule-based-fallback",
        "triggered_rules": triggered,
    }


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    load_all_models()
    yield

app = FastAPI(
    title="Network Anomaly Detection API",
    description="Detects anomalies in telecom network metrics using ML models",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class NetworkMetrics(BaseModel):
    """Input schema: raw network metrics for a single observation."""
    latency_ms: float = Field(..., ge=0, example=15.5)
    packet_loss_pct: float = Field(..., ge=0, le=100, example=0.05)
    cpu_utilization: float = Field(..., ge=0, le=100, example=45.2)
    bandwidth_mbps: float = Field(..., ge=0, example=500.0)
    error_rate: float = Field(..., ge=0, example=3.0)


class BatchMetrics(BaseModel):
    """Input schema: batch of network metrics."""
    records: list[NetworkMetrics]


class PredictionResponse(BaseModel):
    """Output schema for single prediction."""
    is_anomaly: int
    confidence: float
    model_used: str
    model_uri: str
    environment: str
    method: str  # "ml-model" or "rule-based-fallback"


class BatchPredictionResponse(BaseModel):
    """Output schema for batch prediction."""
    predictions: list[PredictionResponse]
    total: int
    anomaly_count: int


class HealthResponse(BaseModel):
    status: str
    environment: str
    model_name: str
    primary_model: str
    challenger_model: Optional[str] = None
    ab_enabled: bool
    ab_split_ratio: float
    uptime_s: float
    ab_stats: dict


# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────

def predict_single(features: dict) -> PredictionResponse:
    """
    Predict anomaly for a single record with A/B routing and fallback.
    """
    global _ab_stats

    # Determine which model to use
    use_challenger = (
        AB_ENABLED
        and _challenger_model is not None
        and random.random() > AB_SPLIT_RATIO
    )

    model = _challenger_model if use_challenger else _primary_model
    model_uri = _challenger_model_uri if use_challenger else _primary_model_uri
    model_label = "challenger" if use_challenger else "primary"

    # Attempt ML prediction
    if model is not None:
        try:
            # Build input DataFrame with feature columns
            if _feature_columns:
                # Pad missing features with 0
                input_data = {col: features.get(col, 0.0) for col in _feature_columns}
                input_df = pd.DataFrame([input_data])
            else:
                input_df = pd.DataFrame([features])

            prediction = model.predict(input_df)
            pred_value = int(prediction[0])

            # Track A/B stats
            if use_challenger:
                _ab_stats["challenger_count"] += 1
            else:
                _ab_stats["primary_count"] += 1

            return PredictionResponse(
                is_anomaly=pred_value,
                confidence=0.9,  # Simplified; real implementation uses predict_proba
                model_used=model_label,
                model_uri=model_uri,
                environment=ENV_NAME,
                method="ml-model",
            )
        except Exception as e:
            logger.error(f"ML prediction failed ({model_label}): {e}")

    # Fallback to rule-based if ML fails
    if FALLBACK_ENABLED:
        _ab_stats["fallback_count"] += 1
        fb = fallback_predict(features)
        return PredictionResponse(
            is_anomaly=fb["is_anomaly"],
            confidence=fb["confidence"],
            model_used="fallback",
            model_uri="rule-based",
            environment=ENV_NAME,
            method="rule-based-fallback",
        )

    raise HTTPException(500, "No model available and fallback is disabled")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe with model and A/B routing status."""
    return HealthResponse(
        status="healthy" if _primary_model or FALLBACK_ENABLED else "degraded",
        environment=ENV_NAME,
        model_name=MODEL_NAME,
        primary_model=_primary_model_uri or "not loaded",
        challenger_model=_challenger_model_uri or None,
        ab_enabled=AB_ENABLED,
        ab_split_ratio=AB_SPLIT_RATIO,
        uptime_s=round(time.time() - _loaded_at, 1) if _loaded_at else 0,
        ab_stats=_ab_stats,
    )


@app.get("/model-info")
async def model_info():
    """Detailed model metadata."""
    info = {
        "primary": {
            "uri": _primary_model_uri or "not loaded",
            "loaded": _primary_model is not None,
        },
        "ab_testing": {
            "enabled": AB_ENABLED,
            "split_ratio": AB_SPLIT_RATIO,
            "challenger_uri": _challenger_model_uri or "not loaded",
            "challenger_loaded": _challenger_model is not None,
        },
        "fallback_enabled": FALLBACK_ENABLED,
        "feature_columns_count": len(_feature_columns) if _feature_columns else 0,
        "environment": ENV_NAME,
    }
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(metrics: NetworkMetrics):
    """
    Predict anomaly for a single network metrics record.

    Example:
    ```json
    {"latency_ms": 150, "packet_loss_pct": 5.0, "cpu_utilization": 88, "bandwidth_mbps": 100, "error_rate": 25}
    ```
    """
    features = metrics.model_dump()
    return predict_single(features)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchMetrics):
    """
    Predict anomalies for a batch of network metrics records.
    """
    predictions = []
    for record in batch.records:
        features = record.model_dump()
        pred = predict_single(features)
        predictions.append(pred)

    anomaly_count = sum(p.is_anomaly for p in predictions)

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        anomaly_count=anomaly_count,
    )


@app.post("/reload")
async def reload_models():
    """Reload models from MLflow registry (for model updates)."""
    load_all_models()
    return {
        "status": "reloaded",
        "primary": _primary_model_uri or "not loaded",
        "challenger": _challenger_model_uri or "not loaded",
    }


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("serving.serve:app", host="0.0.0.0", port=port, reload=False)
