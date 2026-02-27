"""
train_xgboost.py — Supervised Anomaly Detection with XGBoost

Trains a binary XGBoost classifier using labeled anomaly data.
Handles class imbalance with scale_pos_weight.

MLflow tracking:
  - Parameters: all XGBoost hyperparameters
  - Metrics: precision, recall, f1, roc_auc, pr_auc, accuracy
  - Artifacts: confusion matrix, feature importance, ROC curve
  - Model: logged with signature for serving
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature


def load_features(features_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load features and return DataFrame + feature column names."""
    df = pd.read_parquet(features_path)
    exclude = {"timestamp", "node_id", "is_anomaly", "anomaly_type"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def save_confusion_matrix(y_true, y_pred, path: str):
    """Save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — XGBoost")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def save_roc_curve(y_true, y_scores, roc_auc, path: str):
    """Save ROC curve as PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def save_feature_importance(model, feature_cols, path: str, top_n: int = 20):
    """Save top-N feature importance plot as PNG."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), importances[indices], color="#4FC3F7")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"Top {top_n} Feature Importances — XGBoost")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost anomaly classifier")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--min_child_weight", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--features_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    features_path = args.features_path or str(project_root / "data" / "features" / "features.parquet")
    if not os.path.exists(features_path):
        print(f"❌ Features not found at {features_path}")
        print(f"Run 'python features/feature_engineering.py' first.")
        sys.exit(1)

    # ── Load data ──
    df, feature_cols = load_features(features_path)
    X = df[feature_cols].values
    y = df["is_anomaly"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Handle class imbalance
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  XGBoost Anomaly Classifier                     ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Train size       : {len(X_train):>10,}                 ║")
    print(f"║  Test size        : {len(X_test):>10,}                 ║")
    print(f"║  Features         : {len(feature_cols):>10}                 ║")
    print(f"║  scale_pos_weight : {scale_pos_weight:>10.2f}                 ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # ── MLflow tracking ──
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("network-anomaly-detection")

    with mlflow.start_run(run_name="xgboost") as run:
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("min_child_weight", args.min_child_weight)
        mlflow.log_param("subsample", args.subsample)
        mlflow.log_param("colsample_bytree", args.colsample_bytree)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))

        # Train
        print("  Training XGBoost...")
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=args.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # ── Metrics ──
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC AUC   : {roc_auc:.4f}")
        print(f"  PR AUC    : {pr_auc:.4f}")

        # ── Artifacts ──
        artifacts_dir = str(project_root / "outputs" / "xgboost")
        os.makedirs(artifacts_dir, exist_ok=True)

        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        roc_path = os.path.join(artifacts_dir, "roc_curve.png")
        save_roc_curve(y_test, y_proba, roc_auc, roc_path)
        mlflow.log_artifact(roc_path)

        fi_path = os.path.join(artifacts_dir, "feature_importance.png")
        save_feature_importance(model, feature_cols, fi_path)
        mlflow.log_artifact(fi_path)

        report = classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"])
        report_path = os.path.join(artifacts_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        print(f"\n{report}")

        # ── Log model ──
        sample_input = pd.DataFrame(X_test[:5], columns=feature_cols)
        signature = infer_signature(sample_input, y_pred[:5])

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input.iloc[0:1],
            registered_model_name="network-anomaly-detector",
        )

        print(f"\n  ✅ Run ID: {run.info.run_id}")
        print(f"  ✅ Model registered as 'network-anomaly-detector'")
        print()


if __name__ == "__main__":
    main()
