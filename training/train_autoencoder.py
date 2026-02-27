"""
train_autoencoder.py — Deep Learning Anomaly Detection with Autoencoder

Trains a PyTorch autoencoder on NORMAL data only. At inference, high
reconstruction error indicates an anomaly.

Key features:
  - Checkpointing: saves model state every N epochs
  - Resume: can resume training from a checkpoint
  - MLflow tracking: logs loss curve, threshold, reconstruction error distribution
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow.models import infer_signature


# ─────────────────────────────────────────────
# Autoencoder Architecture
# ─────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Symmetric autoencoder for anomaly detection.
    Input → Encoder → Bottleneck → Decoder → Reconstruction
    """
    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super().__init__()
        mid_dim = (input_dim + hidden_dim) // 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mid_dim),
            nn.Dropout(0.2),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mid_dim),
            nn.Dropout(0.2),
            nn.Linear(mid_dim, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ─────────────────────────────────────────────
# Training + Checkpointing
# ─────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0
    for batch_x, in dataloader:
        batch_x = batch_x.to(device)
        output = model(batch_x)
        loss = criterion(output, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    """Validate and return average loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, in dataloader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_x)
            total_loss += loss.item() * len(batch_x)
    return total_loss / len(dataloader.dataset)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint for resume capability."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load checkpoint and return start epoch."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


# ─────────────────────────────────────────────
# Anomaly Scoring
# ─────────────────────────────────────────────

def compute_reconstruction_errors(model, X, device, batch_size=512):
    """Compute per-sample reconstruction error (MSE)."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for batch_x, in loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            mse = ((output - batch_x) ** 2).mean(dim=1)
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


def find_optimal_threshold(errors_normal, errors_all, y_true):
    """Find threshold that maximizes F1 score."""
    thresholds = np.percentile(errors_normal, np.arange(90, 100, 0.5))
    best_f1 = 0
    best_threshold = thresholds[0]

    for threshold in thresholds:
        y_pred = (errors_all > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def save_loss_curve(train_losses, val_losses, path):
    """Save training/validation loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    ax.plot(val_losses, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def save_error_distribution(errors_normal, errors_anomaly, threshold, path):
    """Save reconstruction error distribution for normal vs anomalous samples."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors_normal, bins=100, alpha=0.7, label="Normal", color="#4FC3F7", density=True)
    ax.hist(errors_anomaly, bins=100, alpha=0.7, label="Anomaly", color="#EF5350", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder anomaly detector")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--features_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_path = args.features_path or str(project_root / "data" / "features" / "features.parquet")
    if not os.path.exists(features_path):
        print(f"❌ Features not found at {features_path}")
        print(f"Run 'python features/feature_engineering.py' first.")
        sys.exit(1)

    # ── Load data ──
    df = pd.read_parquet(features_path)
    exclude = {"timestamp", "node_id", "is_anomaly", "anomaly_type"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X = df[feature_cols].values
    y = df["is_anomaly"].values

    # Split: train uses ONLY normal data, test uses all data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Filter to normal-only for autoencoder training
    normal_mask = y_train_full == 0
    X_train_normal = X_train_full[normal_mask]

    # Further split normal data into train/val
    X_train, X_val = train_test_split(
        X_train_normal, test_size=0.15, random_state=args.random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_normal_scaled = scaler.transform(X_train_normal)

    input_dim = X_train_scaled.shape[1]

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Autoencoder Anomaly Detector                   ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Device       : {str(device):<32}║")
    print(f"║  Input dim    : {input_dim:<32}║")
    print(f"║  Hidden dim   : {args.hidden_dim:<32}║")
    print(f"║  Train (normal): {len(X_train):>8,}{' ' * 23}║")
    print(f"║  Validation   : {len(X_val):>8,}{' ' * 23}║")
    print(f"║  Test (all)   : {len(X_test):>8,}{' ' * 23}║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # ── Data loaders ──
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Model ──
    model = Autoencoder(input_dim, args.hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch, last_loss = load_checkpoint(model, optimizer, args.resume)
        print(f"  ↩️  Resumed from epoch {start_epoch} (loss: {last_loss:.6f})")
        start_epoch += 1  # Start from next epoch

    # ── MLflow tracking ──
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("network-anomaly-detection")

    with mlflow.start_run(run_name="autoencoder") as run:
        mlflow.log_param("model_type", "Autoencoder")
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("hidden_dim", args.hidden_dim)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("device", str(device))
        mlflow.log_param("train_normal_size", len(X_train))
        mlflow.log_param("resumed_from_epoch", start_epoch)

        # ── Training loop ──
        train_losses = []
        val_losses = []
        checkpoint_dir = str(project_root / "outputs" / "autoencoder" / "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("  Training...")
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:>3}/{args.epochs}  "
                      f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

            # Checkpoint
            if (epoch + 1) % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
                mlflow.log_artifact(ckpt_path)
                print(f"    💾 Checkpoint saved: epoch {epoch+1}")

        # ── Find optimal anomaly threshold ──
        print("\n  Computing reconstruction errors...")
        errors_train_normal = compute_reconstruction_errors(
            model, X_train_normal_scaled, device
        )
        errors_test = compute_reconstruction_errors(model, X_test_scaled, device)

        threshold, best_f1 = find_optimal_threshold(errors_train_normal, errors_test, y_test)
        y_pred = (errors_test > threshold).astype(int)

        mlflow.log_param("anomaly_threshold", threshold)

        # ── Metrics ──
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, errors_test)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_loss", val_losses[-1])

        print(f"\n  Threshold : {threshold:.6f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC AUC   : {roc_auc:.4f}")

        # ── Artifacts ──
        artifacts_dir = str(project_root / "outputs" / "autoencoder")
        os.makedirs(artifacts_dir, exist_ok=True)

        loss_path = os.path.join(artifacts_dir, "loss_curve.png")
        save_loss_curve(train_losses, val_losses, loss_path)
        mlflow.log_artifact(loss_path)

        errors_normal_test = errors_test[y_test == 0]
        errors_anomaly_test = errors_test[y_test == 1]
        dist_path = os.path.join(artifacts_dir, "error_distribution.png")
        save_error_distribution(errors_normal_test, errors_anomaly_test, threshold, dist_path)
        mlflow.log_artifact(dist_path)

        report = classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"])
        report_path = os.path.join(artifacts_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        print(f"\n{report}")

        # ── Log model ──
        # Note: We log the scaler parameters alongside for serving
        mlflow.log_param("scaler_mean", scaler.mean_.tolist()[:5])  # First 5 for readability
        mlflow.log_param("scaler_scale", scaler.scale_.tolist()[:5])

        # Save scaler for serving
        import pickle
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path)

        # Save threshold
        import json
        config_path = os.path.join(artifacts_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"threshold": float(threshold), "hidden_dim": args.hidden_dim}, f)
        mlflow.log_artifact(config_path)

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="network-anomaly-detector",
        )

        print(f"\n  ✅ Run ID: {run.info.run_id}")
        print(f"  ✅ Model registered as 'network-anomaly-detector'")
        print(f"  ✅ Checkpoints saved to: {checkpoint_dir}")
        print()


if __name__ == "__main__":
    main()
