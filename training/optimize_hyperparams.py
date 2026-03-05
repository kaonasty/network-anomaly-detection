"""
optimize_hyperparams.py — Hyperparameter Optimization with Optuna + MLflow

Runs an Optuna study to find optimal XGBoost hyperparameters.
Each trial is logged as an MLflow child run under a parent "optuna-study" run.

Usage:
    python training/optimize_hyperparams.py --n_trials 30
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import optuna
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from mlflow.models import infer_signature


def load_features(features_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load features and return DataFrame + feature column names."""
    df = pd.read_parquet(features_path)
    exclude = {"timestamp", "node_id", "is_anomaly", "anomaly_type"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=30)
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

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  Optuna Hyperparameter Optimization             ║")
    print(f"╠══════════════════════════════════════════════════╣")
    print(f"║  Trials      : {args.n_trials:<34}║")
    print(f"║  Train size  : {len(X_train):>10,}{' ' * 23}║")
    print(f"║  Test size   : {len(X_test):>10,}{' ' * 23}║")
    print(f"╚══════════════════════════════════════════════════╝")
    print()

    # ── MLflow setup ──
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("network-anomaly-detection")

    with mlflow.start_run(run_name="optuna-study") as parent_run:
        mlflow.log_param("optimizer", "optuna")
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("objective_metric", "f1_score")

        def objective(trial):
            """Optuna objective function — each trial is an MLflow child run."""
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            }

            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                # Log all parameters
                for k, v in params.items():
                    mlflow.log_param(k, v)
                mlflow.log_param("trial_number", trial.number)
                mlflow.log_param("scale_pos_weight", scale_pos_weight)

                # Train
                model = XGBClassifier(
                    **params,
                    scale_pos_weight=scale_pos_weight,
                    random_state=args.random_state,
                    eval_metric="logloss",
                    use_label_encoder=False,
                )
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

                # Evaluate
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_proba)

                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                print(f"  Trial {trial.number:>3} | F1={f1:.4f} | AUC={roc_auc:.4f}")

                return f1

        # ── Run Optuna study ──
        study = optuna.create_study(
            direction="maximize",
            study_name="xgboost-anomaly-detection",
            storage="sqlite:///optuna.db",   # ← add this
            load_if_exists=True,             # ← allows resuming previous study
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        # ── Log best results ──
        best = study.best_trial
        mlflow.log_metric("best_f1_score", best.value)
        mlflow.log_param("best_trial_number", best.number)

        for k, v in best.params.items():
            mlflow.log_param(f"best_{k}", v)

        print()
        print(f"  ───────────────────────────────────────────────")
        print(f"  Best trial  : #{best.number}")
        print(f"  Best F1     : {best.value:.4f}")
        print(f"  Best params :")
        for k, v in best.params.items():
            print(f"    {k:<20} = {v}")

        # ── Train final model with best params ──
        print(f"\n  Training final model with best hyperparameters...")
        best_model = XGBClassifier(
            **best.params,
            scale_pos_weight=scale_pos_weight,
            random_state=args.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred_final = best_model.predict(X_test)
        y_proba_final = best_model.predict_proba(X_test)[:, 1]

        final_f1 = f1_score(y_test, y_pred_final, zero_division=0)
        final_auc = roc_auc_score(y_test, y_proba_final)

        mlflow.log_metric("final_f1_score", final_f1)
        mlflow.log_metric("final_roc_auc", final_auc)

        # Register best model
        sample_input = pd.DataFrame(X_test[:5], columns=feature_cols)
        signature = infer_signature(sample_input, y_pred_final[:5])

        mlflow.xgboost.log_model(
            best_model,
            artifact_path="best_model",
            signature=signature,
            input_example=sample_input.iloc[0:1],
            registered_model_name="network-anomaly-detector",
        )

        print(f"\n  ✅ Study complete — {args.n_trials} trials")
        print(f"  ✅ Best model F1={final_f1:.4f}, AUC={final_auc:.4f}")
        print(f"  ✅ Run ID: {parent_run.info.run_id}")
        print()


if __name__ == "__main__":
    main()
