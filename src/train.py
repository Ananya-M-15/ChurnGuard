"""Model training and experiment tracking pipeline for ChurnGuard.

Trains and evaluates Logistic Regression (baseline), Random Forest, and XGBoost
using 5-Fold Stratified Cross-Validation exclusively on the processed training set.
Tracks experiments, parameters, and metrics in MLflow, selects the champion model
based on PR-AUC, and serializes the winning pipeline to models/pipeline.joblib.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_COMPARISON_PATH,
    MODEL_PIPELINE_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
)
from src.features import create_model_pipeline, get_feature_columns, save_pipeline

# Ensure MLflow tracing agent hints are silenced
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_candidate_models() -> Dict[str, Any]:
    """Define candidate classification models with tuned hyperparameter configurations.

    Includes:
    1. Logistic Regression: Baseline linear model with balanced class weights.
    2. Random Forest: Nonlinear ensemble handling interactions with balanced weights.
    3. XGBoost: Gradient boosted decision trees using scale_pos_weight (~2.77).

    Returns:
        Dictionary mapping model names to instantiated classifier objects.
    """
    candidates = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=2.77,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }
    return candidates


def evaluate_cv_fold(
    pipeline: Any,
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    X_val_fold: pd.DataFrame,
    y_val_fold: pd.Series,
) -> Dict[str, float]:
    """Fit pipeline on fold training data and calculate validation metrics.

    Args:
        pipeline: Unfitted Scikit-Learn Pipeline.
        X_train_fold: Training fold feature matrix.
        y_train_fold: Training fold target series.
        X_val_fold: Validation fold feature matrix.
        y_val_fold: Validation fold target series.

    Returns:
        Dictionary containing metric values for the fold.
    """
    pipeline.fit(X_train_fold, y_train_fold)

    y_pred = pipeline.predict(X_val_fold)
    y_prob = pipeline.predict_proba(X_val_fold)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_val_fold, y_prob)),
        "pr_auc": float(average_precision_score(y_val_fold, y_prob)),
        "f1": float(f1_score(y_val_fold, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val_fold, y_pred, zero_division=0)),
        "precision": float(precision_score(y_val_fold, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_val_fold, y_pred)),
    }
    return metrics


def train_and_compare_models(
    train_path: str | None = None,
) -> Tuple[Any, str, str, pd.DataFrame]:
    """Execute 5-fold cross-validation on all candidate models, track in MLflow,
    and fit the champion pipeline on the entire training set.

    Args:
        train_path: Optional custom path to training Parquet dataset.

    Returns:
        Tuple of (fitted_champion_pipeline, champion_name, champion_run_id, comparison_df).
    """
    data_path = train_path if train_path is not None else str(TRAIN_DATA_PATH)
    logger.info("Loading training data from: %s", data_path)
    train_df = pd.read_parquet(data_path)

    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]

    num_cols, cat_cols = get_feature_columns(X)
    candidate_models = get_candidate_models()

    # Configure MLflow SQLite tracking backend
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(
        "MLflow tracking configured with URI: %s | Experiment: %s",
        MLFLOW_TRACKING_URI,
        MLFLOW_EXPERIMENT_NAME,
    )

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    results_summary: List[Dict[str, Any]] = []
    fitted_pipelines: Dict[str, Any] = {}

    for model_name, classifier in candidate_models.items():
        logger.info("--- Starting 5-Fold Cross-Validation for [%s] ---", model_name)
        start_time = time.time()

        fold_metrics_list: List[Dict[str, float]] = []

        with mlflow.start_run(run_name=f"CV_{model_name}") as run:
            # Log hyperparameters
            params = classifier.get_params()
            clean_params = {
                k: str(v) for k, v in params.items() if isinstance(v, (int, float, str, bool))
            }
            clean_params["random_seed"] = str(RANDOM_SEED)
            clean_params["cv_folds"] = str(CV_FOLDS)
            mlflow.log_params(clean_params)
            mlflow.set_tag("model_type", model_name)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                # Create fresh pipeline per fold so preprocessor is never contaminated
                pipeline = create_model_pipeline(
                    classifier=classifier,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                )

                fold_metrics = evaluate_cv_fold(
                    pipeline, X_train_fold, y_train_fold, X_val_fold, y_val_fold
                )
                fold_metrics_list.append(fold_metrics)
                logger.debug("Fold %d/%d for %s: %s", fold_idx, CV_FOLDS, model_name, fold_metrics)

            training_time = time.time() - start_time

            # Compute mean and standard deviation across folds
            metrics_df = pd.DataFrame(fold_metrics_list)
            mean_metrics = metrics_df.mean().to_dict()
            std_metrics = metrics_df.std().to_dict()

            # Log aggregated metrics to MLflow
            for m_key, m_val in mean_metrics.items():
                mlflow.log_metric(f"cv_{m_key}_mean", float(m_val))
                mlflow.log_metric(f"cv_{m_key}_std", float(std_metrics[m_key]))
            mlflow.log_metric("training_duration_seconds", float(training_time))

            summary_row = {
                "model_name": model_name,
                "cv_pr_auc_mean": round(mean_metrics["pr_auc"], 4),
                "cv_pr_auc_std": round(std_metrics["pr_auc"], 4),
                "cv_roc_auc_mean": round(mean_metrics["roc_auc"], 4),
                "cv_roc_auc_std": round(std_metrics["roc_auc"], 4),
                "cv_f1_mean": round(mean_metrics["f1"], 4),
                "cv_f1_std": round(std_metrics["f1"], 4),
                "cv_recall_mean": round(mean_metrics["recall"], 4),
                "cv_recall_std": round(std_metrics["recall"], 4),
                "cv_precision_mean": round(mean_metrics["precision"], 4),
                "cv_accuracy_mean": round(mean_metrics["accuracy"], 4),
                "training_time_seconds": round(training_time, 2),
                "mlflow_cv_run_id": run.info.run_id,
            }
            results_summary.append(summary_row)

            logger.info(
                "[%s] CV Results: PR-AUC=%.4f (±%.4f) | ROC-AUC=%.4f (±%.4f) | F1=%.4f | Recall=%.4f (Duration: %.2fs)",
                model_name,
                mean_metrics["pr_auc"],
                std_metrics["pr_auc"],
                mean_metrics["roc_auc"],
                std_metrics["roc_auc"],
                mean_metrics["f1"],
                mean_metrics["recall"],
                training_time,
            )

    # 3. Model comparison & selection
    comparison_df = pd.DataFrame(results_summary)
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info("Saved model comparison table to: %s", MODEL_COMPARISON_PATH)

    # Selection Criterion: Primary: cv_pr_auc_mean, secondary: cv_f1_mean, cv_recall_mean
    sorted_df = comparison_df.sort_values(
        by=["cv_pr_auc_mean", "cv_f1_mean", "cv_recall_mean"],
        ascending=[False, False, False],
    )
    champion_name = sorted_df.iloc[0]["model_name"]
    champion_pr_auc = sorted_df.iloc[0]["cv_pr_auc_mean"]
    champion_roc_auc = sorted_df.iloc[0]["cv_roc_auc_mean"]

    logger.info(
        "🏆 Champion Model Selected: [%s] (PR-AUC=%.4f, ROC-AUC=%.4f) based on documented selection criteria.",
        champion_name,
        champion_pr_auc,
        champion_roc_auc,
    )

    # 4. Fit champion on the complete training dataset
    logger.info("Fitting champion [%s] pipeline on complete training dataset...", champion_name)
    champion_classifier = candidate_models[champion_name]
    champion_pipeline = create_model_pipeline(
        classifier=champion_classifier,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    champion_pipeline.fit(X, y)

    # 5. Serialize champion model artifact
    save_pipeline(champion_pipeline, MODEL_PIPELINE_PATH)

    # 6. Log champion run in MLflow
    with mlflow.start_run(run_name=f"Champion_{champion_name}") as champ_run:
        champ_run_id = champ_run.info.run_id
        mlflow.set_tag("role", "champion")
        mlflow.set_tag("champion_model", champion_name)
        mlflow.log_params(
            {
                "champion_model_name": champion_name,
                "selection_criterion": "highest_cv_pr_auc_with_f1_recall",
                "random_seed": str(RANDOM_SEED),
                "features_count": str(len(num_cols) + len(cat_cols)),
            }
        )
        mlflow.log_metrics(
            {
                "cv_pr_auc_champion": float(champion_pr_auc),
                "cv_roc_auc_champion": float(champion_roc_auc),
            }
        )

        # Log complete fitted pipeline artifact to MLflow
        try:
            mlflow.sklearn.log_model(
                sk_model=champion_pipeline,
                name="model",
                serialization_format="cloudpickle",
            )
            mlflow.log_artifact(str(MODEL_PIPELINE_PATH), artifact_path="serialized_pipeline")
            logger.info("Logged champion pipeline to MLflow under run ID: %s", champ_run_id)
        except Exception as exc:
            logger.warning("Could not log sklearn model directly to MLflow: %s", exc)

    # 7. Evaluate champion model on test set and link artifacts to champion run
    from src.evaluate import evaluate_champion_model
    evaluate_champion_model(
        model_path=MODEL_PIPELINE_PATH,
        mlflow_run_id=champ_run_id,
    )

    logger.info("--- Model Training Pipeline Finished Successfully ---")
    return champion_pipeline, champion_name, champ_run_id, comparison_df


if __name__ == "__main__":
    train_and_compare_models()
