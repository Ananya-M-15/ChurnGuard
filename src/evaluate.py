"""Model evaluation and diagnostic reporting module for ChurnGuard.

Evaluates the champion fitted model pipeline on the untouched test dataset exactly
once, computes performance metrics (PR-AUC, ROC-AUC, F1, Recall, Precision, Accuracy),
generates publication-ready figures (Confusion Matrix, ROC Curve, PR Curve, Feature
Importance), logs artifacts to MLflow, and serializes metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    DEFAULT_THRESHOLD,
    EVALUATION_METRICS_PATH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.features import load_pipeline

# Ensure MLflow tracing agent hints are silenced
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Generate and save an annotated confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    labels = [
        f"{count}\n({pct:.1f}%)"
        for count, pct in zip(cm.flatten(), cm_pct.flatten())
    ]
    labels_matrix = np.asarray(labels).reshape(2, 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_title("Confusion Matrix — Test Set Evaluation", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("True Customer Status", fontsize=11, weight="bold")
    ax.set_xlabel("Predicted Customer Status", fontsize=11, weight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved confusion matrix figure to: %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    output_path: Path,
) -> None:
    """Generate and save the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Champion Model (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, weight="bold")
    ax.set_title("ROC Curve — Test Set Evaluation", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved ROC curve figure to: %s", output_path)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    output_path: Path,
) -> None:
    """Generate and save the Precision-Recall (PR) curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    no_skill = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(recall, precision, color="#2ca02c", lw=2.5, label=f"Champion Model (PR-AUC = {pr_auc:.4f})")
    ax.plot([0, 1], [no_skill, no_skill], color="#7f7f7f", lw=1.5, linestyle="--", label=f"No Skill Baseline ({no_skill:.2%})")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall (Coverage of Actual Churners)", fontsize=11, weight="bold")
    ax.set_ylabel("Precision (Accuracy of Churn Alerts)", fontsize=11, weight="bold")
    ax.set_title("Precision-Recall Curve — Test Set Evaluation", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Precision-Recall curve figure to: %s", output_path)


def plot_feature_importance(
    pipeline: Any,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Generate and save feature importance visualization for the champion model."""
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if classifier is None or preprocessor is None:
        logger.warning("Pipeline missing classifier or preprocessor step; skipping feature importance plot.")
        return

    importances = None
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])

    if importances is None:
        logger.warning("Classifier does not expose feature_importances_ or coef_; skipping plot.")
        return

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"Feature_{i}" for i in range(len(importances))])

    # Rank and select top N features
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values(by="importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", color="#1f77b4", ax=ax)
    ax.set_title(f"Top {top_n} Most Influential Features — Champion Model", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Relative Feature Importance", fontsize=11, weight="bold")
    ax.set_ylabel("Transformed Feature", fontsize=11, weight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved feature importance figure to: %s", output_path)


def evaluate_champion_model(
    model_path: Path | str | None = None,
    test_path: Path | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    mlflow_run_id: str | None = None,
) -> Dict[str, Any]:
    """Execute complete post-training evaluation on the untouched test partition.

    Args:
        model_path: Optional path to serialized pipeline. Defaults to MODEL_PIPELINE_PATH.
        test_path: Optional path to test parquet. Defaults to TEST_DATA_PATH.
        threshold: Decision classification threshold. Defaults to 0.5.
        mlflow_run_id: Optional MLflow champion run ID to append test artifacts.

    Returns:
        Dictionary containing verified evaluation metrics and artifact locations.
    """
    logger.info("--- Starting Champion Model Evaluation on Test Set ---")

    # 1. Load fitted champion pipeline
    pipe_path = Path(model_path) if model_path is not None else MODEL_PIPELINE_PATH
    pipeline = load_pipeline(pipe_path)
    classifier_name = pipeline.named_steps["classifier"].__class__.__name__

    # 2. Load untouched test data
    t_path = Path(test_path) if test_path is not None else TEST_DATA_PATH
    logger.info("Loading untouched test dataset from: %s", t_path)
    test_df = pd.read_parquet(t_path)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].to_numpy()

    # 3. Generate predictions using specified threshold
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # 4. Compute verified test metrics
    metrics: Dict[str, Any] = {
        "model_name": classifier_name,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "decision_threshold": float(threshold),
        "test_sample_count": int(len(y_test)),
        "test_churn_count": int(np.sum(y_test)),
        "test_churn_rate": float(np.mean(y_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
    }

    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["confusion_matrix"] = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    logger.info(
        "Verified Test Metrics for [%s]: PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f | Accuracy=%.4f",
        classifier_name,
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["f1"],
        metrics["recall"],
        metrics["precision"],
        metrics["accuracy"],
    )

    # 5. Generate and persist diagnostic figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    roc_path = FIGURES_DIR / "roc_curve.png"
    pr_path = FIGURES_DIR / "precision_recall_curve.png"
    fi_path = FIGURES_DIR / "feature_importance.png"

    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc_curve(y_test, y_prob, metrics["roc_auc"], roc_path)
    plot_precision_recall_curve(y_test, y_prob, metrics["pr_auc"], pr_path)
    plot_feature_importance(pipeline, fi_path, top_n=15)

    # 6. Save evaluation metrics JSON
    EVALUATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved verified evaluation metrics to: %s", EVALUATION_METRICS_PATH)

    # 7. Save comprehensive model metadata JSON
    metadata: Dict[str, Any] = {
        "model_name": classifier_name,
        "model_version": "1.0.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "dataset_info": {
            "train_dataset": "data/processed/train.parquet",
            "test_dataset": "data/processed/test.parquet",
            "train_rows": 5634,
            "test_rows": len(y_test),
            "total_input_features": X_test.shape[1],
            "feature_names": X_test.columns.tolist(),
        },
        "selected_threshold": float(threshold),
        "test_metrics": metrics,
        "mlflow_info": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "champion_run_id": mlflow_run_id or "not_provided",
        },
    }

    MODEL_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to: %s", MODEL_METADATA_PATH)

    # 8. Log evaluation artifacts to MLflow if run_id is active or available
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Open run or create evaluation child run
        active_run = mlflow.active_run()
        if active_run is None and mlflow_run_id is not None:
            context_run = mlflow.start_run(run_id=mlflow_run_id)
        elif active_run is None:
            context_run = mlflow.start_run(run_name=f"Evaluation_{classifier_name}")
        else:
            context_run = active_run

        with context_run:
            mlflow.log_metrics(
                {
                    "test_accuracy": metrics["accuracy"],
                    "test_precision": metrics["precision"],
                    "test_recall": metrics["recall"],
                    "test_f1": metrics["f1"],
                    "test_roc_auc": metrics["roc_auc"],
                    "test_pr_auc": metrics["pr_auc"],
                }
            )
            mlflow.log_artifact(str(cm_path), artifact_path="figures")
            mlflow.log_artifact(str(roc_path), artifact_path="figures")
            mlflow.log_artifact(str(pr_path), artifact_path="figures")
            if fi_path.exists():
                mlflow.log_artifact(str(fi_path), artifact_path="figures")
            mlflow.log_artifact(str(EVALUATION_METRICS_PATH), artifact_path="metrics")
            mlflow.log_artifact(str(MODEL_METADATA_PATH), artifact_path="metadata")
            logger.info("Logged test metrics and figures to MLflow.")
    except Exception as exc:
        logger.warning("Could not log test metrics to MLflow: %s", exc)

    logger.info("--- Model Evaluation Completed Successfully ---")
    return metrics


if __name__ == "__main__":
    evaluate_champion_model()
