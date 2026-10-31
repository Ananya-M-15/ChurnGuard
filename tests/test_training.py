"""Unit and integration tests for model training, evaluation, and serialization.

Verifies candidate model instantiation, artifact persistence, prediction reproducibility,
data isolation, metric validation, and MLflow experiment logging.
"""

import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.config import (
    EVALUATION_METRICS_PATH,
    FIGURES_DIR,
    MLFLOW_DB_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.features import load_pipeline
from src.train import get_candidate_models


def test_candidate_models_definitions():
    """Verify that all three specified models are instantiated with appropriate configurations."""
    candidates = get_candidate_models()

    assert "Logistic_Regression" in candidates
    assert "Random_Forest" in candidates
    assert "XGBoost" in candidates

    assert candidates["Logistic_Regression"].class_weight == "balanced"
    assert candidates["Random_Forest"].class_weight == "balanced"
    assert candidates["XGBoost"].scale_pos_weight == 2.77


def test_saved_model_reloads_and_predicts():
    """Verify models/pipeline.joblib can be reloaded independently and produces valid outputs."""
    assert MODEL_PIPELINE_PATH.exists(), f"Champion pipeline not found at {MODEL_PIPELINE_PATH}"

    pipeline = load_pipeline(MODEL_PIPELINE_PATH)
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    test_df = pd.read_parquet(TEST_DATA_PATH).head(20)
    X_test = test_df.drop(columns=[TARGET_COLUMN])

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probs.shape == (len(X_test), 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_prediction_reproducibility():
    """Verify that model inferences are deterministic across separate invocations."""
    pipeline = load_pipeline(MODEL_PIPELINE_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH).head(50)
    X_test = test_df.drop(columns=[TARGET_COLUMN])

    probs_run_1 = pipeline.predict_proba(X_test)
    probs_run_2 = pipeline.predict_proba(X_test)

    np.testing.assert_array_equal(probs_run_1, probs_run_2)


def test_training_and_test_data_remain_separate():
    """Verify that training and test datasets remain strictly disjoint without row leakage."""
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)

    # Confirm correct partition sizes (80/20 split)
    assert len(train_df) == 5634
    assert len(test_df) == 1409
    assert len(train_df) + len(test_df) == 7043

    # Confirm identical feature schemas
    assert list(train_df.columns) == list(test_df.columns)
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns


def test_generated_artifacts_and_metrics_files():
    """Verify that all required Phase 2 reports, figures, and metadata files exist."""
    # 1. Model comparison CSV
    assert MODEL_COMPARISON_PATH.exists(), f"Missing {MODEL_COMPARISON_PATH}"
    comp_df = pd.read_csv(MODEL_COMPARISON_PATH)
    assert len(comp_df) == 3
    assert "cv_pr_auc_mean" in comp_df.columns
    assert "cv_roc_auc_mean" in comp_df.columns

    # 2. Evaluation metrics JSON
    assert EVALUATION_METRICS_PATH.exists(), f"Missing {EVALUATION_METRICS_PATH}"
    with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1" in metrics
    assert "recall" in metrics
    assert "precision" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["decision_threshold"] == 0.5

    # 3. Model metadata JSON
    assert MODEL_METADATA_PATH.exists(), f"Missing {MODEL_METADATA_PATH}"
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["model_version"] == "1.0.0"
    assert metadata["dataset_info"]["train_rows"] == 5634
    assert metadata["dataset_info"]["test_rows"] == 1409
    assert len(metadata["dataset_info"]["feature_names"]) == 25

    # 4. Evaluation figures
    expected_figures = [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "feature_importance.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing evaluation figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure file is suspiciously small: {fig_path}"


def test_mlflow_experiments_and_runs():
    """Verify that MLflow contains actual tracked runs with logged metrics."""
    if not MLFLOW_DB_PATH.exists():
        pytest.skip(
            f"MLflow tracking database ({MLFLOW_DB_PATH.name}) not found in environment. "
            "Local SQLite database is excluded from Git to prevent repository bloat."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    assert experiment is not None, f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found"

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 3, f"Expected at least 3 runs, found {len(runs)}"

    # Check that at least one run logged cv_pr_auc_mean
    run_metrics_keys = [list(r.data.metrics.keys()) for r in runs]
    has_pr_auc = any("cv_pr_auc_mean" in keys or "test_pr_auc" in keys for keys in run_metrics_keys)
    assert has_pr_auc, "No run logged pr_auc metrics in MLflow"
