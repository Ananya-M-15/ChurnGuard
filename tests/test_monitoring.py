"""Unit and integration tests for ChurnGuard inference logging and drift monitoring.

Verifies:
- SQLite persistence of inference events
- PII and CustomerID redaction/sanitization
- Batch logging integrity and summary telemetry
- Two-sample Kolmogorov-Smirnov numerical drift testing
- Population Stability Index (PSI) categorical drift testing
- Enforcing minimum sample size safeguards
- Report generation and JSON persistence
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    DriftDetector,
    calculate_ks_drift,
    calculate_psi,
    run_drift_analysis,
)
from monitoring.logger import InferenceLogger
from src.config import TRAIN_DATA_PATH


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Fixture providing a temporary SQLite database path."""
    return tmp_path / "test_monitoring.db"


@pytest.fixture
def sample_feature_dict() -> dict:
    """Fixture providing sample feature dictionary with PII to verify redaction."""
    return {
        "City": "Los Angeles",
        "Zip Code": 90001,
        "Lat Long": "34.0522, -118.2437",
        "Latitude": 34.0522,
        "Longitude": -118.2437,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 12,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 50.0,
        "Total Charges": 600.0,
        "CLTV": 3200,
        # PII fields that MUST be stripped
        "CustomerID": "7590-VHVEG",
        "customer_name": "Test User",
        "email": "test@example.com",
    }


def test_logger_initialization_and_single_prediction(temp_db: Path, sample_feature_dict: dict):
    """Verify logging writes single inference event and properly sanitizes PII."""
    logger = InferenceLogger(db_path=temp_db)
    assert logger.get_log_count() == 0

    log_id = logger.log_prediction(
        source="unit-test",
        model_name="XGBClassifier",
        model_version="1.0.0",
        churn_probability=0.7245,
        predicted_class="Yes",
        risk_tier="High",
        features=sample_feature_dict,
    )

    assert log_id == 1
    assert logger.get_log_count() == 1

    df_recent = logger.get_recent_logs(limit=10)
    assert len(df_recent) == 1
    assert df_recent.iloc[0]["prediction_source"] == "unit-test"
    assert df_recent.iloc[0]["model_name"] == "XGBClassifier"
    assert df_recent.iloc[0]["churn_probability"] == 0.7245
    assert df_recent.iloc[0]["predicted_class"] == "Yes"
    assert df_recent.iloc[0]["risk_tier"] == "High"

    # Verify PII was redacted from stored feature payloads
    feat_df = logger.get_feature_dataframe()
    assert len(feat_df) == 1
    assert "CustomerID" not in feat_df.columns
    assert "customer_name" not in feat_df.columns
    assert "email" not in feat_df.columns
    assert "Tenure Months" in feat_df.columns
    assert "Monthly Charges" in feat_df.columns


def test_logger_batch_records_and_summary_telemetry(temp_db: Path, sample_feature_dict: dict):
    """Verify batch insert and summary statistics computation."""
    logger = InferenceLogger(db_path=temp_db)

    batch = [
        {
            "churn_probability": 0.15,
            "predicted_class": "No",
            "risk_tier": "Low",
            "features": sample_feature_dict,
        },
        {
            "churn_probability": 0.45,
            "predicted_class": "No",
            "risk_tier": "Medium",
            "features": sample_feature_dict,
        },
        {
            "churn_probability": 0.85,
            "predicted_class": "Yes",
            "risk_tier": "High",
            "features": sample_feature_dict,
        },
    ]

    inserted = logger.log_batch(
        source="batch-test",
        model_name="XGBClassifier",
        model_version="1.0.0",
        records=batch,
    )

    assert inserted == 3
    assert logger.get_log_count() == 3

    summary = logger.get_summary_stats()
    assert summary["total_inferences"] == 3
    assert summary["predicted_churn_count"] == 1
    assert summary["predicted_retention_count"] == 2
    assert summary["risk_breakdown"]["Low"] == 1
    assert summary["risk_breakdown"]["Medium"] == 1
    assert summary["risk_breakdown"]["High"] == 1
    assert np.isclose(summary["average_churn_probability"], (0.15 + 0.45 + 0.85) / 3, atol=1e-3)


def test_calculate_ks_drift_numerical():
    """Verify KS 2-sample test detects distribution shifts and confirms stability."""
    np.random.seed(42)
    reference = pd.Series(np.random.normal(loc=50.0, scale=10.0, size=1000))

    # 1. Stable sample from identical distribution
    same_sample = pd.Series(np.random.normal(loc=50.0, scale=10.0, size=200))
    res_stable = calculate_ks_drift(reference, same_sample, alpha=0.05)
    assert not res_stable["drift_detected"]
    assert res_stable["p_value"] >= 0.05

    # 2. Shifted sample with significant location shift
    shifted_sample = pd.Series(np.random.normal(loc=90.0, scale=10.0, size=200))
    res_drift = calculate_ks_drift(reference, shifted_sample, alpha=0.05)
    assert res_drift["drift_detected"]
    assert res_drift["p_value"] < 0.05
    assert res_drift["statistic"] > 0.5


def test_calculate_psi_categorical():
    """Verify PSI calculation for stable vs shifted categorical distributions."""
    # Reference proportions: 50% A, 30% B, 20% C
    ref_cats = pd.Series(["A"] * 500 + ["B"] * 300 + ["C"] * 200)

    # 1. Stable sample: 49% A, 31% B, 20% C
    stable_cats = pd.Series(["A"] * 49 + ["B"] * 31 + ["C"] * 20)
    res_stable = calculate_psi(ref_cats, stable_cats, threshold=0.20)
    assert not res_stable["drift_detected"]
    assert res_stable["psi_value"] < 0.10

    # 2. Shifted distribution: 5% A, 15% B, 80% C
    shifted_cats = pd.Series(["A"] * 5 + ["B"] * 15 + ["C"] * 80)
    res_drift = calculate_psi(ref_cats, shifted_cats, threshold=0.20)
    assert res_drift["drift_detected"]
    assert res_drift["psi_value"] >= 0.20


def test_drift_detector_insufficient_samples_guard(tmp_path: Path):
    """Verify drift evaluation gracefully halts and warns when sample count is below minimum."""
    detector = DriftDetector()
    report_file = tmp_path / "drift_insufficient.json"

    # Only 5 samples provided (< 20 minimum)
    small_df = pd.DataFrame(
        {
            "Tenure Months": [1, 2, 3, 4, 5],
            "Monthly Charges": [50.0, 60.0, 70.0, 80.0, 90.0],
            "Contract": ["Month-to-month"] * 5,
        }
    )

    report = detector.evaluate_drift(
        small_df,
        min_samples=20,
        save_report=True,
        report_path=report_file,
    )

    assert report["status"] == "Insufficient monitoring data"
    assert report["inference_sample_count"] == 5
    assert report["min_samples_required"] == 20
    assert report["drift_detected_count"] == 0
    assert report_file.exists()


def test_drift_detector_full_evaluation_and_json_report(tmp_path: Path):
    """Verify end-to-end drift calculation on sufficient samples and JSON persistence."""
    detector = DriftDetector()
    report_file = tmp_path / "drift_test_report.json"

    # Load 50 samples from reference data to simulate stable production traffic
    ref_df = pd.read_parquet(TRAIN_DATA_PATH)
    sample_df = ref_df.sample(n=60, random_state=42).copy()

    report = detector.evaluate_drift(
        sample_df,
        min_samples=20,
        save_report=True,
        report_path=report_file,
    )

    assert report["status"] in ["No significant drift", "Drift detected"]
    assert report["inference_sample_count"] == 60
    assert len(report["numerical_metrics"]) > 0
    assert len(report["categorical_metrics"]) > 0
    assert report_file.exists()

    # Verify JSON file is loadable and conforms to schema
    with open(report_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["inference_sample_count"] == 60
    assert "numerical_metrics" in loaded
    assert "categorical_metrics" in loaded
