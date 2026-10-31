"""Drift detection engine for ChurnGuard production telemetry.

Evaluates data drift between baseline training data and live inference records
using:
1. Two-sample Kolmogorov-Smirnov (KS) tests for numerical features.
2. Population Stability Index (PSI) for categorical features.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    BASE_DIR,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TRAIN_DATA_PATH,
)

logger = logging.getLogger(__name__)

DEFAULT_DRIFT_REPORT_PATH = BASE_DIR / "reports" / "monitoring" / "drift_report.json"
MINIMUM_MONITORING_SAMPLES = 20
KS_ALPHA_DEFAULT = 0.05
PSI_THRESHOLD_DEFAULT = 0.20


def calculate_ks_drift(
    reference: pd.Series,
    current: pd.Series,
    alpha: float = KS_ALPHA_DEFAULT,
) -> Dict[str, Any]:
    """Calculate Kolmogorov-Smirnov 2-sample test for a numerical feature.

    Args:
        reference: Historical/baseline numerical series.
        current: Live inference numerical series.
        alpha: Significance level threshold for rejecting the null hypothesis.

    Returns:
        Dict with statistic, p_value, drift_detected flag, and threshold.
    """
    ref_clean = pd.to_numeric(reference, errors="coerce").dropna().values
    curr_clean = pd.to_numeric(current, errors="coerce").dropna().values

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "threshold": alpha,
            "error": "Empty or non-numeric feature series",
        }

    res = stats.ks_2samp(ref_clean, curr_clean)
    stat = round(float(res.statistic), 4)
    p_val = round(float(res.pvalue), 4)

    return {
        "statistic": stat,
        "p_value": p_val,
        "drift_detected": bool(p_val < alpha),
        "threshold": alpha,
        "interpretation": "Drift detected (p < alpha)" if p_val < alpha else "Stable (p >= alpha)",
    }


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-4,
    threshold: float = PSI_THRESHOLD_DEFAULT,
) -> Dict[str, Any]:
    """Calculate Population Stability Index (PSI) for a categorical feature.

    Formula:
        PSI = sum((P_i - Q_i) * ln(P_i / Q_i))
        where Q_i is the reference proportion, and P_i is the current proportion.

    Args:
        reference: Baseline categorical series.
        current: Live inference categorical series.
        epsilon: Smoothing term for zero-frequency categories.
        threshold: Critical PSI cutoff (default 0.20).

    Returns:
        Dict with psi_value, drift_detected flag, interpretation, and threshold.
    """
    ref_clean = reference.astype(str).str.strip()
    curr_clean = current.astype(str).str.strip()

    all_categories = sorted(list(set(ref_clean.unique()) | set(curr_clean.unique())))
    if not all_categories:
        return {
            "psi_value": 0.0,
            "drift_detected": False,
            "threshold": threshold,
            "interpretation": "Empty series",
        }

    ref_counts = ref_clean.value_counts()
    curr_counts = curr_clean.value_counts()

    total_ref = len(ref_clean)
    total_curr = len(curr_clean)

    # Calculate smoothed relative frequencies
    ref_props = np.array(
        [(ref_counts.get(cat, 0) + epsilon) / (total_ref + epsilon * len(all_categories)) for cat in all_categories]
    )
    curr_props = np.array(
        [(curr_counts.get(cat, 0) + epsilon) / (total_curr + epsilon * len(all_categories)) for cat in all_categories]
    )

    # Normalize to strictly sum to 1
    ref_props = ref_props / ref_props.sum()
    curr_props = curr_props / curr_props.sum()

    # Calculate PSI components
    psi_terms = (curr_props - ref_props) * np.log(curr_props / ref_props)
    psi_value = round(float(np.sum(psi_terms)), 4)

    drift_detected = bool(psi_value >= threshold)
    if psi_value < 0.10:
        interpretation = "Stable (PSI < 0.10)"
    elif psi_value < threshold:
        interpretation = "Moderate shift (0.10 <= PSI < 0.20)"
    else:
        interpretation = f"Significant shift / Drift detected (PSI >= {threshold})"

    return {
        "psi_value": psi_value,
        "drift_detected": drift_detected,
        "threshold": threshold,
        "interpretation": interpretation,
    }


class DriftDetector:
    """Drift evaluation suite comparing inference payloads against training baseline."""

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_path: Optional[Path | str] = None,
    ) -> None:
        if reference_data is not None:
            self.reference_data = reference_data.copy()
        else:
            ref_path = Path(reference_path) if reference_path else TRAIN_DATA_PATH
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Baseline reference data not found at: {ref_path}. "
                    "Ensure Phase 1 data pipeline has generated train.parquet."
                )
            self.reference_data = pd.read_parquet(ref_path)

        # Standardize baseline feature column lists
        self.numerical_cols = [c for c in NUMERICAL_COLUMNS if c in self.reference_data.columns]
        self.categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in self.reference_data.columns]

    def evaluate_drift(
        self,
        current_data: pd.DataFrame,
        min_samples: int = MINIMUM_MONITORING_SAMPLES,
        ks_alpha: float = KS_ALPHA_DEFAULT,
        psi_threshold: float = PSI_THRESHOLD_DEFAULT,
        save_report: bool = True,
        report_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Evaluate covariate drift across all features.

        Args:
            current_data: DataFrame containing live production feature records.
            min_samples: Minimum record count required before computing statistics.
            ks_alpha: P-value threshold for numerical KS test.
            psi_threshold: Critical PSI cutoff for categorical features.
            save_report: Whether to persist JSON report to disk.
            report_path: Custom destination path for the JSON report.

        Returns:
            Structured drift evaluation dictionary.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        out_path = Path(report_path) if report_path else DEFAULT_DRIFT_REPORT_PATH
        n_samples = len(current_data) if current_data is not None else 0

        # Enforce minimum sample requirement to prevent false alarms
        if n_samples < min_samples:
            report = {
                "generated_at": now_utc,
                "status": "Insufficient monitoring data",
                "message": (
                    f"Requires at least {min_samples} production inference samples to perform "
                    f"statistically valid drift hypothesis tests. Found {n_samples} samples."
                ),
                "reference_dataset": str(TRAIN_DATA_PATH.name),
                "reference_sample_count": len(self.reference_data),
                "inference_sample_count": n_samples,
                "min_samples_required": min_samples,
                "drift_detected_count": 0,
                "drifted_features": [],
                "numerical_metrics": {},
                "categorical_metrics": {},
            }
            if save_report:
                self._save_report(report, out_path)
            return report

        numerical_metrics: Dict[str, Any] = {}
        categorical_metrics: Dict[str, Any] = {}
        drifted_features: List[str] = []

        # 1. Numerical drift evaluation (Kolmogorov-Smirnov)
        for col in self.numerical_cols:
            if col in current_data.columns:
                res = calculate_ks_drift(self.reference_data[col], current_data[col], alpha=ks_alpha)
                numerical_metrics[col] = res
                if res["drift_detected"]:
                    drifted_features.append(col)

        # 2. Categorical drift evaluation (Population Stability Index)
        for col in self.categorical_cols:
            if col in current_data.columns:
                res = calculate_psi(self.reference_data[col], current_data[col], threshold=psi_threshold)
                categorical_metrics[col] = res
                if res["drift_detected"]:
                    drifted_features.append(col)

        drift_detected_count = len(drifted_features)
        overall_status = "Drift detected" if drift_detected_count > 0 else "No significant drift"

        summary_msg = (
            f"{overall_status}. {drift_detected_count} feature(s) exceeded drift thresholds "
            f"across {n_samples} inference samples evaluated against baseline."
        )

        report = {
            "generated_at": now_utc,
            "status": overall_status,
            "message": summary_msg,
            "reference_dataset": str(TRAIN_DATA_PATH.name),
            "reference_sample_count": len(self.reference_data),
            "inference_sample_count": n_samples,
            "min_samples_required": min_samples,
            "drift_detected_count": drift_detected_count,
            "drifted_features": drifted_features,
            "numerical_metrics": numerical_metrics,
            "categorical_metrics": categorical_metrics,
        }

        if save_report:
            self._save_report(report, out_path)

        return report

    @staticmethod
    def _save_report(report: Dict[str, Any], path: Path) -> None:
        """Persist structured report to JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("Drift evaluation report saved to: %s", path)
        except Exception as exc:
            logger.error("Failed to save drift report to %s: %s", path, exc)


def run_drift_analysis(
    logger_instance: Optional[Any] = None,
    min_samples: int = MINIMUM_MONITORING_SAMPLES,
    save_report: bool = True,
) -> Dict[str, Any]:
    """Execute drift analysis using feature records stored in the SQLite inference database."""
    from monitoring.logger import get_default_logger

    log_mgr = logger_instance or get_default_logger()
    feature_df = log_mgr.get_feature_dataframe()

    detector = DriftDetector()
    return detector.evaluate_drift(feature_df, min_samples=min_samples, save_report=save_report)
