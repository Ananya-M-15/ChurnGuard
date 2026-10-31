"""ChurnGuard — Production Inference Telemetry & Drift Monitoring Package.

Provides SQLite-backed privacy-compliant request logging, Kolmogorov-Smirnov
numerical drift testing, and Population Stability Index (PSI) categorical drift testing.
"""

from monitoring.logger import InferenceLogger, get_default_logger
from monitoring.drift import DriftDetector, run_drift_analysis

__all__ = [
    "InferenceLogger",
    "get_default_logger",
    "DriftDetector",
    "run_drift_analysis",
]
