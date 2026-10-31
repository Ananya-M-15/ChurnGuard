"""Centralized configuration management for ChurnGuard.

Defines project-relative paths, seeds, dataset parameters, and column
categorizations to ensure reproducibility across modules and environments.
"""

from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data storage paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "Telco_customer_churn.xlsx"

# Processed dataset output paths
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.parquet"

# Models and serialization paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Reports and documentation paths
REPORTS_DIR = BASE_DIR / "reports"
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.md"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"

# MLflow experiment tracking paths
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "ChurnGuard-Experiments"

# Monitoring paths
MONITORING_DIR = BASE_DIR / "monitoring"

# Global reproducibility parameters
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.20
CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.5

# Target definition
TARGET_COLUMN = "Churn Value"

# Prohibited leakage and duplicate target columns (MUST never be included in feature sets)
LEAKAGE_COLUMNS = [
    "Churn Label",   # Duplicate representation of target (Yes/No)
    "Churn Reason",  # 100% retrospective outcome leakage (only populated for churned records)
    "Churn Score",   # Pre-computed proprietary risk score that contaminates model learning
]

# Identifier columns (must be excluded from training features)
IDENTIFIER_COLUMNS = [
    "CustomerID",    # Unique account identifier; causes severe overfitting/memorization
]

# Uninformative / zero-variance constant columns across the California Telco dataset
INVARIANT_COLUMNS = [
    "Count",         # Constant 1 across all records
    "Country",       # Constant "United States" across all records
    "State",         # Constant "California" across all records
]

# All columns to drop prior to feature engineering
COLUMNS_TO_DROP = LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS + INVARIANT_COLUMNS

# Standard feature groups expected by the ML model pipeline
NUMERICAL_COLUMNS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
    "Latitude",
    "Longitude",
    "Zip Code",
]

CATEGORICAL_COLUMNS = [
    "City",
    "Lat Long",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
