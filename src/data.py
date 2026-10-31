"""Data loading, cleaning, validation, and splitting module for ChurnGuard.

Implements robust data preparation routines for the IBM Telco customer churn
dataset, ensuring target leakage prevention, schema integrity, and reproducible
stratified train/test dataset generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    COLUMNS_TO_DROP,
    IDENTIFIER_COLUMNS,
    INVARIANT_COLUMNS,
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SPLIT_RATIO,
    TRAIN_DATA_PATH,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Expected critical columns in the raw IBM Telco dataset
MANDATORY_RAW_COLUMNS = [
    "CustomerID",
    "Monthly Charges",
    "Total Charges",
    "Tenure Months",
    "Churn Value",
    "Churn Label",
]


def load_raw_data(file_path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw Telco customer churn Excel dataset.

    Args:
        file_path: Path to the raw Excel spreadsheet. Defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame containing the raw tabular data.

    Raises:
        FileNotFoundError: If the specified Excel file does not exist.
        ImportError: If openpyxl or another Excel engine is missing.
        ValueError: If the file is empty or missing mandatory columns.
    """
    path = Path(file_path) if file_path is not None else RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at: {path}. "
            "Please ensure the file exists or verify the data directory configuration."
        )

    try:
        logger.info("Loading raw dataset from %s", path)
        df = pd.read_excel(path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError(
            "Missing Excel parsing dependency 'openpyxl'. "
            "Please install it using: pip install openpyxl"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read Excel file at {path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"The loaded dataset at {path} is empty (0 rows).")

    # Validate mandatory schema columns
    missing_cols = [c for c in MANDATORY_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at {path} is missing mandatory columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    logger.info("Raw dataset successfully loaded: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the dataset, sanitize strings, and handle missing values.

    Specific cleaning actions:
    1. Strips whitespace from column names.
    2. Strips surrounding whitespace from string/object columns.
    3. Coerces 'Total Charges' to numeric float64, converting whitespace-only
       strings to NaN.
    4. Missing-value strategy for 'Total Charges': Missing values occur exclusively
       when Tenure Months == 0 (new customers who have not had a full billing cycle).
       These values are imputed with 0.0 to accurately represent total charges accrued
       to date while preserving all customer records.
    5. Validates that the target 'Churn Value' is strictly binary {0, 1}.
    6. Identifies and records duplicate rows (if any).

    Args:
        df: Input raw DataFrame.

    Returns:
        Tuple of (cleaned_df, cleaning_metadata).
    """
    cleaned = df.copy()

    # 1. Strip whitespace from column headers
    cleaned.columns = cleaned.columns.str.strip()

    # 2. Check for duplicate rows
    duplicate_rows_count = int(cleaned.duplicated().sum())
    duplicate_customers_count = int(cleaned["CustomerID"].duplicated().sum()) if "CustomerID" in cleaned.columns else 0

    if duplicate_rows_count > 0:
        logger.warning("Found %d exact duplicate rows in dataset.", duplicate_rows_count)
    else:
        logger.info("Duplicate check passed: 0 duplicate rows found.")

    # 3. Strip whitespace from all string/object cells
    text_cols = cleaned.select_dtypes(include=["object", "string", "str"]).columns
    for col in text_cols:
        cleaned[col] = cleaned[col].apply(lambda val: val.strip() if isinstance(val, str) else val)

    # 4. Safely convert 'Total Charges' to numeric float64
    raw_tc = cleaned["Total Charges"]
    coerced_tc = pd.to_numeric(raw_tc, errors="coerce")
    missing_tc_count = int(coerced_tc.isna().sum())

    # Strategy: Impute missing Total Charges with 0.0 (where Tenure Months == 0)
    cleaned["Total Charges"] = coerced_tc.fillna(0.0).astype(float)
    logger.info(
        "Total Charges cleaned: %d missing/whitespace values coerced to NaN and imputed with 0.0 (tenure=0)",
        missing_tc_count,
    )

    # 5. Validate target column 'Churn Value'
    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    unique_targets = set(cleaned[TARGET_COLUMN].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains non-binary values: {unique_targets}. "
            "Expected binary values {0, 1}."
        )

    if cleaned[TARGET_COLUMN].isna().sum() > 0:
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains {cleaned[TARGET_COLUMN].isna().sum()} missing values.")

    # Clean target dtype to int
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    metadata: dict[str, Any] = {
        "raw_rows": len(df),
        "raw_cols": df.shape[1],
        "duplicate_rows": duplicate_rows_count,
        "duplicate_customers": duplicate_customers_count,
        "total_charges_missing_coerced": missing_tc_count,
        "target_column": TARGET_COLUMN,
        "target_unique_values": list(unique_targets),
        "target_distribution": cleaned[TARGET_COLUMN].value_counts().to_dict(),
        "target_churn_rate": float(cleaned[TARGET_COLUMN].mean()),
    }

    return cleaned, metadata


def drop_unwanted_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Remove target leakage, duplicate target, identifier, and invariant columns.

    Columns dropped:
    - Target Duplicates & Leakage: 'Churn Label', 'Churn Reason', 'Churn Score'
    - Identifiers: 'CustomerID'
    - Invariants: 'Count', 'Country', 'State'

    Args:
        df: Cleaned input DataFrame.

    Returns:
        Tuple of (df_features_with_target, list_of_dropped_columns).
    """
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]

    # Verify invariant columns are genuinely invariant before dropping
    for inv_col in INVARIANT_COLUMNS:
        if inv_col in df.columns:
            n_unique = df[inv_col].nunique()
            if n_unique != 1:
                logger.warning(
                    "Column '%s' is expected to be invariant but has %d unique values: %s",
                    inv_col,
                    n_unique,
                    df[inv_col].unique(),
                )

    filtered_df = df.drop(columns=to_drop)
    logger.info(
        "Dropped %d non-feature / leakage columns: %s. Remaining columns: %d",
        len(to_drop),
        to_drop,
        filtered_df.shape[1],
    )
    return filtered_df, to_drop


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SPLIT_RATIO,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a reproducible, stratified train/test split on the dataset.

    Stratification is performed on the target column to ensure both splits
    maintain identical churn proportion.

    Args:
        df: DataFrame containing model features and TARGET_COLUMN.
        test_size: Proportion of the dataset to allocate to the test split.
        random_seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        ValueError: If TARGET_COLUMN is missing or test_size is invalid.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Cannot perform stratified split: '{TARGET_COLUMN}' is not in DataFrame.")

    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0.0 and 1.0, got: {test_size}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[TARGET_COLUMN],
    )

    # Reset indices to guarantee clean, 0-indexed integer indexing in downstream consumers
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Stratified split complete (seed=%d, test_size=%.2f): Train=%d rows, Test=%d rows",
        random_seed,
        test_size,
        len(train_df),
        len(test_df),
    )
    return train_df, test_df


def separate_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y.

    Args:
        df: Labeled dataset containing TARGET_COLUMN.

    Returns:
        Tuple of (X, y).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"DataFrame does not contain target column '{TARGET_COLUMN}'.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str | None = None,
) -> Tuple[Path, Path]:
    """Save processed train and test datasets in Parquet format with CSV fallback.

    Args:
        train_df: Processed training DataFrame.
        test_df: Processed testing DataFrame.
        output_dir: Destination directory. Defaults to PROCESSED_DATA_DIR.

    Returns:
        Tuple of (train_file_path, test_file_path).
    """
    out_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"

    try:
        logger.info("Saving processed datasets to Parquet: %s and %s", train_path, test_path)
        train_df.to_parquet(train_path, index=False, engine="pyarrow")
        test_df.to_parquet(test_path, index=False, engine="pyarrow")
    except Exception as exc:
        logger.warning(
            "Failed to save Parquet format (%s). Falling back to CSV storage.", exc
        )
        train_path = out_dir / "train.csv"
        test_path = out_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info("Datasets saved as CSV fallback: %s and %s", train_path, test_path)

    return train_path, test_path


def run_data_pipeline(
    raw_data_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the complete Phase 1 data ingestion, cleaning, and preparation pipeline.

    Args:
        raw_data_path: Optional custom path to raw Excel data.
        output_dir: Optional custom path for processed artifacts.

    Returns:
        Summary statistics dictionary suitable for data quality reporting.
    """
    logger.info("--- Starting ChurnGuard Data Preparation Pipeline ---")

    # 1. Ingest raw data
    raw_df = load_raw_data(raw_data_path)

    # 2. Clean data & impute Total Charges
    clean_df, clean_meta = clean_data(raw_df)

    # 3. Drop leakage, identifier, and invariant columns
    prepared_df, dropped_cols = drop_unwanted_columns(clean_df)

    # 4. Stratified train/test split
    train_df, test_df = split_data(prepared_df)

    # 5. Persist processed artifacts
    train_path, test_path = save_processed_data(train_df, test_df, output_dir)

    # 6. Verify reloadability
    if train_path.suffix == ".parquet":
        reloaded_train = pd.read_parquet(train_path)
        reloaded_test = pd.read_parquet(test_path)
    else:
        reloaded_train = pd.read_csv(train_path)
        reloaded_test = pd.read_csv(test_path)

    assert len(reloaded_train) == len(train_df), "Reloaded train row count mismatch"
    assert len(reloaded_test) == len(test_df), "Reloaded test row count mismatch"
    assert set(reloaded_train.columns) == set(prepared_df.columns), "Reloaded schema mismatch"

    # Compute final statistics
    train_churn_pct = float(train_df[TARGET_COLUMN].mean() * 100)
    test_churn_pct = float(test_df[TARGET_COLUMN].mean() * 100)
    overall_churn_pct = float(clean_meta["target_churn_rate"] * 100)

    stats: dict[str, Any] = {
        "raw_rows": clean_meta["raw_rows"],
        "raw_cols": clean_meta["raw_cols"],
        "duplicate_rows": clean_meta["duplicate_rows"],
        "duplicate_customers": clean_meta["duplicate_customers"],
        "total_charges_missing_coerced": clean_meta["total_charges_missing_coerced"],
        "dropped_columns": dropped_cols,
        "dropped_columns_reasons": {
            "Churn Label": "Duplicate target variable (categorical Yes/No)",
            "Churn Reason": "Retrospective outcome leakage (populated only for churned records)",
            "Churn Score": "Pre-computed proprietary risk score causing data contamination",
            "CustomerID": "Unique account identifier; would cause model overfitting",
            "Count": "Zero-variance invariant constant (1)",
            "Country": "Zero-variance invariant constant ('United States')",
            "State": "Zero-variance invariant constant ('California')",
        },
        "final_feature_count": prepared_df.shape[1] - 1,  # Excludes target
        "final_column_count": prepared_df.shape[1],       # Includes target
        "feature_names": [c for c in prepared_df.columns if c != TARGET_COLUMN],
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "overall_churn_count": int(clean_meta["target_distribution"].get(1, 0)),
        "overall_non_churn_count": int(clean_meta["target_distribution"].get(0, 0)),
        "overall_churn_pct": overall_churn_pct,
        "train_churn_pct": train_churn_pct,
        "test_churn_pct": test_churn_pct,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }

    logger.info("--- Data Preparation Pipeline Completed Successfully ---")
    logger.info("Processed Training Set: %s (%d rows)", train_path, stats["train_rows"])
    logger.info("Processed Testing Set:  %s (%d rows)", test_path, stats["test_rows"])
    logger.info("Train Churn Rate: %.2f%% | Test Churn Rate: %.2f%%", train_churn_pct, test_churn_pct)

    return stats


if __name__ == "__main__":
    run_data_pipeline()
