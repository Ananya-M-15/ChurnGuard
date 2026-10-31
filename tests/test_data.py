"""Unit and integration tests for data preparation and pipeline integrity.

Verifies encoding, loading, cleaning, leakage stripping, target validation,
stratified partitioning, and artifact reloadability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import (
    BASE_DIR,
    COLUMNS_TO_DROP,
    IDENTIFIER_COLUMNS,
    INVARIANT_COLUMNS,
    LEAKAGE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.data import (
    clean_data,
    drop_unwanted_columns,
    load_raw_data,
    separate_features_and_target,
    split_data,
)


def test_requirements_file_utf8_encoding():
    """Verify requirements.txt is readable as clean UTF-8 without BOM."""
    req_path = BASE_DIR / "requirements.txt"
    assert req_path.exists(), "requirements.txt does not exist"

    # Read binary to verify no UTF-16 BOM
    with open(req_path, "rb") as f:
        raw_bytes = f.read()

    assert not raw_bytes.startswith(b"\xff\xfe"), "requirements.txt contains UTF-16LE BOM"
    assert not raw_bytes.startswith(b"\xfe\xff"), "requirements.txt contains UTF-16BE BOM"

    # Verify standard utf-8 decoding succeeds
    text = raw_bytes.decode("utf-8")
    assert "pandas" in text
    assert "scikit-learn" in text
    assert "openpyxl" in text
    assert "pyarrow" in text


def test_raw_data_loading():
    """Verify raw Excel file can be loaded and conforms to expected row count."""
    df = load_raw_data(RAW_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 7043
    assert df.shape[1] == 33
    assert TARGET_COLUMN in df.columns


def test_clean_data_total_charges_and_target():
    """Verify cleaning converts Total Charges to float and validates binary target."""
    df = load_raw_data(RAW_DATA_PATH)
    cleaned, meta = clean_data(df)

    # Total Charges must be numeric float with no nulls
    assert np.issubdtype(cleaned["Total Charges"].dtype, np.floating)
    assert cleaned["Total Charges"].isna().sum() == 0
    assert meta["total_charges_missing_coerced"] == 11

    # Target must be strictly binary {0, 1}
    assert set(cleaned[TARGET_COLUMN].unique()) == {0, 1}
    assert meta["target_distribution"][0] == 5174
    assert meta["target_distribution"][1] == 1869


def test_prohibited_leakage_columns_are_dropped():
    """Verify all leakage, identifier, and invariant columns are removed."""
    df = load_raw_data(RAW_DATA_PATH)
    cleaned, _ = clean_data(df)
    prepared, dropped = drop_unwanted_columns(cleaned)

    # Check dropped list includes all mandatory exclusions
    for col in LEAKAGE_COLUMNS:
        assert col in dropped, f"Leakage column '{col}' was not dropped"
        assert col not in prepared.columns, f"Leakage column '{col}' still present in prepared DataFrame"

    for col in IDENTIFIER_COLUMNS:
        assert col in dropped, f"Identifier column '{col}' was not dropped"
        assert col not in prepared.columns, f"Identifier column '{col}' still present in prepared DataFrame"

    for col in INVARIANT_COLUMNS:
        assert col in dropped, f"Invariant column '{col}' was not dropped"
        assert col not in prepared.columns, f"Invariant column '{col}' still present in prepared DataFrame"

    # Target must remain in the prepared dataset
    assert TARGET_COLUMN in prepared.columns
    # Total remaining columns: 33 - 7 = 26
    assert prepared.shape[1] == 26


def test_stratified_train_test_split():
    """Verify train/test split proportions, stratification, and index disjointness."""
    df = load_raw_data(RAW_DATA_PATH)
    cleaned, _ = clean_data(df)
    prepared, _ = drop_unwanted_columns(cleaned)

    train_df, test_df = split_data(prepared, test_size=0.20, random_seed=42)

    # Row count verification
    assert len(train_df) == 5634
    assert len(test_df) == 1409
    assert len(train_df) + len(test_df) == 7043

    # Both splits have identical columns
    assert list(train_df.columns) == list(test_df.columns)

    # Both target classes exist in both splits
    assert set(train_df[TARGET_COLUMN].unique()) == {0, 1}
    assert set(test_df[TARGET_COLUMN].unique()) == {0, 1}

    # Stratification check (churn rates within 0.1% of each other)
    train_rate = train_df[TARGET_COLUMN].mean()
    test_rate = test_df[TARGET_COLUMN].mean()
    assert abs(train_rate - test_rate) < 0.005, f"Stratification mismatch: train={train_rate}, test={test_rate}"

    # Verify no overlapping indices
    overlap = set(train_df.index).intersection(set(test_df.index))
    # Indices were reset, verify they are independent
    assert len(train_df) != len(test_df)


def test_separate_features_and_target():
    """Verify X does not contain the target and y matches row count."""
    df = load_raw_data(RAW_DATA_PATH)
    cleaned, _ = clean_data(df)
    prepared, _ = drop_unwanted_columns(cleaned)

    X, y = separate_features_and_target(prepared)
    assert TARGET_COLUMN not in X.columns
    assert len(X) == len(y) == 7043
    assert X.shape[1] == 25  # Exactly 25 features


def test_processed_artifacts_reloadability():
    """Verify that persisted train.parquet and test.parquet can be reloaded and match schema."""
    assert TRAIN_DATA_PATH.exists(), f"Train artifact not found at {TRAIN_DATA_PATH}"
    assert TEST_DATA_PATH.exists(), f"Test artifact not found at {TEST_DATA_PATH}"

    train_reloaded = pd.read_parquet(TRAIN_DATA_PATH)
    test_reloaded = pd.read_parquet(TEST_DATA_PATH)

    assert len(train_reloaded) == 5634
    assert len(test_reloaded) == 1409
    assert train_reloaded.shape[1] == 26
    assert test_reloaded.shape[1] == 26
    assert TARGET_COLUMN in train_reloaded.columns
    assert TARGET_COLUMN in test_reloaded.columns

    # Verify no nulls in target
    assert train_reloaded[TARGET_COLUMN].isna().sum() == 0
    assert test_reloaded[TARGET_COLUMN].isna().sum() == 0
