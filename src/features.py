"""Feature engineering and preprocessing pipeline for ChurnGuard.

Builds modular Scikit-learn ColumnTransformer and Pipeline objects that handle
both numeric and categorical features, sanitize missing values, prevent target/data
leakage, and guarantee reproducible transformations across training and inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    COLUMNS_TO_DROP,
    MODEL_PIPELINE_PATH,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_feature_columns(
    df: pd.DataFrame,
    explicit_num_cols: list[str] | None = None,
    explicit_cat_cols: list[str] | None = None,
) -> Tuple[list[str], list[str]]:
    """Identify and separate numeric and categorical feature columns.

    Explicitly excludes target, leakage, identifier, and invariant columns
    as specified in src.config to guarantee no data contamination enters
    the feature space.

    Args:
        df: Input DataFrame to inspect.
        explicit_num_cols: Optional predetermined list of numeric columns.
        explicit_cat_cols: Optional predetermined list of categorical columns.

    Returns:
        Tuple of (numeric_columns_list, categorical_columns_list).
    """
    # 1. Strip target, leakage, identifier, and invariant columns
    excluded_set = set(COLUMNS_TO_DROP) | {TARGET_COLUMN}
    valid_cols = [c for c in df.columns if c not in excluded_set]

    if explicit_num_cols is not None and explicit_cat_cols is not None:
        num_cols = [c for c in explicit_num_cols if c in valid_cols]
        cat_cols = [c for c in explicit_cat_cols if c in valid_cols]
        return num_cols, cat_cols

    # 2. Automatically detect numeric and categorical columns
    sub_df = df[valid_cols]
    num_cols = sub_df.select_dtypes(include=[np.number, "int64", "float64"]).columns.tolist()
    cat_cols = sub_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    logger.info(
        "Feature columns separated: %d numeric, %d categorical (total features: %d)",
        len(num_cols),
        len(cat_cols),
        len(valid_cols),
    )
    return num_cols, cat_cols


def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """Construct a Scikit-Learn ColumnTransformer for feature preprocessing.

    - Numeric pipeline: Median imputation -> StandardScaler
    - Categorical pipeline: Constant imputation ('missing') -> OneHotEncoder(handle_unknown='ignore')

    Args:
        num_cols: List of numeric column names.
        cat_cols: List of categorical column names.

    Returns:
        Fitted or unfitted ColumnTransformer object.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def create_model_pipeline(
    classifier: BaseEstimator,
    num_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Combine feature preprocessing and classifier into an end-to-end Pipeline.

    Guarantees that identical preprocessing transformations are executed during
    both training and inference without risk of train-test data leakage.

    Args:
        classifier: Any Scikit-Learn or compatible classifier (e.g. XGBClassifier).
        num_cols: List of numeric column names.
        cat_cols: List of categorical column names.

    Returns:
        End-to-end Scikit-Learn Pipeline.
    """
    preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def save_pipeline(pipeline: Pipeline, file_path: Path | str | None = None) -> Path:
    """Serialize the trained Pipeline to disk using joblib.

    Args:
        pipeline: Fitted Scikit-Learn Pipeline.
        file_path: Optional custom path. Defaults to MODEL_PIPELINE_PATH.

    Returns:
        Path where pipeline was saved.
    """
    dest = Path(file_path) if file_path is not None else MODEL_PIPELINE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)
    logger.info("Model pipeline serialized successfully to: %s", dest)
    return dest


def load_pipeline(file_path: Path | str | None = None) -> Pipeline:
    """Load a serialized Scikit-Learn Pipeline from disk using joblib.

    Args:
        file_path: Optional custom path. Defaults to MODEL_PIPELINE_PATH.

    Returns:
        Loaded Scikit-Learn Pipeline.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
    """
    src = Path(file_path) if file_path is not None else MODEL_PIPELINE_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"Serialized model pipeline not found at: {src}. "
            "Please run the training pipeline first using: python -m src.train"
        )
    pipeline = joblib.load(src)
    logger.info("Model pipeline loaded successfully from: %s", src)
    return pipeline
