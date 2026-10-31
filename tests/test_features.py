"""Unit tests for feature engineering, preprocessing, and pipeline transformations.

Verifies column separation, leakage prevention, handling of unseen categories,
missing value imputation, and pipeline output probabilities.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import (
    COLUMNS_TO_DROP,
    IDENTIFIER_COLUMNS,
    INVARIANT_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
)
from src.features import (
    build_preprocessor,
    create_model_pipeline,
    get_feature_columns,
)


@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Fixture providing a slice of the processed training dataset."""
    assert TRAIN_DATA_PATH.exists(), f"Train dataset not found at {TRAIN_DATA_PATH}"
    df = pd.read_parquet(TRAIN_DATA_PATH)
    return df.head(100).copy()


def test_get_feature_columns_excludes_leakage_and_target(sample_train_data: pd.DataFrame):
    """Verify get_feature_columns rigorously rejects all leakage, target, and identifier columns."""
    contaminated_df = sample_train_data.copy()
    # Inject leakage and identifier columns
    contaminated_df["Churn Value"] = 1
    contaminated_df["Churn Label"] = "Yes"
    contaminated_df["Churn Reason"] = "Price too high"
    contaminated_df["Churn Score"] = 85
    contaminated_df["CustomerID"] = "TEST-1234"
    contaminated_df["Count"] = 1
    contaminated_df["Country"] = "United States"
    contaminated_df["State"] = "California"

    num_cols, cat_cols = get_feature_columns(contaminated_df)
    combined_features = set(num_cols) | set(cat_cols)

    # Check that no prohibited column appears in features
    for col in COLUMNS_TO_DROP + [TARGET_COLUMN]:
        assert col not in combined_features, f"Prohibited column '{col}' was not excluded!"

    assert len(combined_features) == 25, f"Expected 25 features, got {len(combined_features)}"


def test_build_preprocessor_and_transformation(sample_train_data: pd.DataFrame):
    """Verify ColumnTransformer fits and transforms features without errors."""
    X = sample_train_data.drop(columns=[TARGET_COLUMN])
    num_cols, cat_cols = get_feature_columns(X)

    preprocessor = build_preprocessor(num_cols, cat_cols)
    transformed = preprocessor.fit_transform(X)

    # Transformed output should be a 2D matrix with same row count
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(num_cols) + len(cat_cols)


def test_unseen_categories_handling(sample_train_data: pd.DataFrame):
    """Verify OneHotEncoder handle_unknown='ignore' processes previously unseen categories gracefully."""
    X_train = sample_train_data.drop(columns=[TARGET_COLUMN])
    y_train = sample_train_data[TARGET_COLUMN]

    num_cols, cat_cols = get_feature_columns(X_train)
    pipeline = create_model_pipeline(
        classifier=LogisticRegression(max_iter=500),
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    pipeline.fit(X_train, y_train)

    # Construct unseen inference observation
    unseen_obs = X_train.iloc[[0]].copy()
    unseen_obs["City"] = "Mars_Colony_Alpha"
    unseen_obs["Contract"] = "Decade-to-decade"
    unseen_obs["Payment Method"] = "Cryptocurrency"

    # Prediction must succeed without throwing UnknownCategoryError
    pred = pipeline.predict(unseen_obs)
    prob = pipeline.predict_proba(unseen_obs)

    assert pred in [0, 1]
    assert 0.0 <= prob[0, 1] <= 1.0


def test_missing_values_imputation_in_pipeline(sample_train_data: pd.DataFrame):
    """Verify that the pipeline handles injected NaNs in numeric and categorical fields."""
    X_train = sample_train_data.drop(columns=[TARGET_COLUMN])
    y_train = sample_train_data[TARGET_COLUMN]

    num_cols, cat_cols = get_feature_columns(X_train)
    pipeline = create_model_pipeline(
        classifier=LogisticRegression(max_iter=500),
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    pipeline.fit(X_train, y_train)

    # Inject NaNs into test record
    nan_obs = X_train.iloc[[1]].copy()
    nan_obs["Monthly Charges"] = np.nan
    nan_obs["Total Charges"] = np.nan
    nan_obs["Internet Service"] = None

    prob = pipeline.predict_proba(nan_obs)
    assert prob.shape == (1, 2)
    assert not np.isnan(prob).any()


def test_pipeline_produces_valid_probabilities(sample_train_data: pd.DataFrame):
    """Verify probability predictions are strictly bounded in [0, 1] and sum to 1."""
    X_train = sample_train_data.drop(columns=[TARGET_COLUMN])
    y_train = sample_train_data[TARGET_COLUMN]

    num_cols, cat_cols = get_feature_columns(X_train)
    pipeline = create_model_pipeline(
        classifier=LogisticRegression(max_iter=500),
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_train)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
