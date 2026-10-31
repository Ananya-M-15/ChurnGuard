"""ChurnGuard FastAPI Backend Service.

Provides production REST API endpoints for real-time customer churn scoring,
batch file scoring, model health telemetry, and risk classification.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import io
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from api.schemas import (
    MODEL_FEATURE_COLUMNS,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
    customers_to_dataframe,
    determine_risk_tier,
)
from monitoring.drift import run_drift_analysis
from monitoring.logger import get_default_logger
from src.config import (
    DEFAULT_THRESHOLD,
    MODEL_METADATA_PATH,
    MODEL_PIPELINE_PATH,
)
from src.features import load_pipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global container for model state
model_state: Dict[str, Any] = {
    "pipeline": None,
    "model_name": "Unknown",
    "model_version": "1.0.0",
    "model_loaded": False,
    "feature_names": MODEL_FEATURE_COLUMNS,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager that loads the model pipeline on startup."""
    logger.info("Initializing ChurnGuard API service...")
    try:
        if MODEL_PIPELINE_PATH.exists():
            pipeline = load_pipeline(MODEL_PIPELINE_PATH)
            model_state["pipeline"] = pipeline
            model_state["model_loaded"] = True

            # Extract metadata if available
            if MODEL_METADATA_PATH.exists():
                with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    model_state["model_name"] = meta.get("model_name", "XGBClassifier")
                    model_state["model_version"] = meta.get("model_version", "1.0.0")
            else:
                model_state["model_name"] = pipeline.named_steps["classifier"].__class__.__name__

            logger.info(
                "Model [%s v%s] successfully loaded from: %s",
                model_state["model_name"],
                model_state["model_version"],
                MODEL_PIPELINE_PATH,
            )
        else:
            logger.error("Champion model artifact not found at: %s", MODEL_PIPELINE_PATH)
            model_state["model_loaded"] = False
    except Exception as exc:
        logger.error("Failed to load model pipeline on startup: %s", exc)
        model_state["model_loaded"] = False

    yield

    logger.info("Shutting down ChurnGuard API service...")


app = FastAPI(
    title="ChurnGuard API",
    description="Customer Churn Prediction and Risk Telemetry REST Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local and cloud dashboard connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_loaded_pipeline():
    """Retrieve the loaded model pipeline or raise 503 Service Unavailable."""
    pipeline = model_state.get("pipeline")
    if pipeline is None or not model_state.get("model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model pipeline is not loaded or missing. "
                f"Please ensure '{MODEL_PIPELINE_PATH}' exists by running: python -m src.train"
            ),
        )
    return pipeline


@app.get("/", tags=["General"])
def root_endpoint():
    """Welcome endpoint with API details."""
    return {
        "service": "ChurnGuard Customer Churn Prediction API",
        "version": model_state.get("model_version", "1.0.0"),
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """System health telemetry and model availability status."""
    is_loaded = model_state.get("model_loaded", False)
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=model_state.get("model_name", "None"),
        model_version=model_state.get("model_version", "1.0.0"),
        expected_features_count=len(MODEL_FEATURE_COLUMNS),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single_customer(
    features: CustomerFeatures,
    threshold: float = DEFAULT_THRESHOLD,
):
    """Predict churn probability and assign a risk classification for a single customer.

    - **churn_prediction**: 'Yes' if probability >= threshold, else 'No'.
    - **risk_classification**: 'Low' (<0.35), 'Medium' (0.35-0.65), 'High' (>=0.65).
    """
    pipeline = get_loaded_pipeline()

    try:
        df = customers_to_dataframe([features])
        # Strict security verification: ensure CustomerID is NEVER passed to the model
        assert "CustomerID" not in df.columns, "CustomerID must never be passed to model inference pipeline"
        assert df.shape[1] == len(MODEL_FEATURE_COLUMNS), (
            f"Expected exactly {len(MODEL_FEATURE_COLUMNS)} features, got {df.shape[1]}"
        )

        prob = float(pipeline.predict_proba(df)[0, 1])
        retention_prob = round(1.0 - prob, 4)
        prob = round(prob, 4)

        prediction_label = "Yes" if prob >= threshold else "No"
        risk_tier = determine_risk_tier(prob)

        # Log inference to privacy-compliant SQLite telemetry database
        try:
            mon_logger = get_default_logger()
            tier_str = risk_tier.value if hasattr(risk_tier, "value") else str(risk_tier)
            mon_logger.log_prediction(
                source="api",
                model_name=model_state.get("model_name", "ChampionModel"),
                model_version=model_state.get("model_version", "1.0.0"),
                churn_probability=prob,
                predicted_class=prediction_label,
                risk_tier=tier_str,
                features=features.to_model_dict(),
            )
        except Exception as log_exc:
            logger.warning("Telemetry logging failed non-blockingly: %s", log_exc)

        return PredictionResponse(
            customer_id=features.customer_id,
            churn_prediction=prediction_label,
            churn_probability=prob,
            retention_probability=retention_prob,
            risk_classification=risk_tier,
            decision_threshold=threshold,
            model_name=model_state.get("model_name", "ChampionModel"),
            model_version=model_state.get("model_version", "1.0.0"),
        )
    except Exception as exc:
        logger.error("Inference execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {str(exc)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_customers(
    request: BatchPredictionRequest,
    threshold: float = DEFAULT_THRESHOLD,
):
    """Predict churn probabilities and compute batch summary KPIs from a JSON list of customer records."""
    pipeline = get_loaded_pipeline()

    if not request.customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The batch request list must contain at least one customer record.",
        )

    try:
        df = customers_to_dataframe(request.customers)
        # Strict security verification: ensure CustomerID is NEVER passed to the model
        assert "CustomerID" not in df.columns, "CustomerID must never be passed to model inference pipeline"
        assert df.shape[1] == len(MODEL_FEATURE_COLUMNS), (
            f"Expected exactly {len(MODEL_FEATURE_COLUMNS)} features, got {df.shape[1]}"
        )

        probabilities = pipeline.predict_proba(df)[:, 1]

        predictions_list: List[PredictionResponse] = []
        log_records: List[Dict[str, Any]] = []
        churn_count = 0
        low_count = 0
        med_count = 0
        high_count = 0

        for idx, (prob, cust) in enumerate(zip(probabilities, request.customers)):
            prob_val = round(float(prob), 4)
            ret_val = round(1.0 - prob_val, 4)
            label = "Yes" if prob_val >= threshold else "No"
            risk_tier = determine_risk_tier(prob_val)
            tier_str = risk_tier.value if hasattr(risk_tier, "value") else str(risk_tier)

            if label == "Yes":
                churn_count += 1

            if risk_tier == "Low":
                low_count += 1
            elif risk_tier == "Medium":
                med_count += 1
            else:
                high_count += 1

            predictions_list.append(
                PredictionResponse(
                    customer_id=cust.customer_id or f"Customer_{idx+1}",
                    churn_prediction=label,
                    churn_probability=prob_val,
                    retention_probability=ret_val,
                    risk_classification=risk_tier,
                    decision_threshold=threshold,
                    model_name=model_state.get("model_name", "ChampionModel"),
                    model_version=model_state.get("model_version", "1.0.0"),
                )
            )

            log_records.append(
                {
                    "churn_probability": prob_val,
                    "predicted_class": label,
                    "risk_tier": tier_str,
                    "features": cust.to_model_dict(),
                }
            )

        # Batch log to SQLite telemetry
        try:
            mon_logger = get_default_logger()
            mon_logger.log_batch(
                source="api",
                model_name=model_state.get("model_name", "ChampionModel"),
                model_version=model_state.get("model_version", "1.0.0"),
                records=log_records,
            )
        except Exception as log_exc:
            logger.warning("Batch telemetry logging failed non-blockingly: %s", log_exc)

        total = len(predictions_list)
        summary = BatchSummary(
            total_customers=total,
            predicted_churn_count=churn_count,
            predicted_retention_count=total - churn_count,
            predicted_churn_rate=round(churn_count / total, 4) if total > 0 else 0.0,
            low_risk_count=low_count,
            medium_risk_count=med_count,
            high_risk_count=high_count,
            average_churn_probability=round(float(np.mean(probabilities)), 4),
        )

        return BatchPredictionResponse(
            predictions=predictions_list,
            summary=summary,
            model_name=model_state.get("model_name", "ChampionModel"),
            model_version=model_state.get("model_version", "1.0.0"),
        )
    except Exception as exc:
        logger.error("Batch inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(exc)}",
        )


@app.post("/predict/batch/csv", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_from_csv(
    file: UploadFile = File(...),
    threshold: float = DEFAULT_THRESHOLD,
):
    """Upload a CSV file of customer records to generate bulk predictions and summary metrics."""
    pipeline = get_loaded_pipeline()

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a standard CSV (.csv) file.",
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(exc)}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty (0 rows).",
        )

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Validate schema overlap to prevent silent misleading predictions on incompatible files
    known_clean_features = {f.lower().replace(" ", "_") for f in MODEL_FEATURE_COLUMNS}
    uploaded_clean_cols = {c.lower().replace(" ", "_") for c in df.columns}
    matching_features = known_clean_features.intersection(uploaded_clean_cols)

    if len(matching_features) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Incompatible CSV schema. Uploaded file contains only {len(matching_features)} recognizable customer features "
                f"out of {len(MODEL_FEATURE_COLUMNS)} required. At least 5 valid feature columns must be provided to prevent "
                "silent misleading predictions."
            ),
        )

    # Convert records into CustomerFeatures
    customers: List[CustomerFeatures] = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        cust_id = str(row_dict.get("CustomerID", f"Row_{idx+1}"))
        try:
            cust = CustomerFeatures(**row_dict, customer_id=cust_id)
            customers.append(cust)
        except Exception as exc:
            logger.warning("Row %d parsing failed: %s. Using default-filled record.", idx, exc)
            cust = CustomerFeatures(customer_id=cust_id)
            customers.append(cust)

    req = BatchPredictionRequest(customers=customers)
    return predict_batch_customers(req, threshold=threshold)


@app.get("/monitoring/telemetry", tags=["Monitoring"])
def get_monitoring_telemetry():
    """Retrieve aggregate telemetry and volume statistics from inference logs."""
    mon_logger = get_default_logger()
    return mon_logger.get_summary_stats()


@app.get("/monitoring/drift", tags=["Monitoring"])
def evaluate_monitoring_drift(min_samples: int = 20):
    """Run Kolmogorov-Smirnov and PSI covariate drift tests on logged inferences."""
    return run_drift_analysis(min_samples=min_samples)
