"""Integration and unit tests for ChurnGuard FastAPI backend endpoints.

Verifies system health checks, single customer prediction, schema validation,
batch scoring, CSV uploads, error handling, probability boundaries, and risk tiering.
"""

import io
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from api.main import app
from src.config import TEST_DATA_PATH


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a TestClient with lifespan events executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_customer_payload() -> dict:
    """Fixture providing a valid single customer payload."""
    return {
        "City": "Los Angeles",
        "Zip Code": 90003,
        "Lat Long": "33.964131, -118.272783",
        "Latitude": 33.964131,
        "Longitude": -118.272783,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 2,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Mailed check",
        "Monthly Charges": 53.85,
        "Total Charges": 108.15,
        "CLTV": 3239,
        "CustomerID": "3668-QPYBK",
    }


def test_root_endpoint(client: TestClient):
    """Verify root endpoint returns basic service info and status 200."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


def test_health_endpoint(client: TestClient):
    """Verify health endpoint confirms model is loaded and ready."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "XGBClassifier"
    assert data["expected_features_count"] == 25


def test_single_prediction_valid_input(client: TestClient, sample_customer_payload: dict):
    """Verify single customer prediction returns valid prediction and probability fields."""
    response = client.post("/predict", json=sample_customer_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert 0.0 <= data["retention_probability"] <= 1.0
    assert abs(data["churn_probability"] + data["retention_probability"] - 1.0) < 0.001
    assert data["risk_classification"] in ["Low", "Medium", "High"]
    assert data["decision_threshold"] == 0.50
    assert data["model_version"] == "1.0.0"
    assert data["customer_id"] == "3668-QPYBK"


def test_single_prediction_risk_tier_consistency(client: TestClient, sample_customer_payload: dict):
    """Verify risk classification aligns with documented threshold bounds."""
    response = client.post("/predict", json=sample_customer_payload)
    assert response.status_code == 200
    data = response.json()
    prob = data["churn_probability"]
    risk = data["risk_classification"]

    if prob < 0.35:
        assert risk == "Low"
    elif prob < 0.65:
        assert risk == "Medium"
    else:
        assert risk == "High"


def test_single_prediction_invalid_negative_numeric(client: TestClient, sample_customer_payload: dict):
    """Verify Pydantic validation rejects negative numeric values with 422 Unprocessable Entity."""
    invalid_payload = sample_customer_payload.copy()
    invalid_payload["Tenure Months"] = -5  # Must be >= 0

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_batch_prediction_json(client: TestClient, sample_customer_payload: dict):
    """Verify structured batch JSON prediction computes individual and aggregate KPIs."""
    cust1 = sample_customer_payload.copy()
    cust1["CustomerID"] = "Cust_01"
    cust2 = sample_customer_payload.copy()
    cust2["CustomerID"] = "Cust_02"
    cust2["Contract"] = "Two year"
    cust2["Tenure Months"] = 60

    batch_payload = {"customers": [cust1, cust2]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert len(data["predictions"]) == 2
    summary = data["summary"]
    assert summary["total_customers"] == 2
    assert summary["predicted_churn_count"] + summary["predicted_retention_count"] == 2
    assert 0.0 <= summary["predicted_churn_rate"] <= 1.0
    assert 0.0 <= summary["average_churn_probability"] <= 1.0


def test_batch_prediction_empty_list_rejected(client: TestClient):
    """Verify batch endpoint rejects an empty customer list with 422 validation error."""
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422


def test_batch_prediction_csv_upload(client: TestClient):
    """Verify CSV file upload endpoint processes records and returns structured response."""
    assert TEST_DATA_PATH.exists(), f"Test dataset not found at {TEST_DATA_PATH}"
    df = pd.read_parquet(TEST_DATA_PATH).head(10)
    df["CustomerID"] = [f"TEST_CSV_{i}" for i in range(len(df))]

    # Convert to CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    files = {"file": ("test_customers.csv", csv_bytes, "text/csv")}
    response = client.post("/predict/batch/csv", files=files)
    assert response.status_code == 200
    data = response.json()

    assert len(data["predictions"]) == 10
    assert data["summary"]["total_customers"] == 10


def test_batch_prediction_csv_upload_invalid_extension(client: TestClient):
    """Verify uploading non-CSV files returns 400 Bad Request."""
    dummy_bytes = b"Hello, this is not a CSV"
    files = {"file": ("document.txt", dummy_bytes, "text/plain")}
    response = client.post("/predict/batch/csv", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]


def test_batch_prediction_csv_upload_incompatible_schema(client: TestClient):
    """Verify uploading CSV with completely unrelated schema is rejected with 400 Bad Request."""
    bad_csv = "Item,Price,Quantity\nWidget,19.99,2\nGadget,29.99,1\n".encode("utf-8")
    files = {"file": ("inventory.csv", bad_csv, "text/csv")}
    response = client.post("/predict/batch/csv", files=files)
    assert response.status_code == 400
    assert "Incompatible CSV schema" in response.json()["detail"]


def test_customer_id_never_passed_to_model(client: TestClient, sample_customer_payload: dict):
    """Verify CustomerID is strictly treated as metadata and never passed to the ML pipeline."""
    from api.schemas import CustomerFeatures, customers_to_dataframe

    cust = CustomerFeatures(**sample_customer_payload)
    assert cust.customer_id == "3668-QPYBK"

    # Convert to dataframe
    df = customers_to_dataframe([cust])
    assert "CustomerID" not in df.columns
    assert "customer_id" not in df.columns
    assert df.shape == (1, 25)

    # API response retains CustomerID for tracking but passes 25 features to model
    response = client.post("/predict", json=sample_customer_payload)
    assert response.status_code == 200
    assert response.json()["customer_id"] == "3668-QPYBK"


def test_monitoring_endpoints(client: TestClient, sample_customer_payload: dict):
    """Verify monitoring telemetry and drift endpoints respond correctly."""
    # Score a customer to trigger logging
    client.post("/predict", json=sample_customer_payload)

    # 1. Telemetry endpoint
    resp_telemetry = client.get("/monitoring/telemetry")
    assert resp_telemetry.status_code == 200
    stats = resp_telemetry.json()
    assert "total_inferences" in stats
    assert stats["total_inferences"] >= 1
    assert "risk_breakdown" in stats

    # 2. Drift endpoint
    resp_drift = client.get("/monitoring/drift?min_samples=50")
    assert resp_drift.status_code == 200
    drift_data = resp_drift.json()
    assert "status" in drift_data
    assert "reference_sample_count" in drift_data
