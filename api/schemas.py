"""Pydantic v2 schemas for ChurnGuard REST API requests and responses.

Defines validated input payloads for single and batch customer churn predictions,
response models with probabilities and risk classifications, and health status schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RiskClassification(str, Enum):
    """Categorical risk tiers based on predicted churn probability."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Exact feature column ordering expected by the trained pipeline
MODEL_FEATURE_COLUMNS = [
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
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
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]


class CustomerFeatures(BaseModel):
    """Input payload representing one customer's attributes for churn prediction.

    Supports both original column names ('Tenure Months') and clean snake_case
    aliases ('tenure_months').
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    city: str = Field(
        default="Los Angeles",
        validation_alias=AliasChoices("City", "city"),
        description="Customer residence city",
    )
    zip_code: int = Field(
        default=90001,
        validation_alias=AliasChoices("Zip Code", "zip_code", "ZipCode"),
        description="Customer 5-digit zip code",
    )
    lat_long: str = Field(
        default="34.0522, -118.2437",
        validation_alias=AliasChoices("Lat Long", "lat_long", "LatLong"),
        description="Comma-separated latitude and longitude",
    )
    latitude: float = Field(
        default=34.0522,
        validation_alias=AliasChoices("Latitude", "latitude"),
        description="Customer location latitude",
    )
    longitude: float = Field(
        default=-118.2437,
        validation_alias=AliasChoices("Longitude", "longitude"),
        description="Customer location longitude",
    )
    gender: str = Field(
        default="Female",
        validation_alias=AliasChoices("Gender", "gender"),
        description="Customer gender (Male, Female)",
    )
    senior_citizen: str = Field(
        default="No",
        validation_alias=AliasChoices("Senior Citizen", "senior_citizen", "SeniorCitizen"),
        description="Senior citizen status (Yes, No)",
    )
    partner: str = Field(
        default="No",
        validation_alias=AliasChoices("Partner", "partner"),
        description="Has partner (Yes, No)",
    )
    dependents: str = Field(
        default="No",
        validation_alias=AliasChoices("Dependents", "dependents"),
        description="Has dependents (Yes, No)",
    )
    tenure_months: int = Field(
        default=1,
        ge=0,
        validation_alias=AliasChoices("Tenure Months", "tenure_months", "TenureMonths"),
        description="Customer account tenure in months",
    )
    phone_service: str = Field(
        default="Yes",
        validation_alias=AliasChoices("Phone Service", "phone_service", "PhoneService"),
        description="Has phone service (Yes, No)",
    )
    multiple_lines: str = Field(
        default="No",
        validation_alias=AliasChoices("Multiple Lines", "multiple_lines", "MultipleLines"),
        description="Multiple telephone lines (Yes, No, No phone service)",
    )
    internet_service: str = Field(
        default="DSL",
        validation_alias=AliasChoices("Internet Service", "internet_service", "InternetService"),
        description="Internet provider type (DSL, Fiber optic, No)",
    )
    online_security: str = Field(
        default="No",
        validation_alias=AliasChoices("Online Security", "online_security", "OnlineSecurity"),
        description="Online security add-on (Yes, No, No internet service)",
    )
    online_backup: str = Field(
        default="No",
        validation_alias=AliasChoices("Online Backup", "online_backup", "OnlineBackup"),
        description="Online backup add-on (Yes, No, No internet service)",
    )
    device_protection: str = Field(
        default="No",
        validation_alias=AliasChoices("Device Protection", "device_protection", "DeviceProtection"),
        description="Device protection plan (Yes, No, No internet service)",
    )
    tech_support: str = Field(
        default="No",
        validation_alias=AliasChoices("Tech Support", "tech_support", "TechSupport"),
        description="Premium tech support (Yes, No, No internet service)",
    )
    streaming_tv: str = Field(
        default="No",
        validation_alias=AliasChoices("Streaming TV", "streaming_tv", "StreamingTV"),
        description="Streaming TV service (Yes, No, No internet service)",
    )
    streaming_movies: str = Field(
        default="No",
        validation_alias=AliasChoices("Streaming Movies", "streaming_movies", "StreamingMovies"),
        description="Streaming movie service (Yes, No, No internet service)",
    )
    contract: str = Field(
        default="Month-to-month",
        validation_alias=AliasChoices("Contract", "contract"),
        description="Contract duration (Month-to-month, One year, Two year)",
    )
    paperless_billing: str = Field(
        default="Yes",
        validation_alias=AliasChoices("Paperless Billing", "paperless_billing", "PaperlessBilling"),
        description="Paperless billing opted (Yes, No)",
    )
    payment_method: str = Field(
        default="Electronic check",
        validation_alias=AliasChoices("Payment Method", "payment_method", "PaymentMethod"),
        description="Billing payment method",
    )
    monthly_charges: float = Field(
        default=50.0,
        ge=0.0,
        validation_alias=AliasChoices("Monthly Charges", "monthly_charges", "MonthlyCharges"),
        description="Monthly recurring charge amount",
    )
    total_charges: float = Field(
        default=50.0,
        ge=0.0,
        validation_alias=AliasChoices("Total Charges", "total_charges", "TotalCharges"),
        description="Total lifetime billed charges",
    )
    cltv: int = Field(
        default=3000,
        ge=0,
        validation_alias=AliasChoices("CLTV", "cltv"),
        description="Customer Lifetime Value index",
    )

    customer_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CustomerID", "customer_id", "customerId"),
        description="Optional customer identifier for tracking",
    )

    def to_model_dict(self) -> Dict[str, Any]:
        """Convert fields to a dictionary with exact model feature column names."""
        return {
            "City": str(self.city),
            "Zip Code": int(self.zip_code),
            "Lat Long": str(self.lat_long),
            "Latitude": float(self.latitude),
            "Longitude": float(self.longitude),
            "Gender": str(self.gender),
            "Senior Citizen": str(self.senior_citizen),
            "Partner": str(self.partner),
            "Dependents": str(self.dependents),
            "Tenure Months": int(self.tenure_months),
            "Phone Service": str(self.phone_service),
            "Multiple Lines": str(self.multiple_lines),
            "Internet Service": str(self.internet_service),
            "Online Security": str(self.online_security),
            "Online Backup": str(self.online_backup),
            "Device Protection": str(self.device_protection),
            "Tech Support": str(self.tech_support),
            "Streaming TV": str(self.streaming_tv),
            "Streaming Movies": str(self.streaming_movies),
            "Contract": str(self.contract),
            "Paperless Billing": str(self.paperless_billing),
            "Payment Method": str(self.payment_method),
            "Monthly Charges": float(self.monthly_charges),
            "Total Charges": float(self.total_charges),
            "CLTV": int(self.cltv),
        }


def customers_to_dataframe(customers: List[CustomerFeatures]) -> pd.DataFrame:
    """Transform a list of CustomerFeatures into a DataFrame matching model columns."""
    rows = [c.to_model_dict() for c in customers]
    df = pd.DataFrame(rows)
    return df[MODEL_FEATURE_COLUMNS]


def determine_risk_tier(churn_probability: float) -> RiskClassification:
    """Classify risk into Low, Medium, High tiers.

    Risk Tiers:
    - Low Risk:    prob < 0.35
    - Medium Risk: 0.35 <= prob < 0.65
    - High Risk:   prob >= 0.65
    """
    if churn_probability < 0.35:
        return RiskClassification.LOW
    elif churn_probability < 0.65:
        return RiskClassification.MEDIUM
    else:
        return RiskClassification.HIGH


class PredictionResponse(BaseModel):
    """Output prediction response for a single customer."""

    customer_id: Optional[str] = Field(default=None, description="Optional customer ID")
    churn_prediction: str = Field(description="Binary classification ('Yes' or 'No')")
    churn_probability: float = Field(ge=0.0, le=1.0, description="Estimated churn probability")
    retention_probability: float = Field(ge=0.0, le=1.0, description="Estimated retention probability")
    risk_classification: RiskClassification = Field(description="Risk tier: Low, Medium, High")
    decision_threshold: float = Field(default=0.50, description="Decision threshold applied")
    model_name: str = Field(description="Champion model architecture")
    model_version: str = Field(description="Model release version")


class BatchPredictionRequest(BaseModel):
    """Request payload containing a list of customer records for bulk scoring."""

    customers: List[CustomerFeatures] = Field(
        min_length=1,
        description="List of customer records to predict",
    )


class BatchSummary(BaseModel):
    """Aggregated statistics across a batch of customer predictions."""

    total_customers: int
    predicted_churn_count: int
    predicted_retention_count: int
    predicted_churn_rate: float
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    average_churn_probability: float


class BatchPredictionResponse(BaseModel):
    """Complete batch prediction response including row-by-row outputs and summary statistics."""

    predictions: List[PredictionResponse]
    summary: BatchSummary
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """System health check and model status response."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    expected_features_count: int
    timestamp: str
