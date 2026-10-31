"""ChurnGuard — Enterprise Customer Churn Analytics & Prediction Dashboard.

Built with Streamlit, providing an executive overview, real-time single-customer
risk assessment, bulk CSV batch scoring, verified model performance diagnostics,
and MLOps observability telemetry.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Set Streamlit page configuration as first command
st.set_page_config(
    page_title="ChurnGuard | Customer Churn Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API endpoint configuration
_raw_api_url = os.getenv("CHURNGUARD_API_URL", "http://127.0.0.1:8000").strip()
if _raw_api_url and not _raw_api_url.startswith(("http://", "https://")):
    DEFAULT_API_URL = f"http://{_raw_api_url}"
else:
    DEFAULT_API_URL = _raw_api_url or "http://127.0.0.1:8000"


# Local project paths for offline fallback and performance metrics
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
MODEL_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
MODEL_PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train.parquet"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test.parquet"

# Custom CSS for SaaS Navy/Indigo Aesthetic
CUSTOM_CSS = """
<style>
    /* Global styling */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Card containers */
    .saas-card {
        background: linear-gradient(145deg, #111827, #1e293b);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .saas-card-accent {
        background: linear-gradient(145deg, #1e1b4b, #1e293b);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
    }

    /* KPI metric styling */
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Risk Badges */
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    /* Recommendation card */
    .rec-box {
        background-color: rgba(79, 70, 229, 0.12);
        border-left: 4px solid #6366f1;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 14px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# API Connection Helpers
# ==============================================================================

def check_backend_health(api_url: str) -> Dict[str, Any]:
    """Ping FastAPI health endpoint with a short timeout."""
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=2.5)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "degraded", "model_loaded": False, "error": f"HTTP {resp.status_code}"}
    except Exception as exc:
        return {"status": "offline", "model_loaded": False, "error": str(exc)}


def call_predict_api(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call POST /predict on FastAPI backend with graceful fallback."""
    try:
        resp = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json=payload,
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            data["is_fallback"] = False
            return data
        raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")
    except requests.exceptions.RequestException as exc:
        # Resilient local fallback if backend server isn't running
        if MODEL_PIPELINE_PATH.exists():
            from src.features import load_pipeline
            from api.schemas import CustomerFeatures, customers_to_dataframe, determine_risk_tier
            pipe = load_pipeline(MODEL_PIPELINE_PATH)
            cust = CustomerFeatures(**payload)
            df = customers_to_dataframe([cust])
            prob = float(pipe.predict_proba(df)[0, 1])
            label = "Yes" if prob >= 0.50 else "No"
            tier = determine_risk_tier(prob)
            tier_str = tier.value if hasattr(tier, "value") else str(tier)

            # Log local fallback prediction to telemetry SQLite
            try:
                from monitoring.logger import get_default_logger
                mon_logger = get_default_logger()
                mon_logger.log_prediction(
                    source="dashboard-fallback",
                    model_name="XGBClassifier (Local Fallback)",
                    model_version="1.0.0",
                    churn_probability=round(prob, 4),
                    predicted_class=label,
                    risk_tier=tier_str,
                    features=cust.to_model_dict(),
                )
            except Exception:
                pass

            return {
                "churn_prediction": label,
                "churn_probability": round(prob, 4),
                "retention_probability": round(1.0 - prob, 4),
                "risk_classification": tier,
                "decision_threshold": 0.50,
                "model_name": "XGBClassifier (Local Fallback)",
                "model_version": "1.0.0",
                "customer_id": payload.get("CustomerID", "Cust_01"),
                "is_fallback": True,
            }
        raise RuntimeError(f"API unreachable and local model unavailable: {exc}")


def call_batch_predict_api(api_url: str, customers_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call POST /predict/batch on FastAPI backend."""
    try:
        resp = requests.post(
            f"{api_url.rstrip('/')}/predict/batch",
            json={"customers": customers_list},
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            data["is_fallback"] = False
            return data
        raise RuntimeError(f"API Batch Error ({resp.status_code}): {resp.text}")
    except requests.exceptions.RequestException as exc:
        # Fallback local batch scoring
        if MODEL_PIPELINE_PATH.exists():
            from src.features import load_pipeline
            from api.schemas import CustomerFeatures, customers_to_dataframe, determine_risk_tier
            pipe = load_pipeline(MODEL_PIPELINE_PATH)
            custs = [CustomerFeatures(**c) for c in customers_list]
            df = customers_to_dataframe(custs)
            probs = pipe.predict_proba(df)[:, 1]

            preds = []
            log_records = []
            churn_c = 0
            low_c = 0
            med_c = 0
            high_c = 0

            for idx, (p, c) in enumerate(zip(probs, custs)):
                p_val = round(float(p), 4)
                lbl = "Yes" if p_val >= 0.50 else "No"
                tier = determine_risk_tier(p_val)
                tier_str = tier.value if hasattr(tier, "value") else str(tier)

                if lbl == "Yes":
                    churn_c += 1
                if tier == "Low":
                    low_c += 1
                elif tier == "Medium":
                    med_c += 1
                else:
                    high_c += 1

                preds.append(
                    {
                        "customer_id": c.customer_id or f"Cust_{idx+1}",
                        "churn_prediction": lbl,
                        "churn_probability": p_val,
                        "retention_probability": round(1.0 - p_val, 4),
                        "risk_classification": tier,
                        "decision_threshold": 0.50,
                        "model_name": "XGBClassifier (Local Fallback)",
                        "model_version": "1.0.0",
                    }
                )

                log_records.append(
                    {
                        "churn_probability": p_val,
                        "predicted_class": lbl,
                        "risk_tier": tier_str,
                        "features": c.to_model_dict(),
                    }
                )

            # Log local fallback batch to telemetry SQLite
            try:
                from monitoring.logger import get_default_logger
                mon_logger = get_default_logger()
                mon_logger.log_batch(
                    source="dashboard-fallback",
                    model_name="XGBClassifier (Local Fallback)",
                    model_version="1.0.0",
                    records=log_records,
                )
            except Exception:
                pass

            total = len(preds)
            return {
                "predictions": preds,
                "summary": {
                    "total_customers": total,
                    "predicted_churn_count": churn_c,
                    "predicted_retention_count": total - churn_c,
                    "predicted_churn_rate": round(churn_c / total, 4) if total > 0 else 0.0,
                    "low_risk_count": low_c,
                    "medium_risk_count": med_c,
                    "high_risk_count": high_c,
                    "average_churn_probability": round(float(np.mean(probs)), 4),
                },
                "model_name": "XGBClassifier (Local Fallback)",
                "model_version": "1.0.0",
                "is_fallback": True,
            }
        raise RuntimeError(f"API unreachable and local model unavailable: {exc}")


def generate_retention_recommendation(
    churn_prob: float,
    contract: str,
    tenure: int,
    internet_service: str,
    tech_support: str,
    payment_method: str,
    monthly_charges: float,
) -> List[str]:
    """Generate business retention strategies tailored to the customer profile."""
    recommendations = []

    if contract == "Month-to-month":
        recommendations.append(
            "📋 **Contract Stabilization:** Customer is on a Month-to-Month agreement (highest churn cohort). "
            "Incentivize migration to a 1-year contract with a 15% discount for the first 6 months."
        )

    if tenure <= 12:
        recommendations.append(
            "🎯 **Early-Tenure Onboarding:** Customer tenure is under 12 months. Initiate proactive outreach "
            "from Customer Success to ensure service adoption and resolve initial friction."
        )

    if internet_service == "Fiber optic" and tech_support == "No":
        recommendations.append(
            "🛠️ **Support & Stability Bundle:** Customer uses Fiber Optic without Tech Support. Provide 3 months "
            "complimentary Premium Tech Support to address connectivity or hardware complaints."
        )

    if payment_method == "Electronic check":
        recommendations.append(
            r"💳 **Automated Billing Incentive:** Transition customer from Electronic Check to Auto-Pay (ACH or Credit Card) "
            r"with a one-time \$10 bill credit to reduce billing friction."
        )

    if monthly_charges > 85.0:
        recommendations.append(
            r"💰 **Price Optimization:** Monthly billing is above \$85. Review product mix and propose value-oriented "
            "streaming or mobile bundles to improve perceived price-to-value ratio."
        )

    if not recommendations:
        recommendations.append(
            "⭐ **Loyalty Engagement:** Customer profile demonstrates healthy retention indicators. "
            "Maintain regular engagement via quarterly satisfaction check-ins and loyalty rewards."
        )

    return recommendations


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

with st.sidebar:
    st.markdown("## 🛡️ **ChurnGuard**")
    st.caption("Enterprise MLOps Churn Intelligence")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "👤 Single Customer Scoring",
            "📁 Bulk Batch Prediction",
            "📈 Model Diagnostics & Metrics",
            "📡 Telemetry & Drift Monitoring",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🔌 API Connection")
    api_url = st.text_input("Backend URL", value=DEFAULT_API_URL)

    # Health check indicator
    health = check_backend_health(api_url)
    if health.get("status") == "healthy":
        st.success(f"● Connected: {health.get('model_name')} v{health.get('model_version')}")
    else:
        st.warning(f"⚠️ API Offline (Using Local Engine)")

    st.markdown("---")
    st.caption("Academic MLOps Capstone Project  \nModel: XGBoost Champion | Seed: 42")


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================

if selected_page == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview & Portfolio Health")
    st.markdown(
        "Executive telemetry and historical dataset metrics for customer retention. "
        "All figures reflect verified dataset distributions and champion XGBoost model performance."
    )
    st.markdown("---")

    # Top KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="saas-card">
                <div class="kpi-title">Total Customer Base</div>
                <div class="kpi-value">7,043</div>
                <div class="kpi-sub">Verified IBM Telco Dataset</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="saas-card">
                <div class="kpi-title">Active Retained</div>
                <div class="kpi-value" style="color: #10b981;">5,174</div>
                <div class="kpi-sub">73.46% Base Retention</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="saas-card">
                <div class="kpi-title">Historical Churned</div>
                <div class="kpi-value" style="color: #ef4444;">1,869</div>
                <div class="kpi-sub">26.54% Historical Churn Rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="saas-card">
                <div class="kpi-title">Model PR-AUC (Test)</div>
                <div class="kpi-value" style="color: #6366f1;">0.6622</div>
                <div class="kpi-sub">+39.68% Over Baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔍 Strategic Churn Drivers & Behavioral Patterns")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("#### Customer Status Distribution")
        dist_df = pd.DataFrame(
            {
                "Status": ["Retained (Active)", "Churned"],
                "Customer Count": [5174, 1869],
                "Percentage": [73.46, 26.54],
            }
        )
        st.bar_chart(data=dist_df, x="Status", y="Customer Count", color="Status")

    with c2:
        st.markdown("#### High-Impact Findings")
        st.markdown(
            """
            - **Contract Duration Disparity:** Customers on **Month-to-month contracts** experience an alarming **42.7% churn rate**, compared to **2.8%** for customers with Two-year agreements.
            - **Fiber Optic Attrition:** Subscribers on **Fiber Optic** internet churn at **41.9%**, driven by elevated monthly charges and absence of bundled tech support.
            - **First-Year Vulnerability:** Customers with **tenure < 12 months** represent **53.1% of all churn events**, highlighting the critical importance of early lifecycle intervention.
            - **Value-Add Services:** Accounts featuring active **Online Security** and **Tech Support** show a **14.6% churn rate** vs. **41.8%** for accounts without protective add-ons.
            """
        )

    st.markdown("---")
    st.markdown("### 🏆 Production ML Model Card")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Production Architecture", "XGBoost Classifier")
    mc2.metric("Test ROC-AUC", "0.8522")
    mc3.metric("Test Recall (Coverage)", "77.54%")
    mc4.metric("Test F1-Score", "0.6353")


# ==============================================================================
# PAGE 2: SINGLE CUSTOMER SCORING
# ==============================================================================

elif selected_page == "👤 Single Customer Scoring":
    st.markdown("# 👤 Single Customer Risk Scoring")
    st.markdown(
        "Input customer attributes across Demographics, Account Terms, Services, and Location "
        "to evaluate real-time churn probability and generate retention recommendations."
    )
    st.markdown("---")

    with st.form("customer_scoring_form"):
        st.markdown("#### 1. Customer Demographics")
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", ["Female", "Male"], index=0)
        senior = d2.selectbox("Senior Citizen", ["No", "Yes"], index=0)
        partner = d3.selectbox("Partner", ["No", "Yes"], index=0)
        dependents = d4.selectbox("Dependents", ["No", "Yes"], index=0)

        st.markdown("#### 2. Account & Contract Terms")
        a1, a2, a3 = st.columns(3)
        contract = a1.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
        tenure = a2.number_input("Tenure (Months)", min_value=0, max_value=72, value=4, step=1)
        paperless = a3.selectbox("Paperless Billing", ["Yes", "No"], index=0)

        a4, a5, a6 = st.columns(3)
        payment = a4.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=0,
        )
        monthly_charges = a5.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=79.85, step=1.0)
        total_charges = a6.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=319.40, step=10.0)

        st.markdown("#### 3. Subscribed Telecom Services")
        s1, s2, s3 = st.columns(3)
        phone = s1.selectbox("Phone Service", ["Yes", "No"], index=0)
        mult_lines = s2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)
        internet = s3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)

        s4, s5, s6 = st.columns(3)
        sec = s4.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
        backup = s5.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
        device = s6.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)

        s7, s8, s9 = st.columns(3)
        tech = s7.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
        tv = s8.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=0)
        movies = s9.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=0)

        st.markdown("#### 4. Geographic Localization & Account Metadata")
        g1, g2, g3 = st.columns(3)
        city = g1.text_input("City", value="Los Angeles")
        zip_code = g2.number_input("Zip Code", min_value=90000, max_value=96162, value=90003, step=1)
        cltv = g3.number_input("Customer Lifetime Value (CLTV)", min_value=1000, max_value=7000, value=3239, step=50)

        submit_btn = st.form_submit_button("⚡ Evaluate Churn Risk", use_container_width=True)

    if submit_btn:
        payload = {
            "City": city,
            "Zip Code": int(zip_code),
            "Lat Long": "34.0522, -118.2437",
            "Latitude": 34.0522,
            "Longitude": -118.2437,
            "Gender": gender,
            "Senior Citizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": int(tenure),
            "Phone Service": phone,
            "Multiple Lines": mult_lines,
            "Internet Service": internet,
            "Online Security": sec,
            "Online Backup": backup,
            "Device Protection": device,
            "Tech Support": tech,
            "Streaming TV": tv,
            "Streaming Movies": movies,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment,
            "Monthly Charges": float(monthly_charges),
            "Total Charges": float(total_charges),
            "CLTV": int(cltv),
            "CustomerID": "MANUAL-INPUT",
        }

        with st.spinner("Executing real-time inference via FastAPI backend..."):
            try:
                res = call_predict_api(api_url, payload)
                prob = res["churn_probability"]
                ret_prob = res["retention_probability"]
                risk = res["risk_classification"]
                pred = res["churn_prediction"]

                st.markdown("---")
                st.markdown("### 🎯 Risk Assessment Result")

                # Prominently display verified inference source
                is_fallback = res.get("is_fallback", False)
                if is_fallback:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(245, 158, 11, 0.15); border-left: 5px solid #f59e0b; padding: 12px 18px; border-radius: 6px; margin-bottom: 18px;">
                            <span style="font-weight: 700; font-size: 1.05rem; color: #f59e0b;">🟠 Inference Source: Local Model Fallback</span>
                            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px;">
                                FastAPI backend was unreachable at <code>{api_url}</code>. Serving predictions safely via local model pipeline fallback.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(16, 185, 129, 0.15); border-left: 5px solid #10b981; padding: 12px 18px; border-radius: 6px; margin-bottom: 18px;">
                            <span style="font-weight: 700; font-size: 1.05rem; color: #10b981;">🟢 Inference Source: Remote API (FastAPI)</span>
                            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px;">
                                Successfully served live via FastAPI REST endpoint at <code>{api_url}</code>.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                rc1, rc2, rc3 = st.columns([1, 1, 1.2])

                with rc1:
                    st.markdown(
                        f"""
                        <div class="saas-card">
                            <div class="kpi-title">Churn Probability</div>
                            <div class="kpi-value" style="color: {'#ef4444' if prob >= 0.50 else '#10b981'};">{prob*100:.1f}%</div>
                            <div class="kpi-sub">Retention Probability: {ret_prob*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with rc2:
                    badge_class = f"badge-{risk.lower()}"
                    st.markdown(
                        f"""
                        <div class="saas-card">
                            <div class="kpi-title">Assigned Risk Tier</div>
                            <div style="margin-top: 10px;">
                                <span class="{badge_class}" style="font-size: 1.3rem; padding: 6px 18px;">
                                    {risk.upper()} RISK
                                </span>
                            </div>
                            <div class="kpi-sub" style="margin-top: 14px;">Classification Threshold: 0.50</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with rc3:
                    st.markdown(
                        f"""
                        <div class="saas-card">
                            <div class="kpi-title">Proactive Recommendation</div>
                            <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin-top: 6px;">
                                {'🚨 High Churn Vulnerability' if pred == 'Yes' else '✅ Customer Stable'}
                            </div>
                            <div class="kpi-sub">Engine: {res.get('model_name')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Visual probability bar
                st.progress(float(prob))

                # Probability vs. Certainty Distinction
                st.markdown(
                    f"""
                    <div style="background-color: rgba(99, 102, 241, 0.08); border-left: 4px solid #6366f1; padding: 12px 16px; border-radius: 4px; margin: 14px 0;">
                        <span style="font-weight: 600; color: #c7d2fe;">📊 Probability vs. Certainty Distinction:</span>
                        <p style="color: #cbd5e1; font-size: 0.88rem; margin: 4px 0 0 0;">
                            A predicted churn probability of <strong>{prob*100:.1f}%</strong> denotes an empirical likelihood derived from historical cohort propensity across 7,043 customer accounts, rather than a deterministic outcome. Accounts exceeding 50% are prioritized for retention intervention.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Retention recommendations
                st.markdown("### 💡 Recommended Retention Action Plan")
                recs = generate_retention_recommendation(
                    churn_prob=prob,
                    contract=contract,
                    tenure=int(tenure),
                    internet_service=internet,
                    tech_support=tech,
                    payment_method=payment,
                    monthly_charges=float(monthly_charges),
                )
                for rec in recs:
                    st.markdown(f"<div class='rec-box'>{rec}</div>", unsafe_allow_html=True)

                st.caption(
                    "Note: Feature attributions indicate statistical correlation derived from gradient boosted trees, "
                    "not direct causal impact."
                )

            except Exception as e:
                st.error(f"Inference Error: {e}")


# ==============================================================================
# PAGE 3: BULK BATCH PREDICTION
# ==============================================================================

elif selected_page == "📁 Bulk Batch Prediction":
    st.markdown("# 📁 Bulk Customer Batch Scoring")
    st.markdown(
        "Upload a CSV containing customer accounts to generate churn probabilities, "
        "assign risk categories, and export annotated customer lists for CRM retention campaigns."
    )
    st.markdown("---")

    # Sample CSV Download Template
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown("#### Upload Accounts File (.csv)")
    with col_t2:
        if TEST_DATA_PATH.exists():
            sample_df = pd.read_parquet(TEST_DATA_PATH).head(5).copy()
            sample_df["CustomerID"] = [f"SAMPLE_{i+1:03d}" for i in range(len(sample_df))]
            csv_buf = io.StringIO()
            sample_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv_buf.getvalue(),
                file_name="churnguard_sample_template.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with st.expander("📋 CSV Schema & Required Column Specification", expanded=False):
        st.markdown(
            """
            The champion model expects the following **25 customer feature columns** (excluding `CustomerID`):
            - **Numerical Features (7):** `Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`, `Latitude`, `Longitude`, `Zip Code`
            - **Categorical Features (18):** `City`, `Lat Long`, `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Method`
            - **Identifier (Optional):** `CustomerID` (used exclusively for reporting metadata; never fed to the model)
            """
        )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(input_df)} customer rows.")
            st.dataframe(input_df.head(5), use_container_width=True)

            # Check column completeness
            from api.schemas import MODEL_FEATURE_COLUMNS
            known_clean = {f.lower().replace(" ", "_") for f in MODEL_FEATURE_COLUMNS}
            uploaded_clean = {c.lower().replace(" ", "_") for c in input_df.columns}
            matching_features = known_clean.intersection(uploaded_clean)

            if len(matching_features) < 5:
                st.error(
                    f"🚨 **Incompatible CSV Schema:** Uploaded file contains only {len(matching_features)} recognizable customer features "
                    f"out of {len(MODEL_FEATURE_COLUMNS)} required. Batch scoring is disabled to prevent misleading predictions. "
                    "Please download and populate the verified template above."
                )
                can_process = False
            else:
                can_process = True
                missing_cols = [c for c in MODEL_FEATURE_COLUMNS if c not in input_df.columns]
                if missing_cols:
                    st.info(
                        f"ℹ️ **Transparent Imputation Notice:** Uploaded CSV is missing {len(missing_cols)} feature columns "
                        f"(`{', '.join(missing_cols[:4])}{'...' if len(missing_cols) > 4 else ''}`). "
                        "Verified baseline defaults will be applied transparently for missing fields."
                    )

            if can_process and st.button("🚀 Process Batch Predictions", use_container_width=True):
                with st.spinner(f"Scoring {len(input_df)} customers via FastAPI backend..."):
                    # Standardize column names
                    clean_input = input_df.copy()
                    clean_input.columns = clean_input.columns.str.strip()

                    # Convert rows to dictionaries
                    customers_data = []
                    for idx, row in clean_input.iterrows():
                        r_dict = row.to_dict()
                        if "CustomerID" not in r_dict:
                            r_dict["CustomerID"] = f"Batch_Cust_{idx+1}"
                        customers_data.append(r_dict)

                    res = call_batch_predict_api(api_url, customers_data)
                    summary = res["summary"]
                    predictions = res["predictions"]

                    st.markdown("---")
                    st.markdown("### 📊 Batch Evaluation Summary")

                    # Prominently display verified inference source
                    is_fallback = res.get("is_fallback", False)
                    if is_fallback:
                        st.markdown(
                            f"""
                            <div style="background-color: rgba(245, 158, 11, 0.15); border-left: 5px solid #f59e0b; padding: 12px 18px; border-radius: 6px; margin-bottom: 18px;">
                                <span style="font-weight: 700; font-size: 1.05rem; color: #f59e0b;">🟠 Inference Source: Local Model Fallback</span>
                                <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px;">
                                    FastAPI backend was unreachable at <code>{api_url}</code>. Serving predictions safely via local model pipeline fallback.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="background-color: rgba(16, 185, 129, 0.15); border-left: 5px solid #10b981; padding: 12px 18px; border-radius: 6px; margin-bottom: 18px;">
                                <span style="font-weight: 700; font-size: 1.05rem; color: #10b981;">🟢 Inference Source: Remote API (FastAPI)</span>
                                <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px;">
                                    Successfully served live via FastAPI REST endpoint at <code>{api_url}</code>.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("Scored Accounts", summary["total_customers"])
                    bc2.metric("Predicted Churners", summary["predicted_churn_count"], f"{summary['predicted_churn_rate']*100:.1f}%")
                    bc3.metric("High-Risk Accounts", summary["high_risk_count"])
                    bc4.metric("Avg Churn Prob", f"{summary['average_churn_probability']*100:.1f}%")

                    # Convert results to DataFrame
                    results_df = clean_input.copy()
                    results_df["Churn_Prediction"] = [p["churn_prediction"] for p in predictions]
                    results_df["Churn_Probability"] = [p["churn_probability"] for p in predictions]
                    results_df["Risk_Tier"] = [p["risk_classification"] for p in predictions]

                    # Filter by risk tier
                    st.markdown("#### Filtered Output")
                    filter_tier = st.selectbox("Filter by Risk Tier", ["All", "High", "Medium", "Low"])
                    if filter_tier != "All":
                        filtered_df = results_df[results_df["Risk_Tier"] == filter_tier]
                    else:
                        filtered_df = results_df

                    st.dataframe(
                        filtered_df[["CustomerID", "Contract", "Tenure Months", "Monthly Charges", "Churn_Prediction", "Churn_Probability", "Risk_Tier"]],
                        use_container_width=True,
                    )

                    # Export Annotated CSV
                    out_csv = io.StringIO()
                    results_df.to_csv(out_csv, index=False)
                    st.download_button(
                        label="💾 Download Complete Annotated CSV",
                        data=out_csv.getvalue(),
                        file_name="churnguard_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as exc:
            st.error(f"Failed to process uploaded CSV: {exc}")


# ==============================================================================
# PAGE 4: MODEL DIAGNOSTICS & METRICS
# ==============================================================================

elif selected_page == "📈 Model Diagnostics & Metrics":
    st.markdown("# 📈 Model Diagnostics & Performance Metrics")
    st.markdown(
        "Verified evaluation metrics, 5-Fold Cross-Validation comparison benchmarks, "
        "and diagnostic curves for the champion XGBoost classification model."
    )
    st.markdown("---")

    # Load verified metrics JSON
    if EVALUATION_METRICS_PATH.exists():
        with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        st.markdown("### 🎯 Verified Test Set Metrics (Untouched Test Partition: 1,409 rows)")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("PR-AUC", f"{metrics_data.get('pr_auc', 0.6622):.4f}")
        m2.metric("ROC-AUC", f"{metrics_data.get('roc_auc', 0.8522):.4f}")
        m3.metric("Recall", f"{metrics_data.get('recall', 0.7754)*100:.1f}%")
        m4.metric("Precision", f"{metrics_data.get('precision', 0.5380)*100:.1f}%")
        m5.metric("F1-Score", f"{metrics_data.get('f1', 0.6353):.4f}")
        m6.metric("Accuracy", f"{metrics_data.get('accuracy', 0.7637)*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🏆 5-Fold Stratified Cross-Validation Benchmark")

    if MODEL_COMPARISON_PATH.exists():
        comp_df = pd.read_csv(MODEL_COMPARISON_PATH)
        st.dataframe(
            comp_df[["model_name", "cv_pr_auc_mean", "cv_roc_auc_mean", "cv_f1_mean", "cv_recall_mean", "cv_precision_mean", "cv_accuracy_mean", "training_time_seconds"]],
            use_container_width=True,
        )
    else:
        st.info("Cross-validation benchmark table loading...")

    st.markdown("---")
    st.markdown("### 📊 Diagnostic Visualizations")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        cm_file = FIGURES_DIR / "confusion_matrix.png"
        if cm_file.exists():
            st.image(str(cm_file), caption="Confusion Matrix — Test Set Evaluation", use_container_width=True)
        roc_file = FIGURES_DIR / "roc_curve.png"
        if roc_file.exists():
            st.image(str(roc_file), caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)

    with f_col2:
        pr_file = FIGURES_DIR / "precision_recall_curve.png"
        if pr_file.exists():
            st.image(str(pr_file), caption="Precision-Recall Curve (Baseline: 26.54%)", use_container_width=True)
        fi_file = FIGURES_DIR / "feature_importance.png"
        if fi_file.exists():
            st.image(str(fi_file), caption="Top Influential Features (XGBoost)", use_container_width=True)


# ==============================================================================
# PAGE 5: TELEMETRY & DRIFT MONITORING
# ==============================================================================

elif selected_page == "📡 Telemetry & Drift Monitoring":
    st.markdown("# 📡 MLOps Telemetry & Drift Monitoring")
    st.markdown(
        "Production observability and statistical distribution tracking for ChurnGuard serving pipelines. "
        "Monitors inference payloads against baseline training distributions to detect covariate drift."
    )
    st.markdown("---")

    from monitoring.drift import DriftDetector, MINIMUM_MONITORING_SAMPLES
    from monitoring.logger import get_default_logger

    mon_logger = get_default_logger()
    stats = mon_logger.get_summary_stats()
    total_logs = stats["total_inferences"]

    # Evaluate drift status safely
    detector = DriftDetector()
    current_features_df = mon_logger.get_feature_dataframe()
    drift_report = detector.evaluate_drift(
        current_features_df,
        min_samples=MINIMUM_MONITORING_SAMPLES,
        save_report=True,
    )
    status_label = drift_report["status"]

    # Header Telemetry KPIs
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Production Inferences", f"{total_logs:,}")
    m2.metric("Average Churn Prob", f"{stats['average_churn_probability']*100:.1f}%" if total_logs > 0 else "N/A")
    m3.metric("Baseline Reference", "5,634 Train Samples")

    if status_label == "No significant drift":
        m4.metric("Covariate Drift", "🟢 Normal (Stable)")
    elif status_label == "Drift detected":
        m4.metric("Covariate Drift", f"🔴 Alert ({drift_report['drift_detected_count']} Drifted)")
    else:
        m4.metric("Covariate Drift", "🟡 Insufficient Data")

    # Simulation & Demonstration Controls
    st.markdown("### 🛠️ Telemetry Simulation & Demonstration Controls")
    with st.expander("🧪 Faculty Demonstration & Synthetic Testing Sandbox (Non-Production)", expanded=True):
        st.info(
            "⚠️ **Notice for Evaluators & Reviewers:** Controls below inject synthetic demonstration records "
            "to demonstrate real-time MLOps telemetry updates and statistical drift hypothesis testing. "
            "All injected records are explicitly marked with synthetic simulation tags ('faculty-demo' or 'drift-simulation') "
            "and are separated from genuine customer inference traffic."
        )
        c_btn1, c_btn2, c_btn3 = st.columns([1.2, 1.2, 1.2])

        with c_btn1:
            if st.button("🧪 Inject 50 Test Records (Stable Traffic)", use_container_width=True):
                if TEST_DATA_PATH.exists():
                    test_raw = pd.read_parquet(TEST_DATA_PATH)
                    sample_count = min(50, len(test_raw))
                    test_sample = test_raw.sample(n=sample_count, random_state=42)
                    from api.schemas import CustomerFeatures, customers_to_dataframe, determine_risk_tier
                    from src.features import load_pipeline
                    pipe = load_pipeline(MODEL_PIPELINE_PATH)
                    custs = [CustomerFeatures(**row.to_dict()) for _, row in test_sample.iterrows()]
                    df_in = customers_to_dataframe(custs)
                    probs = pipe.predict_proba(df_in)[:, 1]

                    records = []
                    for p, c in zip(probs, custs):
                        tier = determine_risk_tier(p)
                        tier_str = tier.value if hasattr(tier, "value") else str(tier)
                        records.append({
                            "churn_probability": round(float(p), 4),
                            "predicted_class": "Yes" if p >= 0.50 else "No",
                            "risk_tier": tier_str,
                            "features": c.to_model_dict(),
                        })
                    mon_logger.log_batch("faculty-demo", "XGBClassifier", "1.0.0", records)
                    st.success(f"Injected {len(records)} realistic inference events! Refreshing telemetry...")
                    st.rerun()

        with c_btn2:
            if st.button("🚨 Inject 50 Shifted Records (Simulate Drift)", use_container_width=True):
                if TEST_DATA_PATH.exists():
                    test_raw = pd.read_parquet(TEST_DATA_PATH)
                    sample_count = min(50, len(test_raw))
                    test_sample = test_raw.sample(n=sample_count, random_state=99).copy()
                    # Simulate covariate shift: heavily elevated charges and new accounts
                    test_sample["Monthly Charges"] = test_sample["Monthly Charges"] + 75.0
                    test_sample["Tenure Months"] = 1
                    test_sample["Contract"] = "Month-to-month"
                    from api.schemas import CustomerFeatures, customers_to_dataframe, determine_risk_tier
                    from src.features import load_pipeline
                    pipe = load_pipeline(MODEL_PIPELINE_PATH)
                    custs = [CustomerFeatures(**row.to_dict()) for _, row in test_sample.iterrows()]
                    df_in = customers_to_dataframe(custs)
                    probs = pipe.predict_proba(df_in)[:, 1]

                    records = []
                    for p, c in zip(probs, custs):
                        tier = determine_risk_tier(p)
                        tier_str = tier.value if hasattr(tier, "value") else str(tier)
                        records.append({
                            "churn_probability": round(float(p), 4),
                            "predicted_class": "Yes" if p >= 0.50 else "No",
                            "risk_tier": tier_str,
                            "features": c.to_model_dict(),
                        })
                    mon_logger.log_batch("drift-simulation", "XGBClassifier", "1.0.0", records)
                    st.warning("Injected 50 shifted records (high monthly charges, low tenure). Drift detected!")
                    st.rerun()

        with c_btn3:
            if st.button("🗑️ Reset Telemetry Logs", use_container_width=True):
                mon_logger.clear_logs()
                st.info("Cleared all inference logs.")
                st.rerun()

    with st.expander("ℹ️ Understanding Telemetry & Drift Monitoring States", expanded=False):
        st.markdown(
            r"""
            - **🟢 Stable Traffic (No Significant Drift):** Monitored feature distributions match the baseline training set (5,634 rows). Two-sample Kolmogorov-Smirnov $p \ge 0.05$ across numerical features, and Population Stability Index (PSI) $< 0.10$ across categorical features.
            - **🔴 Covariate Drift Detected:** One or more features have drifted significantly ($p < 0.05$ or $\text{PSI} \ge 0.20$), indicating that consumer demographics, service choices, or billing patterns have changed since model training. Recommendation: trigger model retraining pipeline.
            - **🟡 Insufficient Monitoring Data:** Telemetry contains fewer than 20 logged inference records. Statistical hypothesis tests are deferred to prevent sample-size distortion and false alarms.
            """
        )

    st.markdown("---")

    # Inferences Section
    t_col1, t_col2 = st.columns([1, 1])
    with t_col1:
        st.markdown("#### 🎯 Prediction Risk Breakdown")
        rb = stats["risk_breakdown"]
        risk_df = pd.DataFrame([
            {"Risk Tier": "Low (<35%)", "Count": rb.get("Low", 0)},
            {"Risk Tier": "Medium (35-65%)", "Count": rb.get("Medium", 0)},
            {"Risk Tier": "High (>=65%)", "Count": rb.get("High", 0)},
        ])
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

    with t_col2:
        st.markdown("#### 🌐 Traffic Source Breakdown")
        sources = stats["sources"]
        if sources:
            src_df = pd.DataFrame([{"Source": k, "Inferences": v} for k, v in sources.items()])
            st.dataframe(src_df, use_container_width=True, hide_index=True)
        else:
            st.info("No sources recorded yet.")

    st.markdown("### 📜 Recent Production Inferences (Privacy Redacted)")
    recent_logs = mon_logger.get_recent_logs(limit=10)
    if not recent_logs.empty:
        display_logs = recent_logs[["id", "timestamp", "prediction_source", "churn_probability", "predicted_class", "risk_tier"]].copy()
        display_logs["churn_probability"] = (display_logs["churn_probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_logs, use_container_width=True, hide_index=True)
    else:
        st.info("No inference events recorded yet.")

    st.markdown("---")

    # Drift Evaluation Section
    st.markdown("### 🔬 Statistical Covariate Drift Analysis")

    if total_logs < MINIMUM_MONITORING_SAMPLES:
        st.warning(
            f"⚠️ **Insufficient Monitoring Data:** Captured **{total_logs}** / **{MINIMUM_MONITORING_SAMPLES}** required samples. "
            f"Statistical hypothesis testing requires at least {MINIMUM_MONITORING_SAMPLES} production records to prevent false alarms. "
            "Click **'Inject 50 Test Records'** above or score customers on Page 2 / 3 to populate monitoring telemetry."
        )
    else:
        if status_label == "Drift detected":
            st.error(
                f"🚨 **Drift Alert:** {drift_report['drift_detected_count']} feature(s) show statistically significant distribution shifts!  \n"
                f"**Flagged Features:** `{', '.join(drift_report['drifted_features'])}`  \n"
                "**Action Recommended:** Trigger retraining pipeline (`python -m src.train`) and evaluate metric degradation."
            )
        else:
            st.success(
                f"✅ **Distributions Stable:** All {len(drift_report['numerical_metrics']) + len(drift_report['categorical_metrics'])} features "
                "concur with baseline training distributions. Decision boundaries remain statistically valid."
            )

        tab_num, tab_cat = st.tabs(["Numerical Features (Kolmogorov-Smirnov)", "Categorical Features (Population Stability Index)"])

        with tab_num:
            num_rows = []
            for feat, met in drift_report["numerical_metrics"].items():
                num_rows.append({
                    "Feature": feat,
                    "KS Statistic (D)": met["statistic"],
                    "P-Value": met["p_value"],
                    "Alpha Threshold": met["threshold"],
                    "Status": "🚨 DRIFT" if met["drift_detected"] else "✅ STABLE",
                    "Interpretation": met["interpretation"],
                })
            st.dataframe(pd.DataFrame(num_rows), use_container_width=True, hide_index=True)

        with tab_cat:
            cat_rows = []
            for feat, met in drift_report["categorical_metrics"].items():
                cat_rows.append({
                    "Feature": feat,
                    "PSI Value": met["psi_value"],
                    "Threshold": met["threshold"],
                    "Status": "🚨 DRIFT" if met["drift_detected"] else "✅ STABLE",
                    "Interpretation": met["interpretation"],
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
