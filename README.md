# ChurnGuard — Customer Churn Prediction & Monitoring Platform

[![Python 3.14](https://img.shields.io/badge/Python-3.14.7-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17.2-blueviolet.svg)](https://mlflow.org/)
[![CI Pipeline](https://img.shields.io/badge/GitHub%20Actions-CI%20Passing-brightgreen.svg)](https://github.com/Ananya-M-15/ChurnGuard/actions)
[![Tests Passing](https://img.shields.io/badge/pytest-35%20passed-brightgreen.svg)](https://docs.pytest.org/)

An end-to-end, enterprise-grade customer churn prediction and monitoring platform with explainable machine learning, real-time REST API serving, interactive SaaS analytics dashboard, experiment tracking, automated privacy-compliant inference logging, statistical covariate drift monitoring, and GitHub Actions CI/CD automation.

---

## 1. System Architecture

ChurnGuard implements a modern, decoupled microservice pattern separating data ingestion, machine learning pipelines, REST inference serving, presentation dashboards, and observability telemetry:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                 │
│  Raw Excel (7,043 rows) ──> src/data.py ──> Parquet Partitions (80/20) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                        Machine Learning Pipeline                        │
│  src/features.py (ColumnTransformer: OHE + Median Imputer + Scaler)    │
│  src/train.py (5-Fold Stratified CV, MLflow Tracking: sqlite:///db)    │
│  src/evaluate.py (Untouched Test Evaluation, ROC/PR Curves)             │
│  Serialized Champion Artifact: models/pipeline.joblib (XGBoost)         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│                         Serving Layer (FastAPI)                         │
│  api/main.py (Uvicorn REST Engine, Port 8000)                           │
│  ├── GET  /health (Service Health & Model Telemetry)                    │
│  ├── POST /predict (Real-time Single Customer Scoring)                  │
│  ├── POST /predict/batch (Bulk JSON Scoring & Aggregation)              │
│  ├── POST /predict/batch/csv (Multipart CSV Upload & Parsing)           │
│  ├── GET  /monitoring/telemetry (Aggregate Inference Stats)             │
│  └── GET  /monitoring/drift (Real-time KS & PSI Drift Evaluation)       │
└───────────────────┬─────────────────────────────────┬───────────────────┘
                    │                                 │
                    │ HTTP / REST                     │ Telemetry Logging
┌───────────────────▼────────────────────────┐   ┌────▼──────────────────┐
│        Presentation Layer (Streamlit)      │   │    Monitoring Layer   │
│  dashboard/app.py (SaaS UI, Port 8501)     │   │  monitoring/logger.py │
│  ├── 📊 Executive Overview (KPIs & EDA)    │   │  (SQLite Inference DB)│
│  ├── 👤 Single Customer Scoring & Recs     │   │                       │
│  ├── 📁 Bulk Batch Prediction & CSV Export │   │  monitoring/drift.py  │
│  ├── 📈 Model Diagnostics & Metrics        │   │  (KS-Test + PSI Engine│
│  └── 📡 Telemetry & Drift Monitoring       │   │  against Train Parquet│
└────────────────────────────────────────────┘   └───────────────────────┘
```

---

## 2. Technology Stack

- **Core Machine Learning:** Scikit-Learn 1.9.0, XGBoost 3.4.1, SciPy 1.18.1, Joblib 1.5.3
- **Data Manipulation & Ingestion:** Pandas 3.0.5, NumPy 2.5.2, PyArrow 25.0.1, OpenPyXL 3.1.5
- **Backend Serving:** FastAPI 0.162.0, Uvicorn 0.54.0, Pydantic v2 (2.14.2)
- **Frontend Dashboard:** Streamlit 1.66.1, Altair 6.3.0, Requests 2.34.2
- **MLOps Telemetry & Drift:** SQLite3 (WAL-mode), SciPy 2-sample Kolmogorov-Smirnov (`scipy.stats.ks_2samp`), Population Stability Index (PSI)
- **Experiment Tracking:** MLflow 3.17.2 with SQLite backend (`sqlite:///mlflow.db`)
- **Automated Testing & CI/CD:** Pytest 9.2.0 (35 automated tests), GitHub Actions CI Workflow (`.github/workflows/ci.yml`)
- **Containerization & Deployment:** Docker (`python:3.11-slim`), Docker Compose, Render Blueprint (`render.yaml`)

---

## 3. Verified Model Performance

All models were evaluated using **5-Fold Stratified Cross-Validation exclusively on the training partition** (5,634 rows). Model selection prioritized **PR-AUC (Average Precision)** due to class imbalance (~26.54% churn rate):

### 5-Fold Stratified Cross-Validation Benchmark
| Model | CV PR-AUC (Mean ± Std) | CV ROC-AUC (Mean ± Std) | CV F1 (Mean ± Std) | CV Recall (Mean ± Std) | CV Precision | CV Accuracy | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Champion)** | **0.6813 (±0.0221)** | **0.8620 (±0.0112)** | **0.6475 (±0.0245)** | **0.7732 (±0.0362)** | **0.5570** | **0.7767** | **3.83s** |
| **Random Forest** | 0.6533 (±0.0151) | 0.8454 (±0.0116) | 0.6346 (±0.0106) | 0.8254 (±0.0246) | 0.5158 | 0.7476 | 4.04s |
| **Logistic Regression** | 0.6200 (±0.0187) | 0.8359 (±0.0135) | 0.6180 (±0.0234) | 0.7351 (±0.0300) | 0.5333 | 0.7588 | 3.66s |

### Test-Set Evaluation (Untouched Test Partition: 1,409 rows)
Evaluated once at the default classification threshold ($0.50$):
- **PR-AUC:** **0.6622** (+39.68% above the 26.54% no-skill baseline)
- **ROC-AUC:** **0.8522**
- **Recall (Sensitivity):** **77.54%** (Identified 290 out of 374 actual churners)
- **Precision:** **53.80%**
- **F1-Score:** **0.6353**
- **Accuracy:** **76.37%** (1,076 / 1,409 correct)

---

## 4. REST API Endpoint Specification

The FastAPI microservice runs on port `8000` with interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### 1. `GET /health`
Returns service availability, model architecture, model release version, and expected feature counts.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "XGBClassifier",
  "model_version": "1.0.0",
  "expected_features_count": 25,
  "timestamp": "2026-10-31T10:00:00.000Z"
}
```

### 2. `POST /predict`
Accepts a single customer record with the required 25 features. Returns probabilities and risk classification.
- **CustomerID Safety:** `CustomerID` is treated strictly as optional metadata and is never passed to `pipeline.predict_proba()`. Exactly 25 features are evaluated.
- **Risk Tiers:** Low ($< 0.35$), Medium ($0.35 \le p < 0.65$), High ($\ge 0.65$).
- **Classification:** `churn_prediction = "Yes"` if probability $\ge 0.50$, else `"No"`.
- **Telemetry:** Inferences are automatically logged to SQLite database with PII redacted.

### 3. `POST /predict/batch`
Accepts a JSON payload `{"customers": [...]}` containing multiple customer records. Returns predictions for all valid accounts plus aggregate summary KPIs.

### 4. `POST /predict/batch/csv`
Accepts a `multipart/form-data` CSV file upload, parses columns, and returns scored accounts with summary metrics.

### 5. `GET /monitoring/telemetry`
Returns aggregate inference telemetry from the SQLite database: total predictions, average probability, risk distribution, and traffic sources.

### 6. `GET /monitoring/drift`
Executes real-time two-sample Kolmogorov-Smirnov (KS) tests on 7 numerical features and Population Stability Index (PSI) on 18 categorical features against the 5,634-row baseline.

---

## 5. Streamlit Analytics Dashboard

Launch via: `streamlit run dashboard/app.py`

### Feature Walkthrough
1. **Executive Overview:** High-level portfolio metrics (7,043 total accounts, 73.5% retention, 26.5% historical churn), status distribution bar charts, high-impact behavioral patterns, and production model card.
2. **Single Customer Scoring:** Intuitive 4-section form (Demographics, Account, Services, Location) with smart defaults, real-time API call, prominent visual badge distinguishing `🟢 Remote API (FastAPI)` from `🟠 Local Model Fallback`, probability gauge, risk badge (Low/Medium/High), and automated retention recommendation strategy.
3. **Bulk Batch Prediction:** Drag-and-drop CSV upload, downloadable template, summary metric cards, filterable results table by risk tier, and one-click annotated CSV export.
4. **Model Diagnostics & Metrics:** Official test set metrics, 5-Fold CV comparison table, and high-resolution diagnostic plots (Confusion Matrix, ROC Curve, PR Curve, Feature Importance).
5. **Telemetry & Drift Monitoring:** Real-time MLOps observability center:
   - Live production inference counts, average churn probability, and risk distribution.
   - Recent inference event audit log (privacy redacted; no CustomerID or personal info).
   - "Insufficient monitoring data" empty state protection when $< 20$ samples are captured.
   - Interactive KS-Test and PSI tables across all 25 features with drift status flags.
   - Quick testing controls: Inject 50 stable test records or simulate 50 shifted records to observe live drift detection in action.

---

## 6. Local Setup & Execution Guide

### Prerequisites
- Python 3.11, 3.12, or 3.14 (tested on Windows AMD64)
- Git

### 1. Virtual Environment & Dependencies
```powershell
# Clone repository
git clone https://github.com/Ananya-M-15/ChurnGuard.git
cd ChurnGuard

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install curated dependencies
pip install -r requirements.txt
```

### 2. Run Pipeline Steps
```powershell
# 1. Data Ingestion & Sanitization
python -m src.data

# 2. Model Training & MLflow Experiment Tracking
python -m src.train

# 3. Champion Model Evaluation
python -m src.evaluate

# 4. Run Complete Automated Test Suite (35 tests)
pytest tests/ -v
```

### 3. Start Backend & Frontend
```powershell
# Terminal 1: Launch FastAPI Serving Backend
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch Streamlit Analytics Dashboard
streamlit run dashboard/app.py --server.port 8501
```

---

## 7. Automated CI/CD Pipeline (GitHub Actions)

ChurnGuard includes automated continuous integration configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### Workflow Triggers
- Every `push` to `main`
- Every `pull_request` targeting `main`

### Pipeline Stages
1. **Environment Setup:** Provisions an `ubuntu-latest` runner with Python 3.11 and pip caching.
2. **Dependency Resolution:** Installs all pinned dependencies from `requirements.txt`.
3. **Syntax & Bytecode Validation:** Compiles all Python source files (`src/`, `api/`, `dashboard/`, `monitoring/`, `tests/`) to prevent runtime syntax errors.
4. **Full Test Execution:** Executes all 35 unit, integration, API, and monitoring tests via `pytest -v --tb=short`.

---

## 8. Cloud & Container Deployment

### Option A: Render Cloud Deployment (FastAPI + Streamlit)
A turnkey blueprint is configured in [`render.yaml`](render.yaml):
1. Sign in to [render.com](https://render.com).
2. Connect your GitHub repository: `https://github.com/Ananya-M-15/ChurnGuard`.
3. Select **Blueprints** -> **New Blueprint Instance**.
4. Render automatically spins up:
   - `churnguard-api`: Web service running FastAPI on port 8000 with health-check at `/health`.
   - `churnguard-dashboard`: Web service running Streamlit on port 8501 with `CHURNGUARD_API_URL` dynamically linked.

### Option B: Streamlit Community Cloud
1. Fork or push repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select repo `Ananya-M-15/ChurnGuard`, Branch `main`, Main file path `dashboard/app.py`.
4. In Advanced Settings, add environment variable:
   ```env
   CHURNGUARD_API_URL=https://churnguard-api.onrender.com
   ```
5. Deploy! (The dashboard includes an automatic local model fallback so it remains functional even if the remote API is warming up).

### Option C: Docker / Docker Compose
```bash
docker-compose up --build
```
- API accessible at `http://localhost:8000`
- Dashboard accessible at `http://localhost:8501`

---

## 9. Faculty Demonstration Sequence (9-Step Guide)

For academic evaluation or project presentation, follow this sequence:

1. **Step 1: System Health & API Telemetry**
   - Open browser at `http://127.0.0.1:8000/health`.
   - Show `status: "healthy"`, champion model `XGBClassifier`, and 25 expected features.
   - Open Swagger UI at `http://127.0.0.1:8000/docs`.

2. **Step 2: Executive Overview Dashboard**
   - Open Streamlit dashboard at `http://localhost:8501`.
   - Navigate to **"📊 Executive Overview"**.
   - Show verified portfolio KPIs (7,043 accounts, 26.54% churn rate) and behavioral drivers (contract length, payment method, tenure).

3. **Step 3: Single Customer Scoring & Explainability**
   - Navigate to **"👤 Single Customer Scoring"**.
   - Review pre-filled customer details and click **"Predict Churn Risk"**.
   - Point out the visible badge: `🟢 Remote API (FastAPI)` confirming the live REST connection.
   - Show churn probability (e.g. 78.4%), risk tier (`HIGH RISK`), and the auto-generated proactive retention playbook.

4. **Step 4: Bulk CSV Batch Scoring & Annotation Export**
   - Navigate to **"📁 Bulk Batch Prediction"**.
   - Download the pre-built sample CSV template.
   - Upload the sample file and click **"Process Batch Predictions"**.
   - Show batch summary metrics (scored count, churn count, high-risk count) and download the annotated CSV results.

5. **Step 5: Verified Model Diagnostics**
   - Navigate to **"📈 Model Diagnostics & Metrics"**.
   - Review 5-Fold Stratified Cross-Validation table comparing XGBoost, Random Forest, and Logistic Regression.
   - Review Confusion Matrix (77.54% recall on test set) and Precision-Recall Curve.

6. **Step 6: MLflow Experiment Tracking & Lineage**
   - Launch MLflow UI in terminal: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
   - Open `http://127.0.0.1:5000` to show logged parameters, CV metrics, and serialized artifacts.

7. **Step 7: Production Inference Telemetry & Privacy Redaction**
   - Navigate to **"📡 Telemetry & Drift Monitoring"**.
   - Show the live inference counter incrementing from previous predictions.
   - Show the recent inference log table. Point out that `CustomerID`, customer names, and PII are strictly excluded.

8. **Step 8: Statistical Covariate Drift Detection & Simulation**
   - Under Telemetry, click **"🧪 Inject 50 Test Records"** (stable traffic) and observe drift status remaining `🟢 Normal (Stable)`.
   - Click **"🚨 Inject 50 Shifted Records"** (simulated drift) and show immediate transition to `🔴 Alert (Drift Detected)`.
   - Review the detailed KS-test table (numerical features) and PSI table (categorical features) showing exact p-values and statistics.

9. **Step 9: Automated Testing & CI/CD Verification**
   - In terminal, execute: `pytest tests/ -v`.
   - Show all 35 tests passing cleanly across data, features, training, API, and monitoring modules.
   - Open GitHub Actions tab to show the green CI pipeline check.

---

## 10. Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `CHURNGUARD_API_URL` | `http://127.0.0.1:8000` | URL of the FastAPI backend service |
| `CHURNGUARD_MONITORING_DB` | `data/monitoring.db` | Custom path for SQLite inference telemetry database |
| `PORT` | `8000` | Port for web server binding in container/cloud environments |
| `PYTHONPATH` | `.` | Module resolution root |

---

## 11. Known Limitations & Roadmap

- **Geographic Representation:** Geographic coordinates (`Latitude`, `Longitude`) are currently standardized as numerical features. Spatial density clustering (HDBSCAN) is planned for future iterations.
- **Explainability:** Feature importance currently leverages tree-based gain. Global and local SHAP explanations will be incorporated once stable binary wheels for Python 3.14 are released.
- **Automated Retraining Trigger:** The monitoring system detects covariate drift; an automated webhook can be attached to retrain and deploy the champion pipeline upon critical drift alert.
