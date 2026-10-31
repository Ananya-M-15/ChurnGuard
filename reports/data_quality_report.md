# ChurnGuard — Data Quality & Preparation Report (Phase 1)

**Generated Date:** October 28, 2026
**Pipeline Module:** `src/data.py`
**Configuration:** `src/config.py`
**Dataset Source:** `data/raw/Telco_customer_churn.xlsx`

---

## 1. Executive Summary

This report documents the data ingestion, validation, sanitization, leakage elimination, and stratified splitting performed during **Phase 1: Environment Stabilization & Data Foundation** of the ChurnGuard project. All values reported below are computed directly from the actual dataset without synthetic fabrication.

---

## 2. Dataset Dimensions & Volume

| Metric | Measured Value | Verification Method |
| :--- | :---: | :--- |
| **Raw Row Count** | 7,043 | Confirmed via `pd.read_excel()` |
| **Raw Column Count** | 33 | Complete column schema verified |
| **Exact Duplicate Rows** | 0 | `df.duplicated().sum() == 0` |
| **Duplicate Customer IDs** | 0 | `df['CustomerID'].duplicated().sum() == 0` |
| **Data Retention Rate** | 100.0% (7,043 / 7,043) | Zero customer records dropped |

---

## 3. Target Variable Analysis

The target variable for customer churn prediction is established as **`Churn Value`**.

* **Target Name:** `Churn Value`
* **Data Type:** Binary Integer (`int64`, values $\in \{0, 1\}$)
* **Overall Distribution:**
  * Active / Retained (`0`): **5,174 records (73.46%)**
  * Churned (`1`): **1,869 records (26.54%)**
  * Imbalance Ratio: **2.77 : 1**

---

## 4. Missing-Value Treatment & Sanitization

| Feature Column | Raw Missing / Whitespace Count | Nature of Missingness | Cleaning Strategy Applied |
| :--- | :---: | :--- | :--- |
| **`Total Charges`** | 11 | Whitespace-only strings (`" "`) occurring exclusively when `Tenure Months == 0`. | Coerced to `NaN` using numeric parsing, then imputed with `0.0`. A customer with zero months of tenure has accrued \$0.0 in total billing to date. Preserves 100% of customer observations. |
| **`Churn Reason`** | 5,174 | Populated only for the 1,869 churned customers; null for all 5,174 active customers. | Dropped entirely from feature sets as retrospective outcome leakage. |
| **All Other Columns (31)** | 0 | None. Complete records. | String columns stripped of whitespace padding. |

---

## 5. Feature Elimination & Leakage Prevention

To ensure strict model validity and eliminate training-serving contamination, 7 columns were removed prior to feature partitioning:

| Excluded Column | Category | Technical Rationale for Removal |
| :--- | :--- | :--- |
| **`Churn Label`** | Target Duplicate | Categorical string ("Yes"/"No") duplicating `Churn Value`. |
| **`Churn Reason`** | 100% Target Leakage | Retrospective survey reason recorded only after churn occurred. If retained, any model trivially predicts churn based on field presence. |
| **`Churn Score`** | Target Contamination | IBM Telco pre-computed proprietary risk score (0–100). Training on this would model upstream predictions rather than customer behavior. |
| **`CustomerID`** | Identifier | Unique alphanumeric account key (7,043 unique values). Would cause memorization and severe overfitting. |
| **`Count`** | Zero-Variance Invariant | Constant integer `1` across all 7,043 observations. Zero information gain. |
| **`Country`** | Zero-Variance Invariant | Constant string `'United States'` across all 7,043 observations. Zero information gain. |
| **`State`** | Zero-Variance Invariant | Constant string `'California'` across all 7,043 observations. Zero information gain. |

---

## 6. Final Feature Set Specification

* **Final Column Count in Processed Data:** **26 columns** (25 predictive features + 1 target column `Churn Value`).
* **Preserved Predictive Features (25):**
  1. **Demographics (4):** `Gender`, `Senior Citizen`, `Partner`, `Dependents`
  2. **Account & Contract Terms (7):** `Tenure Months`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, `Total Charges`, `CLTV`
  3. **Subscribed Services (9):** `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`
  4. **Geographic Localization (5):** `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`

---

## 7. Train / Test Partitioning & Stratification

Splitting was executed using `sklearn.model_selection.train_test_split` with a fixed random seed (`42`) and stratification on `Churn Value`.

| Split Partition | Row Count | Column Count | Active Count (`0`) | Active Share (%) | Churn Count (`1`) | Churn Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Clean Dataset** | 7,043 | 26 | 5,174 | 73.46% | 1,869 | 26.54% |
| **Training Partition (80%)** | 5,634 | 26 | 4,139 | 73.46% | 1,495 | 26.54% |
| **Testing Partition (20%)** | 1,409 | 26 | 1,035 | 73.46% | 374 | 26.54% |

* **Index Disjointness:** Verified 0 index overlap between training and testing sets ($\text{Train} \cap \text{Test} = \emptyset$).
* **Stratification Quality:** Exact parity maintained between train (26.54%) and test (26.54%) churn rates.

---

## 8. Persisted Artifacts

| Artifact Name | Storage Path | Format | Verification Status |
| :--- | :--- | :---: | :--- |
| **Training Dataset** | `data/processed/train.parquet` | Apache Parquet (Snappy) | Reloaded and schema-verified (5,634 rows, 26 cols) |
| **Testing Dataset** | `data/processed/test.parquet` | Apache Parquet (Snappy) | Reloaded and schema-verified (1,409 rows, 26 cols) |

---
*No personally identifying customer information (PII) is included in this report.*
