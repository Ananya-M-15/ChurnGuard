"""Privacy-compliant, SQLite-backed production inference logger for ChurnGuard.

Captures timestamped model inference events (probabilities, predictions,
risk tiers, and feature payloads) while strictly preventing CustomerID
and personal identifiable information (PII) from being persisted.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import BASE_DIR

logger = logging.getLogger(__name__)

DEFAULT_MONITORING_DB = BASE_DIR / "data" / "monitoring.db"

PROHIBITED_PII_KEYS = {
    "customerid",
    "customer_id",
    "customername",
    "customer_name",
    "name",
    "email",
    "ssn",
    "phone_number",
    "phonenumber",
    "id",
}


class InferenceLogger:
    """Manages recording and retrieval of model inference logs in a local SQLite database."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        if db_path is None:
            env_path = os.getenv("CHURNGUARD_MONITORING_DB")
            self.db_path = Path(env_path) if env_path else DEFAULT_MONITORING_DB
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with appropriate timeout and row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Initialize the inference_logs schema and performance indexes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prediction_source TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    churn_probability REAL NOT NULL,
                    predicted_class TEXT NOT NULL,
                    risk_tier TEXT NOT NULL,
                    features_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON inference_logs(timestamp);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_source ON inference_logs(prediction_source);"
            )
            conn.commit()

    @staticmethod
    def _sanitize_features(features: Dict[str, Any]) -> Dict[str, Any]:
        """Strip CustomerID and any PII keys before serializing to storage."""
        sanitized = {}
        for k, v in features.items():
            key_clean = str(k).strip().lower()
            if key_clean in PROHIBITED_PII_KEYS:
                continue
            # Handle numpy and pandas numeric types for clean JSON
            if hasattr(v, "item"):
                v = v.item()
            sanitized[str(k).strip()] = v
        return sanitized

    def log_prediction(
        self,
        source: str,
        model_name: str,
        model_version: str,
        churn_probability: float,
        predicted_class: str,
        risk_tier: str,
        features: Dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> int:
        """Log a single customer inference event.

        Returns:
            The primary key row ID of the inserted record.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        clean_feats = self._sanitize_features(features)
        feats_str = json.dumps(clean_feats)

        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO inference_logs (
                    timestamp, prediction_source, model_name, model_version,
                    churn_probability, predicted_class, risk_tier, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    timestamp,
                    str(source),
                    str(model_name),
                    str(model_version),
                    float(churn_probability),
                    str(predicted_class),
                    str(risk_tier),
                    feats_str,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def log_batch(
        self,
        source: str,
        model_name: str,
        model_version: str,
        records: List[Dict[str, Any]],
    ) -> int:
        """Log a batch of inference records in a single database transaction.

        Each element in `records` must contain:
        - `churn_probability`: float
        - `predicted_class`: str ("Yes" / "No")
        - `risk_tier`: str ("Low" / "Medium" / "High")
        - `features`: dict
        - Optional `timestamp`: str

        Returns:
            The number of successfully logged records.
        """
        if not records:
            return 0

        now_utc = datetime.now(timezone.utc).isoformat()
        insert_rows = []

        for r in records:
            ts = r.get("timestamp", now_utc)
            prob = float(r["churn_probability"])
            pred = str(r["predicted_class"])
            risk = str(r["risk_tier"])
            clean_feats = self._sanitize_features(r.get("features", {}))
            insert_rows.append(
                (
                    ts,
                    str(source),
                    str(model_name),
                    str(model_version),
                    prob,
                    pred,
                    risk,
                    json.dumps(clean_feats),
                )
            )

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO inference_logs (
                    timestamp, prediction_source, model_name, model_version,
                    churn_probability, predicted_class, risk_tier, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                insert_rows,
            )
            conn.commit()

        return len(insert_rows)

    def get_log_count(self) -> int:
        """Return the total number of logged inference events."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM inference_logs;")
            row = cur.fetchone()
            return row[0] if row else 0

    def get_recent_logs(self, limit: int = 100) -> pd.DataFrame:
        """Fetch the most recent logged inference events as a pandas DataFrame."""
        with self._get_connection() as conn:
            query = """
                SELECT id, timestamp, prediction_source, model_name, model_version,
                       churn_probability, predicted_class, risk_tier, features_json
                FROM inference_logs
                ORDER BY id DESC
                LIMIT ?;
            """
            rows = conn.execute(query, (limit,)).fetchall()

        if not rows:
            return pd.DataFrame(
                columns=[
                    "id",
                    "timestamp",
                    "prediction_source",
                    "model_name",
                    "model_version",
                    "churn_probability",
                    "predicted_class",
                    "risk_tier",
                ]
            )

        records = []
        for r in rows:
            row_dict = {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "prediction_source": r["prediction_source"],
                "model_name": r["model_name"],
                "model_version": r["model_version"],
                "churn_probability": r["churn_probability"],
                "predicted_class": r["predicted_class"],
                "risk_tier": r["risk_tier"],
            }
            records.append(row_dict)

        return pd.DataFrame(records)

    def get_feature_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Extract a clean DataFrame of feature payloads for drift monitoring."""
        with self._get_connection() as conn:
            if limit:
                query = "SELECT features_json FROM inference_logs ORDER BY id DESC LIMIT ?;"
                rows = conn.execute(query, (limit,)).fetchall()
            else:
                query = "SELECT features_json FROM inference_logs ORDER BY id ASC;"
                rows = conn.execute(query).fetchall()

        if not rows:
            return pd.DataFrame()

        feature_dicts = []
        for r in rows:
            try:
                feats = json.loads(r["features_json"])
                feature_dicts.append(feats)
            except Exception:
                continue

        return pd.DataFrame(feature_dicts)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary telemetry KPIs from logged inferences."""
        total = self.get_log_count()
        if total == 0:
            return {
                "total_inferences": 0,
                "average_churn_probability": 0.0,
                "predicted_churn_count": 0,
                "predicted_retention_count": 0,
                "risk_breakdown": {"Low": 0, "Medium": 0, "High": 0},
                "sources": {},
                "last_logged_at": None,
            }

        with self._get_connection() as conn:
            # Average probability and class counts
            agg = conn.execute(
                """
                SELECT AVG(churn_probability),
                       SUM(CASE WHEN predicted_class = 'Yes' THEN 1 ELSE 0 END),
                       MAX(timestamp)
                FROM inference_logs;
                """
            ).fetchone()
            avg_prob = float(agg[0]) if agg[0] is not None else 0.0
            churn_count = int(agg[1]) if agg[1] is not None else 0
            last_ts = agg[2]

            # Risk breakdown
            risk_rows = conn.execute(
                "SELECT risk_tier, COUNT(*) FROM inference_logs GROUP BY risk_tier;"
            ).fetchall()
            risk_dist = {"Low": 0, "Medium": 0, "High": 0}
            for r, c in risk_rows:
                risk_dist[r] = c

            # Source breakdown
            src_rows = conn.execute(
                "SELECT prediction_source, COUNT(*) FROM inference_logs GROUP BY prediction_source;"
            ).fetchall()
            source_dist = {s: c for s, c in src_rows}

        return {
            "total_inferences": total,
            "average_churn_probability": round(avg_prob, 4),
            "predicted_churn_count": churn_count,
            "predicted_retention_count": total - churn_count,
            "risk_breakdown": risk_dist,
            "sources": source_dist,
            "last_logged_at": last_ts,
        }

    def clear_logs(self) -> int:
        """Purge all logged records (used primarily for test cleanup)."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM inference_logs;")
            conn.commit()
            return cur.rowcount


_GLOBAL_LOGGER: Optional[InferenceLogger] = None


def get_default_logger() -> InferenceLogger:
    """Return singleton logger instance."""
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = InferenceLogger()
    return _GLOBAL_LOGGER
