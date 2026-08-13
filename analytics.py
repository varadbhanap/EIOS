"""
EIOS - Step 2: Analytics Engine

This file contains ZERO AI logic. Every function here is plain, deterministic
Python and statistics. The AI agent (built later) will call these functions
through the MCP server, but it never does the math itself, these functions do.

Run this file directly to sanity-check that everything works against your
real database before connecting anything to an AI agent.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

DB_PATH = "data/enterprise_data.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------------
# 1. SQL querying
# ---------------------------------------------------------------------------
def run_query(sql: str) -> list[dict]:
    """
    Executes any SQL query against the enterprise database and returns
    the results as a list of dictionaries (easy for an AI agent to read
    and easy to convert to JSON later).
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn)
        return df.to_dict(orient="records")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2. Anomaly detection
# ---------------------------------------------------------------------------
def detect_anomalies(table: str, column: str, z_thresh: float = 3.0) -> dict:
    """
    Flags statistical outliers in a numeric column using Z-scores.
    A Z-score tells you how many standard deviations a value is from the
    mean, values beyond the threshold (default 3.0) are flagged as anomalies.

    Returns a summary: how many anomalies were found, and the actual
    flagged rows (limited to 20 for readability).
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            f"SELECT SK_ID_CURR, {column} FROM {table} WHERE {column} IS NOT NULL",
            conn,
        )
    finally:
        conn.close()

    mean = df[column].mean()
    std = df[column].std()
    df["z_score"] = (df[column] - mean) / std

    anomalies = df[df["z_score"].abs() > z_thresh].copy()
    anomalies = anomalies.sort_values("z_score", ascending=False)

    return {
        "column": column,
        "mean": round(mean, 2),
        "std_dev": round(std, 2),
        "threshold": z_thresh,
        "total_rows_checked": len(df),
        "anomalies_found": len(anomalies),
        "sample_anomalies": anomalies.head(20).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 3. Clustering / segmentation
# ---------------------------------------------------------------------------
def segment_records(columns: list[str], n_clusters: int = 4) -> dict:
    """
    Groups loan applicants into segments using K-Means clustering, based
    on whichever numeric columns you pass in (e.g. income, credit amount,
    annuity). Returns the size and average profile of each segment.
    """
    conn = get_connection()
    try:
        col_str = ", ".join(columns)
        df = pd.read_sql_query(
            f"SELECT {col_str} FROM loan_applications", conn
        ).dropna()
    finally:
        conn.close()

    # exclude extreme outliers before clustering. K-Means groups by distance,
    # and a single record with an extreme value (this dataset is known to
    # contain at least one AMT_INCOME_TOTAL around 117 million against a
    # typical range of ~100,000-200,000) will always form its own isolated
    # "cluster" no matter how the data is scaled, since it's genuinely far
    # from everything else. Extreme outliers are detect_anomalies()'s job,
    # not segment_records()'s, so we cap each column at the 99th percentile
    # here to keep clustering focused on the typical population.
    for col in columns:
        cap = df[col].quantile(0.99)
        df = df[df[col] <= cap]

    # scale features first, K-Means uses raw distance, so a column with
    # large numbers (like income, sometimes in the hundreds of millions
    # for a handful of records) would otherwise dominate and distort
    # the clusters, even producing tiny single-record "clusters"
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[columns])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(scaled_values)

    segment_summary = (
        df.groupby("cluster")[columns]
        .mean()
        .round(2)
        .to_dict(orient="index")
    )
    segment_sizes = df["cluster"].value_counts().to_dict()

    return {
        "n_clusters": n_clusters,
        "columns_used": columns,
        "segment_sizes": segment_sizes,
        "segment_profiles": segment_summary,
    }


# ---------------------------------------------------------------------------
# 4. Forecasting
# ---------------------------------------------------------------------------
def forecast_timeseries(periods_ahead: int = 6) -> dict:
    """
    Builds a real monthly trend of loan application activity from the
    genuine DAYS_DECISION field in previous_applications (days relative
    to the current application, always negative), re-anchored to a fixed
    reference date so it reads as a normal calendar timeline. Then fits
    an ARIMA model to forecast the next few months.

    No values are invented, only the anchor date is arbitrary, the
    underlying day-offsets are real data.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT DAYS_DECISION FROM previous_applications "
            "WHERE DAYS_DECISION IS NOT NULL",
            conn,
        )
    finally:
        conn.close()

    anchor_date = pd.Timestamp("2024-01-01")
    df["decision_date"] = anchor_date + pd.to_timedelta(df["DAYS_DECISION"], unit="D")

    monthly = (
        df.set_index("decision_date")
        .resample("MS")
        .size()
        .sort_index()
    )
    # keep the most recent 24 months so the trend is meaningful, not noisy
    monthly = monthly.tail(24)

    model = ARIMA(monthly, order=(1, 1, 1))
    fitted = model.fit()
    forecast = fitted.forecast(steps=periods_ahead)

    return {
        "history": {str(k.date()): int(v) for k, v in monthly.items()},
        "forecast": {str(k.date()): round(float(v), 1) for k, v in forecast.items()},
    }


# ---------------------------------------------------------------------------
# 5. Default risk prediction (trained model)
# ---------------------------------------------------------------------------
_risk_model = None
_risk_config = None


def _load_risk_model():
    """
    Loads the trained LightGBM model and its feature config once, then
    reuses it. The model itself is trained separately by
    train_risk_model.py, this function only loads and applies it.
    """
    global _risk_model, _risk_config
    if _risk_model is None:
        _risk_model = lgb.Booster(model_file="models/risk_model.txt")
        with open("models/feature_config.json") as f:
            _risk_config = json.load(f)
    return _risk_model, _risk_config


def predict_default_risk(sk_id_curr: int) -> dict:
    """
    Predicts the default risk for a specific applicant using the trained
    LightGBM model. Looks the applicant up by SK_ID_CURR, applies the same
    cleaning used during training, and returns a risk probability and
    category.
    """
    model, config = _load_risk_model()
    features = config["features"]
    medians = config["medians"]

    conn = get_connection()
    try:
        col_str = ", ".join(features)
        df = pd.read_sql_query(
            f"SELECT {col_str} FROM loan_applications WHERE SK_ID_CURR = {sk_id_curr}",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return {"error": f"No applicant found with SK_ID_CURR = {sk_id_curr}"}

    # apply the same cleaning used at training time, a model must see
    # data prepared identically to how it was trained
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    for col in features:
        df[col] = df[col].fillna(medians[col])

    risk_score = float(model.predict(df[features])[0])

    if risk_score >= 0.5:
        category = "High"
    elif risk_score >= 0.2:
        category = "Medium"
    else:
        category = "Low"

    return {
        "sk_id_curr": sk_id_curr,
        "default_risk_score": round(risk_score, 4),
        "risk_category": category,
    }


# ---------------------------------------------------------------------------
# Quick self-test, run this file directly to sanity check everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing run_query()...")
    result = run_query("SELECT COUNT(*) as total FROM loan_applications")
    print(result)

    print("\nTesting detect_anomalies() on AMT_CREDIT...")
    anomalies = detect_anomalies("loan_applications", "AMT_CREDIT")
    print(f"Found {anomalies['anomalies_found']} anomalies out of {anomalies['total_rows_checked']} rows")

    print("\nTesting segment_records()...")
    segments = segment_records(["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"])
    print(f"Segment sizes: {segments['segment_sizes']}")

    print("\nTesting forecast_timeseries()...")
    forecast = forecast_timeseries()
    print(f"Forecast: {forecast['forecast']}")

    print("\nTesting predict_default_risk()...")
    sample_id = run_query("SELECT SK_ID_CURR FROM loan_applications LIMIT 1")[0]["SK_ID_CURR"]
    risk = predict_default_risk(sample_id)
    print(risk)
