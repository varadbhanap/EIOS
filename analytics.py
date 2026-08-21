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
import re
import os
import pymupdf  # was 'import fitz', PyMuPDF's older import name, now deprecated
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.arima.model import ARIMA

DOCS_DIR = "docs"

DB_PATH = "data/enterprise_data.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------------
# 1. SQL querying
# ---------------------------------------------------------------------------
def _cap_result_size(records: list, max_chars: int = 3000) -> dict:
    """
    Defensive guard against oversized query results. This exists because
    the agent once wrote a SELECT la.*, bc.* join, which multiplied rows
    (one applicant can have many bureau records) and returned a result
    large enough to exceed the LLM's per-request token limit, crashing
    the agent mid-run. Rather than relying on the agent to always write
    narrow queries (its judgment isn't 100% reliable), this caps output
    size at the tool level so oversized results can never reach the LLM,
    regardless of what SQL was written.
    """
    total_rows = len(records)
    serialized = json.dumps(records, default=str)

    if len(serialized) <= max_chars:
        return {"records": records, "row_count": total_rows, "truncated": False}

    trimmed = records
    while trimmed and len(json.dumps(trimmed, default=str)) > max_chars:
        trimmed = trimmed[: max(1, len(trimmed) // 2)]

    return {
        "records": trimmed,
        "row_count": total_rows,
        "truncated": True,
        "note": (
            f"Result had {total_rows} rows and was too large, showing only "
            f"the first {len(trimmed)}. Select specific columns instead of "
            f"*, or add a LIMIT clause, to get complete results."
        ),
    }


def run_query(sql: str) -> dict:
    """
    Executes any SQL query against the enterprise database and returns
    the results. Always returns a dict with a 'records' key (a list of
    rows). If the result is too large, it will be truncated and a 'note'
    explaining this will be included, in that case, write a narrower
    query (select specific columns, or add a LIMIT clause) instead of
    assuming the truncated data is complete.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn)
        records = df.to_dict(orient="records")
    finally:
        conn.close()

    return _cap_result_size(records)


def get_table_schema(table: str) -> dict:
    """
    Returns the real column names and types for a given table. Use this
    whenever you're unsure of exact column names, guessing a column name
    that doesn't exist will cause a SQL error, checking first is faster
    and more reliable than guessing and retrying.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
    finally:
        conn.close()

    if not columns:
        return {"error": f"Table '{table}' not found."}

    return {"table": table, "columns": columns}


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
# 6. Policy document retrieval (RAG)
# ---------------------------------------------------------------------------
_doc_chunks = None
_doc_sources = None
_vectorizer = None
_doc_vectors = None


def _load_documents():
    """
    Loads and indexes every PDF in docs/ once, then reuses the index.
    Each PDF is split into paragraph-level chunks, policy sections are
    naturally separated by blank lines, so each chunk ends up being one
    coherent clause rather than a random fragment of text.
    """
    global _doc_chunks, _doc_sources, _vectorizer, _doc_vectors
    if _doc_chunks is not None:
        return

    chunks, sources = [], []

    for filename in os.listdir(DOCS_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(DOCS_DIR, filename)
        doc = pymupdf.open(path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()

        raw_chunks = re.split(r"\n\s*\n", full_text)
        for chunk in raw_chunks:
            cleaned = " ".join(chunk.split())
            if len(cleaned) > 40:  # skip near-empty fragments and bare headers
                chunks.append(cleaned)
                sources.append(filename)

    if not chunks:
        raise RuntimeError(
            f"No PDF text found in '{DOCS_DIR}/'. Add policy PDFs there before using this tool."
        )

    _doc_chunks = chunks
    _doc_sources = sources
    _vectorizer = TfidfVectorizer(stop_words="english")
    _doc_vectors = _vectorizer.fit_transform(chunks)


def search_policy_documents(query: str, top_k: int = 3) -> dict:
    """
    Searches all PDF documents in docs/ (policy documents, annual reports)
    and returns the most relevant real passages for a given question,
    using TF-IDF similarity, a deterministic text-matching method, not a
    generative one. No text is invented here, only real extracted
    passages from real PDFs are returned, with a genuine relevance score.
    """
    _load_documents()

    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _doc_vectors)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:  # skip completely irrelevant matches
            results.append({
                "source": _doc_sources[idx],
                "text": _doc_chunks[idx],
                "relevance_score": round(float(similarities[idx]), 3),
            })

    return {
        "query": query,
        "results_found": len(results),
        "passages": results,
    }


# ---------------------------------------------------------------------------
# Quick self-test, run this file directly to sanity check everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing run_query()...")
    result = run_query("SELECT COUNT(*) as total FROM loan_applications")
    print(result["records"])

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
    sample_id = run_query("SELECT SK_ID_CURR FROM loan_applications LIMIT 1")["records"][0]["SK_ID_CURR"]
    risk = predict_default_risk(sample_id)
    print(risk)

    print("\nTesting search_policy_documents()...")
    doc_results = search_policy_documents("maximum loan to income ratio allowed")
    print(f"Found {doc_results['results_found']} relevant passages")
    for r in doc_results["passages"]:
        print(f"  [{r['relevance_score']}] ({r['source']}) {r['text'][:100]}...")
