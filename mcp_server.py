"""
EIOS - Step 3: MCP Server

This file exposes the functions from analytics.py as tools an AI agent can
discover and call. It does not contain any analysis logic itself, it is
purely a translation/switchboard layer, on purpose. Keeping this file thin
means the actual math in analytics.py stays fully independent and testable
on its own, which is exactly what we already verified in Step 2.

To run this server on its own (for testing that it starts correctly):
    python mcp_server.py

The agent (built in Step 4) will connect to this server as a client.
"""

from fastmcp import FastMCP
from analytics import (
    run_query,
    detect_anomalies,
    segment_records,
    forecast_timeseries,
    predict_default_risk,
)

mcp = FastMCP("EIOS")


@mcp.tool()
def query_database(sql: str) -> list:
    """
    Executes a SQL query against the enterprise database and returns the
    results. Use this to look up specific records, counts, sums, or any
    other structured data question. Available tables: loan_applications,
    bureau_credit_history, previous_applications. All three can be joined
    on the column SK_ID_CURR.
    """
    return run_query(sql)


@mcp.tool()
def find_anomalies(table: str, column: str, z_threshold: float = 3.0) -> dict:
    """
    Detects statistical outliers in a numeric column of a table using
    Z-score analysis. Use this when asked to find unusual, suspicious,
    or outlier values, for example unusually large loan amounts. Returns
    how many anomalies were found and a sample of the flagged records.
    """
    return detect_anomalies(table, column, z_threshold)


@mcp.tool()
def segment_applicants(columns: list, n_clusters: int = 4) -> dict:
    """
    Groups loan applicants into behavioral segments using K-Means
    clustering on the given numeric columns (for example income, credit
    amount, annuity). Use this when asked to segment, group, or profile
    applicants. Returns segment sizes and the average profile of each
    segment.
    """
    return segment_records(columns, n_clusters)


@mcp.tool()
def forecast_trend(periods_ahead: int = 6) -> dict:
    """
    Forecasts loan application volume for the next few months, based on
    real historical application timing data. Use this when asked about
    future trends, projections, or expected volume. Returns recent
    historical monthly values alongside the forecasted values.
    """
    return forecast_timeseries(periods_ahead)


@mcp.tool()
def assess_default_risk(sk_id_curr: int) -> dict:
    """
    Predicts the likelihood that a specific loan applicant will default,
    using a trained machine learning model. Use this when asked about an
    applicant's risk, likelihood of default, or creditworthiness. Requires
    the applicant's SK_ID_CURR (their unique ID in the database). Returns
    a risk probability score and a category: Low, Medium, or High.
    """
    return predict_default_risk(sk_id_curr)


if __name__ == "__main__":
    mcp.run()
