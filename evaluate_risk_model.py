"""
EIOS - Full Model Evaluation

Runs a complete, proper evaluation of the trained default-risk model on
the exact same held-out test set used during training (same random_state
and split, so this is a genuine re-evaluation, not a new random split).

Beyond AUC and basic precision/recall, this adds: a full confusion
matrix, F1-score, PR-AUC (Average Precision, often more informative than
ROC-AUC for an imbalanced problem like this one), specificity, and
balanced accuracy, the standard, complete set of metrics for evaluating
a binary classifier on imbalanced data.

Run with:  python evaluate_risk_model.py
"""

import json
import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
)

DB_PATH = "data/enterprise_data.db"
MODEL_PATH = "models/risk_model.txt"
FEATURES_PATH = "models/feature_config.json"


def load_and_clean_data():
    with open(FEATURES_PATH) as f:
        config = json.load(f)
    features = config["features"]
    medians = config["medians"]

    conn = sqlite3.connect(DB_PATH)
    cols = ", ".join(features + ["TARGET"])
    df = pd.read_sql_query(f"SELECT {cols} FROM loan_applications", conn)
    conn.close()

    # identical cleaning to train_risk_model.py, so this evaluation
    # matches exactly what the model was trained and originally tested on
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    for col in features:
        df[col] = df[col].fillna(medians[col])

    return df, features


def main():
    df, features = load_and_clean_data()
    X = df[features]
    y = df["TARGET"]

    # identical split parameters to train_risk_model.py, this reproduces
    # the exact same held-out test set the model was originally scored on
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.Booster(model_file=MODEL_PATH)
    y_proba = model.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    print("=" * 60)
    print("FULL MODEL EVALUATION — held-out test set")
    print(f"Test set size: {len(y_test):,} real applications")
    print("=" * 60)

    print(f"\nROC-AUC:            {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC (Avg Prec.):  {average_precision_score(y_test, y_proba):.4f}")
    print(f"Accuracy:            {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy:   {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (class 1): {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (class 1):    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score (class 1):  {f1_score(y_test, y_pred):.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    print(f"Specificity (class 0 recall): {specificity:.4f}")

    print("\nConfusion Matrix:")
    print(f"                 Predicted: No Default   Predicted: Default")
    print(f"Actual: No Default    {tn:>8,}              {fp:>8,}")
    print(f"Actual: Default       {fn:>8,}              {tp:>8,}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    print("Interpretation:")
    print(f"- Of {tp + fn:,} applicants who actually defaulted, the model correctly")
    print(f"  flagged {tp:,} ({recall_score(y_test, y_pred):.1%}), missing {fn:,}.")
    print(f"- Of {tp + fp:,} applicants flagged as high risk, {tp:,} actually defaulted")
    print(f"  ({precision_score(y_test, y_pred):.1%} precision), the rest were false alarms")
    print(f"  requiring manual review that turned out unnecessary, an acceptable")
    print(f"  tradeoff for an audit/risk-flagging tool that prioritizes not missing")
    print(f"  real defaults over minimizing false alarms.")


if __name__ == "__main__":
    main()
