"""
EIOS - Step 2.5: Train the Default Risk Model

This is a ONE-TIME script. Run it once to train and save the model, then
analytics.py loads the saved model file instead of retraining every time.

Uses the real TARGET column already present in loan_applications (1 = the
applicant defaulted, 0 = they didn't), so this is genuine supervised
learning on real labeled outcomes, not something invented for the demo.

Run with:  python train_risk_model.py
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

DB_PATH = "data/enterprise_data.db"
MODEL_PATH = "models/risk_model.txt"
FEATURES_PATH = "models/feature_config.json"

# a lean, real feature set, kept small deliberately rather than throwing
# in all 122 columns, easier to explain, easier to debug, still effective
FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_POPULATION_RELATIVE",
    "REGION_RATING_CLIENT",
]


def load_data():
    conn = sqlite3.connect(DB_PATH)
    cols = ", ".join(FEATURES + ["TARGET", "SK_ID_CURR"])
    df = pd.read_sql_query(f"SELECT {cols} FROM loan_applications", conn)
    conn.close()
    return df


def clean_data(df):
    # known data quality issue in this exact dataset: DAYS_EMPLOYED uses
    # 365243 (roughly 1000 years) as a placeholder code for "not currently
    # employed" instead of a proper missing value. Left as-is, this would
    # badly distort the model, since it looks like a real number rather
    # than "no data". Replacing it with NaN so it gets handled like any
    # other missing value below.
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # median-impute remaining missing values (mainly EXT_SOURCE_1/2/3,
    # which are frequently missing in the real dataset). Save the medians
    # so analytics.py can apply the exact same values at prediction time,
    # a model must be fed data cleaned the same way it was trained on.
    medians = {}
    for col in FEATURES:
        median_val = df[col].median()
        medians[col] = float(median_val) if not pd.isna(median_val) else 0.0
        df[col] = df[col].fillna(medians[col])

    return df, medians


def main():
    import os
    os.makedirs("models", exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} applications")

    print("Cleaning data...")
    df, medians = clean_data(df)

    # known class imbalance in this dataset: roughly 92% did not default,
    # 8% did. Plain accuracy would look artificially high even from a
    # useless model that just predicts "no default" every time, so we use
    # a stratified split (preserves the same ratio in train and test) and
    # tell the model to weight the minority class more heavily.
    X = df[FEATURES]
    y = df["TARGET"]

    default_rate = y.mean()
    print(f"Class balance: {default_rate:.1%} defaulted, {1 - default_rate:.1%} did not")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": True,  # tells LightGBM to account for the class imbalance
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    print("\nTraining model (watch AUC improve round by round)...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, test_data],
        valid_names=["train", "test"],
        callbacks=[lgb.log_evaluation(period=20), lgb.early_stopping(stopping_rounds=20)],
    )

    # evaluate honestly on the held-out test set, this is data the model
    # never saw during training
    test_preds = model.predict(X_test, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, test_preds)
    print(f"\nFinal test AUC: {auc:.4f}")

    print("\nClassification report (threshold = 0.5):")
    print(classification_report(y_test, (test_preds >= 0.5).astype(int)))

    print("\nTop feature importances:")
    importances = dict(zip(FEATURES, model.feature_importance(importance_type="gain")))
    for feat, score in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {score:.0f}")

    # save the model and the exact preprocessing values used, so
    # predictions at inference time are cleaned identically
    model.save_model(MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump({"features": FEATURES, "medians": medians}, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Feature config saved to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
