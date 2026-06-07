"""Predict churn probabilities for new customers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("results/churn_pipeline.pkl")
THRESHOLD_PATH = Path("results/threshold.json")
INPUT_PATH = Path("data/new_customers.csv")
OUTPUT_PATH = Path("results/predictions.csv")


def main():
    pipe = joblib.load(MODEL_PATH)
    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"]

    df = pd.read_csv(INPUT_PATH)
    customer_ids = df["customerID"].copy()
    X = df.drop(columns=["customerID"], errors="ignore")
    X = X.drop(columns=["Churn"], errors="ignore")

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    output = pd.DataFrame(
        {
            "customerID": customer_ids,
            "churn_pred": pred,
            "churn_proba": proba,
        }
    )
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
