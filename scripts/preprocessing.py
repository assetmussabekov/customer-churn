"""Reusable preprocessing transformers for the churn model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create row-local churn features inside the sklearn Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        tenure = pd.to_numeric(X["tenure"], errors="coerce")
        total_charges = pd.to_numeric(X["TotalCharges"], errors="coerce")

        X["charges_per_month_of_tenure"] = total_charges / np.maximum(tenure, 1)
        X["tenure_bucket"] = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12", "13-24", "25-48", "49+"],
        ).astype("object")

        available_service_cols = [col for col in SERVICE_COLUMNS if col in X.columns]
        X["n_services"] = (
            X[available_service_cols]
            .apply(lambda col: ~col.isin(["No", "No internet service", "No phone service"]))
            .sum(axis=1)
        )

        return X
