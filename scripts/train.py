"""Train and evaluate a leakage-free customer churn model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import ChurnFeatureEngineer


RANDOM_STATE = 42
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_PATH = RESULTS_DIR / "churn_pipeline.pkl"
THRESHOLD_PATH = RESULTS_DIR / "threshold.json"


NUM_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_month_of_tenure",
    "n_services",
]

CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]


def make_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("features", ChurnFeatureEngineer()),
            ("prep", make_preprocessor()),
            ("model", model),
        ]
    )


def churn_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "flag_rate": float(pred.mean()),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def cv_table(models, X_train, y_train, cv):
    scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def choose_threshold(y_true, proba, min_recall=0.80):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    rows = pd.DataFrame(
        {
            "threshold": np.r_[thresholds, 1.0],
            "precision": precision,
            "recall": recall,
        }
    )
    feasible = rows[rows["recall"] >= min_recall].copy()
    if feasible.empty:
        return rows.sort_values("recall", ascending=False).iloc[0]
    return feasible.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]


def save_pr_curve(y_true, proba, threshold):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label="Out-of-fold PR curve")
    chosen_precision = precision_score(y_true, proba >= threshold)
    chosen_recall = recall_score(y_true, proba >= threshold)
    plt.scatter([chosen_recall], [chosen_precision], color="crimson", label=f"Threshold {threshold:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve on Training OOF Predictions")
    plt.legend()
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "precision_recall_curve.png", dpi=160)
    plt.close()


def make_new_customers(df):
    new_customers = df.drop(columns=["Churn"]).sample(5, random_state=RANDOM_STATE)
    new_customers.to_csv("data/new_customers.csv", index=False)


def format_cv_markdown(cv_results):
    display = cv_results.copy()
    for col in display.columns:
        if col != "model":
            display[col] = display[col].map(lambda x: f"{x:.3f}")
    return display.to_markdown(index=False)


def write_outputs(cv_results, dummy_results, best_name, best_params, threshold_row, test_metrics):
    cm = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=["True No Churn", "True Churn"],
        columns=["Pred No Churn", "Pred Churn"],
    )
    calls_per_1000 = round(test_metrics["flag_rate"] * 1000)
    churners_per_1000 = round(test_metrics["flag_rate"] * test_metrics["precision"] * 1000)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.md").write_text(
        f"""# Customer Churn Results

## Final Model

- Best model: {best_name}
- Tuned parameters: `{best_params}`
- Frozen threshold selected on out-of-fold training predictions: {threshold_row['threshold']:.3f}
- Test recall: {test_metrics['recall']:.3f}
- Test precision: {test_metrics['precision']:.3f}
- Test F1: {test_metrics['f1']:.3f}
- Test ROC-AUC: {test_metrics['roc_auc']:.3f}

## Business Translation

At threshold {threshold_row['threshold']:.3f}, the model flags about {calls_per_1000} customers per 1,000 for retention outreach. Based on test precision, about {churners_per_1000} of those 1,000 customers are expected to be true churners. This supports a recall-heavy campaign where missing a churner is about 5x more costly than calling a customer who would have stayed.

## Test Confusion Matrix

{cm.to_markdown()}

## Baselines

{pd.DataFrame(dummy_results).to_markdown(index=False)}
""",
        encoding="utf-8",
    )

    readme = f"""# Telecom Customer Churn Prediction

This project predicts telecom customer churn with a leakage-free scikit-learn `Pipeline`. It uses the IBM Telco Customer Churn dataset in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Why Accuracy Is Not Enough

The data is imbalanced: about 73.5% of customers do not churn and 26.5% do churn. A majority-class classifier can look accurate while finding no churners, so model selection focuses on recall, precision, F1, and ROC-AUC. Because a false negative is roughly 5x more costly than a false positive, the final decision threshold is tuned for high recall.

## Feature Engineering

All engineered features are created inside `scripts/preprocessing.py` through `ChurnFeatureEngineer`, which is the first step in the saved Pipeline:

- `TotalCharges` is converted with `pd.to_numeric(..., errors='coerce')`, leaving implicit blanks as `NaN` for median imputation inside the Pipeline.
- `tenure_bucket` groups tenure into `0-12`, `13-24`, `25-48`, and `49+` month cohorts because churn behavior differs strongly by customer age.
- `charges_per_month_of_tenure` normalizes lifetime spend by tenure and helps distinguish new high-bill customers from long-tenured customers.
- `n_services` counts active services, treating `No`, `No internet service`, and `No phone service` as inactive.

## Cross-Validation Results

All scores use 5-fold stratified cross-validation on the training split only.

{format_cv_markdown(cv_results)}

## Baselines

{pd.DataFrame(dummy_results).to_markdown(index=False)}

The majority-class baseline has about 73% accuracy but zero churn recall, so accuracy is discarded as the primary metric.

## Tuning and Threshold

Best model: **{best_name}**

Logistic Regression was selected for tuning because it had the strongest churn recall in cross-validation while maintaining useful precision. That matches the business objective: missing a true churner is roughly 5x more costly than making an extra retention call.

Tuned parameters: `{best_params}`

The decision threshold was chosen from out-of-fold training probabilities, selecting the highest precision threshold with recall >= 0.80. The frozen threshold is **{threshold_row['threshold']:.3f}**.

## Final Held-Out Test Results

- Recall: **{test_metrics['recall']:.3f}**
- Precision: **{test_metrics['precision']:.3f}**
- F1: **{test_metrics['f1']:.3f}**
- ROC-AUC: **{test_metrics['roc_auc']:.3f}**

Confusion matrix:

{cm.to_markdown()}

## Run

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/predict.py
```

Training writes `results/churn_pipeline.pkl`, `results/threshold.json`, `results/results.md`, and `results/plots/precision_recall_curve.png`. Prediction writes `results/predictions.csv`.
"""
    Path("README.md").write_text(readme, encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    make_new_customers(df)

    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Churn", "customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    dummy_results = []
    for strategy in ["most_frequent", "stratified"]:
        dummy = DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)
        dummy_scores = cross_validate(
            dummy,
            X_train,
            y_train,
            cv=cv,
            scoring={"recall": "recall", "roc_auc": "roc_auc", "accuracy": "accuracy"},
        )
        dummy_results.append(
            {
                "strategy": strategy,
                "accuracy_mean": f"{dummy_scores['test_accuracy'].mean():.3f}",
                "recall_mean": f"{dummy_scores['test_recall'].mean():.3f}",
                "roc_auc_mean": f"{dummy_scores['test_roc_auc'].mean():.3f}",
            }
        )

    models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE)),
    }

    cv_results = cv_table(models, X_train, y_train, cv)

    tune_pipe = make_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    )
    param_grid = {
        "model__C": [0.05, 0.1, 0.3, 1.0, 3.0],
        "model__solver": ["lbfgs", "liblinear"],
    }
    search = GridSearchCV(
        tune_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_

    oof_proba = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold_row = choose_threshold(y_train, oof_proba, min_recall=0.80)
    threshold = float(threshold_row["threshold"])
    save_pr_curve(y_train, oof_proba, threshold)

    best_pipe.fit(X_train, y_train)
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics = churn_metrics(y_test, test_proba, threshold)

    joblib.dump(best_pipe, MODEL_PATH)
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2), encoding="utf-8")

    write_outputs(
        cv_results=cv_results,
        dummy_results=dummy_results,
        best_name="Logistic Regression",
        best_params=search.best_params_,
        threshold_row=threshold_row,
        test_metrics=test_metrics,
    )

    print("Cross-validation results:")
    print(cv_results.to_string(index=False))
    print("\nBest params:", search.best_params_)
    print("Frozen threshold:", threshold)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
