# Telecom Customer Churn Prediction

> A leakage-free scikit-learn `Pipeline` that identifies at-risk customers, tuned for high recall because missing a churner is **~5× more costly** than a false alarm.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Why Accuracy Is Not Enough](#why-accuracy-is-not-enough)
- [Model Selection](#model-selection)
- [Baselines](#baselines)
- [Tuning and Decision Threshold](#tuning-and-decision-threshold)
- [Final Results](#final-results)
- [Quick Start](#quick-start)
- [Output Files](#output-files)

---

## Overview

This project trains, evaluates, and deploys a binary classifier that predicts whether a telecom customer will churn. The full workflow — imputation, feature engineering, scaling, and classification — lives inside a single scikit-learn `Pipeline` so that every transformation applied during training is automatically replicated at inference time with no risk of data leakage.

---

## Project Structure

```text
customer-churn/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco training dataset
│   └── new_customers.csv                        # Unlabelled customers for prediction
├── notebook/
│   └── EDA.ipynb                                # Exploratory data analysis
├── scripts/
│   ├── preprocessing.py                         # ChurnFeatureEngineer transformer
│   ├── train.py                                 # Training, CV, tuning, evaluation
│   └── predict.py                               # Batch inference on new customers
├── results/
│   ├── churn_pipeline.pkl                       # Serialised Pipeline
│   ├── threshold.json                           # Frozen decision threshold
│   ├── results.md                               # Full metrics report
│   ├── predictions.csv                          # Predictions for new customers
│   └── plots/
│       └── precision_recall_curve.png           # Precision-recall tradeoff plot
├── requirements.txt
└── README.md
```

---

## Dataset

**IBM Telco Customer Churn** — [`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`](data/WA_Fn-UseC_-Telco-Customer-Churn.csv)

7,043 customers described by 20 features covering demographics, account info, and subscribed services. Binary target column: `Churn`.

| Split       | Share |    Rows |
| :---------- | :---: | ------: |
| Train       |  80%  | ~5,634  |
| Test        |  20%  | ~1,409  |

Class distribution: **73.5% No Churn · 26.5% Churn**

---

## Feature Engineering

All transformations are defined in [`scripts/preprocessing.py`](scripts/preprocessing.py) inside `ChurnFeatureEngineer`, the first step of the saved Pipeline. This guarantees identical preprocessing at train and inference time.

| Feature                      | Description                                                                                                              |
| :--------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| `TotalCharges` (cleaned)     | Coerced to numeric; implicit blanks become `NaN` and are median-imputed by the Pipeline.                                 |
| `tenure_bucket`              | Groups tenure into `0-12`, `13-24`, `25-48`, and `49+` month cohorts — churn rate differs sharply by customer age.      |
| `charges_per_month_of_tenure`| Lifetime spend normalised by tenure; separates new high-bill customers from long-tenured ones.                           |
| `n_services`                 | Count of active add-ons, treating `"No"`, `"No internet service"`, and `"No phone service"` as inactive.                 |

---

## Why Accuracy Is Not Enough

The target is heavily imbalanced (73.5 / 26.5). A majority-class classifier achieves **73.5% accuracy** while catching **zero churners** — confirmed by the baseline table below. Model selection therefore focuses on **recall**, **precision**, **F1**, and **ROC-AUC**, with the constraint that a false negative (missed churner) costs roughly 5× more than a false positive (unnecessary retention call).

---

## Model Selection

5-fold stratified cross-validation on the **training split only** — no test data was touched during selection.

| Model               | Recall | Precision |    F1 | ROC-AUC |
| :------------------ | -----: | --------: | ----: | ------: |
| Logistic Regression |  0.797 |     0.519 | 0.628 |   0.846 |
| Random Forest       |  0.731 |     0.548 | 0.626 |   0.842 |
| Gradient Boosting   |  0.522 |     0.662 | 0.583 |   0.848 |

All values are cross-validation means (std omitted for readability; full breakdown in [`results/results.md`](results/results.md)).

**Logistic Regression** was selected for tuning — highest churn recall while maintaining useful precision, directly matching the business objective.

---

## Baselines

| Strategy                         | Accuracy | Recall | ROC-AUC |
| :------------------------------- | -------: | -----: | ------: |
| Most Frequent (majority class)   |    0.735 |  0.000 |   0.500 |
| Stratified (random proportional) |    0.614 |  0.278 |   0.507 |

The majority-class baseline reaches ~73% accuracy with **zero churn recall**, confirming accuracy is an unreliable metric for this problem.

---

## Tuning and Decision Threshold

**Selected model:** Logistic Regression

**Tuned hyperparameters:** `C = 0.1`, `solver = liblinear`

The decision threshold was chosen from out-of-fold training probabilities as the highest-precision point where **recall ≥ 0.80**.

**Frozen threshold: `0.491`**

At this threshold the model flags approximately **426 customers per 1,000** for retention outreach, of whom ~211 are expected true churners — a recall-heavy campaign consistent with the 5× cost asymmetry.

Precision-recall curve: [`results/plots/precision_recall_curve.png`](results/plots/precision_recall_curve.png)

---

## Final Results

Evaluated **once** on the held-out test set after all tuning was finalised.

| Metric    |  Score |
| :-------- | -----: |
| Recall    |  0.794 |
| Precision |  0.495 |
| F1        |  0.610 |
| ROC-AUC   |  0.842 |

**Confusion Matrix**

|                    | Pred No Churn | Pred Churn |
| :----------------- | ------------: | ---------: |
| True No Churn      |           732 |        303 |
| True Churn         |            77 |        297 |

The model catches **297 of 374 true churners** (79.4% recall) while keeping precision at 49.5%.

Full report: [`results/results.md`](results/results.md)

---

## Quick Start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Train the model**

```bash
python scripts/train.py
```

Runs cross-validation, tunes hyperparameters, selects the decision threshold, evaluates on the held-out test set, and writes all artifacts to [`results/`](results/).

**3. Predict on new customers**

```bash
python scripts/predict.py
```

Loads the saved Pipeline and scores [`data/new_customers.csv`](data/new_customers.csv), writing predictions to [`results/predictions.csv`](results/predictions.csv).

---

## Output Files

| File                                                                                        | Description                                      |
| :------------------------------------------------------------------------------------------ | :----------------------------------------------- |
| [`results/churn_pipeline.pkl`](results/churn_pipeline.pkl)                                  | Serialised scikit-learn Pipeline (preprocessing + model) |
| [`results/threshold.json`](results/threshold.json)                                          | Frozen decision threshold (0.491)                |
| [`results/results.md`](results/results.md)                                                  | Full metrics report with confusion matrix        |
| [`results/predictions.csv`](results/predictions.csv)                                        | Churn predictions for new customers              |
| [`results/plots/precision_recall_curve.png`](results/plots/precision_recall_curve.png)      | Precision-recall curve with threshold annotation |
