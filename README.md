# Telecom Customer Churn Prediction

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

| model               |   recall_mean |   recall_std |   precision_mean |   precision_std |   f1_mean |   f1_std |   roc_auc_mean |   roc_auc_std |
|:--------------------|--------------:|-------------:|-----------------:|----------------:|----------:|---------:|---------------:|--------------:|
| Gradient Boosting   |         0.522 |        0.025 |            0.663 |           0.035 |     0.584 |    0.026 |          0.848 |         0.011 |
| Logistic Regression |         0.797 |        0.035 |            0.519 |           0.017 |     0.628 |    0.021 |          0.846 |         0.011 |
| Random Forest       |         0.672 |        0.031 |            0.582 |           0.018 |     0.624 |    0.022 |          0.841 |         0.009 |

## Baselines

| strategy      |   accuracy_mean |   recall_mean |   roc_auc_mean |
|:--------------|----------------:|--------------:|---------------:|
| most_frequent |           0.735 |         0     |          0.5   |
| stratified    |           0.614 |         0.278 |          0.507 |

The majority-class baseline has about 73% accuracy but zero churn recall, so accuracy is discarded as the primary metric.

## Tuning and Threshold

Best model: **Logistic Regression**

Logistic Regression was selected for tuning because it had the strongest churn recall in cross-validation while maintaining useful precision. That matches the business objective: missing a true churner is roughly 5x more costly than making an extra retention call.

Tuned parameters: `{'model__C': 0.1, 'model__solver': 'liblinear'}`

The decision threshold was chosen from out-of-fold training probabilities, selecting the highest precision threshold with recall >= 0.80. The frozen threshold is **0.491**.

## Final Held-Out Test Results

- Recall: **0.794**
- Precision: **0.495**
- F1: **0.610**
- ROC-AUC: **0.842**

Confusion matrix:

|               |   Pred No Churn |   Pred Churn |
|:--------------|----------------:|-------------:|
| True No Churn |             732 |          303 |
| True Churn    |              77 |          297 |

## Run

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/predict.py
```

Training writes `results/churn_pipeline.pkl`, `results/threshold.json`, `results/results.md`, and `results/plots/precision_recall_curve.png`. Prediction writes `results/predictions.csv`.
