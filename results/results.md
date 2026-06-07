# Customer Churn Results

## Final Model

- Best model: Logistic Regression
- Tuned parameters: `{'model__C': 0.05, 'model__solver': 'liblinear'}`
- Frozen threshold selected on out-of-fold training predictions: 0.490
- Test recall: 0.791
- Test precision: 0.497
- Test F1: 0.610
- Test ROC-AUC: 0.841

## Business Translation

At threshold 0.490, the model flags about 423 customers per 1,000 for retention outreach. Based on test precision, about 210 of those 1,000 customers are expected to be true churners. This supports a recall-heavy campaign where missing a churner is about 5x more costly than calling a customer who would have stayed.

## Test Confusion Matrix

|               |   Pred No Churn |   Pred Churn |
|:--------------|----------------:|-------------:|
| True No Churn |             735 |          300 |
| True Churn    |              78 |          296 |

## Baselines

| strategy      |   accuracy_mean |   recall_mean |   roc_auc_mean |
|:--------------|----------------:|--------------:|---------------:|
| most_frequent |           0.735 |         0     |          0.5   |
| stratified    |           0.614 |         0.278 |          0.507 |
