# Customer Churn Results

## Final Model

- Best model: Logistic Regression
- Tuned parameters: `{'model__C': 0.1, 'model__solver': 'liblinear'}`
- Frozen threshold selected on out-of-fold training predictions: 0.491
- Test recall: 0.794
- Test precision: 0.495
- Test F1: 0.610
- Test ROC-AUC: 0.842

## Business Translation

At threshold 0.491, the model flags about 426 customers per 1,000 for retention outreach. Based on test precision, about 211 of those 1,000 customers are expected to be true churners. This supports a recall-heavy campaign where missing a churner is about 5x more costly than calling a customer who would have stayed.

## Test Confusion Matrix

|               |   Pred No Churn |   Pred Churn |
|:--------------|----------------:|-------------:|
| True No Churn |             732 |          303 |
| True Churn    |              77 |          297 |

## Baselines

| strategy      |   accuracy_mean |   recall_mean |   roc_auc_mean |
|:--------------|----------------:|--------------:|---------------:|
| most_frequent |           0.735 |         0     |          0.5   |
| stratified    |           0.614 |         0.278 |          0.507 |
