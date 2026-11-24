# GridGuard — Evaluation Report

## 1. Evaluation protocol

- Dataset: 20,000 validated records.
- Features: 224.
- Target: `blackout`.
- Initial split: stratified 80% train pool / 20% test.
- Validation: GroupShuffleSplit, 20% of the training pool.
- Cross-validation: 5-fold StratifiedGroupKFold.
- Model selection criterion: recall.
- Final model: Random Forest with all 224 features.
- Decision threshold: 0.50.

## 2. Hyperparameter tuning

Best parameters:

```json
{
  "model__class_weight": "balanced",
  "model__max_depth": 11,
  "model__max_features": 225,
  "model__min_samples_leaf": 2,
  "model__min_samples_split": 6,
  "model__n_estimators": 6
}
```

Best mean cross-validation recall: **0.9723**

## 3. Validation model comparison

```text
              model  validation_recall  validation_precision  validation_f1  validation_average_precision  validation_roc_auc  threshold
Logistic_regression             0.9818                0.8782         0.9271                        0.9764              0.9939     0.5000
Random_forest_Anais             0.9743                0.7568         0.8519                        0.9500              0.9867     0.5000
      Decision_tree             0.9470                0.9616         0.9543                        0.9665              0.9841     0.5000
        Dummy_prior             0.0000                0.0000         0.0000                        0.2081              0.5000     0.5000
```

## 4. Threshold selection

The team selected threshold **0.50**, following Anais's validation analysis.
The threshold comparison is available in `validation_threshold_comparison.csv`.

At threshold 0.50:

- Validation recall: **0.9743**
- Validation precision: **0.7568**
- Validation F1: **0.8519**

## 5. Final test evaluation

The model and threshold were fixed before evaluating the test set.

| Metric | Test result |
|---|---:|
| Accuracy | 0.9345 |
| Precision | 0.7698 |
| Recall | 0.9628 |
| F1-score | 0.8556 |
| ROC-AUC | 0.9821 |
| Average precision | 0.9496 |

### Confusion matrix

| | Predicted no blackout | Predicted blackout |
|---|---:|---:|
| Actual no blackout | 2962 | 232 |
| Actual blackout | 30 | 776 |

- False positives: **232**
- False negatives: **30**

## 6. Error analysis

Test errors are available in `test_errors.csv`.

### Errors by source

```text
source_extract  records  errors  error_rate
      1000_csv      190      12      0.0632
     19000_zip     3810     250      0.0656
```

Feature importances are saved in `feature_importance.csv`.

## 7. Results synthesis

The Random Forest was tuned using recall as the cross-validation scoring metric.
The final threshold was fixed at 0.50 according to the team's decision.

The test metrics provide the final estimate of performance on the reserved test partition.
Recall is especially important because false negatives correspond to missed blackout cases.
Precision is also important because false positives correspond to unnecessary alerts.

## 8. Limitations

The initial outer split is stratified but not group-aware, following Anais's notebook.
Therefore, identical bus configurations may occur in both the training pool and test set.
This limits claims about generalization to completely unseen configurations.

The dataset contains simulated scenarios. These results should not be interpreted as
proof of real-world grid performance.
