# Full-dataset evaluation — GridGuard

## Protocol

- Dataset: 20,000 validated PowerGraph IEEE-24 simulated records.
- Inputs: 224 numerical features.
- Official split: train 13993, validation 3010, test 2997.
- The official all_records_manifest.csv was used.
- The three partition manifests were checked against it.
- Model selection: highest validation average precision.
- Threshold selection: maximum F2 on validation data.
- F2 gives recall more weight than precision.
- The test set was not used for model selection or threshold tuning.

## Validation model comparison

```text
              model  validation_average_precision  validation_roc_auc  validation_recall_at_0_5  validation_precision_at_0_5
      Random_forest                        0.9811              0.9921                    0.9127                       0.9535
      Decision_tree                        0.9768              0.9867                    0.9687                       0.9304
Logistic_regression                        0.9681              0.9881                    0.9588                       0.8584
        Dummy_prior                        0.2017              0.5000                    0.0000                       0.0000
```

## Selected model and threshold

- Model: **Random_forest**
- Threshold: **0.277179**
- Validation F2: **0.9415**

## Final test results

| Metric | Test result |
|---|---:|
| Accuracy | 0.9656 |
| Precision (blackout) | 0.8850 |
| Recall (blackout) | 0.9537 |
| F1 (blackout) | 0.9181 |
| F2 (blackout) | 0.9391 |
| ROC-AUC | 0.9933 |
| Average precision | 0.9849 |

### Confusion matrix

| | Predicted no blackout | Predicted blackout |
|---|---:|---:|
| Actual no blackout | 2317 (TN) | 75 (FP) |
| Actual blackout | 28 (FN) | 577 (TP) |

- False positives: **75**
- False negatives: **28**

A false negative is an actual blackout case that the model misses.
A false positive is a predicted blackout where the actual label is 0.

## Error analysis

Per-record test predictions are saved in `test_predictions.csv`.

Misclassified test records are saved in `test_errors.csv`,
including their record IDs, true labels, predicted probabilities
and error type.

## Figures

- `figures/evaluation/validation_precision_recall.png`
- `figures/evaluation/validation_roc_curve.png`
- `figures/evaluation/test_confusion_matrix.png`
- `figures/evaluation/test_precision_recall.png`

## Limitations

The dataset contains simulated IEEE-24 scenarios.
Results should not be interpreted as validated real-world grid performance.

The project cleaning report also notes that the source documentation
does not establish pre-outcome timing for every input feature.
Possible temporal leakage therefore remains a limitation to investigate.
