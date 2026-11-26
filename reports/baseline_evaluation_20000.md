# Decision Tree Baseline Evaluation — 20,000 Records

**Contributor:** Yutong Cao  
**Project:** GridGuard — Blackout Prediction  
**Module:** Decision Tree Baseline Evaluation  
**Dataset:** PowerGraph IEEE-24 Simulated Scenarios

## 1. Objective

The objective of this experiment is to establish a reproducible Decision Tree baseline using the complete 20,000-record dataset.

The model predicts whether a simulated power-grid scenario contains Demand Not Served (DNS):

- `blackout = 0`: No Demand Not Served.
- `blackout = 1`: Demand Not Served is present.

This is a binary classification task. The target does not necessarily indicate a total power-grid or national blackout.

This experiment is separate from the earlier 1,000-record baseline evaluation.

## 2. Dataset and Integrity Verification

The complete dataset contains 20,000 records and 224 input features, with one binary target column.

| Property | Value |
|---|---:|
| Total records | 20,000 |
| Input features | 224 |
| Blackout cases (1) | 4,032 |
| Normal cases (0) | 15,968 |
| Positive class proportion | 20.16% |

The dataset combines the original 1,000 records and 19,000 additional records.

The dataset integrity checks verified:

- All 20,000 records were matched with the official manifest.
- Record identifiers matched the expected source records.
- Original source indices matched.
- Target labels matched.
- Feature hashes matched.
- No identical feature-hash groups crossed the official train, validation, and test splits.

All implemented integrity checks passed.

## 3. Evaluation Protocol

The official group-aware split was used.

| Split | Records | Blackout (1) | Normal (0) |
|---|---:|---:|---:|
| Train | 13,993 | 2,820 | 11,173 |
| Validation | 3,010 | 607 | 2,403 |
| Test | 2,997 | 605 | 2,392 |
| **Total** | **20,000** | **4,032** | **15,968** |

The roles of the datasets were:

- Training set: model fitting.
- Validation set: model comparison and threshold analysis.
- Test set: final evaluation after the model and threshold were fixed.

The test set was not used for model selection or threshold tuning.

## 4. Baseline Model

The baseline classifier was a scikit-learn `DecisionTreeClassifier` initialized with:

- `random_state = 42`
- Default tree complexity parameters.

The model was trained using all 13,993 training records and 224 input features.

### Model Complexity

| Metric | Value |
|---|---:|
| Tree depth | 28 |
| Number of leaves | 210 |
| Number of nodes | 419 |
| Training samples | 13,993 |
| Input features | 224 |

## 5. Validation Evaluation

The original Decision Tree achieved:

| Metric | Validation Result |
|---|---:|
| Accuracy | 0.9900 |
| Precision | 0.9769 |
| Recall | 0.9736 |
| F1-score | 0.9752 |

Validation confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2,389 | 14 |
| Actual 1 | 16 | 591 |

### 5.1 Controlled Decision Tree Comparison

Two constrained configurations were evaluated against the original baseline.

| Model | Train F1 | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 | Depth | Leaves |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Baseline | 1.0000 | 0.9900 | 0.9769 | 0.9736 | 0.9752 | 28 | 210 |
| Controlled DT 1 | 0.8205 | 0.9166 | 0.7543 | 0.8699 | 0.8080 | 8 | 46 |
| Controlled DT 2 | 0.8476 | 0.9269 | 0.7833 | 0.8814 | 0.8295 | 10 | 59 |

The original baseline achieved the highest validation F1-score.

The controlled models reduced the training-validation F1 gap, but also reduced validation performance. Therefore, the original baseline was retained as the main candidate.

## 6. Validation Threshold Analysis

Thresholds from 0.1 to 0.9 were evaluated using validation probabilities.

All thresholds produced identical classification results:

| Metric | Value |
|---|---:|
| Precision | 0.9769 |
| Recall | 0.9736 |
| F1-score | 0.9752 |
| TN | 2,389 |
| FP | 14 |
| FN | 16 |
| TP | 591 |

The probability distribution contained only two unique values:

- Probability 0.0: 2,405 records.
- Probability 1.0: 605 records.

No validation probabilities fell between 0.1 and 0.9.

This explains why changing the threshold did not affect the predictions. The threshold was retained at 0.5.

The discrete probabilities indicate pure prediction leaves in this fitted tree. They should not be interpreted as calibrated real-world risk probabilities.

## 7. Final Test Evaluation

After fixing the model and threshold, the final test evaluation was performed on the 2,997 test records.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.9923 |
| Precision | 0.9883 |
| Recall | 0.9736 |
| F1-score | 0.9808 |
| ROC-AUC | 0.9853 |

Test confusion matrix:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2,385 | 7 |
| Actual 1 | 16 | 589 |

The model correctly classified 2,974 out of 2,997 test records.

## 8. Test Error Analysis

A total of 23 misclassified records were identified.

| Error Type | Count |
|---|---:|
| False Positives | 7 |
| False Negatives | 16 |
| Total Errors | 23 |

Errors by data source:

| Source | False Positives | False Negatives |
|---|---:|---:|
| Original 1,000 records (`small`) | 1 | 0 |
| Additional 19,000 records (`large`) | 6 | 16 |
| Total | 7 | 16 |

False negatives are particularly important because they represent simulated DNS scenarios that the classifier failed to identify.

This source-level breakdown is descriptive. It does not establish that the additional records are inherently more difficult, because the two sources may differ in their scenario distributions.

## 9. Limitations

Several limitations must be considered:

1. **Simulated data:** The dataset represents simulated IEEE-24 power-grid scenarios, not historical real-world blackout observations.
2. **Generalization:** High test performance on this dataset does not guarantee equivalent performance on other power systems or real operational conditions.
3. **Discrete probabilities:** The model produces only 0.0 and 1.0 probabilities on the validation set, limiting threshold flexibility and preventing meaningful probability calibration analysis with this model output.
4. **False negatives:** Sixteen positive test cases were missed and require further investigation.
5. **Feature interpretation:** The 224 input features retain source preprocessing. Their physical meanings and units should not be assumed without the original dataset documentation.
6. **Group separation:** The group-aware split prevents identical feature-hash groups from crossing splits, but does not by itself prove the absence of every possible form of data leakage.

## 10. Conclusion

The original Decision Tree established a strong baseline on the complete 20,000-record dataset, achieving a test F1-score of 0.9808 and ROC-AUC of 0.9853.

It outperformed the two controlled Decision Tree configurations on validation F1.

The final test error analysis identified 16 false negatives and 7 false positives. These errors, together with the simulated nature of the dataset and the discrete probability outputs, provide important directions for future model development.

This baseline serves as a reference for comparison with more advanced models, including Random Forest and boosting approaches.

**Important:** The 1,000-record and 20,000-record baseline experiments are separate evaluations and their results should not be merged into a single experiment.