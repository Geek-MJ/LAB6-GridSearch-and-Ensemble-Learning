# Baseline Model and Evaluation Report

## 1. Introduction

The objective of this project is to predict whether a simulated power-grid scenario contains Demand Not Served (DNS), represented by the binary target variable `blackout`.

A value of `blackout = 1` indicates that the scenario contains unmet demand, while `blackout = 0` indicates that no unmet demand is recorded.

This report documents the Decision Tree baseline, its validation performance, threshold analysis, final test evaluation, and error analysis.

## 2. Dataset and Evaluation Protocol

The experiments in this report use the 1,000-record subset stored in `data/Blackout_PowerGraph_1000.csv`.

The dataset contains 224 input features and one binary target variable. Its class distribution is:

- No Blackout (0): 798 records
- Blackout (1): 202 records

To avoid leakage caused by identical feature vectors appearing in different subsets, the experiments use the official group-aware split manifests prepared during the data audit.

The 1,000 records were mapped to the official manifests using their record identifiers. The resulting split is:

| Split | Total records | No Blackout | Blackout |
|---|---:|---:|---:|
| Training | 715 | 559 | 156 |
| Validation | 161 | 133 | 28 |
| Test | 124 | 106 | 18 |
| **Total** | **1,000** | **798** | **202** |

The training set was used to fit the model. The validation set was used for model comparison and threshold analysis. The test set was kept separate and used for final evaluation after retaining the baseline configuration.

The official group-aware split prevents exact duplicate feature vectors from crossing the partitions. However, it does not establish complete independence between all simulated scenarios.

## 3. Baseline Model

A Decision Tree Classifier was selected as the baseline model because it is simple, interpretable, and suitable for binary classification.

The model was initialized with:

- Model: `DecisionTreeClassifier`
- Random state: `42`
- Maximum depth: default (unrestricted)
- Minimum samples per leaf: default (`1`)

The model was fitted using the training set only.

The fitted tree had:

- Depth: 14
- Number of leaves: 54
- Number of nodes: 107

The model achieved a training F1-score of 1.0000, indicating that it perfectly classified the training samples. This result motivated further investigation of its generalization performance.

## 4. Validation Results

The baseline was evaluated on the validation set of 161 records.

| Metric | Validation score |
|---|---:|
| Accuracy | 0.9255 |
| Precision (Blackout) | 0.7353 |
| Recall (Blackout) | 0.8929 |
| F1-score (Blackout) | 0.8065 |
| ROC-AUC | 0.9126 |

The validation confusion matrix was:

| | Predicted No Blackout | Predicted Blackout |
|---|---:|---:|
| Actual No Blackout | 124 | 9 |
| Actual Blackout | 3 | 25 |

The model correctly identified 25 of the 28 blackout scenarios, with 3 false negatives and 9 false positives.

The high recall indicates that the baseline detected most positive scenarios in the validation set. However, the false-positive count shows that some normal scenarios were incorrectly flagged.

### 4.1 Overfitting Analysis

The model achieved a training F1-score of 1.0000, compared with a validation F1-score of 0.8065.

This difference of 0.1935 suggests a generalization gap and provides evidence of possible overfitting.

### 4.2 Controlled Decision Tree Comparison

Two additional Decision Tree configurations were evaluated to investigate whether restricting model complexity could improve validation performance.

| Model | Configuration | Train F1 | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 |
|---|---|---:|---:|---:|---:|---:|
| Original Baseline | Default | 1.0000 | 0.9255 | 0.7353 | 0.8929 | **0.8065** |
| Controlled DT 1 | `max_depth=8`, `min_samples_leaf=5` | 0.8440 | 0.9068 | 0.7097 | 0.7857 | 0.7458 |
| Controlled DT 2 | `max_depth=10`, `min_samples_leaf=10` | 0.8199 | 0.8571 | 0.5581 | 0.8571 | 0.6761 |

The controlled models reduced training performance but did not improve validation F1-score. The original Decision Tree therefore remained the preferred configuration within the baseline experiments.

This comparison does not establish that the Decision Tree is the best model for the entire project; other model families should be evaluated separately.

## 5. Threshold Analysis

The baseline's predicted probabilities on the validation set were evaluated using classification thresholds from 0.1 to 0.9.

All tested thresholds produced identical predictions:

| Metric | Result |
|---|---:|
| Precision | 0.7353 |
| Recall | 0.8929 |
| F1-score | 0.8065 |
| True Negatives | 124 |
| False Positives | 9 |
| False Negatives | 3 |
| True Positives | 25 |

This behaviour is explained by the Decision Tree's discrete probability outputs, which were 0.0 or 1.0 for the validation predictions. Therefore, changing the threshold within the tested range did not change the predicted classes.

The default threshold of 0.5 was retained. No alternative threshold was selected.

## 6. Final Test Evaluation

After completing validation-based model selection and threshold analysis, the retained baseline was evaluated on the untouched test set of 124 records.

| Metric | Test score |
|---|---:|
| Accuracy | 0.8306 |
| Precision (Blackout) | 0.4444 |
| Recall (Blackout) | 0.6667 |
| F1-score (Blackout) | 0.5333 |
| ROC-AUC | 0.7626 |

The test confusion matrix was:

| | Predicted No Blackout | Predicted Blackout |
|---|---:|---:|
| Actual No Blackout | 91 | 15 |
| Actual Blackout | 6 | 12 |

The model correctly identified 12 of the 18 blackout scenarios, while missing 6 positive cases and producing 15 false alarms.

Compared with validation, the test F1-score decreased from 0.8065 to 0.5333, and ROC-AUC decreased from 0.9126 to 0.7626.

This performance gap indicates limited generalization and is consistent with the possible overfitting observed during training and validation. The small number of positive test samples should also be considered when interpreting the results.

## 7. Error Analysis

The model made 21 errors on the test set:

| Error type | Count | Interpretation |
|---|---:|---|
| False Positive | 15 | Normal scenarios incorrectly predicted as blackout |
| False Negative | 6 | Blackout scenarios incorrectly predicted as normal |
| **Total** | **21** | |

False negatives are particularly important for this application because they represent blackout-related scenarios that the model failed to detect.

An additional observation is that all incorrect test predictions had discrete predicted probabilities: 0.0 for false negatives and 1.0 for false positives.

This is consistent with the Decision Tree's leaf-based probability estimates. It also explains why threshold changes between 0.1 and 0.9 did not change the validation predictions.

Further investigation of the corresponding feature patterns and decision paths could help explain individual errors, but this was outside the scope of the current baseline evaluation.

## 8. Limitations

Several limitations must be considered when interpreting these results.

1. **Limited test sample size:** The test set contains 124 records, including only 18 blackout cases. Consequently, individual errors have a substantial effect on positive-class metrics.

2. **Generalization gap:** The training F1-score was 1.0000, while the test F1-score was 0.5333. This suggests that the model may be overfitting the training data.

3. **Scenario dependence:** The group-aware split prevents exact duplicate feature vectors from crossing the partitions, but possible dependencies between related simulation scenarios cannot be fully ruled out.

4. **Potential outcome leakage:** The available audit does not establish that all input features are available before the simulated disruption. Therefore, potential outcome leakage cannot be ruled out.

5. **Simulation-to-reality gap:** The experiments use simulated IEEE-24 power-grid scenarios. These results do not establish performance on real-world power grids.

6. **Class distribution:** The dataset's class distribution comes from the selected simulation data and should not be interpreted as the real-world frequency of blackout events.

7. **Probability limitations:** The Decision Tree produces discrete probability estimates in this experiment. These values should not be interpreted as calibrated real-world risk probabilities.

## 9. Conclusion

The Decision Tree provides a simple and interpretable baseline for the blackout prediction task.

It achieved a validation F1-score of 0.8065 but a lower test F1-score of 0.5333, indicating a substantial generalization gap. The test results also reveal a considerable number of false positives and six missed blackout scenarios.

Restricting tree complexity in the two controlled configurations did not improve validation F1-score. Similarly, threshold adjustments between 0.1 and 0.9 did not change validation predictions.

The original Decision Tree is therefore retained as the reference baseline for comparison with more advanced models. Its results highlight the importance of evaluating generalization, false negatives, and data limitations rather than relying only on training or validation performance.