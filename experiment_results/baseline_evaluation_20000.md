# GridGuard — Decision Tree Baseline Evaluation (Official Team Split)

**Author:** Yutong CAO

**Task:** Decision Tree baseline evaluation, model comparison, threshold selection, final test assessment and error analysis

**Dataset:** 20,000 simulated power-grid records

**Target:** `blackout` (0 = Normal, 1 = Blackout)

---

## 1. Objective

The objective of this experiment is to establish and evaluate a Decision Tree baseline for blackout prediction using the official team train/test split.

Three Decision Tree configurations are compared using grouped cross-validation. The selected model is then fitted on the complete training set and evaluated once on the independent test set.

The experiment focuses on classification performance, threshold sensitivity and prediction errors.

## 2. Dataset and Official Split

The dataset contains 20,000 simulated records and 224 numerical input features.

The official team split is:

| Dataset | Records | Normal (0) | Blackout (1) |
|---|---:|---:|---:|
| Training | 16,018 | 12,778 | 3,240 |
| Test | 3,982 | 3,190 | 792 |
| **Total** | **20,000** | **15,968** | **4,032** |

The split was produced using a group-aware procedure. Identical complete input feature vectors were assigned to the same group before the train/test partition was created.

The team verified that no identical complete input feature vectors are shared between the training and test sets.

There is no separate validation CSV in this protocol. Model comparison and threshold analysis are performed using cross-validation within the training set. The test set is reserved for final evaluation.

## 3. Validation Protocol

A five-fold `StratifiedGroupKFold` procedure was used on the 16,018 training records.

Groups were constructed using all 224 input features. Identical feature vectors received the same group identifier.

The training set contained:

- 16,018 records
- 14,863 unique complete feature-vector groups
- 1,155 additional duplicate records grouped with matching vectors

All five folds passed the group-disjoint verification: no group was shared between the training and validation portions of any fold.

This procedure reduces the risk of overly optimistic validation results caused by identical input vectors appearing in both portions of a fold.

## 4. Decision Tree Models

Three configurations were evaluated:

| Model | Parameters |
|---|---|
| Baseline Decision Tree | Default parameters, `random_state=42` |
| Controlled DT 1 | `max_depth=8`, `min_samples_leaf=5` |
| Controlled DT 2 | `max_depth=10`, `min_samples_leaf=10` |

All models used the same five-fold grouped cross-validation procedure.

### 4.1 Grouped Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline Decision Tree | **0.9871** | **0.9711** | 0.9648 | **0.9680** | 0.9788 |
| Controlled DT 1 | 0.9148 | 0.7522 | 0.8636 | 0.8040 | 0.9597 |
| Controlled DT 2 | 0.9293 | 0.7953 | 0.8756 | 0.8336 | 0.9690 |

The Baseline Decision Tree achieved the highest F1-score and was selected for final evaluation.

The controlled models performed less well under this validation protocol. This suggests that the restrictions applied to their tree complexity reduced their ability to capture patterns in this simulated dataset.

The results do not establish that more complex trees are always preferable; they indicate that the unrestricted baseline performed best in this experiment.

## 5. Threshold Analysis

Threshold selection was performed using the selected model's out-of-fold probabilities from the training set.

Thresholds between 0.1 and 0.9 were evaluated.

All tested thresholds produced identical results:

| Threshold | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.1 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |
| 0.2 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |
| 0.3 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |
| 0.4 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |
| 0.5 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |
| 0.6 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |
| 0.7 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |
| 0.8 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |
| 0.9 | 0.9871 | 0.9711 | 0.9648 | 0.9680 |

Only two distinct out-of-fold probability values were observed:

- 0.0: 12,799 records
- 1.0: 3,219 records

This explains why changing the threshold within the tested range did not change the predicted classes.

The default threshold of 0.5 was retained. The model's discrete probability outputs should not be interpreted as calibrated estimates of real-world blackout risk.

## 6. Final Model

The selected Baseline Decision Tree was refitted using all 16,018 training records.

| Model characteristic | Result |
|---|---:|
| Training records | 16,018 |
| Tree depth | 27 |
| Number of leaves | 232 |
| Number of nodes | 463 |

The final model was evaluated on the independent test set using the retained threshold of 0.5.

## 7. Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | **0.9922** |
| Precision | **0.9860** |
| Recall | **0.9747** |
| F1-score | **0.9803** |
| ROC-AUC | **0.9856** |
| Test records | 3,982 |

### 7.1 Confusion Matrix

| Actual / Predicted | Normal (0) | Blackout (1) |
|---|---:|---:|
| Normal (0) | 3,179 | 11 |
| Blackout (1) | 20 | 772 |

The model correctly classified:

- 3,179 normal records
- 772 blackout records

It made 31 classification errors:

- 11 False Positives
- 20 False Negatives

The blackout recall of 0.9747 means that 772 of the 792 blackout records in the test set were correctly identified.

The results show strong classification performance on this simulated test dataset.

## 8. Error Analysis

The 31 misclassified test records were examined separately.

| Error type | Count | Description |
|---|---:|---|
| False Positive | 11 | Normal records incorrectly predicted as blackout |
| False Negative | 20 | Blackout records incorrectly predicted as normal |
| **Total** | **31** | |

The False Negatives represent 64.5% of all classification errors, while False Positives represent 35.5%.

The predicted probabilities of the misclassified records were also examined:

- All 11 False Positives had a predicted probability of 1.0.
- All 20 False Negatives had a predicted probability of 0.0.

This is consistent with the discrete probability outputs observed during threshold analysis.

### 8.1 Feature Difference Analysis

An exploratory comparison was conducted between:

- False Negatives and True Positives
- False Positives and True Negatives

Standardized Mean Differences (SMDs) were calculated to identify features with relatively large differences in group means.

For False Negatives compared with True Positives, some of the largest observed differences involved:

- `line_9_reactive_power_flow`
- `line_7_reactive_power_flow`
- `line_4_reactive_power_flow`
- `bus_20_net_apparent_power`
- `bus_6_net_apparent_power`

For False Positives compared with True Negatives, prominent differences included:

- `bus_3_voltage_magnitude`
- `line_24_reactive_power_flow`
- `bus_14_voltage_magnitude`

These are descriptive comparisons, not evidence that the features caused the classification errors.

The number of errors is relatively small, especially for False Positives (11 records). Consequently, the feature-level findings should be interpreted as exploratory rather than conclusive.

## 9. Discussion

The Baseline Decision Tree achieved an F1-score of 0.9680 during grouped cross-validation and 0.9803 on the independent test set.

The difference between cross-validation and test performance should be interpreted cautiously, as both estimates are based on finite samples from the same simulated dataset.

The model performed well under the official team split, including when identical complete feature vectors were prevented from crossing the train/test boundary.

The controlled trees had lower validation F1-scores, indicating that their restrictions on tree complexity were less suitable for this dataset.

The test results also show that the model identified most blackout records, although 20 blackout cases were missed. In a real operational context, False Negatives could be particularly important because they represent potentially undetected blackout conditions.

The high performance should be interpreted in the context of the simulated dataset and its feature construction. It should not be presented as evidence of equivalent performance on real power-grid events.

## 10. Limitations

Several limitations must be considered:

1. **Simulated data:** The experiment uses simulated power-grid records rather than independent real-world blackout events.
2. **Feature-vector grouping:** The group-aware procedure prevents identical complete input vectors from crossing partitions, but it does not necessarily eliminate all correlations between related simulated operating conditions.
3. **Feature timing:** The available dataset documentation does not establish that every predictor would be available before a real blackout event. Potential temporal or feature-level leakage therefore cannot be completely ruled out.
4. **Probability calibration:** The Decision Tree produced only 0.0 and 1.0 out-of-fold probability values. These outputs are not calibrated risk estimates.
5. **Threshold sensitivity:** The tested thresholds did not change classification outcomes, limiting the usefulness of threshold adjustment for this particular model.
6. **Error analysis sample size:** The relatively small numbers of False Positives and False Negatives limit the strength of conclusions from feature-level comparisons.
7. **Deployment:** The experiment does not establish operational reliability, real-time performance or readiness for deployment in a real power grid.

## 11. Conclusion

The Baseline Decision Tree was the best-performing Decision Tree configuration in the five-fold grouped cross-validation experiment.

It achieved the following results on the independent official test set:

- Accuracy: 99.22%
- Precision: 98.60%
- Recall: 97.47%
- F1-score: 98.03%
- ROC-AUC: 98.56%

The model correctly identified 772 of 792 blackout records, with 20 False Negatives and 11 False Positives.

These results establish a strong Decision Tree baseline for comparison with the advanced classifiers developed by the other team members. They should be understood as results on the current simulated dataset, not as validation of a real-world blackout prediction system.

## 12. Reproducibility and Supporting Files

The experiment outputs are saved in the `experiment_results/` directory:

| File | Description |
|---|---|
| `decision_tree_final_results.csv` | Grouped CV and final test metrics |
| `decision_tree_threshold_analysis.csv` | OOF threshold comparison |
| `decision_tree_test_errors.csv` | Misclassified test records |
| `decision_tree_error_feature_analysis.csv` | Exploratory feature difference analysis |

The main experiment notebook is:

- `notebooks/step4_baseline_20000_team_split.ipynb`

The official split methodology is documented in:

- `experiment_results/team_split.md`

The earlier three-way split experiment is historical and is not mixed with the official two-file split results reported here.