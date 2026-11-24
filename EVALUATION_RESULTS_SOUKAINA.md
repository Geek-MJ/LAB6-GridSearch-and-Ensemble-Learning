# Evaluation Protocol, Threshold Selection, Final Test Evaluation, Error Analysis and Results Synthesis

## 1. Evaluation Protocol

The GridGuard dataset contains 20,000 validated records and 224 input features. The target variable is `blackout`, which indicates whether a blackout occurs.

The dataset was divided into three subsets:
- Training: 12,824 records.
- Validation: 3,176 records.
- Testing: 4,000 records.

The initial split used stratified sampling to preserve the class distribution. The validation split used GroupShuffleSplit, while model hyperparameter tuning used 5-fold StratifiedGroupKFold cross-validation.

A Random Forest classifier was tuned using recall as the optimization metric. The grid search evaluated 1,152 parameter combinations, resulting in 5,760 model fits.

The selected hyperparameters were:
- `class_weight`: balanced
- `max_depth`: 11
- `max_features`: 225
- `min_samples_leaf`: 2
- `min_samples_split`: 6
- `n_estimators`: 6

The best mean cross-validation recall was 97.23%.

## 2. Threshold Selection on Validation Data

The decision threshold determines the probability above which a prediction is classified as a blackout.

Several thresholds between 0.05 and 0.50 were evaluated using validation data.

A lower threshold increases recall but also generates more false alarms. For example, at threshold 0.15, recall reached 99.09%, but there were 298 false positives and 6 false negatives.

At threshold 0.50, recall was 97.43%, precision was 75.68%, and the model produced 207 false positives and 17 false negatives.

Following the team's decision and Anaïs's analysis, the threshold was fixed at 0.50. This provided a better balance between detecting blackouts and limiting unnecessary alerts.

## 3. Final Test Evaluation

After selecting the Random Forest model and fixing the decision threshold at 0.50, the final evaluation was performed on the reserved test set.

The results were:

| Metric | Test result |
|---|---:|
| Accuracy | 93.45% |
| Precision | 76.98% |
| Recall | 96.28% |
| F1-score | 85.56% |
| ROC-AUC | 98.21% |
| Average Precision | 94.96% |

The confusion matrix was:

| | Predicted no blackout | Predicted blackout |
|---|---:|---:|
| Actual no blackout | 2,962 | 232 |
| Actual blackout | 30 | 776 |

The model correctly identified 776 out of 806 actual blackout cases and missed 30. These results show strong blackout detection performance on the reserved test partition.

## 4. Error Analysis

The model made 262 incorrect predictions:
- 232 false positives.
- 30 false negatives.

False negatives are particularly important because they represent actual blackouts that the model failed to detect. The relatively low number of false negatives is consistent with the model's high recall.

The error rates by source were:
- `1000_csv`: 12 errors out of 190 records (6.32%).
- `19000_zip`: 250 errors out of 3,810 records (6.56%).

The error rates are relatively similar across the two sources.

Feature importance analysis showed that `line_5_reactive_power_flow` was the most influential feature, with an importance of approximately 38.15%. It was followed by `line_10_rating` at 7.84% and `line_10_reactive_power_flow` at 5.52%.

These values indicate which features the Random Forest relied on most when making predictions. They do not establish a causal relationship with blackout events.

## 5. Results Synthesis

The final Random Forest achieved 93.45% accuracy and 96.28% recall on the test set. Its high recall indicates that it detected most blackout cases, which is important because missed blackouts can have serious consequences.

The threshold of 0.50 was retained to reduce false alerts compared with lower thresholds. Although the validation comparison showed stronger overall metrics for Logistic Regression, the team retained the Random Forest following the selected approach based on Anaïs's work.

Overall, the results demonstrate that the model can identify simulated blackout scenarios with strong performance. However, the dataset contains simulated scenarios, so these results do not prove real-world grid performance.

One limitation is that the initial train/test split was stratified but not group-aware. Some bus configurations therefore appeared in both the training pool and test set. The results should consequently not be interpreted as a definitive measure of generalization to completely unseen bus configurations.