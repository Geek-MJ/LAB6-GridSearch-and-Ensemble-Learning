# Soukaina - Preprocessing and Baseline Results

Checked on 26 November 2026. This note covers preprocessing, DummyClassifier, Logistic Regression and initial evaluation.

## Setup

- Data: 20,000 simulated PowerGraph IEEE-24 records, 224 numerical inputs and target `blackout`.
- Current evaluator split: 13,944 training, 3,001 validation and 3,055 test records.
- Identical 72-variable bus configurations stay in one partition.
- These partitions differ from the team's existing 16,018/3,982 train/test CSVs. Scores from the two protocols should not be compared directly.

## Baselines

- DummyClassifier uses the training class proportions (`strategy="prior"`).
- Logistic Regression uses `StandardScaler` inside its pipeline. Scaling is fitted only on training data, including within each CV fold.
- Logistic Regression tests `C = 0.1, 1, 10` with five grouped CV folds on training data. Selection uses average precision.

## Validation results

Both baselines were rerun on the current data. The table uses a fixed threshold of 0.50; these are validation scores, not final test scores.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Dummy | 0.7971 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2029 |
| Logistic Regression | 0.9767 | 0.9514 | 0.9327 | 0.9420 | 0.9962 | 0.9864 |

Confusion matrices use rows = actual class and columns = predicted class, in order `[0, 1]`.

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| Dummy | 2392 | 0 | 609 | 0 |
| Logistic Regression | 2363 | 29 | 41 | 568 |

At 0.50, Dummy predicts class 0 for every record. Its 79.71% accuracy comes from the majority class; it detects no positive cases. Logistic Regression detects 568 of the 609 validation positives, with 29 false alerts and 41 missed positives.

## Evidence and limits

The verification completed both baseline searches, predictions, metric calculations and confusion-matrix generation. The final logistic scaler was checked against the 13,944 training rows.

- Code: [`evaluate.py`](../src/evaluate.py), [`split_data.py`](../src/split_data.py).
- Recorded results and environment: [`soukaina_baseline_validation.json`](soukaina_baseline_validation.json).
- The verification used scikit-learn 1.4.2 and pandas 2.2.2; these differ from the pinned requirements, so this does not certify execution with those exact pinned versions.
- This is an internal retrospective check on simulated data, not external or real-grid validation.
- The wider evaluator compares five models. Its final selected-model results are separate from these baseline validation results.

## Earlier experiment

The previous version of this note described a Random Forest experiment using a different split. Those historical outputs remain in [`evaluation_anais/`](evaluation_anais/). They are not results of the current baseline check.
