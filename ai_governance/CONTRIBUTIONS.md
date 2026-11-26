# GridGuard — Team Contributions

**Project:** Classification of supply disruption in simulated IEEE-24 power networks  
**Dataset:** PowerGraph, 20,000 scenarios, 224 input variables and the target `blackout`

This document summarizes the four parts of the project and the corresponding notebooks, scripts and reports.

| Member | Project responsibility |
| --- | --- |
| Mahij Khantigiri GOSAI | Data collection and verification, exploratory analysis, cleaning and splitting |
| Soukaina BAHAMI | Data preprocessing, baseline models and model evaluation |
| Anais KAMTCHOUANG NONO | Advanced models, hyperparameter tuning and feature analysis |
| Yutong CAO | Evaluation, comparison of results, threshold analysis and interpretation |

## 1. Mahij Khantigiri GOSAI — Data Preparation

My part focused on preparing the dataset for the classification experiments.

I examined the two PowerGraph extracts, containing 1,000 and 19,000 scenarios, and checked their columns, missing values, target distribution and repeated inputs. The combined dataset contains 20,000 rows and 224 numerical features. I also prepared exploratory plots showing class distribution, feature groups, duplicates and correlations.

The final data split uses approximately 80% of the records for training and 20% for testing. Identical input vectors are kept in the same partition to prevent duplicate records from appearing in both files. The final outputs are `train.csv` with 16,018 records and `test.csv` with 3,982 records. The split checks found no identical complete input vectors shared between the files.

**Files:**
- `notebooks/01_data_audit_executed.ipynb`
- `src/audit.py`, `src/clean_data.py`, `src/split_data.py`
- `tests/test_clean_data.py`, `tests/test_splits.py`, `tests/test_team_split.py`
- `data/processed/train.csv`, `data/processed/test.csv`
- `figures/01_target_by_extract.png` to `figures/05_feature_correlations.png`
- `reports/data_audit.md`, `reports/cleaning.md`, `reports/team_split.md`

**Report section:** Dataset description, data quality, exploratory analysis, cleaning decisions and train/test splitting.

## 2. Soukaina BAHAMI — Preprocessing and Model Evaluation

Soukaina's assigned work covers preprocessing, DummyClassifier and Logistic Regression baselines, and initial evaluation metrics. Her evaluation work also covers comparisons between classifiers, threshold analysis and classification errors.

The repository includes an evaluation script with preprocessing for Logistic Regression and comparisons involving DummyClassifier, Decision Tree, Random Forest and Gradient Boosting. The evaluation reports include precision, recall, F1-score, confusion matrices and error analysis. The saved results include experiments carried out with an earlier dataset split.

**Files linked to this work:**
- `src/evaluate.py`
- `reports/EVALUATION_RESULTS_SOUKAINA.md`
- `reports/evaluation_anais/evaluation_report.md`
- `reports/evaluation_anais/validation_model_comparison.csv`
- `reports/evaluation_anais/validation_threshold_comparison.csv`
- `reports/evaluation_anais/test_predictions.csv`
- `reports/evaluation_anais/test_errors.csv`

**Report section:** Preprocessing, baseline comparisons, evaluation metrics, threshold decisions and error analysis.

*The evaluation files include work linked to Anais's experiments, so the two contributors should confirm which parts of the shared evaluation they prepared.*

## 3. Anais KAMTCHOUANG NONO — Advanced Models

Anais's part focuses on developing and comparing advanced classification models.

Her notebook contains Random Forest and Gradient Boosting experiments, hyperparameter searches using `GridSearchCV`, comparisons using different sets of features, and analysis of precision, recall and decision thresholds. It also examines feature importance and compares model configurations.

The notebook discusses a Random Forest model using all features and a threshold of 0.50. These results belong to the recorded experiments and should be evaluated again if the final comparison uses the latest training and test files.

**Files:**
- `notebooks/task3.ipynb`
- `reports/evaluation_anais/evaluation_report.md` (shared evaluation work)
- `reports/evaluation_anais/feature_importance.csv` (shared evaluation work)
- `figures/evaluation_anais/random_forest_feature_importance.png` (shared evaluation work)

**Report section:** Advanced classifiers, parameter tuning, feature comparisons, model selection and interpretation.

## 4. Yutong CAO — Evaluation and Decision Tree Experiments

Yutong's work includes Decision Tree baseline experiments and their evaluation. The full-dataset baseline report identifies Yutong as the contributor.

The notebooks cover an initial experiment with 1,000 records and a second experiment using the complete 20,000-record dataset. They include training and validation results, comparisons with trees of limited depth, threshold analysis, confusion matrices and examination of false positives and false negatives.

The full-dataset report records a test F1-score of 0.9808 with 16 false negatives and 7 false positives under the earlier train/validation/test split. Those figures describe that experiment, not the latest two-file split.

**Files:**
- `notebooks/step4_baseline.ipynb`
- `notebooks/step4_baseline_20000.ipynb`
- `reports/baseline_evaluation.md`
- `reports/baseline_evaluation_20000.md`

**Report section:** Decision Tree baseline, validation and test results, threshold comparison, error analysis and interpretation.

## Shared Work

The four parts are connected through the common PowerGraph dataset. Dataset preparation provides the training and test files, the modeling work compares classification methods, and the evaluation work examines predictive performance and errors.

Some reports contain results from earlier dataset partitions. Results included in the final project comparison should use the same agreed evaluation procedure. Each member is responsible for checking their section and confirming any work completed jointly with another member.
