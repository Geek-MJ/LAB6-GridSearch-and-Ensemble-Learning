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
- `experiment_results/data_audit.md`, `experiment_results/cleaning.md`, `experiment_results/team_split.md`

**Report section:** Dataset description, data quality, exploratory analysis, cleaning decisions and train/test splitting.

## 2. Soukaina BAHAMI — Preprocessing and Model Evaluation

Soukaina's assigned work covers preprocessing, DummyClassifier and Logistic Regression baselines, and initial evaluation metrics. Her evaluation work also covers comparisons between classifiers, threshold analysis and classification errors.

The repository includes an evaluation script with preprocessing for Logistic Regression and comparisons involving DummyClassifier, Decision Tree, Random Forest and Gradient Boosting. The evaluation reports include precision, recall, F1-score, confusion matrices and error analysis. The saved results include experiments carried out with an earlier dataset split.

**Files linked to this work:**
- `src/evaluate.py`
- `experiment_results/EVALUATION_RESULTS_SOUKAINA.md`
- `experiment_results/evaluation_anais/evaluation_report.md`
- `experiment_results/evaluation_anais/validation_model_comparison.csv`
- `experiment_results/evaluation_anais/validation_threshold_comparison.csv`
- `experiment_results/evaluation_anais/test_predictions.csv`
- `experiment_results/evaluation_anais/test_errors.csv`

**Report section:** Preprocessing, baseline comparisons, evaluation metrics, threshold decisions and error analysis.

*The evaluation files include work linked to Anais's experiments, so the two contributors should confirm which parts of the shared evaluation they prepared.*

## 3. Anais KAMTCHOUANG NONO — Advanced Models

Anais's part focuses on developing and comparing advanced classification models.

Her notebook contains Random Forest and Gradient Boosting experiments, hyperparameter searches using `GridSearchCV`, comparisons using different sets of features, and analysis of precision, recall and decision thresholds. It also examines feature importance and compares model configurations.

The notebook discusses a Random Forest model using all features and a threshold of 0.50. These results belong to the recorded experiments and should be evaluated again if the final comparison uses the latest training and test files.

**Files:**
- `notebooks/task3.ipynb`
- `experiment_results/evaluation_anais/evaluation_report.md` (shared evaluation work)
- `experiment_results/evaluation_anais/feature_importance.csv` (shared evaluation work)
- `figures/evaluation_anais/random_forest_feature_importance.png` (shared evaluation work)

**Report section:** Advanced classifiers, parameter tuning, feature comparisons, model selection and interpretation.

## 4. Yutong CAO — Evaluation Protocol and Decision Tree Baseline

Yutong is responsible for the evaluation protocol, threshold selection, final test assessment, error analysis and synthesis of evaluation results.

Her work includes developing and evaluating the Decision Tree baseline using the official 20,000-record dataset. She implemented a five-fold Stratified Group Cross-Validation protocol to ensure that identical complete input feature vectors remain in the same fold, preventing exact duplicate inputs from appearing in both training and validation partitions.

The experiments compare a default Decision Tree with two controlled-depth Decision Trees using accuracy, precision, recall, F1-score and ROC-AUC. They also include out-of-fold threshold analysis, final evaluation on the untouched test set, confusion matrices and false-positive / false-negative analysis.

The final selected Decision Tree achieved a test F1-score of **0.9803**, with **20 false negatives** and **11 false positives**, on the official test set of 3,982 records. These results are based on simulated PowerGraph data and should not be interpreted as evidence of real-world grid performance.

**Files:**

- `notebooks/step4_baseline_20000_team_split.ipynb`
- `experiment_results/baseline_evaluation_20000.md`
- `experiment_results/decision_tree_final_results.csv`
- `experiment_results/decision_tree_threshold_analysis.csv`
- `experiment_results/decision_tree_test_errors.csv`
- `experiment_results/decision_tree_error_feature_analysis.csv`

**Report section:** Decision Tree baseline, grouped cross-validation, threshold analysis, final test evaluation, error analysis and interpretation.

## Shared Work

The four parts are connected through the common PowerGraph dataset. Dataset preparation provides the training and test files, the modeling work compares classification methods, and the evaluation work examines predictive performance and errors.

Some reports contain results from earlier dataset partitions. Results included in the final project comparison should use the same agreed evaluation procedure. Each member is responsible for checking their section and confirming any work completed jointly with another member.
