# GridGuard — Individual Contributions

**Project:** Prediction of unsupplied electricity demand in PowerGraph IEEE-24 simulations.  
**Dataset:** 20,000 simulated observations, 224 numerical predictors and the binary target `blackout`.

This document separates the team's **assigned responsibilities** from work visible in the repository. The notebook contents, saved reports, code and relevant Git commits were inspected; the experiments were **not rerun** for this review. All four students should confirm the final attribution before submission.

## Responsibility Matrix

| Student | Assigned responsibilities | Identifiable work |
| --- | --- | --- |
| Mahij Khantigiri GOSAI | Dataset provenance, quality and leakage audit, exploratory analysis, train/test splitting | Data-audit notebook, EDA figures, audit and split code, checked train/test exports |
| Soukaina BAHAMI | Preprocessing, DummyClassifier and Logistic Regression baselines, initial metrics and confusion matrices | Evaluation code committed through Soukaina-associated accounts, model comparisons, threshold and error analysis, named evaluation synthesis |
| Anais KAMTCHOUANG NONO | Two advanced classifiers, tuning, baseline comparisons, feature importance | Uploaded advanced-model notebook studying Random Forest, Gradient Boosting, grid searches and feature analyses |
| Yutong CAO | Evaluation protocol, threshold selection, final test assessment, error analysis, results synthesis | Decision Tree notebooks and baseline reports, including the full-dataset report explicitly crediting Yutong |

## 1. Mahij Khantigiri GOSAI — Data Analysis and Preparation

### Work performed

The data-preparation module documents the origin of the 1,000-row and 19,000-row PowerGraph extracts, their schema and the binary classification target. It examines missing or non-finite values, target distribution, repeated complete input vectors, label consistency, overlap between the supplied extracts and possible information leakage. Exploratory figures describe class counts, feature families, duplicate groups, differences between classes and feature correlations.

The dataset splitting procedure was revised after a conventional stratified outer split allowed duplicate-input overlap. The current procedure assigns identical complete input vectors to the same group, then stratifies the distinct groups and exports **16,018 training** and **3,982 testing** records. Verification found **zero identical complete feature vectors shared across the two exported datasets**. All 20,000 original observations are retained.

### Deliverables and evidence

- `notebooks/01_data_audit_executed.ipynb` — executed provenance, audit, exploratory analysis and splitting notebook.
- `src/audit.py`, `src/clean_data.py`, `src/split_data.py` — reproducible audit, validation and splitting implementations.
- `tests/test_clean_data.py`, `tests/test_splits.py`, `tests/test_team_split.py` — recorded quality and split checks.
- `data/processed/train.csv` and `data/processed/test.csv` — current two-file dataset handoff.
- `figures/01_target_by_extract.png` through `figures/05_feature_correlations.png` — exploratory figures.
- `reports/data_audit.md`, `reports/cleaning.md`, `reports/team_split.md` — data-methodology reports.

### Individual report contribution and limitations

Dataset provenance, integrity findings, exploratory evidence, duplicate-input leakage, split methodology and reproducibility. Exact-input separation does not prove complete independence among related simulations. Whether all source features are available before the simulated disruption has not been independently established.

## 2. Soukaina BAHAMI — Preprocessing, Baselines and Evaluation Implementation

### Assigned responsibilities and work found

The assigned work package covers preprocessing, DummyClassifier and Logistic Regression baselines, and initial metrics and confusion matrices.

Git history associates `src/evaluate.py` and its revisions with `bahamisoukaina1-ai` and `soukaina188`. The current script includes a pipeline using StandardScaler with Logistic Regression, a DummyClassifier reference, Decision Tree, Random Forest and Histogram Gradient Boosting candidates, cross-validation, comparison metrics, threshold selection, confusion-matrix outputs and error-analysis exports. The script was inspected but not executed during this review.

The report `reports/EVALUATION_RESULTS_SOUKAINA.md` documents the earlier model-evaluation protocol, Random Forest hyperparameter and threshold decisions, confusion matrix, errors by source extract and results synthesis. The saved `reports/evaluation_anais/` metrics and figures were also committed through a Soukaina-associated account, although they refer to Anais's model-selection work. This supports evaluation activity but not exclusive authorship of the model-development decisions.

The historical synthesis reports **93.45% test accuracy, 96.28% recall and 30 false negatives** on an earlier train/validation/test protocol. It does **not** establish results on the present train.csv and test.csv.

### Deliverables and evidence

- `src/evaluate.py` and `src/README.md` — evaluation code and execution notes.
- `reports/EVALUATION_RESULTS_SOUKAINA.md` — evaluation and interpretation report.
- `reports/evaluation_anais/evaluation_report.md`, `validation_model_comparison.csv`, `validation_threshold_comparison.csv`, `test_predictions.csv` and `test_errors.csv` — historical saved outputs associated with evaluation work.

### Individual report contribution and review point

Preprocessing/baseline methodology where personally implemented, model-evaluation design, classification results, threshold trade-offs and error analysis. **Soukaina should confirm exactly which baseline implementations and saved result files she produced**, including any shared work with Anais. The latest evaluation script and model claims still require confirmation on the team's current dataset partition.

## 3. Anais KAMTCHOUANG NONO — Advanced Classifier Development

### Work performed in the recorded notebook

The notebook `notebooks/task3.ipynb` was uploaded by the GitHub account `ANANONO`. It includes **Random Forest and Gradient Boosting** classifiers, hyperparameter searches using GridSearchCV, evaluation of alternative feature sets, model comparisons, decision-threshold analysis and feature-importance inspection. Its concluding discussion favors Random Forest with all variables and a decision threshold of 0.50 based on the observed recall and false-alarm trade-off.

Related saved reports under `reports/evaluation_anais/` present a Random Forest comparison, threshold tables and feature importance. Because those reports were committed by a Soukaina-associated account, their exact ownership should be jointly confirmed before including them as Anais's individual deliverables.

### Deliverables and evidence

- `notebooks/task3.ipynb` — advanced-model experiments, grid searches and feature comparisons.
- `reports/evaluation_anais/evaluation_report.md`, `validation_model_comparison.csv` and `feature_importance.csv` — related historical result artifacts, with shared attribution to confirm.
- `figures/evaluation_anais/random_forest_feature_importance.png` and related evaluation plots.

### Individual report contribution and limitations

Advanced-classifier selection, tuning and comparisons, feature analysis and classification-threshold decisions. The inspected notebook references a missing `blackout.csv` and an undefined `redundancy_columns` variable; its tuning/validation sequence also requires methodological review. **These limitations remain unresolved in that notebook**. The report should not claim that its experiments are fully reproducible on the latest train/test exports without a verified new run.

## 4. Yutong CAO — Evaluation and Decision Tree Baseline Analysis

### Work performed in the recorded notebooks

The report `reports/baseline_evaluation_20000.md` explicitly names **Yutong Cao** as the contributor. Commits from `Alaia1023` added `notebooks/step4_baseline.ipynb` and `notebooks/step4_baseline_20000.ipynb`, together with Decision Tree baseline reports.

The notebooks document data checks, group-aware historical partitions, Decision Tree training, comparisons against trees with restricted depth and leaf size, validation metrics, threshold analysis, final test confusion matrices and false-positive/false-negative inspection. The initial evaluation uses a 1,000-record subset; the extended report uses all 20,000 records.

The full-dataset historical report records **99.23% accuracy**, **97.36% recall**, **16 false negatives** and **7 false positives** under the earlier 13,993/3,010/2,997 split. These are **not** test results for the current two-file partition.

### Deliverables and evidence

- `notebooks/step4_baseline.ipynb` and `notebooks/step4_baseline_20000.ipynb`.
- `reports/baseline_evaluation.md` and `reports/baseline_evaluation_20000.md`.
- Historical partition manifests under `data/splits/` used by the recorded experiments.

### Individual report contribution and review point

Baseline evaluation protocols, controlled model comparison, validation and test metrics, threshold analysis, confusion-matrix interpretation and limitations. **Yutong should confirm the relationship between this identifiable Decision Tree work and the assigned responsibility for the final model evaluation and synthesis**, including whether additional final-evaluation artifacts were produced.

## Final Attribution and Consistency Review

Each student should confirm the precise code, analysis, tests and report sections they actually authored. The original task allocations do not match every repository contribution: Decision Tree baseline work is explicitly credited to Yutong, while broader evaluation files have commits associated with Soukaina. Related work must be credited to the appropriate contributors rather than assigned solely from directory names.

Saved model results use different, historical train/validation/test partitions. The current 16,018-row training file and 3,982-row test file are a newer handoff. **The historical metrics cannot be claimed as final performance on these two files.** Any current-protocol comparison or final evaluation must be demonstrated by a separately verified run.

**Status:** Four evidence-based contribution sections drafted. Individual attribution and final submission approval remain subject to each member's review.
