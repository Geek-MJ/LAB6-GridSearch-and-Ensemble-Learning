# GridGuard — Contributions and Module Ownership

This file records the agreed division of responsibilities for the PowerGraph IEEE-24 classification project. Each member must review and confirm the description of their completed work before final submission. Assigned responsibility is not, by itself, proof that every listed deliverable has been completed.

| Module owner | Assigned responsibility | Evidence and deliverables |
| --- | --- | --- |
| **Mahij Khantigiri GOSAI** | Dataset provenance; data quality, duplicate and leakage audit; exploratory data analysis; train/test partitioning | `notebooks/01_data_audit_executed.ipynb`; `src/audit.py`; `src/clean_data.py`; `src/split_data.py`; `tests/test_team_split.py`; `reports/data_audit.md`; `reports/team_split.md`; `data/processed/train.csv` and `test.csv` |
| **Soukaina BAHAMI** | Preprocessing pipeline; DummyClassifier and Logistic Regression baselines; initial metrics and confusion matrices | **To be completed by Soukaina:** identify actual files, experiments, checks and report subsection |
| **Anais KAMTCHOUANG NONO** | Two advanced classifiers; hyperparameter tuning; comparison against baselines; feature importance | **To be completed by Anais:** identify actual files, experiments, checks and report subsection |
| **Yutong CAO** | Evaluation protocol; validation-based threshold selection; final test evaluation; error analysis; results synthesis | **To be completed by Yutong:** identify actual files, experiments, checks and report subsection |

## Mahij Khantigiri GOSAI — Data Analysis and Preparation

The data-preparation module documents the origin and structure of the supplied IEEE-24 simulation extracts, including 20,000 observations, 224 numerical predictors and the binary target `blackout`.

The audit assessed missing and non-finite values, label distribution, repeated feature vectors, conflicting labels, and possible forms of information leakage. It also produced exploratory figures and described limitations of the simulated dataset. A corrected train/test procedure groups identical complete input vectors before assigning them to separate partitions. All original records are preserved.

**Current outputs:** `train.csv` contains 16,018 records; `test.csv` contains 3,982 records. Automated checks verified zero identical complete input vectors across the two exported datasets. The test data are reserved for final evaluation. This does not establish independence of related simulation scenarios or rule out post-outcome features.

**Evidence:** The notebook, splitting script, automated tests, data audit, and splitting report listed above. Previous three-way manifests are retained as evidence of historical experiments and should not be confused with the current split.

## Soukaina BAHAMI — Preprocessing and Baselines

**Assigned scope:** preprocessing; DummyClassifier and Logistic Regression baselines; initial metrics and confusion matrices.

**Completed work, exact file paths, verification and individual report subsection:**  
[Soukaina to complete.]

## Anais KAMTCHOUANG NONO — Advanced Models

**Assigned scope:** two advanced classifiers; hyperparameter tuning; comparisons with baseline models; feature importance.

**Completed work, exact file paths, verification and individual report subsection:**  
[Anais to complete.]

## Yutong CAO — Evaluation and Interpretation

**Assigned scope:** evaluation protocol; threshold selection; final test evaluation; error analysis; results synthesis.

**Completed work, exact file paths, verification and individual report subsection:**  
[Yutong to complete.]

## Final Review

Each student should verify their module description, confirm the evidence links, and be able to explain their technical decisions. Any shared work should be credited to the people who actually performed it. The team must confirm the ownership matrix before submission.
