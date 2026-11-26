# GridGuard — Audit of AI-Assisted Work

This file records errors we found while using AI assistance, how we corrected them and the files we checked afterwards. Each member has a section for an example from their own part of the project.

## 1. Mahij Khantigiri GOSAI — Input Duplication Across Data Partitions

**Initial problem:** The earlier approach used a conventional row-wise stratified train/test split. Stratification preserved the target-class distribution but did not keep identical input records together.

**How I found it:** I checked the earlier split and found **447 test records with complete feature vectors also present in the original training pool**. The same input values were present in training and testing.

**Impact:** Repeated inputs in training and testing can make performance look better than it would on new input configurations.

**Correction:** I grouped identical complete input vectors before splitting, then divided the groups using stratification on `blackout`. All 20,000 records were retained and assigned to training or testing. I did not create a separate validation file.

**Result:** `train.csv` contains **16,018** records and `test.csv` contains **3,982**. The checks found **no identical complete feature vectors across the two files**. The code and results are in `src/split_data.py`, `tests/test_team_split.py`, `notebooks/01_data_audit_executed.ipynb` and `reports/team_split.md`.

**Limitations:** Different records may still describe related simulation conditions. The source documents also do not fully establish when every feature is measured.

## 2. Soukaina BAHAMI — Baseline / Preprocessing Error

**Original AI-output flaw:**  
[Add the AI error identified in this part of the project.]

**How it was detected and why it matters:**  
[Explain how the error was found and why it matters.]

**Correction and validation:**  
[Describe the correction and link to the code, output or test used to check it.]

## 3. Anais KAMTCHOUANG NONO — Advanced Model Error

**Original AI-output flaw:**  
The earlier AI-provided workflow ran the Random Forest GridSearch before creating the validation set. As a result, observations later assigned to validation had already influenced hyperparameter selection.

**How it was detected and why it matters:**  
Reviewing the notebook’s execution order revealed that X_val was extracted from X_train after the GridSearch. This introduced data leakage and could make validation results overly optimistic.
**Correction and validation:**  
The dataset split was moved before all training and hyperparameter searches. Scenarios with identical bus states were kept in the same partition, and feature selection and scaling were placed inside the cross-validation pipelines. Reduced-grid checks on a 1,200-row sample confirmed that selectors used only their fold’s training rows and that the corrected partitions shared no bus-state groups.

## 4. Yutong CAO — Evaluation Workflow Error

**Original AI-output flaw:**  
During the implementation of out-of-fold evaluation, the AI incorrectly assumed that `oof_predictions` was a dictionary containing predictions for each model. In the actual notebook, the variable was a NumPy array with 16,018 elements. Instructions based on the incorrect variable structure would therefore fail when accessing predictions by model name.

**How it was detected and why it matters:**  
The variable's actual type and shape were inspected in the notebook. This revealed that the proposed access pattern did not match the existing implementation. Incorrect assumptions about stored predictions could lead to failed evaluation code or confusion between model outputs.

**Correction and validation:**  
The out-of-fold workflow was corrected to store predictions and probabilities in separate dictionaries keyed by model name. All three Decision Tree models were then evaluated across five grouped folds. Each model produced 16,018 out-of-fold predictions and probabilities, and the checks confirmed zero shared groups between training and validation folds.

The corrected results were used in `notebooks/step4_baseline_20000_team_split.ipynb` and checked against `reports/baseline_evaluation_20000.md` and the supporting evaluation CSV files.

The issue was corrected before the final reported test evaluation; it did not invalidate the final published metrics.

