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

## 4. Yutong CAO — Evaluation Error

**Original AI-output flaw:**

During the out-of-fold prediction workflow, the AI assistant incorrectly assumed that `oof_predictions` was a dictionary indexed by model name. In the notebook at that point, the variable was actually a NumPy array with shape `(16018,)`. The suggested instructions therefore did not match the notebook's real variable structure.

**How it was detected and why it matters:**

I checked the actual variable type and shape in the notebook and found that `oof_predictions` was a NumPy array rather than a dictionary. Following the incorrect assumption would have caused errors when accessing predictions by model name and could have disrupted the comparison of the three Decision Tree models. This was caught before the final evaluation results were reported.

**Correction and validation:**

The out-of-fold workflow was corrected to store predictions and probabilities in separate dictionaries, with one array per model. I reran the five-fold grouped cross-validation and checked that each model had 16,018 out-of-fold predictions and probabilities. I also verified that identical complete feature-vector groups did not overlap between the training and validation portions of any fold. The corrected results were then used for model comparison, threshold analysis and the final evaluation report.

**Files checked:** `notebooks/step4_baseline_20000_team_split.ipynb`, `reports/baseline_evaluation_20000.md`, `reports/decision_tree_final_results.csv`, `reports/decision_tree_threshold_analysis.csv`.

