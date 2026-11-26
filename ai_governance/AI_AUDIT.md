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
[Add the AI error identified in this part of the project.]

**How it was detected and why it matters:**  
[Explain how the error was found and why it matters.]

**Correction and validation:**  
[Describe the correction and link to the code, output or test used to check it.]

## 4. Yutong CAO — Evaluation Error

**Original AI-output flaw:**  
[Add the AI error identified in this part of the project.]

**How it was detected and why it matters:**  
[Explain how the error was found and why it matters.]

**Correction and validation:**  
[Describe the correction and link to the code, output or test used to check it.]
