# GridGuard — AI Interaction Log

We have organized the AI prompts used during the project by student. Each section includes the request, the information provided and the checks carried out on the result.

## 1. Mahij Khantigiri GOSAI — Data Splitting and Leakage Check

**Task:** Correct the training and test split after finding identical input records across partitions.

**Recorded prompt:**
> ok lets do that in the notebook and push the result ok?

This was part of a longer discussion about the split. The agreed requirements were to keep all 20,000 records, create only `train.csv` and `test.csv`, and prevent identical complete input vectors from appearing in both files.

**Data and constraints:** The PowerGraph dataset has 224 numerical features and the binary target `blackout`. The split uses `random_state=42`, target stratification and an approximately 80/20 ratio. Identical feature vectors are assigned to the same group before splitting. The test set is kept separate from model selection; cross-validation can be performed on the training set.

**How I used AI:** I used the response to revise `src/split_data.py`, document the method in my notebook and add checks for duplicate inputs across the exported files.

**Verification:** The final files contain **16,018 training** and **3,982 testing** records, covering all 20,000 observations. The checks found **zero identical complete feature vectors shared between the files**.

**Files:** `notebooks/01_data_audit_executed.ipynb`, `src/split_data.py`, `tests/test_team_split.py`, `reports/team_split.md`.

## 2. Soukaina BAHAMI — Preprocessing and Baseline Models

**Prompt:**  
[Add a prompt used for this part of the project.]

**Task and dataset:**  
[State the real task, input schema and target variable.]

**Constraints:**  
[Describe any instructions on training-only preprocessing, class balance, test-set isolation, and model requirements that were actually given.]

**Response and checks:**  
[Describe the response used, the checks performed and the related files.]

## 3. Anais KAMTCHOUANG NONO — Advanced Model Development

**Prompt:**  
[Add a prompt used for this part of the project.]

**Task and dataset:**  
[State the real task, input schema and models.]

**Constraints:**  
[Describe any actual instructions covering hyperparameter search, cross-validation, leakage prevention and fair model comparison.]

**Response and checks:**  
[Describe the response used, the checks performed and the related files.]

## 4. Yutong CAO — Evaluation and Interpretation

**Prompt:**  
[Add a prompt used for this part of the project.]

**Task and dataset:**  
[State the real evaluation or interpretation task.]

**Constraints:**  
[Describe any actual instructions concerning test isolation, threshold selection, evaluation metrics, or error analysis.]

**Response and checks:**  
[Describe the response used, the checks performed and the related files.]

