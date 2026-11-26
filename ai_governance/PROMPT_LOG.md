# GridGuard — AI Interaction Log

We have organized the AI prompts used during the project by student. Each section includes the request, the information provided and the checks carried out on the result.

## 1. Mahij Khantigiri GOSAI — Dataset Splitting and Duplicate Leakage

**Task:** Prepare training and test files from the complete dataset without identical complete inputs in both partitions.

**Prompt:**
> ok lets do that in the notebook and push the result ok?

**Context:** The data needed to be divided into `train.csv` and `test.csv`, using all 20,000 records. The earlier stratified split had identical inputs in both partitions, so that had to be corrected. No separate validation file was needed.

**Constraints:** The dataset has 224 numerical predictors and the binary target `blackout`. The split uses approximately 80% training and 20% testing, `random_state=42` and target stratification. All original records are kept, identical complete inputs are grouped, and the test file is reserved for final evaluation.

**How I used the response:** I used it to revise the splitting code, update the notebook and check for identical inputs across the two files.

**Checks:** The exported files contain 16,018 training records and 3,982 test records, with no identical complete inputs shared. The work is documented in `src/split_data.py`, `tests/test_team_split.py`, `notebooks/01_data_audit_executed.ipynb` and `reports/team_split.md`.

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

