# GridGuard — AI Interaction Log

The project instructions require **four to six significant interactions with an AI assistant**, with **at least one interaction per student**. Each record must explain the actual request, the technical assumptions and constraints supplied, what the assistant contributed, and how the output was verified.

Use authentic prompts or accurately labeled excerpts. Do not write a reconstructed prompt as though it were a verbatim transcript. Four entries are provided below, one per student. Add up to two further significant interactions only if relevant and documented.

## 1. Mahij Khantigiri GOSAI — Dataset Splitting and Duplicate Leakage

**Purpose:** Produce separate training and test files from the complete dataset without allowing identical complete input vectors to appear in both partitions.

**Recorded request (verbatim):**
> ok lets do that in the notebook and push the result ok?

**Context provided in the discussion:** Split the 20,000 observations into training and test data, avoid identical input records across partitions, save two comma-separated files, and do not create a separate validation dataset. The preceding discussion identified duplicate overlap in the earlier stratified split.

**Dataset and technical constraints:** 20,000 records, 224 numeric predictors, binary target `blackout`; approximately 80% training and 20% testing; `random_state=42`; target stratification; preservation of all original observations; complete input vectors grouped together; final test set excluded from tuning and feature selection.

**AI-assisted work:** Drafted and revised splitting code, notebook explanation, checks for input overlap, and reproducible train/test exports.

**Independent checks and evidence:** `src/split_data.py`, `tests/test_team_split.py`, `notebooks/01_data_audit_executed.ipynb` and `reports/team_split.md` document the actual implementation. The exported data contain 16,018 training and 3,982 test records, with no identical complete input vectors shared between the files. A separate review of feature timing and relationships between different simulation scenarios remains necessary.

## 2. Soukaina BAHAMI — Preprocessing and Baseline Models

**Actual prompt or clearly labeled excerpt:**  
[Paste one significant interaction from your own work.]

**Task and dataset context:**  
[State the real task, input schema and target variable.]

**Constraints and safeguards:**  
[Describe any instructions on training-only preprocessing, class balance, test-set isolation, and model requirements that were actually given.]

**AI output used and verification:**  
[Describe what was used, what you independently checked, and the supporting file path.]

## 3. Anais KAMTCHOUANG NONO — Advanced Model Development

**Actual prompt or clearly labeled excerpt:**  
[Paste one significant interaction from your own work.]

**Task and dataset context:**  
[State the real task, input schema and models.]

**Constraints and safeguards:**  
[Describe any actual instructions covering hyperparameter search, cross-validation, leakage prevention and fair model comparison.]

**AI output used and verification:**  
[Describe what was used, how it was verified, and the supporting file path.]

## 4. Yutong CAO — Evaluation and Interpretation

**Actual prompt or clearly labeled excerpt:**  
[Paste one significant interaction from your own work.]

**Task and dataset context:**  
[State the real evaluation or interpretation task.]

**Constraints and safeguards:**  
[Describe any actual instructions concerning test isolation, threshold selection, evaluation metrics, or error analysis.]

**AI output used and verification:**  
[Describe what was used, how it was verified, and the supporting file path.]

## Submission Check

Before submission, verify that the file contains **four to six real interactions**, **at least one from each student**. For every entry, keep the prompt evidence and describe the schema, assumptions, leakage guardrails, output and validation. Blank sections do not satisfy the requirement.
