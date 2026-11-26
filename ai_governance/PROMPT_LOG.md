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
1. Choice of evaluation metric
“I want to prioritise recall to reduce the number of missed blackout cases. Is this appropriate, and how should we account for false alarms when comparing models?”

2. Preventing data leakage
“Check that validation data is not used to train the model, fit preprocessing steps or select features. How should scenarios belonging to the same group be separated?”

3. Feature selection and hyperparameter tuning
“Can we jointly optimise hyperparameters and feature selection using cross-validation? Explain how to compare Filter, Redundancy, L1 and L2 without data leakage.”

4. Choosing an algorithm beyond the course
“Recommend an algorithm not covered in our course that is suitable for our tabular dataset. Explain its potential benefits compared with the models already tested and provide a scientific paper as a reference.”
**Task and dataset:**  
Binary classification of simulated power-grid scenarios using the IEEE 24-bus subset of PowerGraph. The objective is to predict whether a scenario results in unserved electricity demand, labelled blackout = 1, versus no unserved demand, labelled 0. This label does not necessarily represent a total blackout. Inputs consist of 224 numerical features describing buses and transmission lines. The models studied are Random Forest, Gradient Boosting, their soft-voting ensemble, and TabNet.

**Constraints:**  
Prioritise recall while also reporting precision, F1-score, false positives and false negatives. Use GridSearchCV with the same three group-aware cross-validation folds to compare configurations. Compare all features, ANOVA filtering, redundancy removal, Logistic L1 selection and Logistic L2-based selection. Fit feature selection and scaling only on each fold’s training data. Keep scenarios sharing identical bus states in the same group to prevent overlap between training and evaluation partitions. Use identical evaluation data for model comparisons, select thresholds using validation data, and reserve the test set for final evaluation. Document the search spaces and computational budgets; for soft voting, tune the weights using the previously selected RF and GB configurations.

**Response and checks:**  
Checks included Python syntax validation and reduced-grid runs for Random Forest, Gradient Boosting and soft voting on a 1,200-row sample. These checks confirmed that feature selectors used only the training rows of each fold and that the corrected outer splits shared no bus-state groups. TabNet was not executed during this review because PyTorch was unavailable in the review environment.

## 4. Yutong CAO — Evaluation and Interpretation

**Prompt:**  
[Add a prompt used for this part of the project.]

**Task and dataset:**  
[State the real evaluation or interpretation task.]

**Constraints:**  
[Describe any actual instructions concerning test isolation, threshold selection, evaluation metrics, or error analysis.]

**Response and checks:**  
[Describe the response used, the checks performed and the related files.]

