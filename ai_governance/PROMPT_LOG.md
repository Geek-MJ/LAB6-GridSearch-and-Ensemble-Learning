# GridGuard — AI Interaction Log

We have organized the AI prompts used during the project by student. Each section includes the request, the information provided and the checks carried out on the result.

## 1. Mahij Khantigiri GOSAI — Data Audit and Train/Test Split

**Prompt prepared for a new review (not an earlier chat message):**

> I want you to go through my GridGuard data preparation part properly and check whether everything is correct or not.
> 
> We are using the PowerGraph IEEE-24 dataset with 20,000 records, 224 numerical features and blackout as the target (0 or 1). From my initial audit I found 18,579 unique input combinations and 1,421 repeated records. There are no missing or infinite values.
> 
> Here's what I need you to check:
> 
> 1. First verify the dataset, class distribution, duplicates and whether any identical inputs have different target values.
> 2. Check my train test split. I want approximately 80/20 using random_state=42 and stratification so the blackout ratio stays similar.
> 3. One important thing is the duplicates. I dont want identical input values appearing in both train and test because that can cause data leakage. Keep all the original records but make sure repeated inputs stay in the same split.
> 4. Make sure blackout is only used as the target and not included when grouping the input features.
> 5. Check that preprocessing, feature selection and model tuning are done only on training data. We can use cross validation for this, but the final test set should stay separate.
> 6. Also check for other leakage risks, especially if some simulation records are related or if any features contain information from after the blackout event.
> 
> Once done, give me the exact train and test sizes, class distribution and duplicate overlap. Also make sure the split is reproducible and add proper checks in the code.
> 
> Dont just assume everything is correct. If something is wrong, explain the issue, why it matters and what needs to change. Keep the explanation simple but technically correct.
> 
> Use the actual dataset and code to verify the results. Dont make up numbers or say there's no leakage unless we have checked it properly.

**Why I wrote this:** I wanted a proper check of the data audit and the splitting method, especially duplicate inputs crossing the train/test boundary. I included the dataset structure, split settings, leakage risks and the checks I expected to see in the response.

**Previous results to compare with:** My existing split produced 16,018 training records and 3,982 test records, with no identical complete input vectors shared between the files. Those results come from the earlier work, not from running this new prompt.

**Project files:** `notebooks/01_data_audit_executed.ipynb`, `src/split_data.py`, `tests/test_team_split.py`, `reports/team_split.md`.

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

