# GridGuard — AI Interaction Log

We have organized the AI prompts used during the project by student. Each section includes the request, the information provided and the checks carried out on the result.

## 1. Mahij Khantigiri GOSAI — Data Preparation and Quality Analysis

The following prompts cover the main stages of my data-preparation module. They were prepared together for a complete review of the work; they are not verbatim records of four earlier conversations.

### Prompt 1 — Dataset provenance and quality checks

> I am responsible for the data preparation and analysis part of our GridGuard ML project. We are using the PowerGraph IEEE-24 dataset, provided in two files containing 1,000 and 19,000 records.
>
> I need to combine both files and check whether the dataset is suitable for predicting blackout as a binary classification problem.
>
> Can you help me verify the source, number of records, feature types, missing values, class distribution, duplicate observations and constant features?
>
> Also check if identical input records have different target labels, because that could create problems during training.
>
> I want to keep the original dataset unchanged unless cleaning is actually necessary. Please explain the checks, generate reproducible Python code and summarize the findings for my report. Don't assume the data is correct without verifying it.

### Prompt 2 — Exploratory analysis and feature investigation

> Now that the dataset quality checks are done, I want to prepare the EDA for our GridGuard project.
>
> The dataset contains 224 numerical features related to buses and transmission lines, with blackout as the binary target.
>
> Help me analyse the target distribution, feature distributions, correlations and differences between blackout and non-blackout scenarios.
>
> I want useful visualizations that actually help us understand the dataset rather than generating unnecessary graphs.
>
> Also check whether some features are repeated, highly correlated or potentially problematic for ML models.
>
> Keep the analysis reproducible in Jupyter Notebook, explain the important findings and mention any limitations. The EDA should not be used to select features based on the test data, since that could introduce leakage.

### Prompt 3 — Data splitting and leakage prevention

> I noticed that simply using a random stratified split might not be enough for our dataset because some records have identical input features.
>
> I want to check whether repeated simulation inputs can appear in both training and testing, and how this could affect our model evaluation.
>
> Can you compare the normal stratified splitting approach with a group-aware split?
>
> We need to preserve the blackout class distribution, keep all original records and make sure identical complete input vectors stay together.
>
> Also verify that blackout is excluded when creating the groups and that preprocessing or feature selection does not use information from the test set.
>
> Use a fixed random state, check the group overlap between partitions and explain which splitting method is more suitable for our project.
>
> Please include the actual split sizes, class distribution and verification checks in the notebook.

### Prompt 4 — Final module verification and reproducibility

> I have completed my data preparation module for GridGuard and want to verify everything before sharing it with the other team members.
>
> Please review the dataset audit, cleaning decisions, EDA, splitting scripts, notebook and reports.
>
> Check that the original 20,000 records and 224 input features are preserved, the target values are correct and no identical complete input vectors cross the intended data partitions.
>
> Also verify that the splitting code matches the method described in the notebook and reports. I don't want different split versions being mixed up.
>
> Run the relevant checks where possible and identify any errors, inconsistencies or leakage risks.
>
> I want a clear summary of the verified results, any corrections needed and the remaining limitations.
>
> Keep the project files organized, use relative paths and make sure the other members can reproduce my work without changing their existing notebooks or results.

### Review findings and files

The saved dataset audit reports **20,000 records**, **224 numerical predictors**, **15,968 negative** and **4,032 positive** labels, **zero missing or non-finite values**, **18,579 unique complete input vectors**, **1,421 repeated vectors**, and **zero conflicting labels for identical inputs**.

The saved two-file exports contain **16,018 training** and **3,982 testing** records, with no identical complete input vectors crossing those files. However, the current `src/split_data.py` has since changed to a three-way protocol (approximately 70/15/15, seed 2026) based on the 72 bus variables. This differs from the older two-file method described in `reports/team_split.md`. The current three-way script was inspected but not executed during this review. The group needs one agreed protocol before final model comparisons.

**Files reviewed:** `src/audit.py`, `src/clean_data.py`, `src/split_data.py`, `tests/test_team_split.py`, `data/processed/cleaning_summary.json`, `notebooks/01_data_audit_executed.ipynb` and `reports/team_split.md`.

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


## 4. Yutong CAO — Evaluation and Decision Tree Experiments

**Prompts (summarised from the actual interactions):**

1. Adapt the Decision Tree baseline notebook to the official 20,000-record team split, using only `train.csv` and `test.csv`. Preserve the original experiment and avoid using the test set during model selection.
2. Implement grouped five-fold cross-validation so that identical complete 224-feature input vectors remain in the same fold. Verify that there is no group overlap between training and validation folds.
3. Compare the default Decision Tree with controlled-depth Decision Trees using accuracy, precision, recall, F1-score and ROC-AUC.
4. Analyse out-of-fold predicted probabilities across different classification thresholds. Explain whether threshold selection changes the predictions and whether the probabilities can be interpreted as calibrated risk estimates.
5. Evaluate the selected model once on the untouched test set and analyse false positives, false negatives and feature differences between correctly and incorrectly classified cases.

**Task and dataset:**  
Evaluate a Decision Tree baseline for binary classification of simulated PowerGraph scenarios. The dataset contains 20,000 records, 224 numerical input features and a binary `blackout` target, where 1 indicates demand not served and 0 indicates no demand not served.

**Constraints:**  
Use the official team split (`train.csv`: 16,018 records; `test.csv`: 3,982 records). Group identical complete input vectors during cross-validation. Use grouped cross-validation on the training set for model comparison and threshold analysis. Keep the test set untouched until final evaluation. Report all relevant classification metrics and distinguish simulated-data performance from real-world grid performance.

**Response and checks:**  
The final workflow used `StratifiedGroupKFold` with five folds. It verified that identical feature-vector groups did not overlap between training and validation folds. The default Decision Tree achieved the highest grouped out-of-fold F1-score among the three tested trees. Thresholds from 0.1 to 0.9 produced identical predictions because the tree returned only probabilities of 0.0 and 1.0. The final model was refitted on all training records and evaluated on the test set.

**Files:**  
`notebooks/step4_baseline_20000_team_split.ipynb`, `reports/baseline_evaluation_20000.md`, `reports/decision_tree_final_results.csv`, `reports/decision_tree_threshold_analysis.csv`, `reports/decision_tree_test_errors.csv`, `reports/decision_tree_error_feature_analysis.csv`.
