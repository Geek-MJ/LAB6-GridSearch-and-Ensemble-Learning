# GridGuard — Power Grid Supply Disruption Prediction

# GridGuard – Power Grid Supply Disruption Prediction

## 1. Project Overview

GridGuard is a Machine Learning project focused on predicting power grid supply disruptions.

The objective is to develop a classification model capable of predicting whether a power grid scenario results in a supply disruption.

The project uses simulated power grid data based on the IEEE-24 system.

The target variable is `blackout`:
- `0`: All demand is supplied.
- `1`: Some demand is not supplied.

The goal is to identify potentially critical grid situations and evaluate how effectively Machine Learning models can detect them.

## 2. Dataset Description

The project uses the PowerGraph dataset, containing simulated power grid scenarios.

### Dataset characteristics

| Property | Description |
|---|---|
| Total records | 20,000 |
| Input features | 224 |
| Target variable | `blackout` |
| Class 0 | 15,968 records (79.84%) |
| Class 1 | 4,032 records (20.16%) |
| Missing values | 0 |

The dataset contains numerical features describing electrical grid characteristics, including transmission-line properties and power flows.

### Data preparation

The dataset was validated before model training.

The validation process included:
- Checking missing and infinite values.
- Verifying the target variable.
- Checking duplicate records.
- Verifying feature hashes against the official manifest.
- Checking the integrity of the official data splits.

The official dataset partitions were preserved:

| Partition | Records |
|---|---:|
| Training | 13,993 |
| Validation | 3,010 |
| Testing | 2,997 |

The same official split was used throughout the evaluation process to ensure consistency.

## 3. Methodology

The project follows a structured Machine Learning workflow:

1. Dataset collection and validation.
2. Data exploration and preprocessing.
3. Definition of training, validation, and test sets.
4. Training baseline classification models.
5. Comparing model performance.
6. Selecting the best model using validation results.
7. Selecting a classification threshold using validation data.
8. Evaluating the selected model on the independent test set.
9. Analyzing prediction errors and feature importance.

The test set was reserved for final evaluation.

## 4. Models and Experimental Results

Four models were compared using the validation dataset:

- Dummy Classifier (prior strategy)
- Decision Tree
- Logistic Regression
- Random Forest

### Validation results

| Model | Average Precision | ROC-AUC |
|---|---:|---:|
| Dummy Classifier | 0.2017 | 0.5000 |
| Decision Tree | 0.9768 | 0.9867 |
| Logistic Regression | 0.9681 | 0.9881 |
| Random Forest | 0.9811 | 0.9921 |

The Random Forest achieved the highest Average Precision on the validation set and was therefore selected.

A classification threshold of **0.277179** was selected using the validation set to maximize the F2-score.

## 5. Final Test Evaluation

The selected Random Forest model was evaluated on the independent test dataset.

### Test performance

| Metric | Result |
|---|---:|
| Accuracy | 96.56% |
| Precision | 88.50% |
| Recall | 95.37% |
| F1-score | 91.81% |
| F2-score | 93.91% |
| ROC-AUC | 99.33% |
| Average Precision | 98.49% |

### Confusion Matrix

The confusion matrix is:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2,317 | 75 |
| Actual 1 | 28 | 577 |

The model correctly classified 2,894 out of 2,997 test records.

## 6. Error Analysis

The model produced 103 incorrect predictions:

- 75 False Positives: the model predicted a disruption when the actual label was 0.
- 28 False Negatives: the model failed to detect an actual disruption.

False negatives are particularly important because they represent disruption scenarios that were missed by the model.

### Error analysis by data source

| Data Source | Test Records | Errors | Error Rate |
|---|---:|---:|---:|
| 1000_csv | 124 | 7 | 5.65% |
| 19000_zip | 2,873 | 96 | 3.34% |

Although most errors came from the larger ZIP source, its error rate was lower. The smaller CSV source has fewer test records, so its error rate should be interpreted cautiously.

### Feature importance

The Random Forest feature importance analysis identified several influential transmission-line variables.

The three most important features were:

| Feature | Importance |
|---|---:|
| line_5_reactive_power_flow | 4.29% |
| line_10_reactive_power_flow | 3.92% |
| line_10_reactance | 3.60% |

Other important features included active power flows and transmission-line ratings.

These importance scores describe the model's use of features; they do not establish causal relationships.

## 7. Visualizations

The evaluation process generates the following visualizations:

- Validation Precision-Recall curve.
- Validation ROC curve.
- Test confusion matrix.
- Test Precision-Recall curve.

The generated figures are stored in:

`figures/evaluation/`

## 8. Limitations

Several limitations should be considered:

- The dataset contains simulated scenarios rather than real-world power grid events.
- The results may not generalize directly to operational electricity networks.
- Duplicate grouping helps prevent identical feature records from crossing partitions, but does not guarantee independence between all simulation scenarios.
- The timing and availability of all input features before the predicted outcome should be verified to rule out possible temporal data leakage.

Therefore, the strong test performance should be interpreted within the context of this dataset.

## 9. Hyperparameter Optimization and Ensemble Learning

This section will document the additional experiments required by the project, including:

- GridSearchCV for hyperparameter optimization.
- Bagging models.
- Voting classifiers.
- Stacking classifiers, if implemented.
- Comparison of model performance and computational cost.

The corresponding results and visualizations should be added after these experiments have been completed.

## 10. Project Structure

```text
Lab-5-Machine-Learning-Project/
│
├── data/
│   ├── processed/
│   └── splits/
│
├── src/
│   └── evaluate.py
│
├── reports/
│   └── evaluation/
│       ├── evaluation_report.md
│       ├── evaluation_results.json
│       ├── validation_model_comparison.csv
│       ├── test_predictions.csv
│       ├── test_errors.csv
│       └── selected_model.joblib
│
├── figures/
│   └── evaluation/
│
└── README.md
```

## 11. How to Run the Evaluation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Lab-5-Machine-Learning-Project-
```

Activate your virtual environment if necessary and install the project dependencies.

Run the evaluation script:

```bash
python -m src.evaluate
```

The evaluation reports, predictions, selected model, and figures will be generated in their respective directories.

## 12. Conclusion

The experiments demonstrate that Machine Learning models can achieve strong classification performance on the simulated power grid dataset.

Among the evaluated models, Random Forest achieved the best validation Average Precision and strong performance on the independent test set.

However, model performance alone is not sufficient. Error analysis, feature interpretation, data quality, and potential limitations must also be considered before drawing conclusions about real-world applications.

---

**Project:** GridGuard – Power Grid Supply Disruption Prediction  
**Domain:** Machine Learning / Smart Grids  
**Dataset:** PowerGraph – IEEE-24 simulated scenarioscd "ESILV · Machine Learning · PowerGraph / IEEE-24

## Problem

Predict whether a simulated grid condition and line-failure scenario leaves some electricity demand unsupplied. The binary target is `blackout`: **0** = all demand supplied; **1** = some demand unsupplied, not necessarily a nationwide blackout.

**Dataset:** [PowerGraph](https://figshare.com/articles/dataset/PowerGraph/22820534) · 20,000 simulated records · 224 numerical inputs · 20.16% positive labels.

## Work completed

| Module | Output |
|---|---|
| Data audit and EDA | [Executed notebook](notebooks/01_data_audit_executed.ipynb), [figures](figures/), [audit](reports/data_audit.md) |
| Data cleaning | [Pipeline](src/clean_data.py), [cleaning report](reports/cleaning.md), [summary](data/processed/cleaning_summary.json) |
| Split design | [Grouped 70/15/15 manifests](data/splits/) |
| Initial Decision Tree baseline | [Notebook with outputs](notebooks/step4_baseline.ipynb) (earlier 1,000-row split) |
| Feature selection / advanced models | Planned |

**Cleaning:** No missing or infinite values. There are 1,421 repeated feature rows and 18,579 unique feature vectors. The primary validated dataset keeps all 20,000 scenarios. An optional deduplicated dataset keeps 18,579. Both retain 224 variables; no scaling or feature selection has been done outside training.

## Data files

- `data/Blackout_PowerGraph_1000.csv` and `data/raw/Blackout_PowerGraph_19000.zip` — unchanged sources.
- `data/processed/validated_20000.csv.gz` — validated full dataset.
- `data/processed/unique_18579.csv.gz` — optional deduplicated dataset.
- `data/processed/unique_records_manifest.csv` — original IDs and preserved split assignments.
- `data/processed/cleaning_summary.json` — evidence and hashes.

The cleaning script generates the processed files from the committed raw datasets. **Use the original 20,000-row split for the main study**, and evaluate the deduplicated version separately as a sensitivity analysis. The earlier Decision Tree baseline used a different split and is not directly comparable.

## Reproduce

Python 3.12 recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.audit
python -m src.clean_data
```

For the processed CSVs, read `pd.read_csv("data/processed/validated_20000.csv.gz")` or the unique version. Always join split assignments through the provided record IDs/manifest; don't randomly resplit the cleaned data.

## Limits

These are simulated IEEE-24 records. Duplicate grouping prevents identical complete inputs crossing partitions but does not guarantee independent simulation families. Source feature timing relative to failures still needs verification. Any scaling, feature selection or model tuning must fit on training data only.

## Team

Mahij Khantigiri GOSAI · Anais KAMTCHOUANG NONO · Soukaina BAHAMI · Yutong CAO

