# Blackout Prediction

## 1. Project Overview

This project aims to predict blackout events in an electrical power grid using machine learning techniques.

The objective is to identify grid operating conditions that may lead to situations where part of the electricity demand cannot be satisfied.

## 2. Machine Learning Problem

This is a **binary classification problem**.

The target variable is `blackout`:

- `0`: All electricity demand is satisfied.
- `1`: Part of the electricity demand cannot be satisfied.

The dataset contains 224 explanatory variables describing the power grid conditions.

## 3. Dataset

The project currently uses `Blackout_PowerGraph_1000.csv`, containing 1,000 observations.

- Number of observations: 1,000
- Number of explanatory variables: 224
- Target variable: `blackout`
- Missing values: 0

Class distribution:

- Class 0: 798 observations (79.8%)
- Class 1: 202 observations (20.2%)

The dataset source URL and license information will be added after verification.

## 4. Baseline Model

The baseline model is a **Decision Tree Classifier** implemented using scikit-learn.

It was selected because it is simple, interpretable, and capable of capturing non-linear relationships between grid operating conditions and blackout events.

## 5. Data Splitting Strategy

The dataset is divided into three subsets using stratified sampling:

| Subset | Number of samples | Percentage |
|---|---:|---:|
| Training | 700 | 70% |
| Validation | 150 | 15% |
| Test | 150 | 15% |

The random state is set to 42 to ensure reproducibility.

Stratification preserves the original class distribution across all subsets.

## 6. Evaluation Metrics

The baseline model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Recall and F1-score for the blackout class are particularly important because the dataset is imbalanced.

## 7. Initial Validation Results

The Decision Tree baseline achieved the following validation results:

| Metric | Score |
|---|---:|
| Accuracy | 0.9000 |
| Precision | 0.7586 |
| Recall | 0.7333 |
| F1-score | 0.7458 |
| ROC-AUC | 0.8375 |

Confusion Matrix:

```text
[[113   7]
 [  8  22]]
```

The model correctly identified 22 out of 30 blackout cases in the validation set.

The test set has been reserved for final evaluation.

## 8. Project Structure

```text
.
├── data/
│   └── Blackout_PowerGraph_1000.csv
├── notebooks/
│   └── step4_baseline.ipynb
└── README.md
```

## 9. How to Run

1. Clone this repository.
2. Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Open `notebooks/step4_baseline.ipynb` in Jupyter Notebook or VS Code.
4. Run the notebook cells sequentially.

## 10. Current Status

The initial Decision Tree baseline has been implemented and evaluated on the validation set.

Further model development and comparison will be conducted in subsequent project stages.