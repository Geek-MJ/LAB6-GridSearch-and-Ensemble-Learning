# GridGuard

**Predicting electricity supply disruptions in simulated power grids**

ESILV Machine Learning group project · [PowerGraph (IEEE-24) dataset](https://figshare.com/articles/dataset/PowerGraph/22820534)

A binary classifier predicts whether a simulated scenario results in unmet electricity demand (`blackout = 1`). The source consists of **20,000 scenarios**, **224 numerical input features**, and a binary target. Class 1 occurs in **4,032** scenarios (20.16%).

## Team responsibilities

| Owner | Assigned responsibilities | Work in this repository |
| --- | --- | --- |
| **Mahij Khantigiri GOSAI** | Dataset provenance; quality and leakage audit; exploratory analysis; train/test splitting | [Executed data-audit notebook](notebooks/01_data_audit_executed.ipynb), [audit](src/audit.py), [cleaning](src/clean_data.py), [team split](src/team_split.py), [audit report](reports/data_audit.md), [split report](reports/team_split.md) |
| **Soukaina BAHAMI** | Preprocessing pipeline; DummyClassifier and Logistic Regression baselines; initial metrics and confusion matrices | [Evaluation report](reports/EVALUATION_RESULTS_SOUKAINA.md) and [evaluation script](src/evaluate.py) |
| **Anais KAMTCHOUANG NONO** | Two advanced classifiers; hyperparameter tuning; comparison against baselines; feature importance | [Advanced-model notebook](notebooks/task3.ipynb) and [related evaluation report](reports/evaluation_anais/evaluation_report.md) |
| **Yutong CAO** | Evaluation protocol; grouped cross-validation; threshold analysis; final test evaluation; error analysis; results synthesis | [Decision Tree notebook](notebooks/step4_baseline_20000_team_split.ipynb) and [baseline report](reports/baseline_evaluation_20000.md) |

## Training and test data

The final data preparation uses two files, without a separate validation dataset.

- `data/processed/train.csv`: **16,018 records** (3,240 blackout cases)
- `data/processed/test.csv`: **3,982 records** (792 blackout cases)

The method groups **identical complete feature vectors**, then applies a stratified 80/20 split *to the unique groups*, assigning all 20,000 original rows to train or test. Because groups differ in size, the row counts are approximately, not exactly, 80/20. The verified cross-partition duplicate-input overlap is **zero**.

**Reproduction check:** Run `python -m src.team_split` to regenerate the group assignments in memory and compare every row against the existing CSVs without overwriting them. Run `python -m pytest tests/test_official_team_split.py` for regression checks. The [data-audit notebook](notebooks/01_data_audit_executed.ipynb) contains the procedure and saved results, and the [split report](reports/team_split.md) describes the checks. Model tuning and threshold selection use cross-validation within `train.csv`, with `test.csv` reserved for final evaluation.

Earlier model experiments used a 70/15/15 train/validation/test split. `src/split_data.py` and `src/evaluate.py` still implement a separate historical grouped-bus evaluation workflow and are **not** the reproduction command for the final 80/20 team CSVs. The older manifests are kept in `data/splits/`, and their scores refer to those earlier partitions. The original source files are in `data/`; the complete validated dataset is `data/processed/validated_20000.csv.gz`.

## Repository guide

```text
ai_governance/  Contributions, AI prompts and error audit
data/         Original files, validated data and split manifests
notebooks/    Data audit, baseline and advanced-model notebooks
src/          Reproducible audit, cleaning, splitting and evaluation scripts
tests/        Data-quality and split checks
figures/      EDA and modeling figures
reports/      Data audit and model evaluation reports
```
