# GridGuard

**Predicting electricity supply disruptions in simulated power grids**

ESILV Machine Learning group project · [PowerGraph (IEEE-24) dataset](https://figshare.com/articles/dataset/PowerGraph/22820534)

A binary classifier predicts whether a simulated scenario results in unmet electricity demand (`blackout = 1`). The source consists of **20,000 scenarios**, **224 numerical input features**, and a binary target. Class 1 occurs in **4,032** scenarios (20.16%).

## Who does what?

| Owner | Assigned responsibilities | Work in this repository |
| --- | --- | --- |
| **Mahij Khantigiri GOSAI** | Dataset provenance; quality, duplicate and leakage audit; EDA; train/validation/test splits | [Executed data-audit notebook](notebooks/01_data_audit_executed.ipynb), [audit](src/audit.py), [cleaning](src/clean_data.py), [team split](src/split_data.py), [audit report](reports/data_audit.md), [split report](reports/team_split.md) |
| **Soukaina BAHAMI** | Preprocessing pipeline; DummyClassifier and Logistic Regression baselines; initial metrics and confusion matrices | [Existing evaluation synthesis](reports/EVALUATION_RESULTS_SOUKAINA.md); other work requires owner confirmation |
| **Anais KAMTCHOUANG NONO** | Two advanced classifiers; hyperparameter tuning; comparison against baselines; feature importance | [Advanced experiments notebook](notebooks/task3.ipynb) and [evaluation work](reports/evaluation_anais/evaluation_report.md); confirm exact ownership with the team |
| **Yutong CAO** | Evaluation protocol; validation threshold selection; final test evaluation; error analysis; results synthesis | [Decision Tree notebooks](notebooks/step4_baseline_20000.ipynb) and [report](reports/baseline_evaluation_20000.md); confirm final evaluation responsibilities with the team |

The table shows the **assigned responsibilities from the team**, not a claim that every stored file has been independently verified as the named person's work. Preserve teammates' source notebooks and saved results.

## Current team handoff: train and test

Anais requested **two CSVs only**, with no repeated identical input records across the partitions and no separate validation dataset.

- `data/processed/train.csv`: **16,018 records** (3,240 blackout cases)
- `data/processed/test.csv`: **3,982 records** (792 blackout cases)

The method groups **identical complete feature vectors**, then applies a stratified 80/20 split *to the unique groups*, assigning all 20,000 original rows to train or test. Because groups differ in size, the row counts are approximately, not exactly, 80/20. The verified cross-partition duplicate-input overlap is **zero**.

**Reproduce:** `python -m src.split_data`. The [data-audit notebook](notebooks/01_data_audit_executed.ipynb) shows the complete procedure and validation checks; [split report](reports/team_split.md) explains limitations. Model tuning and threshold selection must use cross-validation **within train.csv**; reserve test.csv for final evaluation.

**Existing team experiments are historical.** Older group-aware 70/15/15 manifests remain under `data/splits/` because existing baseline/evaluation scripts depend on them. Existing saved scores from earlier train/validation/test protocols are **not scores on these new CSVs**. The team must rerun affected models before claiming final results. Original sources remain available in `data/Blackout_PowerGraph_1000.csv` and `data/raw/Blackout_PowerGraph_19000.zip`; the validated full dataset is `data/processed/validated_20000.csv.gz`.

## Repository guide

```text
data/         Original files, one validated modeling dataset and historical/current split manifests
notebooks/    Data audit, baseline and advanced notebooks (saved team work preserved)
src/          Reproducible audit, cleaning, splitting and evaluation scripts
tests/        Data-quality and split checks
figures/      EDA and modeling figures
reports/      Data-audit evidence and existing model evaluation reports
```
