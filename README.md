# GridGuard — Power Grid Supply Disruption Prediction

ESILV · Machine Learning · PowerGraph / IEEE-24

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

