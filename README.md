# GridGuard — Power Grid Supply Disruption Prediction

Machine Learning project · ESILV · PowerGraph / IEEE-24

## Problem

Predict whether a simulated electrical-grid scenario leaves some electricity demand unsupplied.

- **Inputs:** 224 numerical features describing buses and transmission lines.
- **Target:** `blackout` — **0** = all demand supplied; **1** = some demand unsupplied. A positive label does not necessarily mean a nationwide blackout.
- **Source:** [PowerGraph](https://figshare.com/articles/dataset/PowerGraph/22820534).

## Progress

| Work | Status |
|---|---|
| Initial Decision Tree baseline (1,000 rows) | Completed; saved outputs in `notebooks/step4_baseline.ipynb` |
| Data audit and exploratory analysis (20,000 rows) | Completed; saved outputs in `notebooks/01_data_audit_executed.ipynb` |
| Leakage-aware train/validation/test manifests | Completed |
| Advanced model comparison | Not yet completed |

The two supplied extracts contain **20,000 records**, with **4,032 positive labels (20.16%)**. The audit found **zero missing values**, **1,421 repeated input records**, **152 distinct feature patterns shared across extracts**, and **no conflicting labels for identical inputs**.

The split groups identical complete feature vectors so they cannot appear in different partitions:

| Partition | Records | Positive |
|---|---:|---:|
| Train | 13,993 | 2,820 |
| Validation | 3,010 | 607 |
| Test | 2,997 | 605 |

The existing Decision Tree validation results use a separate 1,000-row split; do not compare them directly against new grouped-split results.

## Repository

```text
data/
  Blackout_PowerGraph_1000.csv
  raw/Blackout_PowerGraph_19000.zip
  splits/                      # saved record-level split manifests
figures/                       # five EDA charts
notebooks/
  01_data_audit_executed.ipynb # executed audit and plots
  step4_baseline.ipynb         # original baseline
reports/
  audit_metrics.json
  data_audit.md
src/audit.py                   # reproducible audit, plots and split generation
tests/test_splits.py           # input quality and split checks
requirements.txt
```

## Run locally

Python 3.12 is recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.audit
```

The audit reads the committed CSV and ZIP, regenerates `data/splits/` and `figures/`, and refreshes the metrics report. The notebooks retain their saved outputs on GitHub.

## Limitations

All scenarios are simulated, not real-world outage observations. Input availability before the simulated cascade has not been independently verified. Grouping exact input duplicates prevents that form of overlap, but related scenarios may still be dependent. Exploratory feature differences are descriptive, not proof of causation or validated feature selection.

## Team

Mahij Khantigiri GOSAI · Anais KAMTCHOUANG NONO · Soukaina BAHAMI · Yutong CAO

