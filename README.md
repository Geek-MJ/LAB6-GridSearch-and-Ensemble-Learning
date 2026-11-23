# GridGuard — Power Grid Supply Disruption Prediction

ESILV Machine Learning project · IEEE-24 / PowerGraph simulations

## Objective

Predict whether a simulated grid condition and line-failure scenario will leave any electricity demand unsupplied. The target is `blackout`: **0** = all demand supplied; **1** = some demand unsupplied (not necessarily a nationwide blackout).

## Dataset

The provided extracts contain **20,000 records**, **224 numeric inputs** and one binary label. Class 1 occurs in **20.16%** of records. Source: [PowerGraph](https://figshare.com/articles/dataset/PowerGraph/22820534).

The 1,000-row CSV is committed in `data/`. The separate 19,000-row source ZIP is not uploaded here; put your original `Blackout_PowerGraph_19000.zip` in `data/raw/` to reproduce the full audit.

## Current work

| Stage | Status |
|---|---|
| Initial Decision Tree baseline | Done on 1,000 rows; see `notebooks/step4_baseline.ipynb` |
| Data integrity and provenance | Done on both supplied extracts |
| EDA and grouped 70/15/15 splits | Done locally; audit summary and notebook outputs included |
| Advanced models and final evaluation | Not started in this module |

**Audit findings:** no missing values; **1,421 repeated feature records**, including **152 distinct input patterns shared between extracts**; no label conflicts for identical input vectors. Grouped splitting produced **13,993 train / 3,010 validation / 2,997 test** records, without identical full feature vectors crossing sets.

The existing baseline's validation scores were obtained using a *different* 1,000-row split. They must not be compared directly with future results from the 20,000-row grouped split.

## Run

Python 3.12 recommended:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.audit
```

For the full audit, keep `data/Blackout_PowerGraph_1000.csv` and add the supplied ZIP at `data/raw/Blackout_PowerGraph_19000.zip`. The audit regenerates `data/splits/*_manifest.csv` and `reports/audit_metrics.json`. Open `notebooks/01_data_audit.ipynb` to see recorded audit outputs; it is a compact results notebook, not the full plotting notebook from the local handoff.

## Project files

- `notebooks/step4_baseline.ipynb` — original baseline, preserved with outputs.
- `notebooks/01_data_audit.ipynb` — audit results and saved outputs.
- `src/audit.py` — input checks and deterministic group-aware split generation.
- `reports/data_audit.md` — findings, limitations and next steps.
- `reports/audit_metrics.json` — recorded results of the full-data audit.
- `tests/` — unit checks for split grouping and reproducibility.

**Important limitation:** Feature timing relative to the simulated failure is not yet established, and grouping identical inputs does not remove all possible dependence between related simulations. Results are about simulated IEEE-24 cases, not verified real-grid outage prediction.

## Team

Mahij Khantigiri GOSAI · Anais KAMTCHOUANG NONO · Soukaina BAHAMI · Yutong CAO

Data audit and split design: **Mahij Khantigiri GOSAI**. Other modules are being developed by the group.
