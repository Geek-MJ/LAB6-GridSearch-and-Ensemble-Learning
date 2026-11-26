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

## Data: which version to use?

- **Modeling data:** `data/processed/validated_20000.csv.gz` (20,000 rows, 224 inputs + `blackout`), without dropping duplicate scenarios.
- **Original extracts:** `data/Blackout_PowerGraph_1000.csv` and `data/raw/Blackout_PowerGraph_19000.zip`.
- **Current team-requested 80/20 split:** `data/splits/team_stratified_80_20/`. Use the saved `all_records_manifest.csv` to select records by `row_index`.
- **Earlier 70/15/15 group-aware split:** `data/splits/` at the top level, retained because historical baseline and evaluation notebooks still use it.

The team's 80/20 split uses `train_test_split(test_size=0.20, random_state=42, stratify=y)`, then a bus-group-aware training/validation split. Its saved partitions are **12,728 train / 3,272 validation / 4,000 test**. Unlike the older group-aware protocol, **447 test records share identical complete inputs with records in the initial training pool**. Read [the split report](reports/team_split.md) before making generalization claims.

**Important:** Saved metrics in `reports/evaluation/` and `reports/evaluation_anais/` were obtained under different protocols and must not be presented as one controlled comparison. The earlier Anais evaluation also used a different inner split (12,824/3,176/4,000), so rerun affected models before claiming the current manifest's test results.

## Repository guide

```text
data/         Original files, one validated modeling dataset and historical/current split manifests
notebooks/    Data audit, baseline and advanced notebooks (saved team work preserved)
src/          Reproducible audit, cleaning, splitting and evaluation scripts
tests/        Data-quality and split checks
figures/      EDA and modeling figures
reports/      Data-audit evidence and existing model evaluation reports
```

## Reproduce Mahij's module

Python 3.12 recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.split_data
```

This checks the existing data and regenerates current split manifests. For full audit regeneration, run `python -m src.audit`, followed by `python -m src.clean_data`. Those commands also regenerate the older audit protocol; they do not train models.

## Submission status

**Not yet submission-ready.** LAB 5 requires a final PDF report, an agreed experimental methodology, and the `ai_governance/CONTRIBUTIONS.md`, `PROMPT_LOG.md` and `AI_AUDIT.md` files in the Moodle ZIP. Each student must verify and document their genuine contributions and AI interactions. Do not invent these records.

The repository preserves the team's experiments without treating historical and current splits as equivalent.
