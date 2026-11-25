# GridGuard — Power Grid Supply Disruption Prediction

Machine learning course project using PowerGraph / IEEE-24 simulated electrical-grid scenarios. The goal is to predict unmet electricity demand (`blackout=1`); `0` means all demand is supplied.

## Start here

| Stage | What to read |
| --- | --- |
| 1. Data audit | [Audit report](reports/data_audit.md) · [Executed notebook](notebooks/01_data_audit_executed.ipynb) |
| 2. Data cleaning | [Cleaning report](reports/cleaning.md) · [Dataset guide](data/README.md) |
| 3. Baseline | [Baseline notebook](notebooks/step4_baseline.ipynb) · [Baseline report](reports/baseline_evaluation.md) |
| 4. Advanced experiments | [Task 3 notebook](notebooks/task3.ipynb) |
| 5. Evaluation | [Report index](reports/README.md) · [Figures](figures/README.md) |

## Dataset

[PowerGraph](https://figshare.com/articles/dataset/PowerGraph/22820534), IEEE-24: 20,000 records, 224 numerical input features, 4,032 positive labels (20.16%), no missing or infinite values, 18,579 unique input patterns. The primary dataset retains all records; the optional unique-input dataset is separate.

## Saved evaluation results

Two full-dataset experiments are documented separately because their splits and selection protocols differ.

| Protocol | Manifest-based evaluation | Anaïs evaluation |
| --- | ---: | ---: |
| Train / validation / test | 13,993 / 3,010 / 2,997 | 12,824 / 3,176 / 4,000 |
| Selection metric | Average precision | Recall |
| Random Forest threshold | 0.277179 (validation F2) | 0.50 |
| Test accuracy | 96.56% | 93.45% |
| Test recall | 95.37% | 96.28% |
| Test ROC-AUC | 99.33% | 98.21% |
| Report | [Results](reports/evaluation/evaluation_report.md) | [Results](reports/evaluation_anais/evaluation_report.md) |

These are saved results, not new experiments or a head-to-head comparison. The earlier 1,000-row baseline is a separate experiment. Feature importance is not feature selection.

## Repository map

```text
data/          Original extracts, processed datasets, split manifests
src/           Audit, cleaning and current evaluation implementation
notebooks/     Executed audit, baseline and advanced experiments
reports/       Audit, cleaning, baseline and evaluation results
figures/       Exploratory plots and separate evaluation plots
tests/         Data validation and split integrity tests
.github/       CI workflow
docs/          Preserved previous project overview
```

## Reproduce

Python 3.12 recommended. From the repository root:

```bash
git clone --branch feature/step4-baseline https://github.com/Geek-MJ/Lab-5-Machine-Learning-Project-.git
cd Lab-5-Machine-Learning-Project-
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.audit
python -m src.clean_data
python -m src.evaluate
```

The last three commands regenerate outputs. The current evaluation script reproduces the Anaïs protocol, not the earlier manifest-based protocol. See [source guide](src/README.md) and [test guide](tests/README.md).

## Limitations

Simulated scenarios may not generalize to real networks. Exact-input grouping does not guarantee independent simulation families. The Anaïs outer test split is not group-aware; overlap is reported. Feature timing relative to the outcome requires verification. Fit preprocessing and feature selection on training data only.

## Team

Mahij Khantigiri GOSAI · Anais KAMTCHOUANG NONO · Soukaina BAHAMI · Yutong CAO

[Previous README (preserved verbatim)](docs/previous_readme.md)
