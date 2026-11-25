# Source code

GridGuard predicts whether a simulated electrical-grid scenario results in a blackout. The source code is organized into three sequential stages.

## Execution order

| Stage | Module | Responsibility |
| --- | --- | --- |
| 1 | [`audit.py`](audit.py) | Inspect raw data, identify identical inputs, generate exploratory figures and create split manifests. |
| 2 | [`clean_data.py`](clean_data.py) | Validate the 20,000 records and 224 input features, check manifest alignment and export validated datasets. |
| 3 | [`evaluate.py`](evaluate.py) | Compare models, tune Random Forest, analyze thresholds and evaluate the final model. |

`__init__.py` marks this directory as a Python package.

## Run the pipeline

From the repository root, with project dependencies installed and the original datasets available:

```bash
python -m src.audit
python -m src.clean_data
python -m src.evaluate
```

These commands generate or regenerate project outputs. Run them in order; cleaning requires the audit manifest, and evaluation requires the validated dataset and manifest.

## Inputs and outputs

| Stage | Inputs | Outputs |
| --- | --- | --- |
| Audit | Original 1,000-row CSV and 19,000-row ZIP | `data/splits/`, audit metrics and feature inventory in `reports/`, exploratory plots in `figures/` |
| Cleaning | Original extracts and `data/splits/all_records_manifest.csv` | `data/processed/`: complete and unique-input datasets, unique-record manifest and cleaning summary |
| Evaluation | `data/processed/validated_20000.csv.gz` and the audit manifest | `reports/evaluation_anais/` and `figures/evaluation_anais/`: comparisons, metrics, predictions, error analysis, feature importance, model and figures |

## Evaluation protocol

The evaluation module creates its own stratified 80/20 outer split, then uses group-aware training/validation splitting and cross-validation. The final Random Forest uses all features and a threshold of 0.50. The outer test split is not group-aware; the script reports group overlap. Feature importance is reported, but this is not a feature-selection stage.

## Navigation note

The existing module names and locations are retained because the scripts resolve paths relative to the repository root and are invoked as `src` modules. This guide documents the existing implementation without changing it.
