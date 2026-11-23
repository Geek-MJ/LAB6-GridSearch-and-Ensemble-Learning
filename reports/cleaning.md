# Dataset cleaning — GridGuard

**Module:** Data preparation · **Owner:** Mahij Khantigiri GOSAI

## Decision

Both source extracts are already numeric and complete. We therefore do not impute missing values, clip outliers, rescale features, or alter the target. Such operations would be unjustified at this stage. Scaling and feature selection should be fitted only on the training partition, inside a model pipeline (LAB 4).

| Validation | Finding |
|---|---:|
| Combined input records | 20,000 |
| Features | 224 numerical |
| Missing / infinite values | 0 / 0 |
| Invalid target values | 0 |
| Constant input columns | 0 |
| Extra copies of identical feature vectors | 1,421 |
| Conflicting labels for identical feature vectors | 0 |
| Unique feature vectors | 18,579 |

Six input columns are exact copies of other feature columns. They remain in the cleaned datasets so that the original 224-variable schema is stable. Later feature-selection experiments can test whether removing redundant columns helps.

## Outputs

- `data/processed/validated_20000.csv.gz`: 20,000 scenarios; all feature values and records preserved.
- `data/processed/unique_18579.csv.gz`: one representative per identical complete feature vector. This optional version has 14,859 class-0 and 3,720 class-1 records.
- `data/processed/unique_records_manifest.csv`: each retained record's original ID, original train/validation/test split and duplicate-group size.
- `data/processed/cleaning_summary.json`: counts, duplicate-column pairs, file hashes and validation notes.

For repository size, GitHub tracks the unique dataset and its manifest. The 20,000-record validated CSV can be regenerated from the two committed original extracts using `python -m src.clean_data`.

| Split | Original records | Unique records |
|---|---:|---:|
| Training | 13,993 | 13,005 |
| Validation | 3,010 | 2,787 |
| Test | 2,997 | 2,787 |

The existing split assignments are retained; duplicates are only removed *within* their assigned split. **Do not** regenerate new random splits on the unique version, or tune feature selection using validation/test labels.

## Interpretation and limitations

Repeated input vectors can be genuine simulated scenarios. Removing them changes the empirical weighting of simulations and slightly changes class prevalence. The original full-data version should remain available for sensitivity comparisons. The available source metadata cannot prove that all features are measured before the simulated cascade. That leakage risk must be resolved before making operational claims.

Reproduce: `python -m src.clean_data`. Verify: `python -m pytest -q tests/test_clean_data.py`.
