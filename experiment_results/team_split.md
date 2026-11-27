# GridGuard — Training and Test Split

**Owner:** Mahij Khantigiri GOSAI

The split uses all **20,000 simulated records** and produces two files:

| CSV | Rows | `blackout=0` | `blackout=1` |
| --- | ---: | ---: | ---: |
| `data/processed/train.csv` | 16,018 | 12,778 | 3,240 |
| `data/processed/test.csv` | 3,982 | 3,190 | 792 |
| **Total** | **20,000** | **15,968** | **4,032** |

## Method

The dataset contains 224 numerical input features and the target `blackout`. We grouped records with identical values across all input features, applied `train_test_split(test_size=0.20, random_state=42, stratify=y)` to the distinct groups, then assigned every original record to its group's partition.

The split is approximately 80/20 by row count because some groups contain multiple records. The saved training and test sets have **no identical complete input vectors in common**.

No separate validation file was created. Model comparisons, feature selection, hyperparameter tuning and threshold selection use cross-validation within `train.csv`. The test file is reserved for final evaluation.

## Files and checks

- The splitting code and saved results are included in [the notebook](../notebooks/01_data_audit_executed.ipynb).
- The official two-file split is reproduced in `src/team_split.py`; run `python -m src.team_split` from the repository root. This verifies the saved CSVs row-by-row without overwriting them. `src/split_data.py` is retained for a separate historical 70/15/15 experiment.
- `tests/test_official_team_split.py` checks the saved row counts, label balance, group separation, conflicting-label rejection and reproducibility. The older `tests/test_team_split.py` belongs to the historical three-way workflow.
- The two output CSVs retain all 224 original features, `blackout`, and all source observations.

## Limitations

The grouping check covers identical complete inputs, but different records may still represent related simulation conditions. The source documentation also does not establish when every feature was measured relative to the disruption.

Earlier three-way splits were used for previous model experiments. Their saved results refer to those partitions; direct comparisons on the current two-file split require new model evaluations.
