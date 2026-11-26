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
- The reusable split code is in `src/split_data.py`; run `python -m src.split_data` from the repository root.
- `tests/test_team_split.py` verifies row counts, label balance, disjoint complete input vectors and deterministic output.
- The two output CSVs retain all 224 original features, `blackout`, and all source observations.

## Limitations

The grouping check covers identical complete inputs, but different records may still represent related simulation conditions. The source documentation also does not establish when every feature was measured relative to the disruption.

Earlier three-way splits were used for previous model experiments. Their saved results refer to those partitions; direct comparisons on the current two-file split require new model evaluations.
