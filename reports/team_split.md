# GridGuard — two-file train/test handoff

**Owner:** Mahij Khantigiri GOSAI

In accordance with the team's latest instructions, this project uses the **full 20,000 simulated records** and delivers **two CSV files only**:

| CSV | Rows | `blackout=0` | `blackout=1` |
| --- | ---: | ---: | ---: |
| `data/processed/train.csv` | 16,018 | 12,778 | 3,240 |
| `data/processed/test.csv` | 3,982 | 3,190 | 792 |
| **Total** | **20,000** | **15,968** | **4,032** |

## Method

The data have 224 numerical features and target `blackout`. Identical **complete input vectors** receive the same group ID (all rows retained). Use `train_test_split(test_size=0.20, random_state=42, stratify=y)` on the **distinct groups** with their common target labels; then assign every original record by its group.

This is approximately an 80/20 split by records (precisely 80/20 by groups), not a vanilla row-wise stratified split. The row counts differ slightly from 16,000/4,000 because duplicates must stay together. The test/production holdout has **zero shared complete feature vectors** with training.

No validation CSV is created: use cross-validation on `train.csv` for model comparison, feature selection, hyperparameter tuning and threshold selection. Keep `test.csv` untouched until the final evaluation.

## Reproduction and evidence

- The exact readable implementation is in [the executed notebook](../notebooks/01_data_audit_executed.ipynb).
- The reusable split code is in `src/split_data.py`; run `python -m src.split_data` from the repository root.
- `tests/test_team_split.py` verifies row counts, label balance, disjoint complete input vectors and deterministic output.
- The two output CSVs retain all 224 original features, `blackout`, and all source observations.

## Limitations

This excludes *exactly matching complete inputs* across partitions. Other correlated operating conditions or related simulations may still cross partitions. The available documentation does not prove that every feature is measured before the disruption. Do not describe the resulting model as validated on independent real-world grid events.

Historical three-way manifests and older saved model metrics were produced with **other** splits. Keep those as historical results; the group must refit/re-evaluate models on these two CSVs before claiming final performance.
