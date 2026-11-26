# Team dataset split

**Dataset:** PowerGraph IEEE-24, 20,000 simulated records with 224 numeric features and target `blackout`.

This implements the team's requested `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`. The 16,000-row training pool is then divided into train and validation subsets using `GroupShuffleSplit(test_size=0.20, random_state=42)` on 72 bus variables, matching the existing `src/evaluate.py` protocol.

| Partition | Records | `blackout=0` | `blackout=1` |
|---|---:|---:|---:|
| Train | 12,824 | 10,259 | 2,565 |
| Validation | 3,176 | 2,515 | 661 |
| Test | 4,000 | 3,194 | 806 |
| **Total** | **20,000** | **15,968** | **4,032** |

The initial 80/20 split itself preserves the target-class distribution. The group-aware validation split does not enforce exact stratification.

## Files

The `data/splits/team_stratified_80_20/` directory contains the overall row-level manifest and three subset manifests, plus `split_summary.json`. Every manifest includes the original row index and record ID. The input files are unchanged.

Run `python -m src.split_data` from the repository root to regenerate the same split. The saved manifests are authoritative for this protocol, and downstream models must use their row IDs.

## Limitation

This follows the requested **ordinary** stratified outer split. It is not a group-aware test split: **447 test rows** have identical full input vectors in the 16,000-row training pool, and **457 distinct bus-pattern groups** occur in both. The train and validation subsets are bus-group-disjoint. The overlap should be reported when interpreting test performance.

This protocol is different from the earlier 13,993/3,010/2,997 group-aware manifest. Keep that previous experiment separate; do not combine metrics across the two.
