# GridGuard — Data audit

**Owner:** Mahij Khantigiri GOSAI  
**Scope:** Data provenance, exploratory analysis and leakage-aware split design.

## Inputs

The supplied IEEE-24 / PowerGraph extracts contain 1,000 and 19,000 simulated scenarios. The larger ZIP includes a README and source-index mapping; the smaller CSV has no source-index mapping. Reference: [PowerGraph dataset](https://figshare.com/articles/dataset/PowerGraph/22820534).

The target `blackout=1` means *some demand remains unsupplied*, not necessarily a nationwide blackout.

## Verified results (20,000 records)

| Check | Result |
|---|---:|
| Input variables | 224 |
| Class 0 / 1 | 15,968 / 4,032 |
| Positive rate | 20.16% |
| Missing values | 0 |
| Non-finite values | 0 |
| Unique full feature vectors | 18,579 |
| Repeated records beyond first | 1,421 |
| Shared distinct patterns between extracts | 152 |
| Conflicting labels for identical input vectors | 0 |

The 224 inputs consist of three groups of 24 bus measurements and four groups of 38 line measurements. These descriptions reflect the flattened CSV and do not reconstruct the grid topology.

## Split design

We assign group IDs from hashes of complete input vectors, then use fixed-seed GroupShuffleSplit candidates. This prevents identical full feature vectors from crossing partitions. We keep duplicate records instead of deleting them.

| Set | Rows | Class-1 rows |
|---|---:|---:|
| Training | 13,993 | 2,820 |
| Validation | 3,010 | 607 |
| Test | 2,997 | 605 |

All results come from the completed local audit. Run `python -m src.audit` with both supplied extracts to regenerate the split manifests. The large source ZIP and generated manifests are not stored in GitHub; see README.

## Key limitations

1. Source documentation supplied with these CSVs does not establish which inputs were available *before* the cascade. Feature timing is still an open leakage risk.
2. Grouping exact feature matches does not isolate every simulation family or operating-point dependency.
3. The original dataset is simulated, not a historical record of real-world grid outages.
4. Label prevalence in this selected simulation sample is not an estimate of the operational event rate.
5. Descriptive class differences across all records are **not** valid training-time feature selection. Any feature selection must be fitted inside training cross-validation (LAB 4).

## Handoff

Use the generated split manifests for all subsequent experiments, and do not use the test partition for tuning. The existing 1,000-row Decision Tree baseline in this repository uses a different split and cannot be compared directly with future scores from these partitions. This module does not claim a trained or deployed predictive model.
