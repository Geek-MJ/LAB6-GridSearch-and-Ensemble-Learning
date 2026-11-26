# GridGuard — team-requested splitting

**Owner:** Mahij Khantigiri GOSAI · **Input:** 20,000-row validated dataset

The team requested an **80/20 stratified train/test split**. Reproduce with `python -m src.split_data`. It uses:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

A `GroupShuffleSplit(test_size=0.20, random_state=42)` using the 72 `bus_` features then produces the training and validation subsets from the initial 16,000-row training pool.

| Partition | Rows | Normal (0) | Blackout (1) |
| --- | ---: | ---: | ---: |
| Train | 12,728 | 10,130 | 2,598 |
| Validation | 3,272 | 2,644 | 628 |
| Test | 4,000 | 3,194 | 806 |
| **Total** | **20,000** | **15,968** | **4,032** |

The saved split IDs and full summary are in `data/splits/team_stratified_80_20/`. The `all_records_manifest.csv` includes the `row_index` needed to load each partition consistently.

**Limitations:** The initial stratified outer split is not group-disjoint. **447 test rows** have identical complete feature vectors within the 16,000-row training pool; **457 bus-configuration groups** occur in both. The training and validation partitions share no bus groups. These facts must be disclosed in the final report.

**Historical protocol:** The older audit split is a separate **13,993 train / 3,010 validation / 2,997 test** grouped experiment. It remains in the repository because existing team notebooks refer to it. Do not combine historical metrics with current split results.

**Results warning:** A saved older Anais experiment reports 12,824/3,176/4,000. Those are *not* this manifest's partitions. The group must refit/re-evaluate models on the agreed protocol before its final comparison.
