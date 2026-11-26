# GridGuard — data quality and cleaning

**Owner:** Mahij Khantigiri GOSAI

The supplied PowerGraph IEEE-24 extracts contain **1,000 + 19,000 = 20,000** simulated scenarios, with **224 numeric predictors** and the binary target `blackout`.

| Source-data check | Result |
| --- | ---: |
| Rows | 20,000 |
| Target counts (0 / 1) | 15,968 / 4,032 |
| Missing cells | 0 |
| Non-finite feature values | 0 |
| Invalid target values | 0 |
| Constant features | 0 |
| Distinct complete input vectors | 18,579 |
| Repeated input vectors beyond first occurrence | 1,421 |
| Conflicting labels for identical input vectors | 0 |

Six numerical input columns match another column exactly; this was documented, not silently removed.

**Cleaning decision:** No imputation, scaling, clipping, target changes, or duplicate removal. Repeated inputs can represent simulated outcomes and removing them changes their relative frequencies. Model preprocessing and supervised feature selection must be fitted on training data only.

**Primary output:** `data/processed/validated_20000.csv.gz`, the full dataset with all 224 source features retained. The `src/clean_data.py` command validates inputs and recreates that file and `data/processed/cleaning_summary.json`. The two original source files are preserved.

**Splitting:** The latest team-requested 80/20 stratified procedure is in `src/split_data.py` and described in [team_split.md](team_split.md). The earlier group-aware audit split still exists for historical model notebooks; the two procedures must not be mixed.

**Limits:** Data are simulated rather than observations of real-world blackouts. The source documents do not prove that every predictor would be known before the predicted event. The 80/20 test set has repeated-input overlap with the training pool.
