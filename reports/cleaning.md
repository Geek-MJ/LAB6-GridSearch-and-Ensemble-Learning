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

Six numerical input columns match another column exactly. We recorded this finding and kept the columns in the validated dataset.

**Cleaning:** We kept the original values, labels and repeated records. No missing-value imputation, scaling, clipping or duplicate removal was applied at this stage. Model preprocessing and supervised feature selection use training data.

**Validated dataset:** `data/processed/validated_20000.csv.gz`, the full dataset with all 224 source features retained. The `src/clean_data.py` command validates inputs and recreates that file and `data/processed/cleaning_summary.json`. The two original source files are preserved.

**Train/test split:** The current procedure is in `src/split_data.py` and [team_split.md](team_split.md). The earlier three-way manifests are still available for the original experiments.

**Limitations:** The dataset contains simulations rather than real-world blackout observations, and the source documents do not confirm when every predictor is measured. The current training and test files have **zero identical complete input vectors in common**.
