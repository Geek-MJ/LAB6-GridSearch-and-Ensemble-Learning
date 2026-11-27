# Evaluation workflows

**Official team train/test data:** Run `python -m src.team_split` to verify the two existing 80/20 CSV exports without changing them. The canonical files are `data/processed/train.csv` and `data/processed/test.csv`; model selection on this protocol uses grouped cross-validation within training data.

**Separate historical evaluator:** `python -m src.evaluate --jobs 2` uses `src/split_data.py`, a grouped-bus three-way protocol (~70/15/15), and writes results under `reports/corrected/` and figures under `figures/corrected/`. Its results should not be directly compared with the official two-file test results.

Install pinned packages with `python -m pip install -r requirements.txt` from the repository root. These data describe simulations and do not establish real-world blackout prediction performance.
