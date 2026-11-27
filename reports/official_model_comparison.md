# Official team-split comparison (retrospective)

**Dataset:** the group's fixed PowerGraph IEEE-24 **20,000-row** dataset, with
`data/processed/train.csv` (16,018 rows) and `test.csv` (3,982 rows).
Both files preserve 224 numerical inputs and have **zero identical complete
input vectors across them**. The official CSVs have been regenerated locally
and their Git blob SHA-1 hashes matched the current files byte-for-byte.

## Run

```bash
python -m pip install -r requirements.txt
python -m src.team_split
python -m pytest tests/test_official_team_split.py
python -m src.compare_official
```

The comparison applies the **same five stratified group folds** to all five
fixed candidates: DummyClassifier, scaled Logistic Regression, Decision Tree,
Random Forest and Histogram Gradient Boosting. Group IDs use all 224 inputs,
never the target. Candidate selection is based on training-only pooled
out-of-fold F1. The published holdout is used for a **retrospective check**,
because earlier team experiments already exposed the test outcomes.

### Observed results (training five-fold out-of-fold)

| Model | F1 | Recall | ROC-AUC |
| --- | ---: | ---: | ---: |
| Dummy majority | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.9361 | 0.9207 | 0.9907 |
| Decision Tree | 0.9710 | **0.9716** | 0.9820 |
| Random Forest | 0.9503 | 0.9228 | 0.9950 |
| Histogram Gradient Boosting | **0.9758** | 0.9651 | **0.9978** |

The highest-CV-F1 candidate, Hist. Gradient Boosting, achieved **0.99247
accuracy** and **0.98084 F1** on the previously examined test file, compared
with the Decision Tree's **0.99221 accuracy** and **0.98032 F1**.
That is **30 versus 31 errors**, not evidence of a meaningful superiority.
The exact paired McNemar test gives **p = 1.00**.

### Sensitivity to related inputs

Grouping entire **72-variable bus configurations**, rather than just identical
224-feature vectors, produced Decision Tree F1 **0.97246** and Gradient Boosting
F1 **0.97566**. This does not establish independence between all simulated
scenario families.

**Limitations:** uncertain feature timing, possible correlation among related
simulation scenarios, simulated rather than operational grid data, and
previous exposure to the held-out test set.

**Evidence:** `reports/official_model_comparison.json`,
`reports/baseline_evaluation_20000.md`, `reports/team_split.md`.
