# Evaluation

This script compares five models on the same grouped dataset split.

Run from the project folder:

```bash
python -m pip install -r requirements.txt
python -m src.evaluate --jobs 2
```

The model and threshold are chosen on validation data. The test set is used at the end.

Results: `reports/corrected/`. Plots: `figures/corrected/`.

The data are simulated, so these scores do not prove real-grid performance.
