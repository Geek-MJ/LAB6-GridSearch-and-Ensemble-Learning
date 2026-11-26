"""Create the team's 80/20 train/test CSVs without duplicate-input overlap.

Duplicates are grouped first, then the distinct groups are split with
stratify=their blackout labels. All 20,000 rows are retained; there is no
separate validation dataset. Model selection should use CV within train.csv.

Run from repository root: python -m src.split_data
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "validated_20000.csv.gz"
OUTPUT_DIR = ROOT / "data" / "processed"
TARGET = "blackout"
SEED = 42


def make_splits(save: bool = True):
    """Split unique feature-vector groups; keep every source record."""
    df = pd.read_csv(DATA_PATH)
    if df.shape != (20000, 225) or TARGET not in df:
        raise ValueError("Expected 20,000 observations, 224 features and blackout.")
    if df.isna().any().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Missing or invalid data values.")

    X = df.drop(columns=TARGET)
    y = df[TARGET].astype(int)

    # Every exact identical input vector receives the same group ID.
    fingerprints = pd.util.hash_pandas_object(X, index=False)
    group_ids, _ = pd.factorize(fingerprints, sort=False)
    group_table = pd.DataFrame({"group": group_ids, TARGET: y})
    if group_table.groupby("group")[TARGET].nunique().max() != 1:
        raise ValueError("Identical feature vectors have conflicting labels.")
    unique_groups = group_table.drop_duplicates(subset="group")

    # Stratify UNIQUE groups, not individual records: duplicates cannot cross.
    train_groups, test_groups = train_test_split(
        unique_groups["group"].to_numpy(),
        test_size=0.20,
        random_state=SEED,
        stratify=unique_groups[TARGET],
    )
    train_mask = np.isin(group_ids, train_groups)
    test_mask = np.isin(group_ids, test_groups)
    assert np.all(train_mask ^ test_mask)
    assert set(fingerprints[train_mask]).isdisjoint(set(fingerprints[test_mask]))

    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()

    summary = {
        "dataset_rows": int(len(df)),
        "input_features": int(X.shape[1]),
        "method": "train_test_split on unique complete feature-vector groups",
        "test_size_of_groups": 0.20,
        "random_state": SEED,
        "stratify": TARGET,
        "unique_input_groups": int(len(unique_groups)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive": int(train[TARGET].sum()),
        "test_positive": int(test[TARGET].sum()),
        "train_positive_rate": float(train[TARGET].mean()),
        "test_positive_rate": float(test[TARGET].mean()),
        "identical_feature_groups_in_both": 0,
        "caution": (
            "Exact duplicates are separated; related simulations may still "
            "be dependent and some features may describe post-event outcomes."
        ),
    }
    if save:
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        train.to_csv(OUTPUT_DIR / "train.csv", index=False, float_format="%.17g")
        test.to_csv(OUTPUT_DIR / "test.csv", index=False, float_format="%.17g")
        (OUTPUT_DIR / "train_test_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        # Verify the exported, reloaded CSVs—not just the in-memory split.
        train_check = pd.read_csv(OUTPUT_DIR / "train.csv")
        test_check = pd.read_csv(OUTPUT_DIR / "test.csv")
        if train_check.shape != train.shape or test_check.shape != test.shape:
            raise ValueError("Exported CSV sizes differ from the original split.")
        h_train = set(pd.util.hash_pandas_object(
            train_check.drop(columns=TARGET), index=False
        ))
        h_test = set(pd.util.hash_pandas_object(
            test_check.drop(columns=TARGET), index=False
        ))
        if h_train.intersection(h_test):
            raise ValueError("Duplicate input vectors cross the exported CSVs!")
    return train, test, summary


if __name__ == "__main__":
    _, _, report = make_splits()
    print(json.dumps(report, indent=2))
