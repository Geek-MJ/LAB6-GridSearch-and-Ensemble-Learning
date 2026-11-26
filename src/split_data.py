"""Make the stratified train/test split requested by the GridGuard group.

This reproduces the protocol in src/evaluate.py:
- 80/20 train-pool/test with train_test_split(stratify=y, random_state=42).
- Group-aware 80/20 train/validation inside the training pool.

Run from the repository root: python -m src.split_data
The original datasets and existing group-aware manifests are not modified.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "validated_20000.csv.gz"
ORIGINAL_MANIFEST = ROOT / "data" / "splits" / "all_records_manifest.csv"
OUTPUT = ROOT / "data" / "splits" / "team_stratified_80_20"
TARGET = "blackout"
SEED = 42


def make_splits(save=True):
    """Return original row indices and a summary; no fitting or feature selection."""
    df = pd.read_csv(DATA)
    if df.shape != (20000, 225) or TARGET not in df:
        raise ValueError("Expected 20,000 rows and 224 input columns plus blackout.")
    if df.isna().any().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Missing or invalid data; inspect the original file.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    # This first split is exactly what Anais requested.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    pool_idx = X_train.index.to_numpy()
    test_idx = X_test.index.to_numpy()

    # Match the validation split already used in src/evaluate.py.
    bus_columns = [name for name in X_train if name.startswith("bus_")]
    if len(bus_columns) != 72:
        raise ValueError("Unexpected bus-feature schema.")
    pool_groups = pd.util.hash_pandas_object(
        X_train[bus_columns], index=False
    ).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=SEED
    )
    train_relative, validation_relative = next(
        splitter.split(X_train, y_train, groups=pool_groups)
    )
    train_idx = pool_idx[train_relative]
    validation_idx = pool_idx[validation_relative]

    partitions = {
        "train": train_idx,
        "validation": validation_idx,
        "test": test_idx,
    }
    assert len(set(np.concatenate(list(partitions.values())))) == len(df)
    assert set(pool_groups[train_relative]).isdisjoint(
        pool_groups[validation_relative]
    )

    # Keep the original record IDs so the evaluation outputs are traceable.
    original = pd.read_csv(
        ORIGINAL_MANIFEST, dtype={"input_hash_uint64": "string"}
    )
    if len(original) != len(df) or original["record_id"].duplicated().any():
        raise ValueError("Original record manifest is missing rows or has duplicate IDs.")
    # The validated CSV is a floating-point text roundtrip. Exact bitwise
    # hashes may differ from the raw CSV even if the numerical inputs match.
    from src.audit import read_inputs
    raw_data, raw_features, _, _ = read_inputs()
    if raw_features != X.columns.tolist() or len(raw_data) != len(X):
        raise ValueError("Processed input schema does not match the raw data.")
    if not np.allclose(
        raw_data[raw_features].to_numpy(),
        X.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Processed numeric features differ from the original inputs.")
    raw_hash = pd.util.hash_pandas_object(
        raw_data[raw_features], index=False
    ).astype(str)
    if not (original["input_hash_uint64"].to_numpy() == raw_hash.to_numpy()).all():
        raise ValueError("The original audit manifest does not match source features.")
    if not original[TARGET].reset_index(drop=True).equals(y.reset_index(drop=True)):
        raise ValueError("Labels do not match the original audit.")
    full_hash = pd.util.hash_pandas_object(X, index=False).astype(str)

    assigned = np.empty(len(df), dtype=object)
    for name, idx in partitions.items():
        assigned[idx] = name
    manifest = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "record_id": original["record_id"],
        "source_extract": original["source_extract"],
        "blackout": y,
        "split": assigned,
        "input_hash_uint64": full_hash,
    })

    pool_hashes = set(full_hash.iloc[pool_idx])
    test_duplicate_rows = int(full_hash.iloc[test_idx].isin(pool_hashes).sum())
    groups_test = pd.util.hash_pandas_object(
        X_test[bus_columns], index=False
    ).to_numpy()
    overlap_bus = int(len(set(pool_groups).intersection(set(groups_test))))

    summary = {
        "dataset_rows": len(df),
        "input_features": X.shape[1],
        "method": "train_test_split then GroupShuffleSplit",
        "outer": {
            "test_size": 0.20,
            "random_state": SEED,
            "stratify": TARGET,
            "training_pool_rows": len(pool_idx),
            "test_rows": len(test_idx),
        },
        "validation": {
            "method": "GroupShuffleSplit on 72 bus features",
            "test_size": 0.20,
            "random_state": SEED,
        },
        "partitions": {
            name: {
                "records": int(len(idx)),
                "blackout_1": int(y.iloc[idx].sum()),
                "blackout_0": int(len(idx) - y.iloc[idx].sum()),
                "positive_rate": float(y.iloc[idx].mean()),
            }
            for name, idx in partitions.items()
        },
        "leakage_check": {
            "test_rows_with_identical_complete_inputs_in_training_pool": test_duplicate_rows,
            "bus_configuration_groups_shared_by_training_pool_and_test": overlap_bus,
            "train_validation_bus_group_overlap": 0,
            "caution": (
                "The stratified outer split is not group-aware. Reported test "
                "metrics may be optimistic for genuinely unseen configurations."
            ),
        },
    }

    if save:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(OUTPUT / "all_records_manifest.csv", index=False)
        for name in partitions:
            manifest.loc[manifest["split"] == name].to_csv(
                OUTPUT / f"{name}_manifest.csv", index=False
            )
        (OUTPUT / "split_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return partitions, summary


if __name__ == "__main__":
    _, summary = make_splits()
    print(json.dumps(summary, indent=2))
