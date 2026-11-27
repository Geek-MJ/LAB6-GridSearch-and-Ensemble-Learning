"""Official GridGuard 80/20 split supplied to the model-development team.

Identical complete 224-feature vectors belong to the same group. The binary
label is used for stratification, never to define input groups. Existing
train.csv and test.csv are checked, not silently overwritten.

Run: python -m src.team_split
"""
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "validated_20000.csv.gz"
TRAIN = ROOT / "data" / "processed" / "train.csv"
TEST = ROOT / "data" / "processed" / "test.csv"
TARGET = "blackout"
SEED = 42


def build_team_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign every original row to one of two group-disjoint partitions."""
    if data.shape != (20_000, 225) or TARGET not in data or not data.columns.is_unique:
        raise ValueError("Expected 20,000 rows, 224 inputs, and a blackout target.")
    if data.isna().any().any() or not all(pd.api.types.is_numeric_dtype(t) for t in data.dtypes):
        raise ValueError("Dataset contains missing or non-numeric entries.")
    values = data.to_numpy()
    if not np.isfinite(values).all() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("Dataset contains non-finite values or invalid target labels.")

    X = data.drop(columns=[TARGET])
    hashes = pd.util.hash_pandas_object(X, index=False)
    groups, _ = pd.factorize(hashes, sort=False)
    group_table = pd.DataFrame({"group": groups, TARGET: data[TARGET].to_numpy()})
    if group_table.groupby("group")[TARGET].nunique().max() != 1:
        raise ValueError("The same input vector has conflicting target labels.")
    unique_groups = group_table.drop_duplicates(subset="group")
    train_groups, test_groups = train_test_split(
        unique_groups["group"], test_size=0.20, random_state=SEED,
        stratify=unique_groups[TARGET],
    )
    train_mask = np.isin(groups, train_groups.to_numpy())
    test_mask = np.isin(groups, test_groups.to_numpy())
    if not np.all(train_mask ^ test_mask):
        raise ValueError("Row coverage is incomplete or overlaps.")
    if not set(hashes[train_mask]).isdisjoint(set(hashes[test_mask])):
        raise ValueError("Identical feature vectors cross train/test.")

    train = data.loc[train_mask].reset_index(drop=True)
    test = data.loc[test_mask].reset_index(drop=True)
    return train, test


def verify_saved_split() -> dict:
    """Compare regenerated partitions with committed CSVs without altering them."""
    data = pd.read_csv(SOURCE)
    train, test = build_team_split(data)
    for name, regenerated, path in (("train", train, TRAIN), ("test", test, TEST)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing official {name} CSV: {path}")
        saved = pd.read_csv(path)
        # Compare actual rows and their order, not just sizes/class counts.
        try:
            assert_frame_equal(regenerated, saved, check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise ValueError(f"{name}.csv does not match the documented 80/20 procedure.") from error
    return {
        "seed": SEED,
        "method": "stratified 80/20 split of distinct complete 224-feature input groups",
        "train_rows": len(train),
        "test_rows": len(test),
        "train_blackout_1": int(train[TARGET].sum()),
        "test_blackout_1": int(test[TARGET].sum()),
        "exact_input_overlap": 0,
        "reproduces_saved_csvs": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(verify_saved_split(), indent=2))
