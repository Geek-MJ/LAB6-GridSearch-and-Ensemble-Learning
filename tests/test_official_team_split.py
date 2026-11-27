"""Official two-file GridGuard handoff must be reproducible and leakage-safe."""
import numpy as np
import pandas as pd
import pytest
from src.team_split import build_team_split, verify_saved_split


def test_official_csvs_match_reproduced_grouped_split():
    result = verify_saved_split()
    assert result["train_rows"] == 16018
    assert result["test_rows"] == 3982
    assert result["train_blackout_1"] == 3240
    assert result["test_blackout_1"] == 792
    assert result["exact_input_overlap"] == 0


def test_same_input_group_stays_together():
    # Build a data-shaped fixture with repeated complete inputs.
    rng = np.random.default_rng(42)
    distinct = rng.normal(size=(18579, 224))
    X = np.vstack((distinct, distinct[:1421]))
    y = np.tile([0, 1], 10000)
    # Duplicate labels must match their original group.
    y[18579:] = y[:1421]
    columns = [f"feature_{i}" for i in range(224)]
    frame = pd.DataFrame(X, columns=columns)
    frame["blackout"] = y
    train, test = build_team_split(frame)
    assert len(train) + len(test) == 20000
    left = set(pd.util.hash_pandas_object(train.drop(columns="blackout"), index=False))
    right = set(pd.util.hash_pandas_object(test.drop(columns="blackout"), index=False))
    assert left.isdisjoint(right)


def test_conflicting_labels_are_rejected():
    # The fixture violates the unique-input/unique-label assumption.
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(20000, 224)))
    frame["blackout"] = np.arange(20000) % 2
    frame.iloc[1, :-1] = frame.iloc[0, :-1].to_numpy()
    with pytest.raises(ValueError, match="conflicting target labels"):
        build_team_split(frame)
