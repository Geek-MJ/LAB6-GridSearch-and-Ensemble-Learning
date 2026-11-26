"""Checks for the two-file, group-stratified 80/20 split."""
import pandas as pd
from src.split_data import make_splits

def test_counts_class_balance_and_no_shared_inputs():
    train, test, report = make_splits(save=False)
    assert len(train) == 16018
    assert len(test) == 3982
    assert len(train) + len(test) == 20000
    assert len(train.columns) == len(test.columns) == 225
    assert int(train.blackout.sum()) == 3240
    assert int(test.blackout.sum()) == 792
    assert abs(train.blackout.mean() - test.blackout.mean()) < 0.01
    h1 = set(pd.util.hash_pandas_object(train.drop(columns="blackout"), index=False))
    h2 = set(pd.util.hash_pandas_object(test.drop(columns="blackout"), index=False))
    assert h1.isdisjoint(h2)
    assert report["identical_feature_groups_in_both"] == 0

def test_is_deterministic():
    a_train, a_test, _ = make_splits(save=False)
    b_train, b_test, _ = make_splits(save=False)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)
