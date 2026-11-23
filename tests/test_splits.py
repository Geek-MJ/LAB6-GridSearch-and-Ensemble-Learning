"""Checks for input quality, split isolation, and label consistency."""
from src.audit import read_inputs, make_groups, choose_splits


def test_input_quality():
    data, features, _, _ = read_inputs()
    assert len(data) == 20000
    assert len(features) == 224
    assert data[features].notna().all().all()
    assert data.blackout.isin([0, 1]).all()


def test_group_isolation_and_coverage():
    data, features, _, _ = read_inputs()
    groups, _ = make_groups(data, features)
    splits, _ = choose_splits(data, groups)
    assert sum(len(idx) for idx in splits.values()) == len(data)
    group_sets = [set(groups[idx]) for idx in splits.values()]
    assert all(group_sets[i].isdisjoint(group_sets[j])
               for i in range(3) for j in range(i + 1, 3))


def test_identical_inputs_have_matching_labels():
    data, features, _, _ = read_inputs()
    groups, _ = make_groups(data, features)
    assert data.groupby(groups).blackout.nunique().max() == 1
