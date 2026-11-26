"""The canonical protocol must exclude shared configurations from every split."""
import numpy as np
import pytest
from src.split_data import make_splits, choose_partitions, check_isolation


def test_real_data_split_is_disjoint_and_complete():
    parts, summary = make_splits(save=False)
    assert sorted(np.concatenate(list(parts.values())).tolist()) == list(range(20000))
    assert all(v == 0 for v in summary["leakage_check"].values())
    assert all(summary["partitions"][name]["blackout_1"] > 0 for name in parts)


def test_grouped_split_is_deterministic_without_fitting_any_model():
    groups = np.repeat(np.arange(300), 2)
    y = np.repeat(np.arange(300) % 2, 2)
    a, _ = choose_partitions(y, groups)
    b, _ = choose_partitions(y, groups)
    for name in a:
        assert np.array_equal(a[name], b[name])
    check_isolation(a, groups, len(y))


def test_overlap_is_rejected():
    with pytest.raises(ValueError, match="Group overlap"):
        check_isolation({"train": np.array([0, 1]), "test": np.array([2, 3])}, np.array([7, 8, 7, 9]), 4)
