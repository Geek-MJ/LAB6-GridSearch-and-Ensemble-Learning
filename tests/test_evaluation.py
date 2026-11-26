"""Small protocol checks; full-dataset training is deliberately not run in tests."""
import numpy as np
import pytest

from src.evaluate import (
    bootstrap_intervals, classification_metrics, select_candidate,
    select_threshold, validate_partitions,
)


def test_selection_follows_validation_not_a_forced_forest():
    rows = [
        {"model": "Random_forest", "validation_average_precision": 0.90},
        {"model": "Logistic_regression", "validation_average_precision": 0.95},
    ]
    assert select_candidate(rows) == "Logistic_regression"
    rows[0]["validation_average_precision"] = 0.95
    assert select_candidate(rows) == "Logistic_regression"
    assert select_candidate(list(reversed(rows))) == "Logistic_regression"


def test_threshold_optimizes_declared_validation_f2():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
    threshold, _ = select_threshold(y, p)
    # Low-probability true positive matters more than the extra false alert.
    assert threshold == 0.2
    scores = [classification_metrics(y, p, value)["f2"] for value in np.unique(p)]
    assert classification_metrics(y, p, threshold)["f2"] == max(scores)


def test_partitions_reject_group_overlap_even_when_rows_are_disjoint():
    parts = {"train": np.array([0, 1]), "validation": np.array([2, 3]), "test": np.array([4, 5])}
    y = np.array([0, 1, 0, 1, 0, 1])
    validate_partitions(parts, np.arange(6), y)
    with pytest.raises(ValueError, match="group crosses"):
        validate_partitions(parts, np.array([0, 1, 2, 3, 0, 5]), y)
    with pytest.raises(ValueError, match="exactly once"):
        validate_partitions({**parts, "test": np.array([4, 4])}, np.arange(6), y)


def test_cluster_bootstrap_keeps_repeated_rows_together():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.8, 0.8, 0.3, 0.3, 0.9, 0.9])
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    result = bootstrap_intervals(y, p, groups, 0.5, repeats=50, seed=7)
    assert result == bootstrap_intervals(y, p, groups, 0.5, repeats=50, seed=7)
    assert result["test_bus_groups"] == 4
    assert result["valid_repetitions"] + result["skipped_single_class_repetitions"] == 50
    assert result["intervals"]["recall"] == {"lower": 1.0, "upper": 1.0}


def test_cluster_bootstrap_preserves_sampled_group_multiplicity(monkeypatch):
    class FixedDraws:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 4, 4)
            return np.array([0, 1, 0, 2])

    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedDraws())
    y = np.array([1, 1, 0, 0, 0, 1, 0])
    probabilities = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1])
    groups = np.array([1, 1, 2, 2, 2, 3, 4])
    result = bootstrap_intervals(y, probabilities, groups, 0.5, repeats=1)
    # Group 1 is drawn twice: four false negatives, not two; group 3 adds one TP.
    assert result["intervals"]["recall"] == {"lower": 0.2, "upper": 0.2}
