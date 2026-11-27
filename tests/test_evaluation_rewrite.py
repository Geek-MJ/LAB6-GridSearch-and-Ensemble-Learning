"""Regression tests for scientific boundaries and publication of a corrected run.

The integration tests use tiny synthetic frames and cheap estimators, not the
project dataset, serialized models or expensive scientific reruns.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.evaluate as evaluation


@pytest.mark.parametrize("labels, probabilities, threshold", [
    ([0, 1.5], [0.1, 0.9], 0.5),
    ([0, 2], [0.1, 0.9], 0.5),
    ([0, np.nan], [0.1, 0.9], 0.5),
    ([0, 1], [0.1, np.nan], 0.5),
    ([0, 1], [0.1, np.inf], 0.5),
    ([0, 1], [-0.1, 1.1], 0.5),
    ([0, 1], [0.1], 0.5),
    ([0, 1], [[0.1], [0.9]], 0.5),
    ([0, 1], [0.1, 0.9], np.nan),
    ([0, 1], [0.1, 0.9], 1.1),
])
def test_metrics_reject_invalid_scientific_inputs(labels, probabilities, threshold):
    """Never silently truncate fractional labels or score invalid probabilities."""
    with pytest.raises((ValueError, TypeError)):
        evaluation.classification_metrics(labels, probabilities, threshold)


@pytest.mark.parametrize("broken", ["fractional", "negative", "incomplete", "duplicate", "group_overlap"])
def test_partition_validator_rejects_leakage_and_invalid_row_indices(broken):
    parts = {"train": np.array([0, 1]), "validation": np.array([2, 3]), "test": np.array([4, 5])}
    groups = np.arange(6)
    if broken == "fractional":
        parts["train"] = np.array([0.25, 1.25])
    elif broken == "negative":
        parts["train"] = np.array([-1, 1])
    elif broken == "incomplete":
        parts["test"] = np.array([4])
    elif broken == "duplicate":
        parts["test"] = np.array([4, 4])
    else:
        groups[4] = groups[0]
    with pytest.raises((ValueError, TypeError)):
        evaluation.validate_partitions(parts, groups, np.array([0, 1, 0, 1, 0, 1]))


def test_candidate_selection_uses_validation_ap_with_order_independent_ties():
    rows = [
        {"model": "Random_forest", "validation_average_precision": 0.91},
        {"model": "Logistic_regression", "validation_average_precision": 0.96},
        {"model": "Decision_tree", "validation_average_precision": 0.96},
    ]
    assert evaluation.select_candidate(rows) == "Logistic_regression"
    assert evaluation.select_candidate(list(reversed(rows))) == "Logistic_regression"
    rows[0]["validation_average_precision"] = 0.97
    assert evaluation.select_candidate(rows) == "Random_forest"


def test_threshold_exact_f2_tie_chooses_higher_threshold():
    # At .1 there are TP=2, FP=8, FN=0; at .9 TP=1, FP=0, FN=1.
    # Both yield F2=5/9, so the declared conservative tie break chooses .9.
    y = np.array([1, 1] + [0] * 8)
    probabilities = np.array([.9, .1] + [.1] * 8)
    threshold, _ = evaluation.select_threshold(y, probabilities)
    assert threshold == .9
    low = evaluation.classification_metrics(y, probabilities, .1)["f2"]
    high = evaluation.classification_metrics(y, probabilities, .9)["f2"]
    assert low == pytest.approx(high)


FLOW = {"events": [], "train_rows": set(), "test_rows": set(), "output": None}


class AuditedScaler(StandardScaler):
    def fit(self, X, y=None, **kwargs):
        rows = set(X.index)
        assert rows <= FLOW["train_rows"], "Learned preprocessing used held-out rows"
        FLOW["events"].append(("scaler_fit", tuple(X.index)))
        return super().fit(X, y, **kwargs)


class AuditedClassifier(ClassifierMixin, BaseEstimator):
    """A cheap deterministic classifier that audits scientific boundaries."""
    def fit(self, X, y):
        assert set(X.index) <= FLOW["train_rows"], "Classifier fit used held-out rows"
        FLOW["events"].append(("model_fit", tuple(X.index)))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        rows = set(X.index)
        if rows & FLOW["test_rows"]:
            assert rows == FLOW["test_rows"], "Partial/mixed test inference"
            freezes = list(FLOW["output"].glob("runs/*/frozen_selection.json"))
            assert len(freezes) == 1, "Test was inferred before freezing selection"
            assert json.loads(freezes[0].read_text())["model"] == "Logistic_regression"
            FLOW["events"].append(("test_prediction", tuple(X.index)))
        p = np.where(X["signal"].to_numpy() > 0, .8, .2)
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    """Patch I/O only: keep genuine grouped CV, searches, selection and outputs."""
    n = 50
    y = np.tile([0, 1], n//2)
    X = pd.DataFrame({"signal": y.astype(float), "other": np.arange(n)/n})
    parts = {"train": np.arange(30), "validation": np.arange(30, 40), "test": np.arange(40, 50)}
    groups = np.arange(n)
    manifest = pd.DataFrame({"record_id": [f"synthetic_{i}" for i in range(n)],
                             "source_extract": "synthetic", "blackout": y})
    summary = {"protocol_version": "synthetic-test-only", "partitions": {
        name: {"records": len(idx)} for name, idx in parts.items()}}
    output, figures = tmp_path / "experiment_results/corrected", tmp_path / "figures/corrected"
    for name in ("data/processed/validated_20000.csv.gz", "data/splits/all_records_manifest.csv",
                 "data/Blackout_PowerGraph_1000.csv", "data/raw/Blackout_PowerGraph_19000.zip",
                 "src/clean_data.py", "src/split_data.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic test fixture, not project data\n")
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "DATA_PATH", tmp_path / "data/processed/validated_20000.csv.gz")
    monkeypatch.setattr(evaluation, "MANIFEST_PATH", tmp_path / "data/splits/all_records_manifest.csv")
    monkeypatch.setattr(evaluation, "OUTPUT_DIR", output)
    monkeypatch.setattr(evaluation, "FIGURE_DIR", figures)
    monkeypatch.setattr(evaluation, "load_problem", lambda: (X, y, manifest, groups, parts, summary))
    scaler = AuditedScaler().set_output(transform="pandas")
    specs = {
        "Dummy_prior": (DummyClassifier(strategy="prior"), {}),
        "Logistic_regression": (Pipeline([("scaler", scaler), ("model", AuditedClassifier())]), {}),
    }
    monkeypatch.setattr(evaluation, "candidate_specs", lambda: specs)
    FLOW.update(events=[], train_rows=set(parts["train"]), test_rows=set(parts["test"]), output=output)
    return output, figures


def test_small_run_keeps_training_boundaries_freezes_once_and_publishes_complete(synthetic_run):
    output, _ = synthetic_run
    result = evaluation.run_evaluation(jobs=1)
    assert result["model_selection"]["model"] == "Logistic_regression"
    assert sum(name == "test_prediction" for name, _ in FLOW["events"]) == 1
    fits = [rows for name, rows in FLOW["events"] if name == "model_fit"]
    scaler_fits = [rows for name, rows in FLOW["events"] if name == "scaler_fit"]
    assert len(fits) == len(scaler_fits) == 6  # five CV fits and training-only refit
    assert any(set(rows) == FLOW["train_rows"] for rows in fits)
    pointer = json.loads((output / "latest.json").read_text())
    run = output / pointer["run_directory"]
    assert pointer["run_id"] == result["run_id"]
    completion = json.loads((run / "complete.json").read_text())
    for name, digest in completion["artifacts"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
    saved = pd.read_csv(run / "test_predictions.csv", float_precision="round_trip")
    assert set(saved.row_index) == FLOW["test_rows"]
    assert saved.record_id.is_unique
    recalculated = evaluation.classification_metrics(saved.blackout, saved.blackout_probability,
                                                    result["model_selection"]["threshold"])
    assert recalculated == result["final_test_metrics"]
    assert (output / "evaluation_results.json").read_bytes() == (run / "evaluation_results.json").read_bytes()


def test_failed_fit_keeps_last_completed_results(synthetic_run, monkeypatch):
    output, figures = synthetic_run
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    old = {output / "latest.json": b'{"run_id":"previous"}',
           output / "evaluation_results.json": b'{"result":"previous"}',
           figures / "test_roc.png": b"previous figure"}
    for path, content in old.items():
        path.write_bytes(content)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic fit failure")
    monkeypatch.setattr(evaluation, "fit_candidates", fail)
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        evaluation.run_evaluation(jobs=1)
    assert all(path.read_bytes() == content for path, content in old.items())
    assert not list(output.glob("runs/*/complete.json"))
    assert not any(name == "test_prediction" for name, _ in FLOW["events"])


def completed_fixture(output, content=b"new result"):
    run = output / "runs" / "new-run"
    (run / "figures").mkdir(parents=True)
    (run / "evaluation_results.json").write_bytes(content)
    (run / "test_predictions.csv").write_bytes(b"row_index,blackout\n0,0\n")
    artifacts = {p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in run.rglob("*") if p.is_file()}
    (run / "complete.json").write_text(json.dumps({"run_id": run.name, "artifacts": artifacts}))
    return run


def test_publication_rejects_corrupt_completed_artifact_without_touching_latest(tmp_path, monkeypatch):
    output, figures = tmp_path / "experiment_results", tmp_path / "figures"
    monkeypatch.setattr(evaluation, "OUTPUT_DIR", output)
    monkeypatch.setattr(evaluation, "FIGURE_DIR", figures)
    run = completed_fixture(output)
    (run / "test_predictions.csv").write_bytes(b"corrupted after completion")
    old = {output / "latest.json": b'{"run_id":"previous"}',
           output / "evaluation_results.json": b"previous result"}
    for path, content in old.items():
        path.write_bytes(content)
    with pytest.raises(ValueError):
        evaluation.publish_latest(run)
    assert all(path.read_bytes() == content for path, content in old.items())
    assert not (output / "test_predictions.csv").exists()


def test_caught_publication_failure_restores_mirrors_and_previous_pointer(tmp_path, monkeypatch):
    output, figures = tmp_path / "experiment_results", tmp_path / "figures"
    monkeypatch.setattr(evaluation, "OUTPUT_DIR", output)
    monkeypatch.setattr(evaluation, "FIGURE_DIR", figures)
    run = completed_fixture(output)
    old = {output / "latest.json": b'{"run_id":"previous"}',
           output / "evaluation_results.json": b"previous result",
           output / "test_predictions.csv": b"previous predictions"}
    for path, content in old.items():
        path.write_bytes(content)
    real_atomic = evaluation.atomic_bytes
    written = []
    def fail_once(path, content):
        written.append(Path(path))
        if len(written) == 2:
            raise OSError("synthetic publication interruption")
        return real_atomic(path, content)
    monkeypatch.setattr(evaluation, "atomic_bytes", fail_once)
    with pytest.raises(OSError, match="synthetic publication interruption"):
        evaluation.publish_latest(run)
    assert len(written) >= 3  # proves a mirror changed, then rollback restored it
    assert all(path.read_bytes() == content for path, content in old.items())
