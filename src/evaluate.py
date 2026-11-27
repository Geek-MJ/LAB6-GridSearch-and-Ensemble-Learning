"""Compare models on a fixed grouped split, then evaluate one frozen selection.

Run from the repository root: python -m src.evaluate --jobs 2
Completed runs are immutable under experiment_results/corrected/runs. latest.json is the
authoritative pointer; compatible report/figure copies are published last.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.split_data import group_ids, make_splits

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/validated_20000.csv.gz"
MANIFEST_PATH = ROOT / "data/splits/all_records_manifest.csv"
OUTPUT_DIR, FIGURE_DIR = ROOT / "experiment_results/corrected", ROOT / "figures/corrected"
TARGET, SEED = "blackout", 42
MODEL_ORDER = ("Dummy_prior", "Logistic_regression", "Decision_tree", "Random_forest", "Hist_gradient_boosting")
METRICS = ("accuracy", "precision", "recall", "f1", "f2", "roc_auc", "average_precision")
SELECTION = "Highest validation average precision; exact ties use declared model order."
THRESHOLD_RULE = "Maximum validation F2; highest threshold on exact ties."
F2_ASSUMPTION = "F2 emphasizes recall; engineering assumption, not a confirmed business cost model."
SCOPE = "Internal retrospective validation: this dataset and older test results were examined previously."
CONTEXT = "PowerGraph uses pre-outage graph features with the initial contingency known to assess cascading-failure risk, not to predict an unknown first outage."
CAVEATS = (SCOPE, CONTEXT, F2_ASSUMPTION,
    "Simulated IEEE-24 results do not establish real-world grid performance.",
    "Source documentation describes pre-outage attributes; the local MATLAB-to-CSV mapping and label alignment remain independently unverified.",
    "Bus hashes proxy dependence; unknown simulation-family dependencies may remain.",
    "Bootstrap intervals condition on the fitted model and split, excluding retraining, selection uncertainty and deployment shift.",
    "Only the selected training-fitted model is tested; validation is not added to its training data.")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, BaseEstimator):
        return {"class": f"{type(value).__module__}.{type(value).__name__}", "parameters": value.get_params(deep=False)}
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, default=json_default, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def checked_predictions(y, probabilities, threshold):
    y, p = np.asarray(y), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or not len(y) or len(y) != len(p):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1].")
    if not np.isscalar(threshold) or not isinstance(threshold, (int, float, np.number)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1].")
    return y.astype(int), p


def classification_metrics(y, probabilities, threshold):
    y, p = checked_predictions(y, probabilities, threshold)
    prediction = (p >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "f2": float(fbeta_score(y, prediction, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def validate_partitions(parts, groups, y):
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or pd.isna(groups).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels and groups must be aligned binary-label vectors.")
    if set(parts) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test partitions.")
    arrays = [np.asarray(parts[name]) for name in ("train", "validation", "test")]
    if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in arrays):
        raise ValueError("Partition row indices must be integer vectors.")
    rows = np.concatenate(arrays)
    if len(rows) != len(y) or not np.array_equal(np.sort(rows), np.arange(len(y))):
        raise ValueError("Partitions must cover each original row exactly once.")
    sets = [set(groups[a]) for a in arrays]
    for i, a in enumerate(arrays):
        if len(np.unique(y[a])) != 2:
            raise ValueError("Every partition must contain both target classes.")
        if any(sets[i] & other for other in sets[:i]):
            raise ValueError("Bus configuration group crosses partitions.")


def training_folds(y, groups):
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, groups))
    for train, held in folds:
        if set(groups[train]) & set(groups[held]):
            raise ValueError("Group overlap inside cross-validation.")
        if any(len(np.unique(y[idx])) != 2 for idx in (train, held)):
            raise ValueError("Both classes must occur in every CV training and validation fold.")
    return folds


def select_candidate(rows):
    if not rows or any(row["model"] not in MODEL_ORDER or not np.isfinite(row["validation_average_precision"])
                       or not 0 <= row["validation_average_precision"] <= 1 for row in rows):
        raise ValueError("Invalid candidate name or average precision.")
    return min(rows, key=lambda row: (-row["validation_average_precision"], MODEL_ORDER.index(row["model"])))["model"]


def select_threshold(y, probabilities):
    y, p = checked_predictions(y, probabilities, 0.5)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both target classes.")
    precision, recall, thresholds = precision_recall_curve(y, p)
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(5 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    table = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1], "f2": f2})
    best = table.sort_values(["f2", "threshold"], ascending=False).iloc[0]
    return float(best.threshold), table


def bootstrap_intervals(y, probabilities, groups, threshold, repeats=300, seed=SEED):
    """Resample entire bus groups with multiplicity; do not refit or retune."""
    y, p = checked_predictions(y, probabilities, threshold)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(y) or pd.isna(groups).any() or isinstance(repeats, (bool, np.bool_)) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("Bootstrap needs aligned groups and a positive integer repeat count.")
    unique, inverse = np.unique(groups, return_inverse=True)
    group_rows = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    values, skipped, rng = {name: [] for name in METRICS}, 0, np.random.default_rng(seed)
    for _ in range(repeats):
        idx = np.concatenate([group_rows[i] for i in rng.integers(0, len(unique), size=len(unique))])
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        metrics = classification_metrics(y[idx], p[idx], threshold)
        for name in METRICS:
            values[name].append(metrics[name])
    intervals = {name: dict(zip(("lower", "upper"), map(float, np.quantile(v, [0.025, 0.975]))))
                 if v else {"lower": None, "upper": None} for name, v in values.items()}
    return {"method": "bus-group percentile bootstrap", "confidence_level": 0.95,
        "requested_repetitions": repeats, "valid_repetitions": repeats - skipped,
        "skipped_single_class_repetitions": skipped, "seed": seed, "test_bus_groups": len(unique),
        "intervals": intervals, "scope": "Conditional internal retrospective uncertainty; no refitting or selection uncertainty."}


def candidate_specs():
    return {
        "Dummy_prior": (DummyClassifier(strategy="prior"), {}),
        "Logistic_regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)),
                                {"logisticregression__C": [0.1, 1.0, 10.0]}),
        "Decision_tree": (DecisionTreeClassifier(random_state=SEED), {"max_depth": [8, None], "min_samples_leaf": [2, 10]}),
        "Random_forest": (RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
                          {"max_depth": [12, None], "min_samples_leaf": [2, 5]}),
        "Hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=SEED,
                                                                 early_stopping=False), {"max_leaf_nodes": [15, 31]}),
    }


def load_problem():
    data = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"input_hash_uint64": "string"})
    if data.shape != (20000, 225) or TARGET not in data or not data.columns.is_unique:
        raise ValueError("Expected 20,000 rows, 224 unique numerical features and blackout.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
        raise ValueError("The dataset must contain only numerical columns.")
    if not np.isfinite(data.to_numpy()).all() or not data[TARGET].isin([0, 1]).all():
        raise ValueError("Non-finite data or non-binary target.")
    required = {"record_id", "source_extract", TARGET}
    if not required <= set(manifest) or manifest[list(required)].isna().any().any():
        raise ValueError("Missing original manifest fields.")
    # The splitter also verifies the original source values, row identities and locked protocol.
    parts, summary = make_splits(save=True, data=data, original=manifest)
    X, y = data.drop(columns=TARGET), data[TARGET].to_numpy(dtype=int)
    groups = group_ids(X)
    validate_partitions(parts, groups, y)
    return X, y, manifest, groups, parts, summary


def make_config(run_id, jobs, specs, summary):
    hashes = {name: file_sha256(path) for name, path in {
        "dataset_sha256": DATA_PATH, "original_manifest_sha256": MANIFEST_PATH,
        "evaluator_sha256": Path(__file__), "splitter_sha256": ROOT / "src/split_data.py"}.items()}
    sources = ("data/Blackout_PowerGraph_1000.csv", "data/raw/Blackout_PowerGraph_19000.zip", "src/clean_data.py")
    return {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(), **hashes,
        "provenance": "AI-authored rewrite; no student authorship is asserted.",
        "seed": SEED, "jobs": jobs, "python": platform.python_version(), "platform": platform.platform(),
        "package_versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "threadpoolctl")},
        "source_hashes": {name: file_sha256(ROOT / name) for name in sources}, "split_summary": summary,
        "selection": SELECTION, "candidate_order": list(MODEL_ORDER),
        "parameter_grids": {name: grid for name, (_, grid) in specs.items()},
        "candidate_estimators": {name: estimator for name, (estimator, _) in specs.items()},
        "cross_validation": "Five fixed training-only StratifiedGroupKFold folds",
        "threshold_rule": THRESHOLD_RULE, "threshold_assumption": F2_ASSUMPTION,
        "bootstrap_repetitions": 300, "evaluation_scope": SCOPE, "prediction_context": CONTEXT,
        "source_reference": "PowerGraph dataset paper, arXiv:2402.02827v3; local flattening remains unverified."}


def fit_candidates(X, y, train, validation, folds, specs, jobs):
    rows, models, probabilities = [], {}, {}
    with threadpool_limits(limits=jobs), joblib.parallel_backend("loky", inner_max_num_threads=1):
        for name, (estimator, grid) in specs.items():
            print(f"Training {name}: {len(ParameterGrid(grid)) * len(folds)} CV fits + refit", flush=True)
            started = perf_counter()
            search = GridSearchCV(estimator, grid, scoring={"ap": "average_precision", "recall": "recall"},
                                  refit="ap", cv=folds, n_jobs=jobs, error_score="raise")
            search.fit(X.iloc[train], y[train])
            elapsed, model, best = perf_counter() - started, search.best_estimator_, search.best_index_
            p = model.predict_proba(X.iloc[validation])[:, list(model.classes_).index(1)]
            metrics = classification_metrics(y[validation], p, 0.5)
            rows.append({"model": name, "fit_seconds": elapsed, "best_parameters": search.best_params_,
                "full_parameters": model, "train_cv_average_precision_mean": float(search.cv_results_["mean_test_ap"][best]),
                "train_cv_average_precision_std": float(search.cv_results_["std_test_ap"][best]),
                "train_cv_recall_mean": float(search.cv_results_["mean_test_recall"][best]),
                **{f"validation_{key}": metrics[key] for key in ("average_precision", "roc_auc")},
                **{f"validation_{key}_at_0_5": metrics[key] for key in ("precision", "recall", "f1")}})
            models[name], probabilities[name] = model, p
            print(f"  finished in {elapsed:.1f}s; validation AP={metrics['average_precision']:.4f}", flush=True)
    return rows, models, probabilities


def render_report(results):
    selection, metrics = results["model_selection"], results["final_test_metrics"]
    lines = ["# GridGuard evaluation", "", f"Run: `{results['run_id']}`", "", SCOPE, "",
        "All candidates share fixed bus-group-disjoint partitions and training-only grouped CV. "
        "Scaling is inside the logistic pipeline only. The selected training model is frozen before testing.", "",
        f"Split sizes: {results['split_sizes']}", "", "## Validation comparison", "",
        "| Candidate | Training CV AP | Validation AP | Recall at 0.50 | Fit seconds |", "|---|---:|---:|---:|---:|"]
    for row in results["validation_model_comparison"]:
        lines.append(f"| {row['model']} | {row['train_cv_average_precision_mean']:.4f} ± {row['train_cv_average_precision_std']:.4f} | {row['validation_average_precision']:.4f} | {row['validation_recall_at_0_5']:.4f} | {row['fit_seconds']:.1f} |")
    lines += ["", f"Selected **{selection['model']}** by validation AP; threshold **{selection['threshold']:.8f}** by validation F2.",
        F2_ASSUMPTION, "", "## Retrospective test", "", "| Metric | Estimate | Conditional 95% group-bootstrap interval |", "|---|---:|---|"]
    for name, ci in results["bootstrap_confidence_intervals"]["intervals"].items():
        interval = f"[{ci['lower']:.4f}, {ci['upper']:.4f}]" if ci["lower"] is not None else "Unavailable"
        lines.append(f"| {name} | {metrics[name]:.4f} | {interval} |")
    lines += ["", f"Confusion matrix `[[TN, FP], [FN, TP]]`: `{metrics['confusion_matrix']}`.", "", "## Limits", ""]
    return "\n".join(lines + [f"- {item}" for item in results["caveats"]]) + "\n"


def save_figures(folder, y_val, val_probabilities, y_test, test_probabilities, threshold):
    folder.mkdir()
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, p in val_probabilities.items():
        PrecisionRecallDisplay.from_predictions(y_val, p, name=name, ax=ax)
    ax.set_title("Grouped validation")
    figures = [(fig, "validation_precision_recall.png")]
    for display, filename, values in (
        (ConfusionMatrixDisplay, "test_confusion_matrix.png", (test_probabilities >= threshold).astype(int)),
        (PrecisionRecallDisplay, "test_precision_recall.png", test_probabilities),
        (RocCurveDisplay, "test_roc.png", test_probabilities)):
        fig, ax = plt.subplots(figsize=(7, 5))
        display.from_predictions(y_test, values, ax=ax)
        ax.set_title("Internal retrospective grouped test")
        figures.append((fig, filename))
    for fig, filename in figures:
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=150)
        plt.close(fig)


def atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_latest(run_dir):
    """Publish only a completed run; restore previous mirrors on a caught failure."""
    complete = run_dir / "complete.json"
    if not complete.is_file():
        raise ValueError("Cannot publish an incomplete run.")
    receipt = json.loads(complete.read_text(encoding="utf-8"))
    artifacts = receipt.get("artifacts", {})
    actual = {p.relative_to(run_dir).as_posix(): p for p in run_dir.rglob("*") if p.is_file() and p != complete}
    if receipt.get("run_id") != run_dir.name or not artifacts or set(artifacts) != set(actual):
        raise ValueError("Completion manifest does not match the run artifacts.")
    if any(file_sha256(actual[name]) != digest for name, digest in artifacts.items()):
        raise ValueError("Completed run artifact checksum mismatch.")
    destinations = {OUTPUT_DIR / p.name: p for p in run_dir.iterdir() if p.suffix in (".json", ".csv", ".md")
                    and p.name not in ("complete.json", "config.json", "frozen_selection.json")}
    destinations.update({FIGURE_DIR / p.name: p for p in (run_dir / "figures").iterdir()})
    previous = {target: target.read_bytes() if target.exists() else None for target in destinations}
    changed = []
    try:
        for target, source in destinations.items():
            atomic_bytes(target, source.read_bytes())
            changed.append(target)
        pointer = {"run_id": run_dir.name, "run_directory": f"runs/{run_dir.name}", "complete_sha256": file_sha256(complete)}
        atomic_bytes(OUTPUT_DIR / "latest.json", (json.dumps(pointer, indent=2) + "\n").encode())
    except Exception:
        for target in reversed(changed):
            if previous[target] is None:
                target.unlink(missing_ok=True)
            else:
                atomic_bytes(target, previous[target])
        raise


def run_evaluation(jobs=2):
    if type(jobs) is not int or jobs not in (1, 2):
        raise ValueError("Use one or two worker processes.")
    print("Validating source data, fixed partitions and CV folds...", flush=True)
    X, y, manifest, groups, parts, summary = load_problem()
    train, validation, test = (parts[name] for name in ("train", "validation", "test"))
    folds, specs = training_folds(y[train], groups[train]), candidate_specs()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    run_dir = OUTPUT_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = make_config(run_id, jobs, specs, summary)
    # Normalize estimator objects to explicit class/default-parameter dictionaries.
    config = json.loads(json.dumps(config, default=json_default, allow_nan=False))
    write_json(run_dir / "config.json", config)
    rows, models, val_p = fit_candidates(X, y, train, validation, folds, specs, jobs)
    rows = json.loads(json.dumps(rows, default=json_default, allow_nan=False))
    selected = select_candidate(rows)
    threshold, threshold_table = select_threshold(y[validation], val_p[selected])
    winner = next(row for row in rows if row["model"] == selected)
    frozen = {"model": selected, "threshold": threshold, "parameters": winner["best_parameters"],
        "full_parameters": winner["full_parameters"], "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_selection": SELECTION, "threshold_rule": THRESHOLD_RULE}
    write_json(run_dir / "frozen_selection.json", frozen)
    print(f"Frozen {selected}, threshold={threshold:.6f}; evaluating selected model once...", flush=True)
    with threadpool_limits(limits=jobs):
        model = models[selected]
        test_p = model.predict_proba(X.iloc[test])[:, list(model.classes_).index(1)]
    metrics = classification_metrics(y[test], test_p, threshold)
    print("Computing 300 conditional bus-group bootstrap intervals...", flush=True)
    intervals = bootstrap_intervals(y[test], test_p, groups[test], threshold)
    results = {"run_id": run_id, "dataset": "PowerGraph IEEE-24 simulated scenarios", "rows": len(X),
        "input_features": X.shape[1], "target": TARGET, "split_sizes": {k: len(v) for k, v in parts.items()},
        "split_summary": summary, "configuration": config, "model_selection": frozen,
        "validation_model_comparison": rows, "threshold_selection": {"threshold": threshold, "criterion": THRESHOLD_RULE, "assumption": F2_ASSUMPTION},
        "validation_metrics_at_selected_threshold": classification_metrics(y[validation], val_p[selected], threshold),
        "final_test_metrics": metrics, "bootstrap_confidence_intervals": intervals, "caveats": list(CAVEATS)}
    predictions = manifest.iloc[test][["record_id", "source_extract", TARGET]].copy().reset_index(drop=True)
    predictions.insert(0, "row_index", test)
    predictions["blackout_probability"], predictions["predicted_blackout"] = test_p, (test_p >= threshold).astype(int)
    predictions["error_type"] = np.select([(y[test] == 1) & (test_p < threshold), (y[test] == 0) & (test_p >= threshold)],
                                          ["false_negative", "false_positive"], default="correct")
    tables = {"validation_model_comparison": pd.DataFrame(rows), "validation_threshold_comparison": threshold_table,
              "test_predictions": predictions, "test_errors": predictions.loc[predictions.error_type != "correct"]}
    for name, table in tables.items():
        table.to_csv(run_dir / f"{name}.csv", index=False)
    for name, payload in (("evaluation_results", results), ("bootstrap_intervals", intervals)):
        write_json(run_dir / f"{name}.json", payload)
    (run_dir / "evaluation_report.md").write_text(render_report(results), encoding="utf-8")
    joblib.dump({"model": model, "features": X.columns.tolist(), "threshold": threshold, "run_id": run_id}, run_dir / "selected_model.joblib")
    save_figures(run_dir / "figures", y[validation], val_p, y[test], test_p, threshold)
    artifacts = {str(p.relative_to(run_dir)).replace("\\", "/"): file_sha256(p) for p in run_dir.rglob("*") if p.is_file()}
    write_json(run_dir / "complete.json", {"run_id": run_id, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts})
    publish_latest(run_dir)
    print(f"Completed run {run_id}; latest pointer published.", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    result = run_evaluation(parser.parse_args().jobs)
    print(json.dumps({"run_id": result["run_id"], "selected_model": result["model_selection"]["model"],
                      "final_test_metrics": result["final_test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
