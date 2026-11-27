"""Reproduce the official GridGuard comparison using the team's fixed 80/20 CSVs.

Run from repository root:
    python -m src.compare_official

This is a retrospective benchmark: earlier team experiments have already
examined the test set. Do not describe this as fresh external validation.
Only training folds are used for candidate selection. No dataset is modified.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "experiment_results" / "official_model_comparison.json"
TARGET = "blackout"
SEED = 42


def candidate_models():
    return {
        "Dummy majority": lambda: DummyClassifier(strategy="prior"),
        "Logistic regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=1, random_state=SEED)),
        "Decision tree": lambda: DecisionTreeClassifier(random_state=SEED),
        "Random forest": lambda: RandomForestClassifier(
            n_estimators=100, max_features="sqrt", min_samples_leaf=2,
            n_jobs=2, random_state=SEED),
        "Histogram gradient boosting": lambda: HistGradientBoostingClassifier(
            max_iter=110, max_leaf_nodes=25, learning_rate=0.1, random_state=SEED),
    }


def metrics(y, predicted, probability):
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def cross_validate(name, factory, X, y, groups):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = np.full(len(y), -1, dtype=int)
    probability = np.full(len(y), np.nan)
    fold_rows = []
    for fold, (train_idx, validation_idx) in enumerate(splitter.split(X, y, groups), 1):
        assert set(groups[train_idx]).isdisjoint(set(groups[validation_idx]))
        estimator = factory()
        estimator.fit(X.iloc[train_idx], y[train_idx])
        fold_pred = estimator.predict(X.iloc[validation_idx])
        fold_prob = estimator.predict_proba(X.iloc[validation_idx])[:, 1]
        predictions[validation_idx] = fold_pred
        probability[validation_idx] = fold_prob
        fold_rows.append({
            "fold": fold, "train_rows": len(train_idx),
            "validation_rows": len(validation_idx),
            **metrics(y[validation_idx], fold_pred, fold_prob),
        })
    assert (predictions >= 0).all() and np.isfinite(probability).all()
    return {
        "model": name,
        **metrics(y, predictions, probability),
        "mean_fold_f1": float(np.mean([row["f1"] for row in fold_rows])),
        "std_fold_f1": float(np.std([row["f1"] for row in fold_rows], ddof=1)),
        "folds": fold_rows,
    }


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    X = train.drop(columns=[TARGET])
    y = train[TARGET].to_numpy(dtype=int)
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET].to_numpy(dtype=int)
    if len(train) != 16018 or len(test) != 3982 or len(X.columns) != 224:
        raise ValueError("Unexpected official team split.")
    if not X.columns.equals(X_test.columns):
        raise ValueError("Training and test feature schemas differ.")
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    test_groups = pd.util.hash_pandas_object(X_test, index=False).to_numpy()
    if not set(groups).isdisjoint(set(test_groups)):
        raise ValueError("Identical complete inputs cross train/test.")

    factories = candidate_models()
    comparisons = [cross_validate(name, factory, X, y, groups)
                   for name, factory in factories.items()]
    winner = max(comparisons, key=lambda row: row["f1"])["model"]

    selected = factories[winner]()
    selected.fit(X, y)
    selected_pred = selected.predict(X_test)
    selected_test = metrics(y_test, selected_pred, selected.predict_proba(X_test)[:, 1])

    original_tree = factories["Decision tree"]()
    original_tree.fit(X, y)
    tree_pred = original_tree.predict(X_test)
    tree_test = metrics(y_test, tree_pred, original_tree.predict_proba(X_test)[:, 1])

    bus_cols = [name for name in X if name.startswith("bus_")]
    if len(bus_cols) != 72:
        raise ValueError("Expected 72 bus variables.")
    bus_groups = pd.util.hash_pandas_object(X[bus_cols], index=False).to_numpy()
    sensitivity = [
        cross_validate(name + " (bus groups)", factories[name], X, y, bus_groups)
        for name in ("Decision tree", "Histogram gradient boosting")
    ]
    error_a = selected_pred != y_test
    error_b = tree_pred != y_test
    a_only = int((error_a & ~error_b).sum())
    b_only = int((~error_a & error_b).sum())
    paired_p = float(binomtest(min(a_only, b_only), n=a_only + b_only,
                              p=0.5, alternative="two-sided").pvalue)
    result = {
        "status": "retrospective_internal_comparison",
        "warning": "Earlier test exposure: this is not new external validation.",
        "dataset_rows": len(train) + len(test), "train_rows": len(train),
        "test_rows": len(test), "features": len(X.columns),
        "split": "fixed official grouped ~80/20", "cv": "five StratifiedGroupKFold folds",
        "candidate_selection": "highest pooled training-only out-of-fold F1",
        "candidates": comparisons, "selected_model": winner,
        "selected_test": selected_test, "decision_tree_test": tree_test,
        "bus_group_sensitivity": sensitivity,
        "paired_test": {
            "selected_wrong_tree_right": a_only, "tree_wrong_selected_right": b_only,
            "both_wrong": int((error_a & error_b).sum()),
            "both_right": int((~error_a & ~error_b).sum()),
            "mcnemar_exact_p": paired_p,
        },
        "caveats": [
            "The original source does not prove that every flattened feature is pre-outcome.",
            "Repeated simulations can still share scenario dependence.",
            "Results on simulated data are not operational grid validation.",
            "Test set was previously examined in other project experiments.",
        ],
    }
    RESULTS.parent.mkdir(exist_ok=True, parents=True)
    RESULTS.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Selected by training CV:", winner)
    for row in comparisons:
        print(f'{row["model"]:30s}  CV F1={row["f1"]:.4f}, recall={row["recall"]:.4f}')
    print("Retrospective test:", json.dumps(selected_test, indent=2))
    print("Wrote", RESULTS)


if __name__ == "__main__":
    main()
