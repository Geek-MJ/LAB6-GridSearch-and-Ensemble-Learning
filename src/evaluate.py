
"""Evaluation protocol following Anais's GridGuard methodology.

Run from the repository root:
    python -m src.evaluate

Protocol:
1. Stratified 80/20 train-pool/test split.
2. GroupShuffleSplit on the train pool to create validation.
3. StratifiedGroupKFold CV for Random Forest tuning.
4. Recall-based hyperparameter selection.
5. Random Forest with all features and threshold 0.50.
6. Final test evaluation after model selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "validated_20000.csv.gz"
MANIFEST_PATH = ROOT / "data" / "splits" / "all_records_manifest.csv"

OUTPUT_DIR = ROOT / "reports" / "evaluation_anais"
FIGURE_DIR = ROOT / "figures" / "evaluation_anais"

TARGET = "blackout"
RANDOM_STATE = 42
THRESHOLD = 0.50


def load_data():
    """Load the dataset and verify its alignment with the manifest."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(MANIFEST_PATH)

    data = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"input_hash_uint64": "string"},
    )

    if len(data) != 20000:
        raise ValueError(f"Expected 20,000 rows, got {len(data)}.")

    if len(manifest) != len(data):
        raise ValueError("Dataset and manifest have different row counts.")

    if TARGET not in data.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if not manifest[TARGET].reset_index(drop=True).equals(
        data[TARGET].reset_index(drop=True)
    ):
        raise ValueError("Target labels do not match manifest row order.")

    features = [c for c in data.columns if c != TARGET]

    if len(features) != 224:
        raise ValueError(f"Expected 224 features, got {len(features)}.")

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if data[features].isna().to_numpy().any():
        raise ValueError("Missing feature values detected.")

    if not np.isfinite(data[features].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values detected.")

    # Verify the feature values and row order against the manifest.
    feature_hashes = pd.util.hash_pandas_object(
        data[features],
        index=False,
    ).astype(str)

    if not (
        feature_hashes.to_numpy()
        == manifest["input_hash_uint64"].astype(str).to_numpy()
    ).all():
        raise ValueError("Feature hashes do not match the manifest.")

    bus_columns = [c for c in features if c.startswith("bus_")]

    if not bus_columns:
        raise ValueError("No bus_ columns found for group construction.")

    if len(bus_columns) != 72:
        print(f"WARNING: Found {len(bus_columns)} bus features, not 72.")

    print("Dataset validation successful.")
    print("Rows:", len(data))
    print("Features:", len(features))
    print("Bus features:", len(bus_columns))
    print("Target distribution:")
    print(data[TARGET].value_counts().sort_index())

    return data, manifest, features, bus_columns


def get_metrics(y_true, probabilities, threshold=THRESHOLD):
    """Calculate classification metrics at a fixed threshold."""

    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD AND VALIDATE DATA
    # ---------------------------------------------------------

    data, manifest, features, bus_columns = load_data()

    X = data[features]
    y = data[TARGET].astype(int)

    # ---------------------------------------------------------
    # 2. INITIAL STRATIFIED 80/20 SPLIT
    # ---------------------------------------------------------

    all_indices = np.arange(len(data))

    train_pool_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_pool = X.iloc[train_pool_idx].copy()
    y_pool = y.iloc[train_pool_idx].copy()

    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    test_manifest = manifest.iloc[test_idx].reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. GROUP-BASED TRAIN / VALIDATION SPLIT
    # ---------------------------------------------------------

    groups_pool = pd.util.hash_pandas_object(
        X_pool[bus_columns],
        index=False,
    ).to_numpy()

    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        group_splitter.split(X_pool, y_pool, groups=groups_pool)
    )

    X_train = X_pool.iloc[train_idx].copy()
    y_train = y_pool.iloc[train_idx].copy()
    groups_train = groups_pool[train_idx]

    X_val = X_pool.iloc[val_idx].copy()
    y_val = y_pool.iloc[val_idx].copy()
    groups_val = groups_pool[val_idx]

    # Verify that training and validation groups are separate.
    train_val_overlap = len(
        set(groups_train).intersection(set(groups_val))
    )

    if train_val_overlap:
        raise ValueError("Group overlap detected between train and validation.")

    print("\n========== DATA SPLITS ==========")

    for name, labels in [
        ("Train", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:
        print(
            f"{name:12s}: {len(labels):5d} records | "
            f"blackout rate: {labels.mean():.3%} | "
            f"classes: {labels.value_counts().sort_index().to_dict()}"
        )

    print("Train/validation group overlap:", train_val_overlap)

    # The outer split follows Anais's stratified random split.
    # It is not group-aware, so test group overlap is reported.
    groups_test = pd.util.hash_pandas_object(
        X_test[bus_columns],
        index=False,
    ).to_numpy()

    train_test_overlap = len(
        set(groups_train).intersection(set(groups_test))
    )

    pool_test_overlap = len(
        set(groups_pool).intersection(set(groups_test))
    )

    print("Train/test group overlap:", train_test_overlap)
    print("Pool/test group overlap:", pool_test_overlap)

    # ---------------------------------------------------------
    # 4. GROUP-AWARE CROSS-VALIDATION
    # ---------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Anais's pipeline and parameter grid.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "model__n_estimators": [2, 4, 6],
        "model__max_depth": [4, 8, 11, None],
        "model__min_samples_split": [2, 4, 6],
        "model__min_samples_leaf": [2, 3, 5, 7],
        "model__max_features": [200, 30, 100, 225],
        "model__class_weight": [None, "balanced"],
    }

    print("\n========== RANDOM FOREST GRID SEARCH ==========")
    print("Scoring: recall")
    print("Cross-validation: 5-fold StratifiedGroupKFold")
    print("Parameter combinations:", 1152)
    print("Expected fits:", 5760)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
        verbose=1,
    )

    grid_search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        "Best mean cross-validation recall:",
        round(grid_search.best_score_, 4),
    )

    # ---------------------------------------------------------
    # 5. VALIDATION MODEL COMPARISON
    # ---------------------------------------------------------

    models = {
        "Dummy_prior": DummyClassifier(strategy="prior"),

        "Decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=5,
        ),

        "Logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "Random_forest_Anais": grid_search.best_estimator_,
    }

    comparison_rows = []
    validation_probabilities = {}
    fitted_models = {}

    for name, model in models.items():

        print(f"\nEvaluating {name} on validation...")

        if name != "Random_forest_Anais":
            model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        fitted_models[name] = model
        validation_probabilities[name] = probabilities

        metrics = get_metrics(y_val, probabilities, THRESHOLD)

        comparison_rows.append({
            "model": name,
            "validation_recall": metrics["recall"],
            "validation_precision": metrics["precision"],
            "validation_f1": metrics["f1"],
            "validation_average_precision": metrics["average_precision"],
            "validation_roc_auc": metrics["roc_auc"],
            "threshold": THRESHOLD,
        })

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_recall", "validation_precision"],
        ascending=False,
    )

    comparison.to_csv(
        OUTPUT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    print("\n========== VALIDATION MODEL COMPARISON ==========")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Follow Anais's final decision: Random Forest with all features.
    selected_name = "Random_forest_Anais"
    selected_model = fitted_models[selected_name]

    selected_val_prob = validation_probabilities[selected_name]

    # ---------------------------------------------------------
    # 6. VALIDATION THRESHOLD ANALYSIS
    # ---------------------------------------------------------

    threshold_rows = []

    for threshold in np.round(np.arange(0.05, 0.55, 0.05), 2):

        metrics = get_metrics(y_val, selected_val_prob, float(threshold))

        tn, fp, fn, tp = np.array(
            metrics["confusion_matrix"]
        ).ravel()

        threshold_rows.append({
            "threshold": float(threshold),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positives": int(fp),
            "false_negatives": int(fn),
        })

    threshold_comparison = pd.DataFrame(threshold_rows)

    threshold_comparison.to_csv(
        OUTPUT_DIR / "validation_threshold_comparison.csv",
        index=False,
    )

    validation_metrics = get_metrics(
        y_val,
        selected_val_prob,
        THRESHOLD,
    )

    print("\n========== THRESHOLD ANALYSIS ==========")
    print(threshold_comparison.round(4).to_string(index=False))

    print("\nSelected threshold:", THRESHOLD)
    print("Validation recall:", round(validation_metrics["recall"], 4))
    print("Validation precision:", round(validation_metrics["precision"], 4))

    # ---------------------------------------------------------
    # 7. FINAL TEST EVALUATION
    # ---------------------------------------------------------

    # The model and threshold are now fixed.
    # Do not use test results to change either one.
    print("\n========== FINAL TEST EVALUATION ==========")

    test_probabilities = selected_model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= THRESHOLD
    ).astype(int)

    test_metrics = get_metrics(
        y_test,
        test_probabilities,
        THRESHOLD,
    )

    # ---------------------------------------------------------
    # 8. TEST PREDICTIONS AND ERROR ANALYSIS
    # ---------------------------------------------------------

    test_results = test_manifest[
        [
            "record_id",
            "source_extract",
            "source_scenario_index",
            TARGET,
        ]
    ].copy()

    test_results["blackout_probability"] = test_probabilities
    test_results["predicted_blackout"] = test_predictions
    test_results["error_type"] = "correct"

    test_results.loc[
        (test_results[TARGET] == 1) & (test_predictions == 0),
        "error_type",
    ] = "false_negative"

    test_results.loc[
        (test_results[TARGET] == 0) & (test_predictions == 1),
        "error_type",
    ] = "false_positive"

    test_results.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    errors = test_results[
        test_results["error_type"] != "correct"
    ].copy()

    errors.to_csv(
        OUTPUT_DIR / "test_errors.csv",
        index=False,
    )

    error_counts = test_results["error_type"].value_counts().to_dict()

    source_analysis = (
        test_results.groupby("source_extract")
        .agg(
            records=(TARGET, "size"),
            errors=("error_type", lambda s: (s != "correct").sum()),
        )
        .reset_index()
    )

    source_analysis["error_rate"] = (
        source_analysis["errors"] / source_analysis["records"]
    )

    source_analysis.to_csv(
        OUTPUT_DIR / "error_analysis_by_source.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 9. FEATURE IMPORTANCE
    # ---------------------------------------------------------

    rf = selected_model.named_steps["model"]

    importances = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    importances.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 10. SAVE MODEL
    # ---------------------------------------------------------

    joblib.dump(
        {
            "model": selected_model,
            "features": features,
            "threshold": THRESHOLD,
            "selected_model_name": selected_name,
            "selection_metric": "StratifiedGroupKFold recall",
            "threshold_selection": "Anais team decision: fixed 0.50",
            "best_parameters": grid_search.best_params_,
        },
        OUTPUT_DIR / "selected_model.joblib",
    )

    # ---------------------------------------------------------
    # 11. FIGURES
    # ---------------------------------------------------------

    # Validation precision-recall curves.
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, probabilities in validation_probabilities.items():
        PrecisionRecallDisplay.from_predictions(
            y_val,
            probabilities,
            name=name,
            ax=ax,
        )

    ax.set_title("Validation Precision-Recall Curves")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "validation_precision_recall.png",
        dpi=160,
    )
    plt.close(fig)

    # Validation recall by threshold.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        threshold_comparison["threshold"],
        threshold_comparison["recall"],
        marker="o",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Recall")
    ax.set_title("Validation Recall vs Decision Threshold")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "validation_recall_threshold.png",
        dpi=160,
    )
    plt.close(fig)

    # Test confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=[0, 1],
        display_labels=["No blackout", "Blackout"],
        values_format="d",
        ax=ax,
    )

    ax.set_title("Final Test Confusion Matrix — Threshold 0.50")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "test_confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)

    # Top 15 feature importances.
    top_features = importances.head(15).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title("Random Forest — Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "random_forest_feature_importance.png",
        dpi=160,
    )
    plt.close(fig)

    # ---------------------------------------------------------
    # 12. RESULTS JSON
    # ---------------------------------------------------------

    tn, fp, fn, tp = np.array(
        test_metrics["confusion_matrix"]
    ).ravel()

    results = {
        "dataset": "Validated PowerGraph IEEE-24 dataset",
        "rows": int(len(data)),
        "features": int(len(features)),
        "bus_features": int(len(bus_columns)),
        "target": TARGET,

        "split_method": {
            "outer": "train_test_split, test_size=0.20, stratify=y, random_state=42",
            "validation": "GroupShuffleSplit, test_size=0.20, random_state=42",
            "cross_validation": "5-fold StratifiedGroupKFold, random_state=42",
            "train_records": int(len(X_train)),
            "validation_records": int(len(X_val)),
            "test_records": int(len(X_test)),
        },

        "group_overlap": {
            "train_validation": int(train_val_overlap),
            "train_test": int(train_test_overlap),
            "pool_test": int(pool_test_overlap),
        },

        "model_selection": {
            "criterion": "Recall",
            "selected_model": selected_name,
            "best_parameters": grid_search.best_params_,
            "best_mean_cv_recall": float(grid_search.best_score_),
            "validation_comparison": comparison.to_dict(orient="records"),
        },

        "threshold_selection": {
            "threshold": THRESHOLD,
            "reason": "Team decision following Anais's validation analysis",
            "validation_metrics": validation_metrics,
            "threshold_comparison": threshold_comparison.to_dict(orient="records"),
        },

        "final_test_metrics": test_metrics,

        "test_errors": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "correct": int(error_counts.get("correct", 0)),
        },

        "error_analysis_by_source": source_analysis.to_dict(orient="records"),

        "caveats": [
            "The initial stratified outer split follows Anais's notebook and is not group-aware.",
            "Identical bus configurations may therefore appear in the training pool and test set.",
            "The inner train/validation split and cross-validation are group-aware.",
            "The test set was evaluated only after model and threshold selection.",
            "The dataset contains simulated scenarios and does not establish real-world grid performance.",
            "The local dataset has not been independently verified as byte-for-byte identical to Anais's original blackout.csv.",
        ],
    }

    (OUTPUT_DIR / "evaluation_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 13. MARKDOWN REPORT
    # ---------------------------------------------------------

    report = f"""# GridGuard — Evaluation Report

## 1. Evaluation protocol

- Dataset: 20,000 validated records.
- Features: 224.
- Target: `blackout`.
- Initial split: stratified 80% train pool / 20% test.
- Validation: GroupShuffleSplit, 20% of the training pool.
- Cross-validation: 5-fold StratifiedGroupKFold.
- Model selection criterion: recall.
- Final model: Random Forest with all 224 features.
- Decision threshold: 0.50.

## 2. Hyperparameter tuning

Best parameters:

```json
{json.dumps(grid_search.best_params_, indent=2)}
```

Best mean cross-validation recall: **{grid_search.best_score_:.4f}**

## 3. Validation model comparison

```text
{comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}")}
```

## 4. Threshold selection

The team selected threshold **0.50**, following Anais's validation analysis.
The threshold comparison is available in `validation_threshold_comparison.csv`.

At threshold 0.50:

- Validation recall: **{validation_metrics['recall']:.4f}**
- Validation precision: **{validation_metrics['precision']:.4f}**
- Validation F1: **{validation_metrics['f1']:.4f}**

## 5. Final test evaluation

The model and threshold were fixed before evaluating the test set.

| Metric | Test result |
|---|---:|
| Accuracy | {test_metrics['accuracy']:.4f} |
| Precision | {test_metrics['precision']:.4f} |
| Recall | {test_metrics['recall']:.4f} |
| F1-score | {test_metrics['f1']:.4f} |
| ROC-AUC | {test_metrics['roc_auc']:.4f} |
| Average precision | {test_metrics['average_precision']:.4f} |

### Confusion matrix

| | Predicted no blackout | Predicted blackout |
|---|---:|---:|
| Actual no blackout | {tn} | {fp} |
| Actual blackout | {fn} | {tp} |

- False positives: **{fp}**
- False negatives: **{fn}**

## 6. Error analysis

Test errors are available in `test_errors.csv`.

### Errors by source

```text
{source_analysis.to_string(index=False, float_format=lambda x: f"{x:.4f}")}
```

Feature importances are saved in `feature_importance.csv`.

## 7. Results synthesis

The Random Forest was tuned using recall as the cross-validation scoring metric.
The final threshold was fixed at 0.50 according to the team's decision.

The test metrics provide the final estimate of performance on the reserved test partition.
Recall is especially important because false negatives correspond to missed blackout cases.
Precision is also important because false positives correspond to unnecessary alerts.

## 8. Limitations

The initial outer split is stratified but not group-aware, following Anais's notebook.
Therefore, identical bus configurations may occur in both the training pool and test set.
This limits claims about generalization to completely unseen configurations.

The dataset contains simulated scenarios. These results should not be interpreted as
proof of real-world grid performance.
"""

    (OUTPUT_DIR / "evaluation_report.md").write_text(
        report,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 14. FINAL CONSOLE SUMMARY
    # ---------------------------------------------------------

    print("\n========== FINAL TEST RESULTS ==========")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        print(f"{metric:20s}: {test_metrics[metric]:.4f}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(test_metrics["confusion_matrix"])

    print("False positives:", fp)
    print("False negatives:", fn)

    print("\nEvaluation report:", OUTPUT_DIR / "evaluation_report.md")
    print("Results JSON:", OUTPUT_DIR / "evaluation_results.json")
    print("Figures:", FIGURE_DIR)


if __name__ == "__main__":
    main()
