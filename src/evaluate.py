
"""Full evaluation protocol for the GridGuard PowerGraph dataset.

Run from the repository root:
    python -m src.evaluate

Uses the official train/validation/test assignments.
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
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "validated_20000.csv.gz"
MANIFEST_PATH = ROOT / "data" / "splits" / "all_records_manifest.csv"
SPLIT_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "reports" / "evaluation"
FIGURE_DIR = ROOT / "figures" / "evaluation"

TARGET = "blackout"
RANDOM_STATE = 42


def load_and_validate_data():
    """Load data and validate the official manifests."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing processed dataset: {DATA_PATH}")

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing official manifest: {MANIFEST_PATH}")

    # IMPORTANT:
    # round_trip preserves the exact floating-point values
    # used to generate the official feature hashes.
    data = pd.read_csv(
        DATA_PATH,
        float_precision="round_trip"
    )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"input_hash_uint64": "string"}
    )

    required = {
        "record_id",
        "source_extract",
        "source_scenario_index",
        TARGET,
        "split",
        "input_hash_uint64",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}"
        )

    if TARGET not in data.columns:
        raise ValueError(
            f"Dataset has no target column '{TARGET}'."
        )

    if len(data) != 20000 or len(manifest) != len(data):
        raise ValueError(
            f"Expected 20,000 aligned records; "
            f"got data={len(data)}, manifest={len(manifest)}."
        )

    if not manifest["record_id"].is_unique:
        raise ValueError(
            "Duplicate record_id values in all_records_manifest.csv."
        )

    # Check target-label alignment.
    if not manifest[TARGET].reset_index(drop=True).equals(
        data[TARGET].reset_index(drop=True)
    ):
        raise ValueError(
            "Manifest labels do not match dataset row order. "
            "Stop: do not train."
        )

    if set(manifest["split"].unique()) != {
        "train", "validation", "test"
    }:
        raise ValueError(
            "Unexpected or missing split labels in the full manifest."
        )

    # Verify that partition files match the full manifest.
    for split in ("train", "validation", "test"):

        part_path = SPLIT_DIR / f"{split}_manifest.csv"

        if not part_path.is_file():
            raise FileNotFoundError(
                f"Missing split file: {part_path}"
            )

        part = pd.read_csv(
            part_path,
            dtype={"input_hash_uint64": "string"}
        )

        expected_ids = set(
            manifest.loc[
                manifest["split"] == split,
                "record_id"
            ]
        )

        actual_ids = set(part["record_id"])

        if (
            actual_ids != expected_ids
            or len(part) != len(expected_ids)
        ):
            raise ValueError(
                f"{split}_manifest.csv does not match "
                "all_records_manifest.csv."
            )

        if not part["record_id"].is_unique:
            raise ValueError(
                f"Duplicate IDs in {part_path.name}."
            )

    # Identical feature vectors must not cross partitions.
    split_count_by_hash = (
        manifest.groupby("input_hash_uint64")["split"].nunique()
    )

    if split_count_by_hash.max() != 1:
        raise ValueError(
            "Identical feature vectors cross data partitions."
        )

    # Identify the 224 input features.
    features = [
        column for column in data.columns
        if column != TARGET
    ]

    if len(features) != 224:
        raise ValueError(
            f"Expected 224 input features; found {len(features)}."
        )

    # Verify exact feature-row alignment.
    feature_hashes = pd.util.hash_pandas_object(
        data[features],
        index=False
    ).astype(str)

    if not (
        manifest["input_hash_uint64"].astype(str)
        == feature_hashes
    ).all():
        raise ValueError(
            "Manifest feature hashes do not match dataset row order. "
            "Stop: do not train."
        )

    if data[features].isna().to_numpy().any():
        raise ValueError("Missing feature values detected.")

    if not np.isfinite(
        data[features].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Infinite or non-numeric feature values detected."
        )

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    print("Data validation successful.")
    print("All 20,000 feature hashes match the official manifest.")

    return data, manifest, features


def metrics_at_threshold(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            fbeta_score(
                y_true,
                predictions,
                beta=1,
                zero_division=0
            )
        ),

        "f2": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),

        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            predictions
        ).tolist(),
    }


def choose_f2_threshold(y_true, probabilities):
    """Choose threshold maximizing F2 on validation data only."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    if len(thresholds) == 0:
        return 0.5, 0.0

    p = precision[:-1]
    r = recall[:-1]

    f2 = np.divide(
        5 * p * r,
        4 * p + r,
        out=np.zeros_like(p),
        where=(4 * p + r) != 0
    )

    best_index = int(np.argmax(f2))

    return (
        float(thresholds[best_index]),
        float(f2[best_index])
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load and validate the dataset.
    data, manifest, features = load_and_validate_data()

    X = data[features]
    y = data[TARGET].astype(int)

    split = manifest["split"]

    train_idx = split.eq("train")
    validation_idx = split.eq("validation")
    test_idx = split.eq("test")

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    X_val = X.loc[validation_idx]
    y_val = y.loc[validation_idx]

    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    test_manifest = (
        manifest.loc[test_idx].reset_index(drop=True)
    )

    # Display official split information.
    print("\n========== OFFICIAL DATA SPLITS ==========")

    for name, mask in (
        ("train", train_idx),
        ("validation", validation_idx),
        ("test", test_idx),
    ):

        labels = y.loc[mask]

        print(
            f"{name:10s}: {len(labels):5d} rows | "
            f"blackout rate={labels.mean():.3%} | "
            f"class counts="
            f"{labels.value_counts().sort_index().to_dict()}"
        )

    # Define the candidate models.
    models = {

        "Dummy_prior": DummyClassifier(
            strategy="prior"
        ),

        "Decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=5
        ),

        "Logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        ),

        "Random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    fitted_models = {}
    validation_probabilities = {}
    comparison_rows = []

    # Train models using training data only.
    for name, model in models.items():

        print(f"\nTraining {name} ...")

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        fitted_models[name] = model
        validation_probabilities[name] = probabilities

        comparison_rows.append({

            "model": name,

            "validation_average_precision":
                average_precision_score(
                    y_val,
                    probabilities
                ),

            "validation_roc_auc":
                roc_auc_score(
                    y_val,
                    probabilities
                ),

            "validation_recall_at_0_5":
                recall_score(
                    y_val,
                    (probabilities >= 0.5).astype(int),
                    zero_division=0
                ),

            "validation_precision_at_0_5":
                precision_score(
                    y_val,
                    (probabilities >= 0.5).astype(int),
                    zero_division=0
                ),
        })

        print(
            f"Validation AP="
            f"{comparison_rows[-1]['validation_average_precision']:.4f}, "
            f"ROC-AUC="
            f"{comparison_rows[-1]['validation_roc_auc']:.4f}"
        )

    # Compare models using validation average precision.
    comparison = pd.DataFrame(
        comparison_rows
    ).sort_values(
        "validation_average_precision",
        ascending=False
    )

    comparison.to_csv(
        OUTPUT_DIR / "validation_model_comparison.csv",
        index=False
    )

    # Select the best model using validation data only.
    selected_name = str(
        comparison.iloc[0]["model"]
    )

    selected_model = fitted_models[selected_name]

    selected_val_prob = validation_probabilities[
        selected_name
    ]

    # Select threshold using validation F2.
    selected_threshold, validation_best_f2 = (
        choose_f2_threshold(
            y_val,
            selected_val_prob
        )
    )

    validation_metrics = metrics_at_threshold(
        y_val,
        selected_val_prob,
        selected_threshold
    )

    print("\n========== MODEL SELECTION ==========")

    print(f"Selected model: {selected_name}")

    print(
        "Validation-selected threshold (max F2): "
        f"{selected_threshold:.6f}"
    )

    print(
        "Validation F2 at selected threshold: "
        f"{validation_best_f2:.4f}"
    )

    # Final test evaluation.
    # The model and threshold are now frozen.
    print("\n========== FINAL TEST EVALUATION ==========")

    test_prob = selected_model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = metrics_at_threshold(
        y_test,
        test_prob,
        selected_threshold
    )

    test_predictions = (
        test_prob >= selected_threshold
    ).astype(int)

    # Save individual test predictions.
    test_results = test_manifest[
        [
            "record_id",
            "source_extract",
            "source_scenario_index",
            TARGET
        ]
    ].copy()

    test_results["blackout_probability"] = test_prob

    test_results["predicted_blackout"] = test_predictions

    test_results["error_type"] = "correct"

    test_results.loc[
        (test_results[TARGET] == 1)
        & (test_predictions == 0),
        "error_type"
    ] = "false_negative"

    test_results.loc[
        (test_results[TARGET] == 0)
        & (test_predictions == 1),
        "error_type"
    ] = "false_positive"

    test_results.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    # Export only incorrect predictions.
    test_results.loc[
        test_results["error_type"] != "correct"
    ].to_csv(
        OUTPUT_DIR / "test_errors.csv",
        index=False
    )

    # Save the selected model and threshold.
    joblib.dump(
        {
            "model": selected_model,
            "features": features,
            "threshold": selected_threshold,
            "selected_model_name": selected_name,
            "selection_metric": "validation average precision",
            "threshold_metric": "validation F2",
        },
        OUTPUT_DIR / "selected_model.joblib"
    )

    # Validation precision-recall curves.
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, probabilities in validation_probabilities.items():

        PrecisionRecallDisplay.from_predictions(
            y_val,
            probabilities,
            name=name,
            ax=ax
        )

    ax.set_title("Validation precision-recall curves")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "validation_precision_recall.png",
        dpi=160
    )

    plt.close(fig)

    # Validation ROC curve.
    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_val,
        selected_val_prob,
        ax=ax
    )

    ax.set_title(
        f"Validation ROC curve — {selected_name}"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "validation_roc_curve.png",
        dpi=160
    )

    plt.close(fig)

    # Final test confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        display_labels=[
            "No blackout",
            "Blackout"
        ],
        values_format="d",
        ax=ax
    )

    ax.set_title(
        "Final test confusion matrix — "
        f"threshold {selected_threshold:.3f}"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "test_confusion_matrix.png",
        dpi=160
    )

    plt.close(fig)

    # Final test precision-recall curve.
    fig, ax = plt.subplots(figsize=(8, 6))

    PrecisionRecallDisplay.from_predictions(
        y_test,
        test_prob,
        name=f"{selected_name} (test)",
        ax=ax
    )

    ax.set_title("Final test precision-recall curve")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "test_precision_recall.png",
        dpi=160
    )

    plt.close(fig)

    # Prepare complete results.
    results = {

        "dataset": (
            "PowerGraph IEEE-24 simulated scenarios"
        ),

        "rows": int(len(data)),

        "input_features": int(len(features)),

        "target": TARGET,

        "split_sizes": {
            "train": int(train_idx.sum()),
            "validation": int(validation_idx.sum()),
            "test": int(test_idx.sum()),
        },

        "target_counts": {
            str(k): int(v)
            for k, v in y.value_counts().sort_index().items()
        },

        "model_selection": {

            "criterion": (
                "highest validation average precision"
            ),

            "selected_model": selected_name,

            "validation_model_comparison":
                comparison.to_dict(orient="records"),
        },

        "threshold_selection": {

            "criterion": "maximum validation F2",

            "threshold": selected_threshold,

            "validation_metrics_at_selected_threshold":
                validation_metrics,
        },

        "final_test_metrics": test_metrics,

        "test_error_counts": {

            "false_positives": int(
                (
                    test_results["error_type"]
                    == "false_positive"
                ).sum()
            ),

            "false_negatives": int(
                (
                    test_results["error_type"]
                    == "false_negative"
                ).sum()
            ),

            "correct": int(
                (
                    test_results["error_type"]
                    == "correct"
                ).sum()
            ),
        },

        "caveats": [
            "All model fitting used training data only.",
            "Model selection and threshold selection used validation data only.",
            "The test set was evaluated after the model and threshold were frozen.",
            "Records are simulated IEEE-24 scenarios, not real-world grid events.",
            "Source documentation does not establish pre-outcome timing for every input feature.",
        ],
    }

    (OUTPUT_DIR / "evaluation_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    # Generate the final Markdown report.
    cm = np.array(
        test_metrics["confusion_matrix"]
    )

    tn, fp, fn, tp = cm.ravel()

    report = f"""# Full-dataset evaluation — GridGuard

## Protocol

- Dataset: 20,000 validated PowerGraph IEEE-24 simulated records.
- Inputs: 224 numerical features.
- Official split: train {int(train_idx.sum())}, validation {int(validation_idx.sum())}, test {int(test_idx.sum())}.
- The official all_records_manifest.csv was used.
- The three partition manifests were checked against it.
- Model selection: highest validation average precision.
- Threshold selection: maximum F2 on validation data.
- F2 gives recall more weight than precision.
- The test set was not used for model selection or threshold tuning.

## Validation model comparison

```text
{comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}")}
```

## Selected model and threshold

- Model: **{selected_name}**
- Threshold: **{selected_threshold:.6f}**
- Validation F2: **{validation_best_f2:.4f}**

## Final test results

| Metric | Test result |
|---|---:|
| Accuracy | {test_metrics['accuracy']:.4f} |
| Precision (blackout) | {test_metrics['precision']:.4f} |
| Recall (blackout) | {test_metrics['recall']:.4f} |
| F1 (blackout) | {test_metrics['f1']:.4f} |
| F2 (blackout) | {test_metrics['f2']:.4f} |
| ROC-AUC | {test_metrics['roc_auc']:.4f} |
| Average precision | {test_metrics['average_precision']:.4f} |

### Confusion matrix

| | Predicted no blackout | Predicted blackout |
|---|---:|---:|
| Actual no blackout | {tn} (TN) | {fp} (FP) |
| Actual blackout | {fn} (FN) | {tp} (TP) |

- False positives: **{fp}**
- False negatives: **{fn}**

A false negative is an actual blackout case that the model misses.
A false positive is a predicted blackout where the actual label is 0.

## Error analysis

Per-record test predictions are saved in `test_predictions.csv`.

Misclassified test records are saved in `test_errors.csv`,
including their record IDs, true labels, predicted probabilities
and error type.

## Figures

- `figures/evaluation/validation_precision_recall.png`
- `figures/evaluation/validation_roc_curve.png`
- `figures/evaluation/test_confusion_matrix.png`
- `figures/evaluation/test_precision_recall.png`

## Limitations

The dataset contains simulated IEEE-24 scenarios.
Results should not be interpreted as validated real-world grid performance.

The project cleaning report also notes that the source documentation
does not establish pre-outcome timing for every input feature.
Possible temporal leakage therefore remains a limitation to investigate.
"""

    (OUTPUT_DIR / "evaluation_report.md").write_text(
        report,
        encoding="utf-8"
    )

    # Print final results.
    print("\n========== FINAL TEST RESULTS ==========")

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "average_precision",
    ):

        print(
            f"{key:20s}: {test_metrics[key]:.4f}"
        )

    print(
        "\nConfusion matrix [[TN, FP], [FN, TP]]: "
        f"{cm.tolist()}"
    )

    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")

    print(f"\nSaved outputs in: {OUTPUT_DIR}")
    print(f"Saved figures in: {FIGURE_DIR}")

    print(
        "\nRead reports/evaluation/evaluation_report.md "
        "for the synthesis."
    )


if __name__ == "__main__":
    main()
