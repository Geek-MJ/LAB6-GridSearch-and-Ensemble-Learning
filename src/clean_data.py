"""Validate the PowerGraph extracts and export clean, reproducible datasets.

Run from the repository root: python -m src.clean_data
The validated export preserves every record and feature. Duplicates are
reported rather than silently removed. Scaling and feature selection
belong inside the model training pipeline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "blackout"
EXPECTED_FEATURES = 224
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources(root: Path):
    small_file = root / "data" / "Blackout_PowerGraph_1000.csv"
    archive_file = root / "data" / "raw" / "Blackout_PowerGraph_19000.zip"
    if not small_file.is_file() or not archive_file.is_file():
        raise FileNotFoundError("The 1,000-row CSV or 19,000-row ZIP is missing.")
    small = pd.read_csv(small_file)
    with zipfile.ZipFile(archive_file) as archive:
        large = pd.read_csv(archive.open("Blackout_PowerGraph_19000.csv"))
        metadata = json.loads(archive.read("source_indices.json"))
    if not small.columns.equals(large.columns):
        raise ValueError("The source schemas differ.")
    if not small.columns.is_unique or TARGET not in small or len(small.columns) != EXPECTED_FEATURES + 1:
        raise ValueError("Unexpected feature schema.")
    if len(small) != 1000 or len(large) != 19000:
        raise ValueError("Unexpected number of source rows.")
    features = [col for col in small if col != TARGET]
    data = pd.concat((small, large), ignore_index=True)
    if not all(pd.api.types.is_numeric_dtype(data[col]) for col in data):
        raise ValueError("Non-numeric source column detected.")
    if data.isna().to_numpy().any():
        raise ValueError("Missing values found: inspect the source rather than imputing without review.")
    if not np.isfinite(data[features].to_numpy()).all():
        raise ValueError("Non-finite input values found.")
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("Invalid target labels.")
    if len(metadata["source_scenario_indices_zero_based"]) != len(large):
        raise ValueError("Incorrect source scenario-index mapping.")
    return data, features, small_file, archive_file


def check_manifest(root: Path, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    path = root / "data" / "splits" / "all_records_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError("Run the original audit to create the split manifest first.")
    manifest = pd.read_csv(path, dtype={"input_hash_uint64": "string"})
    needed = {"record_id", "source_extract", "source_scenario_index", TARGET, "split", "input_hash_uint64"}
    if not needed.issubset(manifest.columns):
        raise ValueError("Split manifest is missing required fields.")
    expected_ids = ([f"small_{i:05d}" for i in range(1000)] +
                    [f"large_{i:05d}" for i in range(19000)])
    if manifest["record_id"].tolist() != expected_ids or manifest["record_id"].duplicated().any():
        raise ValueError("Manifest record IDs do not match source row order.")
    if not manifest[TARGET].equals(data[TARGET]):
        raise ValueError("Manifest target labels differ from source labels.")
    if not manifest["split"].isin(SPLITS).all():
        raise ValueError("Unexpected partition name.")
    hashes = pd.util.hash_pandas_object(data[features], index=False).astype(str)
    if not (manifest["input_hash_uint64"] == hashes).all():
        raise ValueError("Feature hashes differ from the original data audit.")
    if manifest.groupby("input_hash_uint64")["split"].nunique().max() != 1:
        raise ValueError("Identical feature vectors cross data partitions.")
    if data.groupby(manifest["input_hash_uint64"])[TARGET].nunique().max() != 1:
        raise ValueError("Identical input vectors have conflicting labels.")
    return manifest


def clean(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    data, features, small_file, archive_file = load_sources(root)
    manifest = check_manifest(root, data, features)

    # Count repeated inputs for the audit. Retain all original records.
    unique_mask = ~data.duplicated(subset=features, keep="first")

    # Identical columns are documented, not silently removed (feature selection comes later).
    first_for_hash = {}
    equal_columns = []
    for name in features:
        key = int(pd.util.hash_pandas_object(data[name], index=False).sum())
        if key in first_for_hash and data[name].equals(data[first_for_hash[key]]):
            equal_columns.append({"equivalent_to": first_for_hash[key], "column": name})
        else:
            first_for_hash[key] = name

    counts = data[TARGET].value_counts().sort_index()
    unique_counts = data.loc[unique_mask, TARGET].value_counts().sort_index()
    summary = {
        "source": "PowerGraph IEEE-24 simulated scenarios",
        "original_rows": len(data),
        "original_target_counts": {str(k): int(v) for k, v in counts.items()},
        "input_features_retained": len(features),
        "missing_cells": int(data.isna().sum().sum()),
        "infinite_input_cells": int((~np.isfinite(data[features].to_numpy())).sum()),
        "constant_columns": [col for col in features if data[col].nunique(dropna=False) == 1],
        "exact_duplicate_feature_rows": int((~unique_mask).sum()),
        "conflicting_duplicate_labels": 0,
        "identical_columns": equal_columns,
        "unique_input_patterns": int(unique_mask.sum()),
        "unique_target_counts": {str(k): int(v) for k, v in unique_counts.items()},
        "original_split_rows": {part: int((manifest["split"] == part).sum()) for part in SPLITS},
        "raw_file_sha256": {small_file.name: sha256(small_file), archive_file.name: sha256(archive_file)},
        "notes": [
            "No imputation, outlier clipping, scaling, or feature selection was performed.",
            "The validated export retains all records, including repeated simulation inputs.",
            "Existing audit split assignments are preserved and checked.",
            "The 224 input columns are retained for training-only feature selection.",
            "Source documentation does not establish pre-outcome timing for all inputs."
        ]
    }
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    gzip_options = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    full_path = processed / "validated_20000.csv.gz"
    data.to_csv(full_path, index=False, float_format="%.17g", compression=gzip_options)
    summary["processed_file_sha256"] = {full_path.name: sha256(full_path)}
    (processed / "cleaning_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(clean(), indent=2))
