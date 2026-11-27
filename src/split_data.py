"""Historical three-way grouped-bus GridGuard evaluation protocol.

This module remains an input to src.evaluate and is NOT the official team
train/test CSV reproduction. For the official two-file 80/20 procedure, use
src.team_split. This historical workflow uses 72-variable bus groups and an
approximately 70/15/15 split. No estimator or metric is used to choose a split.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "validated_20000.csv.gz"
ORIGINAL_MANIFEST = ROOT / "data" / "splits" / "all_records_manifest.csv"
OUTPUT = ROOT / "data" / "splits" / "corrected_grouped"
TARGET = "blackout"
SEED = 2026
PROTOCOL_VERSION = "grouped-bus-v1"


def group_ids(X):
    """Hash bus configurations without labels, record IDs or source IDs."""
    bus_columns = [name for name in X if name.startswith("bus_")]
    if len(bus_columns) != 72:
        raise ValueError("Expected exactly 72 bus features.")
    return pd.util.hash_pandas_object(X[bus_columns], index=False).to_numpy()


def choose_partitions(y, groups, seed=SEED, candidates=60):
    """Choose approximate 70/15/15 groups by size and class balance only."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(y) != len(groups) or len(y) == 0:
        raise ValueError("Labels and groups must be nonempty and aligned.")
    indices = np.arange(len(y))
    best = None
    for offset in range(candidates):
        outer = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed+offset)
        train, held = next(outer.split(indices, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed+1000+offset)
        validation_rel, test_rel = next(inner.split(held, y[held], groups[held]))
        validation, test = held[validation_rel], held[test_rel]
        parts = (train, validation, test)
        if any(len(np.unique(y[p])) < 2 for p in parts):
            continue
        cost = 10*np.square(np.array([len(p)/len(y) for p in parts])-[.7,.15,.15]).sum()
        cost += np.square(np.array([y[p].mean() for p in parts])-y.mean()).sum()
        if best is None or cost < best[0]:
            best = (float(cost), offset, parts)
    if best is None:
        raise ValueError("No group-disjoint partition containing both classes.")
    parts = dict(zip(("train", "validation", "test"), best[2]))
    check_isolation(parts, groups, len(y))
    return parts, best[1]


def check_isolation(parts, groups, n):
    combined = np.concatenate(list(parts.values()))
    if len(combined) != n or not np.array_equal(np.sort(combined), np.arange(n)):
        raise ValueError("Partitions must cover each row exactly once.")
    names = list(parts)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            if set(groups[parts[name]]) & set(groups[parts[other]]):
                raise ValueError(f"Group overlap: {name}/{other}")


def frame_fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def make_splits(save=True, *, data=None, original=None):
    """Reuse already loaded frames when supplied; validate source values once."""
    df = pd.read_csv(DATA) if data is None else data
    if df.shape != (20000, 225) or TARGET not in df or not df.columns.is_unique:
        raise ValueError("Expected 20,000 rows, 224 inputs and one target.")
    X = df.drop(columns=TARGET)
    y = df[TARGET].astype(int).to_numpy()
    if not np.isfinite(df.to_numpy()).all() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Nonfinite data or invalid labels.")
    if original is None:
        original = pd.read_csv(ORIGINAL_MANIFEST, dtype={"input_hash_uint64": "string"})
    if len(original) != len(df) or original.record_id.duplicated().any():
        raise ValueError("Source manifest row IDs must be unique and aligned.")
    expected_ids = [f"small_{i:05d}" for i in range(1000)] + [f"large_{i:05d}" for i in range(19000)]
    if original.record_id.tolist() != expected_ids or not np.array_equal(original[TARGET], y):
        raise ValueError("Source manifest order/labels do not match the validated data.")
    # Compare values, not pandas hashes across CSV parsers and library versions.
    from src.clean_data import load_sources
    raw, features, _, _ = load_sources(ROOT)
    if list(raw.columns) != list(df.columns) or not np.allclose(raw.to_numpy(), df.to_numpy(), rtol=1e-14, atol=1e-14):
        raise ValueError("Validated dataset values differ from original source rows.")
    groups = group_ids(X)
    parts, offset = choose_partitions(y, groups)
    full_groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    check_isolation(parts, full_groups, len(df))
    assigned = np.empty(len(df), dtype=object)
    for name, idx in parts.items():
        assigned[idx] = name
    manifest = pd.DataFrame({"row_index": np.arange(len(df)), "record_id": original.record_id,
                             "source_extract": original.source_extract, TARGET: y, "split": assigned,
                             "bus_group": groups.astype(str), "input_hash_uint64": full_groups.astype(str)})
    summary = {
        "protocol_version": PROTOCOL_VERSION, "seed": SEED, "candidate_offset": offset,
        "method": "group-disjoint approximately 70/15/15, balancing counts/labels only",
        "group_definition": "identical complete 72 bus features; no claim of physical scenario independence",
        "dataset_rows": len(df), "input_features": X.shape[1], "dataset_fingerprint": frame_fingerprint(df),
        "partitions": {name: {"records": len(idx), "blackout_1": int(y[idx].sum()),
                              "blackout_0": int(len(idx)-y[idx].sum()), "groups": len(set(groups[idx]))}
                       for name, idx in parts.items()},
        "leakage_check": {"train_validation_bus_group_overlap": 0, "train_test_bus_group_overlap": 0,
                          "validation_test_bus_group_overlap": 0, "complete_input_overlap": 0},
        "caveats": ["Retrospective internal benchmark: the dataset and earlier test scores were already examined.",
                    "A new split does not create independent external validation or erase prior analyst exposure.",
                    "The source paper documents pre-outage graph attributes; exact local flattening alignment remains unverified.",
                    "Bus grouping does not establish independence of all physical simulation families."]}
    if save:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        # Refuse silent replacement of a previously locked partition.
        saved = OUTPUT / "all_records_manifest.csv"
        if saved.exists():
            previous = pd.read_csv(saved, dtype={"bus_group": "string", "input_hash_uint64": "string"})
            if previous.record_id.tolist() != manifest.record_id.tolist() or previous.split.tolist() != manifest.split.tolist():
                raise ValueError("Locked corrected manifest differs: use an explicitly versioned new protocol.")
        manifest.to_csv(saved, index=False)
        for name in parts:
            manifest.loc[manifest.split == name].to_csv(OUTPUT / f"{name}_manifest.csv", index=False)
        (OUTPUT / "split_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return parts, summary


if __name__ == "__main__":
    _, result = make_splits()
    print(json.dumps(result, indent=2))
