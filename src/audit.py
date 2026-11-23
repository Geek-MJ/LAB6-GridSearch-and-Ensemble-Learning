"""GridGuard data audit and reproducible group-based splits.

Run: python -m src.audit
Data are simulated. Nothing here trains or tunes a predictive model.
"""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SPLITS = DATA / "splits"
REPORTS = ROOT / "reports"
TARGET = "blackout"
SEED = 2026

def load_data():
    small_path = RAW / "Blackout_PowerGraph_1000.csv"
    if not small_path.exists():
        small_path = DATA / "Blackout_PowerGraph_1000.csv"
    large_path = RAW / "Blackout_PowerGraph_19000.zip"
    if not large_path.exists():
        raise FileNotFoundError(
            "Place Blackout_PowerGraph_19000.zip in data/raw/ (original supplied project archive)."
        )
    small = pd.read_csv(small_path)
    with zipfile.ZipFile(large_path) as archive:
        large = pd.read_csv(archive.open("Blackout_PowerGraph_19000.csv"))
        meta = json.loads(archive.read("source_indices.json"))
    assert small.columns.equals(large.columns), "The two source schemas differ"
    assert TARGET in small.columns and len(small.columns) == 225
    assert small[TARGET].isin([0, 1]).all() and large[TARGET].isin([0, 1]).all()
    assert len(meta["source_scenario_indices_zero_based"]) == len(large)
    small.insert(0, "record_id", [f"small_{i:05d}" for i in range(len(small))])
    large.insert(0, "record_id", [f"large_{i:05d}" for i in range(len(large))])
    small.insert(1, "source_extract", "1000_csv")
    large.insert(1, "source_extract", "19000_zip")
    small.insert(2, "source_scenario_index", -1)
    large.insert(2, "source_scenario_index", meta["source_scenario_indices_zero_based"])
    features = [c for c in small if c not in {
        "record_id", "source_extract", "source_scenario_index", TARGET
    }]
    return pd.concat([small, large], ignore_index=True), features

def select_partitions(frame, group_ids):
    """Keep identical complete input vectors within one partition."""
    y = frame[TARGET].to_numpy()
    idx = np.arange(len(frame))
    best = None
    for i in range(120):
        first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED + i)
        train, hold = next(first.split(idx, y, group_ids))
        second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 1000 + i)
        v_rel, t_rel = next(second.split(hold, y[hold], group_ids[hold]))
        val, test = hold[v_rel], hold[t_rel]
        partitions = {"train": train, "validation": val, "test": test}
        sizes = np.array([len(train), len(val), len(test)]) / len(frame)
        rates = np.array([y[k].mean() for k in (train, val, test)])
        score = 10 * np.square(sizes - [0.70, 0.15, 0.15]).sum()
        score += np.square(rates - y.mean()).sum()
        if best is None or score < best[0]:
            best = score, partitions, i
    return best[1], best[2]

def audit(save=True):
    frame, features = load_data()
    hashes = pd.util.hash_pandas_object(frame[features], index=False).to_numpy(dtype="uint64")
    groups, _ = pd.factorize(hashes)
    parts, seed_offset = select_partitions(frame, groups)
    sets = [set(groups[parts[k]]) for k in ("train", "validation", "test")]
    assert all(a.isdisjoint(b) for i, a in enumerate(sets) for b in sets[i + 1:])
    assert sum(len(x) for x in parts.values()) == len(frame)
    conflicts = int(frame.groupby(groups)[TARGET].nunique().gt(1).sum())
    assert conflicts == 0, "Identical input vectors have different labels"
    a = frame.loc[frame.source_extract.eq("1000_csv"), features].drop_duplicates()
    b = frame.loc[frame.source_extract.eq("19000_zip"), features].drop_duplicates()
    overlap = len(a.merge(b, on=features))
    metrics = {
        "records": len(frame),
        "input_features": len(features),
        "target_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts().sort_index().items()},
        "positive_rate": float(frame[TARGET].mean()),
        "missing_cells": int(frame[features + [TARGET]].isna().sum().sum()),
        "nonfinite_input_cells": int((~np.isfinite(frame[features].to_numpy())).sum()),
        "unique_feature_vectors": int(len(np.unique(groups))),
        "repeated_feature_records": int(frame.duplicated(subset=features).sum()),
        "shared_feature_patterns_between_extracts": int(overlap),
        "conflicting_labels": conflicts,
        "chosen_seed_offset": int(seed_offset),
        "splits": {}
    }
    membership = np.empty(len(frame), dtype=object)
    for name, rows in parts.items():
        membership[rows] = name
        metrics["splits"][name] = {
            "records": int(len(rows)), "positive": int(frame.iloc[rows][TARGET].sum()),
            "positive_rate": float(frame.iloc[rows][TARGET].mean())
        }
    if save:
        SPLITS.mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)
        frame[["record_id", "source_extract", "source_scenario_index", TARGET]].assign(
            split=membership, input_hash_uint64=hashes.astype(str)
        ).to_csv(SPLITS / "all_records_manifest.csv", index=False)
        for name in parts:
            manifest = pd.read_csv(SPLITS / "all_records_manifest.csv")
            manifest.loc[manifest.split.eq(name)].to_csv(
                SPLITS / f"{name}_manifest.csv", index=False
            )
        (REPORTS / "audit_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
