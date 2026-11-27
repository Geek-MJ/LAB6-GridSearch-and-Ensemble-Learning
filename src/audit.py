"""Dataset checks and split design for the GridGuard classification project.

Run from the repository root: python -m src.audit
The raw data are left unchanged; no models are trained here.
"""
from __future__ import annotations
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "experiment_results"
SEED = 2026
TARGET = "blackout"


def read_inputs():
    small_file = RAW / "Blackout_PowerGraph_1000.csv"
    if not small_file.exists():
        small_file = ROOT / "data" / "Blackout_PowerGraph_1000.csv"
    archive = RAW / "Blackout_PowerGraph_19000.zip"
    with zipfile.ZipFile(archive) as z:
        large = pd.read_csv(z.open("Blackout_PowerGraph_19000.csv"))
        readme = z.read("README.txt").decode("utf-8")
        metadata = json.loads(z.read("source_indices.json"))
    small = pd.read_csv(small_file)
    assert small.columns.equals(large.columns), "Schemas differ; inspect source data."
    assert TARGET in small.columns
    assert small[TARGET].isin([0, 1]).all() and large[TARGET].isin([0, 1]).all()
    assert len(metadata["source_scenario_indices_zero_based"]) == len(large)
    # Only the large extract provides original scenario indices.
    small.insert(0, "record_id", [f"small_{i:05d}" for i in range(len(small))])
    large.insert(0, "record_id", [f"large_{i:05d}" for i in range(len(large))])
    small.insert(1, "source_extract", "1000_csv")
    large.insert(1, "source_extract", "19000_zip")
    small.insert(2, "source_scenario_index", -1)
    large.insert(2, "source_scenario_index", metadata["source_scenario_indices_zero_based"])
    combined = pd.concat([small, large], ignore_index=True)
    feature_names = [c for c in small.columns if c not in
                     ["record_id", "source_extract", "source_scenario_index", TARGET]]
    return combined, feature_names, readme, metadata


def make_groups(data, features):
    # Identical complete feature vectors belong to one group.
    hashes = pd.util.hash_pandas_object(data[features], index=False).to_numpy(dtype="uint64")
    labels, _ = pd.factorize(hashes, sort=False)
    return labels, hashes


def choose_splits(data, groups):
    # Use group-disjoint splits; the target is used only to balance proportions.
    target = data[TARGET].to_numpy()
    n = len(data)
    all_idx = np.arange(n)
    best = None
    for i in range(120):
        first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED + i)
        train, hold = next(first.split(all_idx, target, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.50,
                                   random_state=SEED + 1000 + i)
        val_rel, test_rel = next(second.split(hold, target[hold], groups[hold]))
        val, test = hold[val_rel], hold[test_rel]
        parts = (train, val, test)
        frac = np.array([len(p) / n for p in parts])
        rate = np.array([target[p].mean() for p in parts])
        cost = np.square(frac - [0.70, 0.15, 0.15]).sum() * 10
        cost += np.square(rate - target.mean()).sum()
        if best is None or cost < best[0]:
            best = (cost, parts, i)
    train, val, test = best[1]
    assert set(groups[train]).isdisjoint(groups[val])
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(groups[val]).isdisjoint(groups[test])
    assert len(np.unique(np.concatenate([train, val, test]))) == n
    return {"train": train, "validation": val, "test": test}, best[2]


def feature_family(name):
    if name.startswith("bus_"):
        return "bus_" + name.split("_", 2)[2]
    if name.startswith("line_"):
        return "line_" + name.split("_", 2)[2]
    return "other"


def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=190, bbox_inches="tight")
    plt.close()


def plot_eda(df, features, group_ids):
    plt.figure(figsize=(7.6, 4.1))
    ct = df.groupby(["source_extract", TARGET]).size().unstack(fill_value=0)
    ct.rename(columns={0: "Supplied (0)", 1: "Unmet demand (1)"}).plot(
        kind="bar", ax=plt.gca(), rot=0)
    plt.title("Outcomes in the supplied extracts")
    plt.ylabel("Scenarios")
    save_plot("01_target_by_extract.png")

    fam = pd.Series([feature_family(c) for c in features]).value_counts().sort_values(ascending=False)
    plt.figure(figsize=(9, 4.5))
    fam.plot(kind="bar")
    plt.title("Input variables by engineering feature family")
    plt.ylabel("Number of variables")
    plt.xticks(rotation=40, ha="right")
    save_plot("02_feature_families.png")

    counts = pd.Series(group_ids).value_counts()
    plt.figure(figsize=(7.2, 4))
    counts.value_counts().sort_index().plot(kind="bar")
    plt.title("Identical input-vector group sizes")
    plt.xlabel("Records with the same feature vector")
    plt.ylabel("Number of groups")
    plt.xticks(rotation=0)
    save_plot("03_duplicate_group_sizes.png")

    # These class differences are descriptive, not feature-selection scores.
    y = df[TARGET].to_numpy()
    mean0 = df.loc[y == 0, features].mean()
    mean1 = df.loc[y == 1, features].mean()
    scale = df[features].std(ddof=0).replace(0, np.nan)
    effects = ((mean1 - mean0) / scale).abs().sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 5.5))
    effects.iloc[::-1].plot(kind="barh")
    plt.xlabel("Absolute standardized class-mean difference")
    plt.title("Largest univariate group differences (descriptive only)")
    save_plot("04_univariate_differences.png")

    top_var = df[features].var().nlargest(25).index
    corr = df[top_var].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(top_var)), top_var, rotation=90, fontsize=6)
    plt.yticks(range(len(top_var)), top_var, fontsize=6)
    plt.colorbar(label="Pearson correlation")
    plt.title("Correlations among 25 highest-variance inputs")
    save_plot("05_feature_correlations.png")
    return effects


def main():
    for folder in (SPLITS, FIGURES, REPORTS):
        folder.mkdir(exist_ok=True, parents=True)
    df, features, readme, meta = read_inputs()
    group_ids, hashes = make_groups(df, features)
    partitions, choice = choose_splits(df, group_ids)
    dup = df.duplicated(subset=features, keep="first")
    duplicated_all = df.duplicated(subset=features + [TARGET]).sum()
    conflicts = df.groupby(group_ids)[TARGET].nunique().gt(1).sum()
    overlap = len(pd.merge(
        df[df.source_extract == "1000_csv"][features].drop_duplicates(),
        df[df.source_extract == "19000_zip"][features].drop_duplicates(),
        on=features))
    zero_missing = int(df[features + [TARGET]].isna().sum().sum())
    nonfinite = int((~np.isfinite(df[features].to_numpy())).sum())
    const = [f for f in features if df[f].nunique(dropna=False) == 1]
    group_counts = pd.Series([feature_family(f) for f in features]).value_counts().to_dict()
    source_indices = df.loc[df.source_extract == "19000_zip", "source_scenario_index"]
    source_unique = source_indices.is_unique
    for name, idx in partitions.items():
        x = df.iloc[idx][["record_id", "source_extract", "source_scenario_index", TARGET]].copy()
        x["input_hash_uint64"] = hashes[idx].astype(str)
        x.to_csv(SPLITS / f"{name}_manifest.csv", index=False)
    membership = np.empty(len(df), dtype=object)
    for name, idx in partitions.items():
        membership[idx] = name
    df[["record_id", "source_extract", "source_scenario_index", TARGET]].assign(
        split=membership, input_hash_uint64=hashes.astype(str)
    ).to_csv(SPLITS / "all_records_manifest.csv", index=False)
    effects = plot_eda(df, features, group_ids)
    split_report = {
        name: {
            "rows": len(idx),
            "share": round(len(idx) / len(df), 5),
            "positive_count": int(df.iloc[idx][TARGET].sum()),
            "positive_rate": round(float(df.iloc[idx][TARGET].mean()), 5),
            "unique_feature_groups": int(len(set(group_ids[idx])))
        } for name, idx in partitions.items()
    }
    checks = {
        "rows": len(df), "columns_in_source": len(features) + 1,
        "input_features": len(features),
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "positive_rate": float(df[TARGET].mean()),
        "feature_families": group_counts,
        "missing_cells": zero_missing,
        "nonfinite_input_cells": nonfinite,
        "constant_columns": const,
        "unique_feature_vectors": int(len(np.unique(group_ids))),
        "repeated_records": int(dup.sum()),
        "exact_duplicate_full_records": int(duplicated_all),
        "conflicting_labels_for_identical_inputs": int(conflicts),
        "shared_distinct_feature_vectors_between_extracts": overlap,
        "source_index_unique_in_19000": bool(source_unique),
        "source_index_available_in_1000": False,
        "split_choice_seed_offset": int(choice),
        "split": split_report,
        "top_univariate_differences": {k: round(float(v), 5) for k, v in effects.items()},
        "known_source_url": meta["source"],
        "source_archive_file_id": meta["source_archive_file_id"]
    }
    (REPORTS / "audit_metrics.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    pd.DataFrame({
        "feature": features,
        "family": [feature_family(x) for x in features],
        "missing": df[features].isna().sum().to_numpy(),
        "unique_values": df[features].nunique().to_numpy(),
        "std": df[features].std().to_numpy()
    }).to_csv(REPORTS / "feature_inventory.csv", index=False)
    print(json.dumps({k: v for k, v in checks.items() if k not in
                      ["top_univariate_differences", "feature_families"]}, indent=2))
    return checks


if __name__ == "__main__":
    main()
