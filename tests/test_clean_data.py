"""Regression checks for cleaned PowerGraph data and split integrity."""
import pandas as pd
from src.clean_data import ROOT, check_manifest, load_sources


def test_source_schema_and_values():
    data, features, _, _ = load_sources(ROOT)
    assert data.shape == (20_000, 225)
    assert len(features) == 224
    assert data.isna().sum().sum() == 0
    assert data.blackout.value_counts().to_dict() == {0: 15968, 1: 4032}


def test_duplicate_counts_and_labels():
    data, features, _, _ = load_sources(ROOT)
    manifest = check_manifest(ROOT, data, features)
    unique = data.drop_duplicates(subset=features)
    assert len(unique) == 18_579
    assert unique.blackout.value_counts().to_dict() == {0: 14859, 1: 3720}
    assert manifest.groupby("input_hash_uint64").blackout.nunique().max() == 1
    assert manifest.groupby("input_hash_uint64")["split"].nunique().max() == 1


def test_cleaning_keeps_original_partitions():
    data, features, _, _ = load_sources(ROOT)
    manifest = check_manifest(ROOT, data, features)
    kept = ~data.duplicated(subset=features)
    assert manifest.loc[kept, "split"].value_counts().to_dict() == {
        "train": 13005, "validation": 2787, "test": 2787
    }
