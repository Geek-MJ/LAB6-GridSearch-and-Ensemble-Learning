"""Small tests for the shipped audit pipeline. Full-data checks require both extracts."""
import numpy as np
import pandas as pd
from src.audit import select_partitions

def test_groups_do_not_cross_splits():
    groups = np.repeat(np.arange(40), 3)
    frame = pd.DataFrame({"blackout": (groups % 5 == 0).astype(int)})
    splits, _ = select_partitions(frame, groups)
    seen = [set(groups[splits[k]]) for k in ("train", "validation", "test")]
    assert all(a.isdisjoint(b) for i, a in enumerate(seen) for b in seen[i+1:])
    assert sum(len(x) for x in splits.values()) == len(frame)

def test_reproducible_partition():
    groups = np.repeat(np.arange(40), 3)
    frame = pd.DataFrame({"blackout": (groups % 5 == 0).astype(int)})
    one, _ = select_partitions(frame, groups)
    two, _ = select_partitions(frame, groups)
    assert all(np.array_equal(one[k], two[k]) for k in one)
