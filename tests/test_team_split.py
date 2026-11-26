"""Check that the requested stratified split matches the team's evaluation."""
from src.split_data import make_splits

def test_split_sizes_and_labels():
    parts, report = make_splits(save=False)
    assert [len(parts[k]) for k in ("train", "validation", "test")] == [12824, 3176, 4000]
    assert [report["partitions"][k]["blackout_1"] for k in ("train", "validation", "test")] == [2565, 661, 806]
    assert report["outer"]["random_state"] == 42
    assert report["leakage_check"]["train_validation_bus_group_overlap"] == 0

def test_split_is_deterministic():
    a, _ = make_splits(save=False)
    b, _ = make_splits(save=False)
    for name in ("train", "validation", "test"):
        assert (a[name] == b[name]).all()
