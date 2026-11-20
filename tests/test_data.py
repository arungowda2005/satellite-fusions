import numpy as np
import pandas as pd
import pytest

from src.data import make_masks, apply_mask


def test_same_seed_reproducibility():
    a = make_masks(100, "random", 0.3, 42)
    b = make_masks(100, "random", 0.3, 42)
    assert np.array_equal(a, b)


def test_different_seeds():
    a = make_masks(100, "random", 0.3, 42)
    b = make_masks(100, "random", 0.3, 43)
    assert not np.array_equal(a, b)


def test_random_missing_rate():
    masks = make_masks(3000, "random", 0.3, 42)
    missing_rate = (~masks).mean()
    assert abs(missing_rate - 0.3) < 0.02


def test_block_mask():
    masks = make_masks(100, "block", 0.5, 42)

    for mask in masks:
        # Sentinel-1 must remain available.
        assert mask[:, 0].all()

        missing = np.where(~mask[:, 1])[0]

        # Exactly six consecutive Sentinel-2 months are missing.
        assert len(missing) == 6
        assert np.all(np.diff(missing) == 1)


def test_no_s1():
    masks = make_masks(10, "no_s1", 0.3, 42)
    assert not masks[:, :, 0].any()
    assert masks[:, :, 1].all()


def test_no_s2():
    masks = make_masks(10, "no_s2", 0.3, 42)
    assert masks[:, :, 0].all()
    assert not masks[:, :, 1].any()


def test_edge_rates():
    none_missing = make_masks(10, "random", 0.0, 42)
    all_missing = make_masks(10, "random", 1.0, 42)

    assert none_missing.all()
    assert not all_missing.any()


def test_invalid_rate():
    with pytest.raises(ValueError):
        make_masks(10, "random", 1.5, 42)


def test_invalid_kind():
    with pytest.raises(ValueError):
        make_masks(10, "unknown", 0.3, 42)


def test_apply_mask():
    X = np.ones((1, 12, 13), dtype=np.float32)
    M = np.ones((1, 12, 2), dtype=bool)

    M[0, 0, 0] = False
    M[0, 1, 1] = False

    result = apply_mask(X, M)

    # One S1 month removes two bands.
    assert np.all(result[0, 0, 0:2] == 0)

    # One S2 month removes eleven bands.
    assert np.all(result[0, 1, 2:13] == 0)

    # Remaining observations are unchanged.
    assert result[0, 2:].sum() == 10 * 13
    assert result[0, 0, 2:13].sum() == 11
    assert result[0, 1, 0:2].sum() == 2


def test_split_integrity():
    df = pd.read_csv(
        "data/splits/optimized_spatial_seed42_cell1.csv"
    )

    assert df["fname"].is_unique
    assert df["split"].isin(["train", "val", "test"]).all()

    # No geographic cell can appear in multiple splits.
    assert (
        df.groupby("cell")["split"].nunique().max() == 1
    )

    # Every class must occur in all three splits.
    class_counts = pd.crosstab(
        df["classification_label"], df["split"]
    )

    assert set(class_counts.columns) == {"train", "val", "test"}
    assert (class_counts > 0).all().all()

    # Check expected total sample count.
    assert len(df) == 29749


def test_masked_sensor_bands():
    X = np.random.default_rng(42).normal(
        size=(5, 12, 13)
    ).astype(np.float32)

    M = make_masks(5, "no_s1", 0.3, 42)
    result = apply_mask(X, M)

    assert np.all(result[:, :, 0:2] == 0)
    assert np.array_equal(result[:, :, 2:13], X[:, :, 2:13])