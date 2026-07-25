"""Phase 5c: perturbation-function correctness, checked against synthetic
data with a known right answer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.robustness import ablate_features, inject_gaussian_noise


def _toy_features(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "record_id": [f"r{i}" for i in range(n)],
        "feat_a": rng.normal(10.0, 2.0, size=n),
        "feat_b": rng.normal(0.0, 1.0, size=n),
    })


def test_zero_noise_fraction_is_identity():
    features = _toy_features()
    stds = features[["feat_a", "feat_b"]].std()

    out = inject_gaussian_noise(features, ["feat_a", "feat_b"], stds, noise_fraction=0.0, seed=1)

    pd.testing.assert_series_equal(out["feat_a"], features["feat_a"])
    pd.testing.assert_series_equal(out["feat_b"], features["feat_b"])


def test_noise_scales_with_requested_fraction_of_std():
    features = _toy_features()
    stds = features[["feat_a", "feat_b"]].std()

    out = inject_gaussian_noise(features, ["feat_a", "feat_b"], stds, noise_fraction=0.5, seed=1)
    added_noise = out["feat_a"].to_numpy() - features["feat_a"].to_numpy()

    expected_std = 0.5 * float(stds["feat_a"])
    # empirical std of the injected noise should be close to the requested std
    assert abs(added_noise.std() - expected_std) < 0.1 * expected_std
    assert abs(added_noise.mean()) < 0.05 * expected_std  # zero-mean


def test_noise_reproducible_given_same_seed():
    features = _toy_features()
    stds = features[["feat_a", "feat_b"]].std()

    out_a = inject_gaussian_noise(features, ["feat_a"], stds, noise_fraction=0.3, seed=42)
    out_b = inject_gaussian_noise(features, ["feat_a"], stds, noise_fraction=0.3, seed=42)

    pd.testing.assert_frame_equal(out_a, out_b)


def test_ablate_features_replaces_only_named_columns_with_median():
    features = _toy_features()
    medians = features[["feat_a", "feat_b"]].median()

    out = ablate_features(features, ["feat_a"], medians)

    assert (out["feat_a"] == medians["feat_a"]).all()
    pd.testing.assert_series_equal(out["feat_b"], features["feat_b"])  # untouched
    pd.testing.assert_series_equal(out["record_id"], features["record_id"])  # untouched
