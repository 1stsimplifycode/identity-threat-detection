"""Phase 5b: calibration/bootstrap-CI/significance-test correctness,
checked against synthetic cases with a known right answer -- not eyeballed.
"""
from __future__ import annotations

import numpy as np

from evaluation.calibration import bootstrap_metric_ci, compute_calibration, paired_bootstrap_significance


def test_calibration_well_calibrated_scores_have_near_zero_ece_and_low_brier():
    scores, labels = [], []
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        n = 200
        n_pos = int(round(p * n))
        scores.extend([p] * n)
        labels.extend([1] * n_pos + [0] * (n - n_pos))
    y_score = np.array(scores)
    y_true = np.array(labels)

    result = compute_calibration("test_model", y_true, y_score, n_bins=10)

    assert result.ece < 0.02, f"expected near-zero ECE for perfectly-calibrated scores, got {result.ece}"
    assert result.brier_score < 0.25  # a real (non-degenerate) score, well below the 0.25 chance-level ceiling


def test_calibration_badly_calibrated_scores_have_high_ece():
    # Model always says 90% confident, but the true positive rate is only 10%.
    n = 500
    y_score = np.full(n, 0.9)
    y_true = np.array([1 if i < 50 else 0 for i in range(n)])  # 10% positive

    result = compute_calibration("overconfident_model", y_true, y_score, n_bins=10)

    assert result.ece > 0.5, f"expected a large ECE for badly-miscalibrated scores, got {result.ece}"
    assert result.brier_score > 0.5


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    rng = np.random.default_rng(0)
    n = 2000
    y_true = (rng.random(n) < 0.05).astype(int)
    y_score = np.clip(y_true * 0.7 + rng.random(n) * 0.3, 0, 1)
    y_pred = (y_score >= 0.5).astype(int)

    ci = bootstrap_metric_ci(y_true, y_score, y_pred, n_bootstrap=200, seed=1)

    for metric_name, (point, lo, hi) in ci.items():
        assert lo <= hi, f"{metric_name}: CI lower bound {lo} > upper bound {hi}"
        # the point estimate should sit inside or very near its own bootstrap distribution
        assert lo - 1e-6 <= point <= hi + 1e-6, f"{metric_name}: point {point} outside CI [{lo}, {hi}]"


def test_bootstrap_reproducible_given_same_seed():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = (rng.random(n) < 0.1).astype(int)
    y_score = rng.random(n)
    y_pred = (y_score >= 0.5).astype(int)

    ci_a = bootstrap_metric_ci(y_true, y_score, y_pred, n_bootstrap=100, seed=7)
    ci_b = bootstrap_metric_ci(y_true, y_score, y_pred, n_bootstrap=100, seed=7)

    assert ci_a == ci_b, "same seed must reproduce byte-identical bootstrap CIs"


def test_significance_identical_scores_give_p_value_near_one():
    rng = np.random.default_rng(2)
    n = 2000
    y_true = (rng.random(n) < 0.1).astype(int)
    score = np.clip(y_true * 0.6 + rng.random(n) * 0.4, 0, 1)

    result = paired_bootstrap_significance("model_a", "model_a_copy", y_true, score, score.copy(), n_bootstrap=200)

    assert result.diff == 0.0
    assert result.p_value > 0.9, f"identical models should show no significant difference, got p={result.p_value}"


def test_significance_clearly_different_models_give_small_p_value():
    rng = np.random.default_rng(3)
    n = 2000
    y_true = (rng.random(n) < 0.1).astype(int)
    # model_a: strong real signal. model_b: pure noise, uninformative.
    score_a = np.clip(y_true * 0.8 + rng.random(n) * 0.2, 0, 1)
    score_b = rng.random(n)

    result = paired_bootstrap_significance("strong_model", "noise_model", y_true, score_a, score_b, n_bootstrap=200)

    assert result.value_a > result.value_b
    assert result.p_value < 0.05, f"expected a significant PR-AUC gap, got p={result.p_value}"
    assert result.diff_ci_lo > 0, "95% CI on the difference should exclude 0 given a clearly better model"
