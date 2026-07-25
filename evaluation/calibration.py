"""Phase 5b (evaluation rigor): calibration quality, bootstrap confidence
intervals, and paired significance testing -- pure post-hoc statistics over
already-computed (y_true, y_score, y_pred) arrays. Deliberately independent
of `evaluation/report.py`'s `ModelEvaluation`/`evaluate_model()` (no new
fields added there, no retraining triggered) so this can be layered on top
of the model suite's existing checkpoints without invalidating them -- see
`evaluation/run_rigor_analysis.py`, which is the entrypoint that wires this
against real checkpoint data.

Everything stochastic (bootstrap resampling) takes an explicit seed and
defaults to a fixed one, so re-running produces byte-identical output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class CalibrationResult:
    model_name: str
    brier_score: float
    ece: float
    n_bins: int
    reliability_bins: pd.DataFrame


def compute_calibration(model_name: str, y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> CalibrationResult:
    """Brier score (mean squared error between predicted probability and
    the 0/1 outcome -- lower is better, 0 is perfect) and Expected
    Calibration Error (the standard equal-width-bin definition: partition
    [0, 1] into `n_bins`, and for each non-empty bin take
    |mean predicted probability - empirical positive rate|, weighted by the
    bin's share of all examples). Only meaningful for scores that are
    actually probabilities in [0, 1] -- see run_rigor_analysis.py's model
    selection for which models that excludes and why.
    """
    brier = float(brier_score_loss(y_true, y_score))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, bin_edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin_lo": bin_edges[b], "bin_hi": bin_edges[b + 1], "count": 0,
                "mean_predicted": float("nan"), "empirical_rate": float("nan"),
            })
            continue
        mean_pred = float(y_score[mask].mean())
        empirical = float(y_true[mask].mean())
        rows.append({
            "bin_lo": bin_edges[b], "bin_hi": bin_edges[b + 1], "count": count,
            "mean_predicted": mean_pred, "empirical_rate": empirical,
        })
        ece += (count / n) * abs(mean_pred - empirical)

    return CalibrationResult(
        model_name=model_name, brier_score=brier, ece=float(ece), n_bins=n_bins,
        reliability_bins=pd.DataFrame(rows),
    )


def _point_estimates(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    has_both = len(set(y_true)) > 1
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both else float("nan"),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray,
    n_bootstrap: int = 300, ci: float = 0.95, seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Percentile bootstrap CIs for precision/recall/f1/roc_auc/pr_auc:
    resample test-set INDICES with replacement `n_bootstrap` times (the
    model's predictions are fixed -- only which test rows get sampled
    varies), recompute each metric per resample, and take the
    [alpha/2, 1-alpha/2] percentiles of the resulting distribution. This is
    the standard nonparametric bootstrap answer to "how much would this
    metric plausibly vary on a different sample from the same underlying
    population." Resamples with only one class present are skipped (AUC/
    PR-AUC undefined there) rather than silently inserting a NaN or 0.

    Returns {metric_name: (point_estimate, ci_lo, ci_hi)}.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collected: dict[str, list[float]] = {"precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys, yp = y_true[idx], y_score[idx], y_pred[idx]
        if len(set(yt)) < 2:
            continue
        collected["precision"].append(precision_score(yt, yp, zero_division=0))
        collected["recall"].append(recall_score(yt, yp, zero_division=0))
        collected["f1"].append(f1_score(yt, yp, zero_division=0))
        collected["roc_auc"].append(roc_auc_score(yt, ys))
        collected["pr_auc"].append(average_precision_score(yt, ys))

    point = _point_estimates(y_true, y_score, y_pred)
    alpha = (1.0 - ci) / 2.0
    out: dict[str, tuple[float, float, float]] = {}
    for name, values in collected.items():
        arr = np.array(values)
        lo, hi = np.percentile(arr, [alpha * 100, (1 - alpha) * 100]) if len(arr) > 0 else (float("nan"), float("nan"))
        out[name] = (point[name], float(lo), float(hi))
    return out


@dataclass
class SignificanceResult:
    metric: str
    model_a: str
    model_b: str
    value_a: float
    value_b: float
    diff: float
    diff_ci_lo: float
    diff_ci_hi: float
    p_value: float


def paired_bootstrap_significance(
    model_a: str, model_b: str,
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
    metric: str = "pr_auc", n_bootstrap: int = 300, seed: int = 42,
) -> SignificanceResult:
    """Is `model_a` really better than `model_b` on `metric` (default
    PR-AUC, the problem statement's headline metric under imbalance), or
    is the observed gap within the noise you'd expect from resampling the
    SAME test set? Both models are scored on the SAME resampled indices
    each iteration (paired -- controls for both models seeing identical
    examples per resample, unlike two independent bootstraps). Two-sided
    p-value = 2 * min(fraction of resamples with diff <= 0, fraction with
    diff >= 0), the standard bootstrap significance estimate; a 95% CI on
    the difference that excludes 0 is the same claim from the interval
    side.
    """
    metric_fn = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}[metric]
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(set(yt)) < 2:
            continue
        diffs.append(metric_fn(yt, score_a[idx]) - metric_fn(yt, score_b[idx]))

    diffs_arr = np.array(diffs)
    value_a = float(metric_fn(y_true, score_a))
    value_b = float(metric_fn(y_true, score_b))
    lo, hi = np.percentile(diffs_arr, [2.5, 97.5])
    frac_le0 = float((diffs_arr <= 0).mean())
    frac_ge0 = float((diffs_arr >= 0).mean())
    p_value = float(min(1.0, 2 * min(frac_le0, frac_ge0)))

    return SignificanceResult(
        metric=metric, model_a=model_a, model_b=model_b,
        value_a=value_a, value_b=value_b, diff=value_a - value_b,
        diff_ci_lo=float(lo), diff_ci_hi=float(hi), p_value=p_value,
    )
