"""Lightweight streaming explainability approximation -- the other half of
constraint #4's explicit tradeoff (`explainability/shap_batch.py` is the
exact, offline-batch half). Running `shap.TreeExplainer` per live event
would mean a full tree-traversal pass on every streaming prediction; this
module instead reuses two things precomputed ONCE, offline:

1. Global per-feature weights (`shap_batch.compute_global_feature_weights`)
   -- derived from real exact-SHAP output on a batch sample, not a
   separately-invented importance metric.
2. A per-feature baseline mean/std (`compute_feature_baseline`), computed
   from the TRAIN split only (same leakage discipline as
   `feature_engineering/cold_start.py`'s priors).

At streaming-scoring time, each live event's approximate "contribution"
per feature is `weight * |z-score|` -- how much this feature usually
matters, times how far today's value sits from the established baseline.
This is a genuine approximation, not exact attribution: it has no sense of
feature INTERACTIONS or this specific event's model path, only "usually
important, and unusual right now." It is labeled as approximate in every
explanation string it produces, and in the dashboard wherever it's shown,
per the constraint's own instruction not to hide the tradeoff.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from explainability.feature_glossary import describe
from feature_engineering.pipeline import FEATURE_COLUMNS

_MIN_STD = 1e-6  # floor to avoid divide-by-zero on a constant feature


def compute_feature_baseline(reference_features: pd.DataFrame) -> pd.DataFrame:
    """`reference_features` should be TRAIN-split rows only. Returns one
    row per feature in `FEATURE_COLUMNS` with `mean` and `std` columns.
    """
    stats = reference_features[FEATURE_COLUMNS].agg(["mean", "std"]).T
    stats["std"] = stats["std"].fillna(0.0).clip(lower=_MIN_STD)
    return stats


class StreamingApproxExplainer:
    """Holds the two precomputed artifacts and applies them cheaply, per
    event, in the streaming path. Construction (from `weights`/`baseline`)
    is the only place any batch computation happens; `explain_event` is
    O(n_features) arithmetic, safe to call on every live event.
    """

    def __init__(self, weights: dict[str, float], baseline: pd.DataFrame) -> None:
        self.weights = weights
        self.baseline = baseline

    def explain_event(self, feature_row: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        contributors = []
        for feat in FEATURE_COLUMNS:
            value = float(feature_row.get(feat, 0.0))
            mean = float(self.baseline.loc[feat, "mean"])
            std = float(self.baseline.loc[feat, "std"])
            z = (value - mean) / std
            approx_score = self.weights.get(feat, 0.0) * abs(z)
            contributors.append((feat, approx_score, value, z))

        contributors.sort(key=lambda c: c[1], reverse=True)
        top = contributors[:top_k]

        return {
            "top_features": [
                {
                    "feature": feat,
                    "approx_score": round(score, 6),
                    "feature_value": round(value, 6),
                    "z_score": round(z, 3),
                    "description": describe(feat),
                }
                for feat, score, value, z in top
            ],
            "explanation": _build_approx_explanation(top),
        }


def _build_approx_explanation(top: list[tuple[str, float, float, float]]) -> str:
    if not top or top[0][1] <= 0:
        return "Approximate explanation (not exact SHAP): no feature far from its established baseline."
    parts = []
    for feat, _score, value, z in top:
        direction = "unusually high" if z > 0 else "unusually low"
        parts.append(f"{feat}={value:.3g} ({direction}, z={z:+.2f})")
    return "Approximate explanation (not exact SHAP): " + ", ".join(parts) + "."


def explain_stream_batch(
    explainer: StreamingApproxExplainer, features: pd.DataFrame, record_ids: list[str], top_k: int = 5,
) -> pd.DataFrame:
    """Applies the approximation to a set of already-computed feature rows
    (e.g. River's flagged test events) -- used offline here purely to
    populate the dashboard's precomputed data, standing in for what the
    same `StreamingApproxExplainer.explain_event()` call would do live,
    one event at a time, in a deployed streaming process.
    """
    subset = features[features["record_id"].isin(record_ids)]
    rows = []
    for record in subset.to_dict("records"):
        result = explainer.explain_event(record, top_k=top_k)
        rows.append({
            "record_id": record["record_id"],
            "top_features": result["top_features"],
            "explanation": result["explanation"],
        })
    return pd.DataFrame(rows, columns=["record_id", "top_features", "explanation"])
