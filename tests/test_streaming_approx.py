"""Phase 4: the lightweight streaming explainability approximation must
(1) rank a genuinely anomalous feature above a baseline-typical one, (2)
stay cheap (no model inference, pure arithmetic against precomputed
weights/baseline), and (3) always label itself as approximate.
"""
from __future__ import annotations

import pandas as pd

from explainability.streaming_approx import StreamingApproxExplainer, compute_feature_baseline
from feature_engineering.pipeline import FEATURE_COLUMNS


def _uniform_reference(n: int = 200) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    data = {feat: rng.normal(loc=1.0, scale=0.1, size=n) for feat in FEATURE_COLUMNS}
    return pd.DataFrame(data)


def test_baseline_has_one_row_per_feature_with_mean_and_std():
    baseline = compute_feature_baseline(_uniform_reference())
    assert set(baseline.index) == set(FEATURE_COLUMNS)
    assert {"mean", "std"} <= set(baseline.columns)
    assert (baseline["std"] > 0).all()


def test_anomalous_feature_ranks_above_typical_feature():
    baseline = compute_feature_baseline(_uniform_reference())
    # Equal weights, so ranking is driven purely by deviation from baseline.
    weights = {feat: 1.0 / len(FEATURE_COLUMNS) for feat in FEATURE_COLUMNS}
    explainer = StreamingApproxExplainer(weights, baseline)

    event = {feat: 1.0 for feat in FEATURE_COLUMNS}
    event["geo_distance_from_home_km"] = 50.0  # far outside the ~1.0 +/- 0.1 baseline

    result = explainer.explain_event(event, top_k=3)
    assert result["top_features"][0]["feature"] == "geo_distance_from_home_km"
    assert result["explanation"].startswith("Approximate explanation (not exact SHAP):")


def test_all_baseline_values_produce_no_standout_explanation():
    baseline = compute_feature_baseline(_uniform_reference())
    weights = {feat: 1.0 / len(FEATURE_COLUMNS) for feat in FEATURE_COLUMNS}
    explainer = StreamingApproxExplainer(weights, baseline)

    # Exactly at the baseline's own mean -> z=0 for every feature, guaranteed
    # (unlike a fixed constant, which only approximates the sampled mean).
    event = {feat: float(baseline.loc[feat, "mean"]) for feat in FEATURE_COLUMNS}
    result = explainer.explain_event(event, top_k=3)
    assert "no feature far from its established baseline" in result["explanation"]
