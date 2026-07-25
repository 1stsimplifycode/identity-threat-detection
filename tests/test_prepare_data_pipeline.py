"""Phase 4: exercises dashboard/prepare_data.py's actual building blocks
end-to-end (denormalized test-event detail + exact SHAP + streaming
approximation + feature history) against a real generated run, the same
way `dashboard.prepare_data.main()` assembles them -- without invoking the
Hydra CLI entrypoint directly, consistent with how the rest of this test
suite drives cfg objects.
"""
from __future__ import annotations

import numpy as np

from dashboard.prepare_data import RAW_EVENT_FIELDS, _build_test_detail
from evaluation.model_suite import PRIMARY_MODEL_NAME, build_model_suite
from explainability.shap_batch import (
    compute_global_feature_weights,
    explain_flagged_events,
    select_flagged_sample,
)
from explainability.streaming_approx import StreamingApproxExplainer, compute_feature_baseline, explain_stream_batch
from feature_engineering.pipeline import FEATURE_COLUMNS
from models.xgboost_classifier import score_xgboost
from tests.test_model_suite import _write_run


def test_full_dashboard_data_prep_pipeline(graph_verification_cfg, tmp_path):
    run_dir = tmp_path / "run"
    _write_run(graph_verification_cfg, run_dir)
    suite = build_model_suite(run_dir, graph_verification_cfg, include_hf=False)

    xgb_test_scores = score_xgboost(suite.primary_model, suite.test_features)
    detail = _build_test_detail(suite, xgb_test_scores)

    assert len(detail) == suite.split.n_test
    assert set(RAW_EVENT_FIELDS) <= set(detail.columns)
    assert set(FEATURE_COLUMNS) <= set(detail.columns)
    assert {"score", "predicted_class", "severity", "department"} <= set(detail.columns)
    assert not detail["score"].isna().any()

    xgb_train_scores = score_xgboost(suite.primary_model, suite.train_features)["xgb_anomaly_score"]
    slider_min = float(np.percentile(xgb_train_scores, float(graph_verification_cfg.dashboard.threshold_slider_min_percentile)))

    flagged_ids = select_flagged_sample(detail, "score", slider_min, max_samples=None, seed=1)
    assert len(flagged_ids) > 0

    shap_explanations = explain_flagged_events(
        suite.primary_model, suite.test_features, xgb_test_scores["xgb_predicted_class"], flagged_ids, top_k=5,
    )
    assert set(shap_explanations["record_id"]) == set(flagged_ids)

    global_weights = compute_global_feature_weights(suite.primary_model, suite.test_features, flagged_ids)
    assert abs(sum(global_weights.values()) - 1.0) < 1e-6
    baseline = compute_feature_baseline(suite.train_features)
    explainer = StreamingApproxExplainer(global_weights, baseline)
    approx_explanations = explain_stream_batch(explainer, suite.test_features, flagged_ids, top_k=5)
    assert set(approx_explanations["record_id"]) == set(flagged_ids)

    merged = detail.merge(
        shap_explanations.rename(columns={"explanation": "shap_explanation"})[["record_id", "shap_explanation"]],
        on="record_id", how="left",
    )
    flagged_rows = merged[merged["record_id"].isin(flagged_ids)]
    assert flagged_rows["shap_explanation"].notna().all()
    unflagged_rows = merged[~merged["record_id"].isin(flagged_ids)]
    if len(unflagged_rows) > 0:
        assert unflagged_rows["shap_explanation"].isna().all()
