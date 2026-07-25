"""Phase 4: unit tests for dashboard/app.py's pure helper functions -- the
parts of the live threshold slider and filter logic that don't need a
running Streamlit session to verify. `_threshold_metrics` in particular is
the function backing the dashboard's "live-updates the precision/recall
tradeoff" requirement, so its precision/recall/FP-per-day arithmetic is
checked directly against hand-computed expectations.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dashboard.app import _parse_top_features, _threshold_metrics, apply_filters


def _toy_detail() -> pd.DataFrame:
    # 2 true attacks (score 0.9, 0.6), 3 benign (score 0.7, 0.2, 0.1).
    # At threshold 0.5: flagged = {0.9(TP), 0.6(TP), 0.7(FP)} -> precision=2/3, recall=2/2=1.0
    return pd.DataFrame({
        "record_id": ["a", "b", "c", "d", "e"],
        "score": [0.9, 0.6, 0.7, 0.2, 0.1],
        "is_attack": [True, True, False, False, False],
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00", "2026-01-01 12:00", "2026-01-02 00:00",
            "2026-01-02 12:00", "2026-01-03 00:00",
        ]),
        "department": ["Engineering", "Sales", "Engineering", "IT", "Sales"],
        "user_id": ["u1", "u2", "u3", "u4", "u5"],
    })


def test_threshold_metrics_matches_hand_computed_values():
    df = _toy_detail()
    m = _threshold_metrics(df, threshold=0.5)
    assert m["n_flagged"] == 3
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 0
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)


def test_threshold_metrics_at_max_threshold_flags_nothing():
    df = _toy_detail()
    m = _threshold_metrics(df, threshold=1.5)
    assert m["n_flagged"] == 0
    assert m["tp"] == 0
    assert m["fp"] == 0
    # recall is 0/2 = 0.0 (well-defined; no division-by-zero since 2 true positives exist)
    assert m["recall"] == 0.0


def test_apply_filters_department_and_time_range():
    df = _toy_detail()
    filters = {
        "departments": ["Engineering"], "attack_types": [], "severities": [],
        "date_range": (df["timestamp"].min(), df["timestamp"].max()), "user_search": "",
    }
    filtered = apply_filters(df, filters)
    assert set(filtered["record_id"]) == {"a", "c"}


def test_apply_filters_user_search_is_case_insensitive():
    df = _toy_detail()
    filters = {
        "departments": [], "attack_types": [], "severities": [],
        "date_range": (df["timestamp"].min(), df["timestamp"].max()), "user_search": "U1",
    }
    filtered = apply_filters(df, filters)
    assert set(filtered["record_id"]) == {"a"}


def test_parse_top_features_handles_missing_and_valid_json():
    assert _parse_top_features(None) == []
    assert _parse_top_features(float("nan")) == []
    assert _parse_top_features('[{"feature": "x", "shap_value": 0.1}]') == [{"feature": "x", "shap_value": 0.1}]
