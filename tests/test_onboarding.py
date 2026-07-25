"""Unit tests for dashboard/onboarding.py's pure logic -- FAQ matching and
the contextual "why was this flagged" answer, the parts that don't need a
live Streamlit session to verify (matching test_dashboard_app.py's own
discipline: test the pure helper functions directly, not the UI rendering).
"""
from __future__ import annotations

from dashboard.onboarding import STEP_BY_ID, TOTAL_STEPS, TOUR_STEPS, answer_question


def test_all_tour_steps_have_non_empty_content():
    assert TOTAL_STEPS == len(TOUR_STEPS) > 0
    for step in TOUR_STEPS:
        assert step.title.strip()
        assert step.story.strip()
        assert step.basic.strip()
        assert step.advanced.strip()


def test_tour_step_ids_are_unique():
    ids = [s.id for s in TOUR_STEPS]
    assert len(ids) == len(set(ids))
    assert STEP_BY_ID.keys() == set(ids)


def test_answer_question_matches_shap_keyword():
    answer = answer_question("What is SHAP?")
    assert "shap" in answer.lower()


def test_answer_question_matches_mitre_keyword():
    answer = answer_question("Can you explain MITRE ATT&CK?")
    assert "mitre" in answer.lower()


def test_answer_question_falls_back_for_unknown_topic():
    answer = answer_question("What's the weather like today?")
    assert "don't have a canned answer" in answer.lower()


def test_why_flagged_without_context_asks_to_select_event():
    answer = answer_question("why was this flagged")
    assert "select a flagged event" in answer.lower()


def test_why_flagged_with_context_lists_real_evidence():
    context = {"evidence": [
        {"text": "Anomaly score at the 99.9% percentile", "category": "score"},
        {"text": "Login-failure ratio 3.0 std. devs above this user's own norm", "category": "login_failure"},
    ]}
    answer = answer_question("why was this flagged?", context=context)
    assert "2 evidence signal" in answer
    assert "Anomaly score at the 99.9% percentile" in answer
    assert "Login-failure ratio 3.0 std. devs above this user's own norm" in answer
