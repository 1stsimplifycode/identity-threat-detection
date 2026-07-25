"""Dumb, rule-based baseline (constraint #3): every subsequent model is
compared against this in every results table.

Deliberately does NOT use `feature_engineering/`'s output at all -- it
reads only raw event fields, computing three independent static-threshold
flags per event:
    rule_failed_login_flag -- >= N failed logins by this user in the last
                               W minutes (a simple rolling count, not the
                               EMA-smoothed baseline in behavioral.py)
    rule_new_country_flag  -- this event's country has never appeared
                               before in this user's own history
    rule_off_hours_flag    -- generator/events.py's own is_off_hours field

`rule_risk_score` is the sum of the three flags (0-3); `rule_predicted_anomaly`
flags rows at or above `cfg.models.baseline.risk_score_threshold`.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd
from omegaconf import DictConfig

BASELINE_COLUMNS: list[str] = [
    "rule_failed_login_flag",
    "rule_new_country_flag",
    "rule_off_hours_flag",
    "rule_risk_score",
]


def compute_rule_based_scores(events: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    baseline_cfg = cfg.models.baseline
    window = pd.Timedelta(minutes=float(baseline_cfg.failed_login_window_minutes))
    failure_threshold = int(baseline_cfg.failed_login_count_threshold)

    events_sorted = events.sort_values("timestamp")
    recent_failures: dict[str, deque] = defaultdict(deque)
    seen_countries: dict[str, set[str]] = defaultdict(set)

    record_ids: list[str] = []
    failed_flags: list[int] = []
    new_country_flags: list[int] = []
    off_hours_flags: list[int] = []

    for row in events_sorted.itertuples():
        user_id = row.user_id
        ts = row.timestamp

        failures = recent_failures[user_id]
        while failures and (ts - failures[0]) > window:
            failures.popleft()
        failed_flags.append(1 if len(failures) >= failure_threshold else 0)

        countries = seen_countries[user_id]
        new_country_flags.append(0 if row.geo_country in countries else 1)
        countries.add(row.geo_country)

        off_hours_flags.append(1 if row.is_off_hours else 0)
        record_ids.append(row.record_id)

        if row.auth_result == "failure":
            failures.append(ts)

    risk_score = [a + b + c for a, b, c in zip(failed_flags, new_country_flags, off_hours_flags)]

    return pd.DataFrame({
        "record_id": record_ids,
        "rule_failed_login_flag": failed_flags,
        "rule_new_country_flag": new_country_flags,
        "rule_off_hours_flag": off_hours_flags,
        "rule_risk_score": risk_score,
    })


def predict_anomaly(scores: pd.DataFrame, cfg: DictConfig) -> pd.Series:
    threshold = int(cfg.models.baseline.risk_score_threshold)
    return (scores["rule_risk_score"] >= threshold).astype(int)
