"""Phase 3: cold-start priors must actually change cold-start rows'
COLD_START_ELIGIBLE_COLUMNS toward a department prior, and must leave
established (non-cold-start) rows completely untouched.
"""
from __future__ import annotations

import pandas as pd

from attacks.injector import inject_attacks
from feature_engineering.cold_start import COLD_START_ELIGIBLE_COLUMNS, apply_cold_start_priors, compute_department_priors
from feature_engineering.pipeline import compute_feature_table
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, load_mitre_mapping
from generator.seeding import set_global_seed


def test_cold_start_priors_change_only_cold_start_rows(audit_cfg):
    rng = set_global_seed(int(audit_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(audit_cfg, rng)
    events = generate_login_events(users, audit_cfg, rng)
    events = events.reset_index(drop=True)
    events["record_id"] = [f"r{i}" for i in range(len(events))]

    features = compute_feature_table(events, users, audit_cfg)
    priors = compute_department_priors(features, events, users)
    assert set(priors.columns) == set(COLD_START_ELIGIBLE_COLUMNS)

    adjusted = apply_cold_start_priors(features, events, users, priors, audit_cfg)

    context = events[["record_id", "user_id", "timestamp"]].merge(users[["user_id", "join_date"]], on="user_id")
    window = pd.Timedelta(days=float(audit_cfg.feature_engineering.cold_start.window_days))
    context["is_cold_start"] = (context["timestamp"] - context["join_date"]) < window

    non_cold_ids = context.loc[~context["is_cold_start"], "record_id"]
    cold_ids = context.loc[context["is_cold_start"], "record_id"]

    orig_non_cold = features[features["record_id"].isin(non_cold_ids)].set_index("record_id").sort_index()
    adj_non_cold = adjusted[adjusted["record_id"].isin(non_cold_ids)].set_index("record_id").sort_index()
    pd.testing.assert_frame_equal(orig_non_cold, adj_non_cold)

    assert len(cold_ids) > 0, "expected at least one cold-start row at audit_cfg scale (10% of users join within 60 days)"
    # for cold-start rows, at least one eligible column should differ from
    # the raw computed value for at least some rows (not asserting ALL
    # rows change, since a prior can coincidentally equal the raw value)
    orig_cold = features[features["record_id"].isin(cold_ids)].set_index("record_id").sort_index()
    adj_cold = adjusted[adjusted["record_id"].isin(cold_ids)].set_index("record_id").sort_index()
    any_changed = (orig_cold[COLD_START_ELIGIBLE_COLUMNS] != adj_cold[COLD_START_ELIGIBLE_COLUMNS]).any().any()
    assert any_changed, "expected cold-start rows to differ from raw computed values after prior substitution"
