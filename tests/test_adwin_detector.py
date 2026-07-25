"""Phase 3: ADWIN drift detector -- must actually detect the configured
drift event on real generated data, at a reasonable lag, using the
event-count-binned signal (see drift_detection/adwin_detector.py's module
docstring for why raw per-event and calendar-day aggregation both failed
during development).
"""
from __future__ import annotations

import pandas as pd

from attacks.injector import inject_attacks
from drift_detection.adwin_detector import bin_by_event_count, evaluate_against_drift_log, run_adwin
from feature_engineering.pipeline import compute_feature_table
from generator.drift import resolve_drift_schedule
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, _assemble_events_and_labels, _assign_bookkeeping_fields, load_mitre_mapping
from generator.seeding import set_global_seed


def test_adwin_detects_configured_drift(audit_cfg):
    rng = set_global_seed(int(audit_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")
    users = generate_population(audit_cfg, rng)
    drift, drift_log = resolve_drift_schedule(users, audit_cfg, rng)
    assert len(drift_log) > 0, "audit_cfg inherits small_dev's drift schedule -- expected at least one configured event"

    benign_events = generate_login_events(users, audit_cfg, rng, drift=drift)
    attack_events, attack_meta = inject_attacks(users, benign_events, audit_cfg, mitre_mapping, rng)
    events, labels = _assemble_events_and_labels(benign_events, attack_events, attack_meta)
    events, labels = _assign_bookkeeping_fields(events, labels, int(audit_cfg.attacks.num_generation_batches), rng)

    features = compute_feature_table(events, users, audit_cfg)
    events_sorted = events.sort_values("timestamp").reset_index(drop=True)
    merged = events_sorted[["record_id", "timestamp"]].merge(features[["record_id", "geo_distance_from_home_km"]], on="record_id")

    binned = bin_by_event_count(merged["geo_distance_from_home_km"], merged["timestamp"], int(audit_cfg.models.adwin.bin_size))
    detected = run_adwin(binned, audit_cfg)

    start_date = pd.Timestamp(audit_cfg.events.start_date)
    result = evaluate_against_drift_log(detected, drift_log, start_date)

    assert result["detected"].all(), f"expected every configured drift event to be detected: {result.to_dict('records')}"
    # a real detector shouldn't need more than a few days' lag on this scale
    assert (result["detection_lag_days"] < 5).all()
