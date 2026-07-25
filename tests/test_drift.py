"""Phase 3: concept-drift simulation sanity checks -- the ground-truth
drift_log and ResolvedDrift must actually correspond to a real behavioral
change in the generated events, not just metadata that looks right.
"""
from __future__ import annotations

from generator.drift import resolve_drift_schedule
from generator.events import generate_login_events
from generator.population import generate_population
from generator.seeding import set_global_seed


def test_drift_log_matches_configured_schedule(audit_cfg):
    rng = set_global_seed(int(audit_cfg.seed))
    users = generate_population(audit_cfg, rng)
    drift, drift_log = resolve_drift_schedule(users, audit_cfg, rng)

    assert len(drift_log) == len(audit_cfg.events.drift.schedule)
    for row, configured in zip(drift_log.itertuples(), audit_cfg.events.drift.schedule):
        assert row.day == configured.day
        assert row.change_type == configured.change_type
        # affected_user_count should be close to the configured fraction of
        # active (non-terminated) users -- allow generous slack since exact
        # rounding depends on the active-user count.
        n_active = int((users["employment_status"] != "terminated").sum())
        expected = n_active * configured.affected_fraction
        assert abs(row.affected_user_count - expected) <= max(5, expected * 0.1)


def test_drifted_users_actually_change_behavior(audit_cfg):
    """The whole point of the ground-truth log: a user tagged as drifted
    for remote_work_shift must show a real, observable change in their
    generated events from their individual rollout day forward -- not just
    a metadata entry with no effect on the data.
    """
    rng = set_global_seed(int(audit_cfg.seed))
    users = generate_population(audit_cfg, rng)
    drift, drift_log = resolve_drift_schedule(users, audit_cfg, rng)
    events = generate_login_events(users, audit_cfg, rng, drift=drift)

    remote_shift_users = drift.drift_days.get("remote_work_shift", {})
    assert len(remote_shift_users) > 0, "expected the configured remote_work_shift drift to affect some users"

    checked_any = False
    for user_id, drift_day in list(remote_shift_users.items())[:20]:
        drifted_country = drift.drifted_values["remote_work_shift"][user_id]["home_country"]
        user_events = events[events["user_id"] == user_id].sort_values("timestamp")
        if len(user_events) < 2:
            continue  # too little data for this user to check pre/post reliably
        start_date = user_events["timestamp"].min().normalize()
        day_index = (user_events["timestamp"].dt.normalize() - start_date).dt.days
        pre_drift = user_events[day_index < drift_day]
        post_drift = user_events[day_index >= drift_day]
        if len(pre_drift) == 0 or len(post_drift) == 0:
            continue
        checked_any = True
        # post-drift events should predominantly show the new country;
        # pre-drift events should not (allowing for the rare travel_event
        # that legitimately sends anyone anywhere).
        post_drift_match_rate = (post_drift["geo_country"] == drifted_country).mean()
        assert post_drift_match_rate > 0.5, (
            f"user {user_id} tagged as drifted to {drifted_country} but only "
            f"{post_drift_match_rate:.0%} of post-drift events show that country"
        )

    assert checked_any, "no drifted user had enough pre/post events to verify -- check test scale"


def test_drift_disabled_by_default_config_still_produces_empty_log(tiny_cfg):
    """tiny_cfg overrides population/events scale but inherits small_dev's
    drift schedule (day=7) -- this test instead directly disables drift to
    confirm the "no drift configured" path still returns a valid, empty,
    correctly-headered log rather than erroring.
    """
    from omegaconf import OmegaConf

    cfg_no_drift = OmegaConf.merge(tiny_cfg, {"events": {"drift": {"enabled": False}}})
    rng = set_global_seed(int(cfg_no_drift.seed))
    users = generate_population(cfg_no_drift, rng)
    drift, drift_log = resolve_drift_schedule(users, cfg_no_drift, rng)

    assert not drift.enabled
    assert len(drift_log) == 0
    assert list(drift_log.columns) == ["day", "change_type", "description", "affected_user_count"]
