"""Schema validation: every generated table must satisfy its schema (all
required fields present, correct types, categorical values in range, numeric
ranges respected).
"""
from __future__ import annotations

from attacks.injector import inject_attacks
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, _assemble_events_and_labels, _assign_bookkeeping_fields, load_mitre_mapping
from generator.seeding import set_global_seed
from preprocessing.schema import ATTACKS_SCHEMA, EVENTS_SCHEMA, LABELS_SCHEMA, USERS_SCHEMA, validate_dataframe


def _generate_full_run(cfg):
    rng = set_global_seed(int(cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(cfg, rng)
    benign_events = generate_login_events(users, cfg, rng)
    attack_events, attack_meta = inject_attacks(users, benign_events, cfg, mitre_mapping, rng)
    events, labels = _assemble_events_and_labels(benign_events, attack_events, attack_meta)
    events, labels = _assign_bookkeeping_fields(events, labels, int(cfg.attacks.num_generation_batches), rng)
    return users, events, labels, attack_meta


def test_all_tables_satisfy_their_schema(tiny_cfg):
    users, events, labels, attack_meta = _generate_full_run(tiny_cfg)

    validate_dataframe(users, USERS_SCHEMA, "users")
    validate_dataframe(events, EVENTS_SCHEMA, "events")
    validate_dataframe(labels, LABELS_SCHEMA, "labels")
    assert len(attack_meta) > 0, "expected at least one attack campaign at the configured imbalance ratio"
    validate_dataframe(attack_meta, ATTACKS_SCHEMA, "attacks")


def test_events_and_labels_are_row_aligned(tiny_cfg):
    _, events, labels, _ = _generate_full_run(tiny_cfg)

    assert len(events) == len(labels)
    assert list(events["record_id"]) == list(labels["record_id"])
    # events table itself must never carry the label directly
    assert "is_attack" not in events.columns
    assert "attack_id" not in events.columns


def test_events_are_chronologically_sorted(tiny_cfg):
    _, events, _, _ = _generate_full_run(tiny_cfg)
    assert events["timestamp"].is_monotonic_increasing


def test_session_id_present_and_resource_access_rows_have_no_auth_fields(audit_cfg):
    """Phase 2a schema expansion: every row has a session_id; resource_access
    rows (added this phase) are auth-not-applicable, but still carry a normal
    resource_accessed value like a benign successful login would (see
    attacks/credential_misuse.py, attacks/lateral_movement.py, and
    attacks/device_spoofing.py's login-row fix for why this matters -- a
    successful login with a null resource would itself be a leakage-adjacent
    tell, since no benign successful login ever has one).
    """
    _, events, _, _ = _generate_full_run(audit_cfg)

    assert events["session_id"].notna().all()
    assert (events["session_id"].str.len() > 0).all()

    resource_rows = events[events["event_type"] == "resource_access"]
    assert len(resource_rows) > 0, "expected at least one resource_access row at audit_cfg scale"
    assert resource_rows["auth_result"].isna().all()
    assert resource_rows["auth_method"].isna().all()
    assert (resource_rows["mfa_used"] == False).all()  # noqa: E712 -- sentinel, not a real auth event
    assert resource_rows["resource_accessed"].notna().all()

    successful_logins = events[(events["event_type"] == "login_attempt") & (events["auth_result"] == "success")]
    assert successful_logins["resource_accessed"].notna().all(), (
        "every successful login must carry a normal resource_accessed value -- "
        "a null resource on a successful login would itself be an attack tell"
    )
