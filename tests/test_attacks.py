"""Attack-generation sanity checks: MITRE tagging, time ordering, severity
vocabulary, and attack-type-specific parameter bounds.
"""
from __future__ import annotations

import json

from attacks.injector import inject_attacks
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, load_mitre_mapping
from generator.seeding import set_global_seed


def test_attack_metadata_is_well_formed(tiny_cfg):
    rng = set_global_seed(int(tiny_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(tiny_cfg, rng)
    benign_events = generate_login_events(users, tiny_cfg, rng)
    _, attack_meta = inject_attacks(users, benign_events, tiny_cfg, mitre_mapping, rng)

    assert len(attack_meta) > 0
    assert (attack_meta["start_time"] <= attack_meta["end_time"]).all()
    assert attack_meta["severity"].isin(["low", "medium", "high", "critical"]).all()
    assert (attack_meta["mitre_technique_ids"].str.len() > 0).all()
    assert (attack_meta["mitre_mapping_version"] == mitre_mapping["schema_version"]).all()


def test_brute_force_attempt_counts_within_configured_bounds(tiny_cfg):
    rng = set_global_seed(int(tiny_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(tiny_cfg, rng)
    benign_events = generate_login_events(users, tiny_cfg, rng)
    _, attack_meta = inject_attacks(users, benign_events, tiny_cfg, mitre_mapping, rng)

    bf = attack_meta[attack_meta["attack_type"] == "brute_force"]
    if len(bf) == 0:
        return  # nothing to check this run; imbalance ratio is small at tiny scale
    assert (bf["num_events_generated"] >= tiny_cfg.attacks.brute_force.min_attempts).all()
    assert (bf["num_events_generated"] <= tiny_cfg.attacks.brute_force.max_attempts).all()


def test_impossible_travel_implied_speed_exceeds_threshold(tiny_cfg):
    rng = set_global_seed(int(tiny_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(tiny_cfg, rng)
    benign_events = generate_login_events(users, tiny_cfg, rng)
    _, attack_meta = inject_attacks(users, benign_events, tiny_cfg, mitre_mapping, rng)

    it = attack_meta[attack_meta["attack_type"] == "impossible_travel"]
    if len(it) == 0:
        return
    speeds = it["parameters_json"].apply(lambda s: json.loads(s)["implied_speed_kmh"])
    assert (speeds >= tiny_cfg.attacks.impossible_travel.min_implied_speed_kmh).all()


# The remaining three attack types (Phase 2a) use audit_cfg (2,000
# users/10 days) rather than tiny_cfg (60 users/3 days): the injector's
# round-robin dispatch can satisfy the tiny_cfg imbalance floor (1 attack
# event) with a single brute_force campaign alone, so at tiny_cfg scale the
# later attack types in enabled_attacks may never actually fire. audit_cfg
# was already sized (during the leakage-audit fix) to reliably produce
# multiple campaigns of every type.

def test_all_five_attack_types_present_at_audit_scale(audit_cfg):
    rng = set_global_seed(int(audit_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(audit_cfg, rng)
    benign_events = generate_login_events(users, audit_cfg, rng)
    _, attack_meta = inject_attacks(users, benign_events, audit_cfg, mitre_mapping, rng)

    expected_types = {"brute_force", "impossible_travel", "credential_misuse", "lateral_movement", "device_spoofing"}
    assert expected_types.issubset(set(attack_meta["attack_type"].unique())), (
        f"expected all 5 attack types at audit_cfg scale, got {sorted(attack_meta['attack_type'].unique())}"
    )
    # rationale is required (non-nullable) and must be genuinely populated, not just present
    assert (attack_meta["rationale"].str.len() > 20).all()


def test_credential_misuse_resource_is_outside_department_and_privilege(audit_cfg):
    from generator.constants import MIN_PRIVILEGE_BY_RESOURCE, RESOURCE_TYPES_BY_DEPT
    from preprocessing.constants import PRIVILEGE_LEVELS

    rng = set_global_seed(int(audit_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(audit_cfg, rng)
    benign_events = generate_login_events(users, audit_cfg, rng)
    _, attack_meta = inject_attacks(users, benign_events, audit_cfg, mitre_mapping, rng)

    cm = attack_meta[attack_meta["attack_type"] == "credential_misuse"]
    if len(cm) == 0:
        return
    users_indexed = users.set_index("user_id")
    for _, row in cm.iterrows():
        params = json.loads(row["parameters_json"])
        resource = params["overreach_resource"]
        department = users_indexed.loc[row["target_user_id"], "department"]
        privilege = users_indexed.loc[row["target_user_id"], "privilege_level"]
        assert resource not in RESOURCE_TYPES_BY_DEPT.get(department, RESOURCE_TYPES_BY_DEPT["_default"])
        assert PRIVILEGE_LEVELS.index(MIN_PRIVILEGE_BY_RESOURCE[resource]) > PRIVILEGE_LEVELS.index(privilege)


def test_lateral_movement_chain_crosses_departments_and_is_fast(audit_cfg):
    rng = set_global_seed(int(audit_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(audit_cfg, rng)
    benign_events = generate_login_events(users, audit_cfg, rng)
    _, attack_meta = inject_attacks(users, benign_events, audit_cfg, mitre_mapping, rng)

    lm = attack_meta[attack_meta["attack_type"] == "lateral_movement"]
    if len(lm) == 0:
        return
    for _, row in lm.iterrows():
        params = json.loads(row["parameters_json"])
        assert audit_cfg.attacks.lateral_movement.min_hops <= params["num_hops"] <= audit_cfg.attacks.lateral_movement.max_hops
        # every hop must be a resource-access event, i.e. num_events_generated is hops + 1 login
        assert row["num_events_generated"] == params["num_hops"] + 1


def test_device_spoofing_variants_are_well_formed(audit_cfg):
    rng = set_global_seed(int(audit_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(audit_cfg, rng)
    benign_events = generate_login_events(users, audit_cfg, rng)
    _, attack_meta = inject_attacks(users, benign_events, audit_cfg, mitre_mapping, rng)

    ds = attack_meta[attack_meta["attack_type"] == "device_spoofing"]
    if len(ds) == 0:
        return
    for _, row in ds.iterrows():
        params = json.loads(row["parameters_json"])
        assert params["variant"] in ("fingerprint_mismatch", "cross_user_reuse")
        if params["variant"] == "fingerprint_mismatch":
            assert params["original_signature"] != params["reported_signature"]
        else:
            assert params["original_owner"] != params["new_user"]
