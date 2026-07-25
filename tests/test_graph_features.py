"""Phase 2b explicit verification (per the approved plan): the graph
features must actually spike on the attacks they're meant to catch, checked
with real assertions -- not eyeballed in a notebook.

Uses `graph_verification_cfg` (higher imbalance_ratio than real small_dev)
so every attack type gets a large-enough sample within one run; see
tests/conftest.py's comment on why the real small_dev profile can starve
non-brute_force types of a fair sample via the injector's round-robin.
"""
from __future__ import annotations

import json

from attacks.injector import inject_attacks
from feature_engineering.pipeline import compute_feature_table
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, _assemble_events_and_labels, _assign_bookkeeping_fields, load_mitre_mapping
from generator.seeding import set_global_seed


def _generate_full_run_with_features(cfg):
    rng = set_global_seed(int(cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(cfg, rng)
    benign_events = generate_login_events(users, cfg, rng)
    attack_events, attack_meta = inject_attacks(users, benign_events, cfg, mitre_mapping, rng)
    events, labels = _assemble_events_and_labels(benign_events, attack_events, attack_meta)
    events, labels = _assign_bookkeeping_fields(events, labels, int(cfg.attacks.num_generation_batches), rng)

    features = compute_feature_table(events, users, cfg)
    merged = events.merge(features, on="record_id").merge(
        labels[["record_id", "is_attack", "attack_type", "attack_id"]], on="record_id"
    ).merge(attack_meta[["attack_id", "parameters_json"]], on="attack_id", how="left")
    return merged


def test_access_chain_distance_spikes_on_lateral_movement(graph_verification_cfg):
    merged = _generate_full_run_with_features(graph_verification_cfg)
    with_resource = merged[merged["resource_accessed"].notna()]

    lm_rows = with_resource[with_resource["attack_type"] == "lateral_movement"]
    benign_rows = with_resource[with_resource["is_attack"] == False]  # noqa: E712

    assert len(lm_rows) >= 10, f"expected a meaningful lateral_movement sample, got {len(lm_rows)}"

    lm_mean = lm_rows["access_chain_distance"].mean()
    benign_mean = benign_rows["access_chain_distance"].mean()

    assert lm_mean > benign_mean, (
        f"access_chain_distance did not spike on lateral_movement: "
        f"lateral_movement mean={lm_mean:.3f} <= benign mean={benign_mean:.3f}"
    )
    # not just "slightly higher" -- meaningfully so
    assert lm_mean > 1.5 * benign_mean


def test_is_new_edge_spikes_on_device_spoofing_cross_user_reuse(graph_verification_cfg):
    """Only the cross_user_reuse variant is expected to spike is_new_edge --
    fingerprint_mismatch reuses the SAME (user, device) pair, so the
    bipartite user-device edge already exists and is_new_edge is correctly
    ~0 for that variant. `is_new_edge` (and the other 4 original graph
    features) track (user, device) pairing identity/topology, not device
    fingerprint consistency -- that gap is covered separately by
    `device_fingerprint_mismatch`, see
    `test_device_fingerprint_mismatch_spikes_on_device_spoofing_fingerprint_variant`
    below and `docs/phase_5_recall_investigation.md`.
    """
    merged = _generate_full_run_with_features(graph_verification_cfg)
    ds_rows = merged[merged["attack_type"] == "device_spoofing"].copy()
    ds_rows["variant"] = ds_rows["parameters_json"].apply(
        lambda s: json.loads(s)["variant"] if s is not None else None
    )
    benign_rows = merged[merged["is_attack"] == False]  # noqa: E712

    cross_user_rows = ds_rows[ds_rows["variant"] == "cross_user_reuse"]
    assert len(cross_user_rows) >= 3, f"expected a meaningful cross_user_reuse sample, got {len(cross_user_rows)}"

    cross_user_mean = cross_user_rows["is_new_edge"].mean()
    benign_mean = benign_rows["is_new_edge"].mean()

    assert cross_user_mean > benign_mean, (
        f"is_new_edge did not spike on device_spoofing (cross_user_reuse): "
        f"attack mean={cross_user_mean:.3f} <= benign mean={benign_mean:.3f}"
    )
    assert cross_user_mean > 0.5  # should be strongly, not marginally, elevated


def test_device_fingerprint_mismatch_spikes_on_device_spoofing_fingerprint_variant(graph_verification_cfg):
    """The variant `is_new_edge` cannot see (same user, same device_id,
    changed device_type/os) -- `device_fingerprint_mismatch` exists
    specifically to catch it. See `docs/phase_5_recall_investigation.md`
    for the measurement that motivated adding this feature.
    """
    merged = _generate_full_run_with_features(graph_verification_cfg)
    ds_rows = merged[merged["attack_type"] == "device_spoofing"].copy()
    ds_rows["variant"] = ds_rows["parameters_json"].apply(
        lambda s: json.loads(s)["variant"] if s is not None else None
    )
    benign_rows = merged[merged["is_attack"] == False]  # noqa: E712

    mismatch_rows = ds_rows[ds_rows["variant"] == "fingerprint_mismatch"]
    assert len(mismatch_rows) >= 3, f"expected a meaningful fingerprint_mismatch sample, got {len(mismatch_rows)}"

    attack_mean = mismatch_rows["device_fingerprint_mismatch"].mean()
    benign_mean = benign_rows["device_fingerprint_mismatch"].mean()

    assert attack_mean > benign_mean, (
        f"device_fingerprint_mismatch did not spike on device_spoofing (fingerprint_mismatch): "
        f"attack mean={attack_mean:.3f} <= benign mean={benign_mean:.3f}"
    )
    assert attack_mean > 0.5  # should be strongly, not marginally, elevated


def test_session_features_spike_on_lateral_movement(graph_verification_cfg):
    """access_chain_distance (a single direct-transition cost) left
    lateral_movement at 0% recall even after Stage B's device-spoofing fix
    -- session_foreign_resource_count / session_hop_seconds target the
    session-WIDE mechanism attacks/lateral_movement.py actually generates
    (multiple cross-department resources, touched fast, in one session).
    See docs/phase_5_recall_investigation.md.
    """
    merged = _generate_full_run_with_features(graph_verification_cfg)
    with_resource = merged[merged["resource_accessed"].notna()]

    lm_rows = with_resource[with_resource["attack_type"] == "lateral_movement"]
    benign_rows = with_resource[with_resource["is_attack"] == False]  # noqa: E712
    assert len(lm_rows) >= 10, f"expected a meaningful lateral_movement sample, got {len(lm_rows)}"

    lm_foreign_mean = lm_rows["session_foreign_resource_count"].mean()
    benign_foreign_mean = benign_rows["session_foreign_resource_count"].mean()
    assert lm_foreign_mean > benign_foreign_mean, (
        f"session_foreign_resource_count did not spike on lateral_movement: "
        f"attack mean={lm_foreign_mean:.3f} <= benign mean={benign_foreign_mean:.3f}"
    )
    assert lm_foreign_mean > 1.5 * max(benign_foreign_mean, 1e-6)

    lm_hop_mean = lm_rows["session_hop_seconds"].mean()
    benign_hop_mean = benign_rows["session_hop_seconds"].mean()
    assert lm_hop_mean < benign_hop_mean, (
        f"session_hop_seconds did not drop (faster hopping) on lateral_movement: "
        f"attack mean={lm_hop_mean:.1f}s >= benign mean={benign_hop_mean:.1f}s"
    )


def test_dual_mode_state_is_deterministic_given_replayed_order(graph_verification_cfg):
    """Sanity check on the dual-mode design contract: replaying the exact
    same chronologically-sorted event stream through a fresh state object
    twice must produce identical feature values -- there is no hidden
    non-determinism (e.g. dict-ordering, unseeded randomness) in the
    feature computation itself. This does not yet test true batch-vs-live
    streaming equivalence (no live streaming loop exists before Phase 4),
    only that compute_batch() is a pure function of its input.
    """
    from feature_engineering.pipeline import compute_feature_table
    from generator.events import generate_login_events as _gen_events

    rng = set_global_seed(int(graph_verification_cfg.seed))
    users = generate_population(graph_verification_cfg, rng)
    events = _gen_events(users, graph_verification_cfg, rng)
    events = events.reset_index(drop=True)
    events["record_id"] = [f"r{i}" for i in range(len(events))]

    feats_a = compute_feature_table(events, users, graph_verification_cfg)
    feats_b = compute_feature_table(events, users, graph_verification_cfg)

    import pandas as pd
    pd.testing.assert_frame_equal(feats_a, feats_b)
