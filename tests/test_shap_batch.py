"""Phase 4: offline exact SHAP over flagged events must (1) select only
flagged record_ids, (2) return well-formed top-k contributors that are
FEATURE_COLUMNS names with genuine SHAP values, and (3) never crash on the
degenerate all-benign-predicted case.
"""
from __future__ import annotations

import pandas as pd

from evaluation.chronological_split import apply_split, chronological_split
from explainability.shap_batch import explain_flagged_events, select_flagged_sample
from feature_engineering.pipeline import FEATURE_COLUMNS, compute_feature_table
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, _assemble_events_and_labels, _assign_bookkeeping_fields, load_mitre_mapping
from generator.seeding import set_global_seed
from attacks.injector import inject_attacks
from models.xgboost_classifier import score_xgboost, train_xgboost


def _generate_with_features(cfg):
    rng = set_global_seed(int(cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")
    users = generate_population(cfg, rng)
    benign_events = generate_login_events(users, cfg, rng)
    attack_events, attack_meta = inject_attacks(users, benign_events, cfg, mitre_mapping, rng)
    events, labels = _assemble_events_and_labels(benign_events, attack_events, attack_meta)
    events, labels = _assign_bookkeeping_fields(events, labels, int(cfg.attacks.num_generation_batches), rng)
    features = compute_feature_table(events, users, cfg)
    return events, labels, features


def test_shap_explanations_are_well_formed_for_flagged_events(graph_verification_cfg):
    events, labels, features = _generate_with_features(graph_verification_cfg)
    split = chronological_split(events, train_frac=0.7)
    train_features = apply_split(features, split, "train")
    test_features = apply_split(features, split, "test")
    train_labels = apply_split(labels, split, "train")

    model = train_xgboost(train_features, train_labels, "smote", graph_verification_cfg)
    scores = score_xgboost(model, test_features)

    threshold = scores["xgb_anomaly_score"].quantile(0.9)
    flagged_ids = select_flagged_sample(scores, "xgb_anomaly_score", threshold, max_samples=None, seed=42)
    assert len(flagged_ids) > 0

    explanations = explain_flagged_events(
        model, test_features, scores["xgb_predicted_class"], flagged_ids, top_k=5,
    )

    assert set(explanations["record_id"]) == set(flagged_ids)
    assert explanations["explanation"].str.startswith("Flagged as").all()

    for top_features in explanations["top_features"]:
        assert 1 <= len(top_features) <= 5
        for entry in top_features:
            assert entry["feature"] in FEATURE_COLUMNS
            assert isinstance(entry["shap_value"], float)
            assert isinstance(entry["feature_value"], float)
            assert entry["description"]


def test_select_flagged_sample_respects_max_samples_cap():
    scores = pd.DataFrame({
        "record_id": [f"r{i}" for i in range(100)],
        "score": [1.0] * 100,  # everything flagged at threshold 0.5
    })
    sampled = select_flagged_sample(scores, "score", threshold=0.5, max_samples=10, seed=1)
    assert len(sampled) == 10
    assert set(sampled) <= set(scores["record_id"])
