"""Phase 3: XGBoost multi-class classifier -- all 3 imbalance-handling
conditions must train without error (including on classes with very few
train examples, where SMOTE's k_neighbors-shrinking / RandomOverSampler
fallback matters), and scores must be well-formed.
"""
from __future__ import annotations

from evaluation.chronological_split import apply_split, chronological_split
from feature_engineering.pipeline import compute_feature_table
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, _assemble_events_and_labels, _assign_bookkeeping_fields, load_mitre_mapping
from generator.seeding import set_global_seed
from attacks.injector import inject_attacks
from models.xgboost_classifier import ATTACK_TYPE_CLASSES, score_xgboost, train_xgboost


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


def test_all_imbalance_methods_train_and_score(graph_verification_cfg):
    events, labels, features = _generate_with_features(graph_verification_cfg)
    split = chronological_split(events, train_frac=0.7)
    train_features = apply_split(features, split, "train")
    test_features = apply_split(features, split, "test")
    train_labels = apply_split(labels, split, "train")

    for method in graph_verification_cfg.models.xgboost.imbalance_methods:
        model = train_xgboost(train_features, train_labels, str(method), graph_verification_cfg)
        scores = score_xgboost(model, test_features)

        assert set(scores.columns) == {"record_id", "xgb_anomaly_score", "xgb_predicted_class"}
        assert len(scores) == len(test_features)
        assert scores["xgb_anomaly_score"].between(0, 1).all()
        assert scores["xgb_predicted_class"].isin(ATTACK_TYPE_CLASSES).all()
