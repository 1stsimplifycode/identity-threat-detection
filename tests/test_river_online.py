"""Phase 3: River online learner -- the streaming loop must score every
event (train and test alike), and the model must actually have LEARNED
something (not just returned well-formed output) -- see
online_learning/river_model.py's module docstring: a single
HoeffdingTreeClassifier never split even once on this project's real data,
found only by checking the model's own internal state, not by eyeballing
score shape. ARFClassifier is checked here for exactly that: real,
non-trivial score variance, not just "the function ran."
"""
from __future__ import annotations

from attacks.injector import inject_attacks
from evaluation.chronological_split import chronological_split
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, _assemble_events_and_labels, _assign_bookkeeping_fields, load_mitre_mapping
from generator.seeding import set_global_seed
from models.xgboost_classifier import ATTACK_TYPE_CLASSES
from online_learning.river_model import run_river_online


def test_river_online_scores_every_event_and_learns_something(graph_verification_cfg):
    rng = set_global_seed(int(graph_verification_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")
    users = generate_population(graph_verification_cfg, rng)
    benign_events = generate_login_events(users, graph_verification_cfg, rng)
    attack_events, attack_meta = inject_attacks(users, benign_events, graph_verification_cfg, mitre_mapping, rng)
    events, labels = _assemble_events_and_labels(benign_events, attack_events, attack_meta)
    events, labels = _assign_bookkeeping_fields(events, labels, int(graph_verification_cfg.attacks.num_generation_batches), rng)

    split = chronological_split(events, train_frac=0.7)
    scores = run_river_online(events, users, labels, split.train_record_ids, graph_verification_cfg)

    assert len(scores) == len(events)
    assert set(events["record_id"]) == set(scores["record_id"])
    assert scores["river_anomaly_score"].between(0, 1).all()
    assert scores["river_predicted_class"].isin(ATTACK_TYPE_CLASSES).all()

    # The real regression check: a model stuck at a single root leaf (never
    # split) produces a near-constant score for every event, regardless of
    # input. Real learning shows up as meaningful score variance.
    assert scores["river_anomaly_score"].std() > 0.01, (
        "river_anomaly_score has near-zero variance -- the online model may "
        "never have split (see online_learning/river_model.py's docstring "
        "for the investigation this exact failure mode came from)"
    )
