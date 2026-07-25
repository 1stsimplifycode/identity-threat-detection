"""Phase 4: `evaluation/model_suite.py` is the shared model-training path
both the evaluation report and the dashboard depend on -- verifies its
returned structure (models dict, normalized test_scores, evaluation
lookup) is well-formed, using include_hf=False to keep this fast (HF
fine-tuning/inference isn't what this test is checking).
"""
from __future__ import annotations

from evaluation.model_suite import PRIMARY_MODEL_NAME, build_model_suite
from generator.drift import resolve_drift_schedule
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, _assemble_events_and_labels, _assign_bookkeeping_fields, load_mitre_mapping
from generator.seeding import set_global_seed
from attacks.injector import inject_attacks


def _write_run(cfg, out_dir):
    rng = set_global_seed(int(cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")
    users = generate_population(cfg, rng)
    drift, drift_log = resolve_drift_schedule(users, cfg, rng)
    benign_events = generate_login_events(users, cfg, rng, drift=drift)
    attack_events, attack_meta = inject_attacks(users, benign_events, cfg, mitre_mapping, rng)
    events, labels = _assemble_events_and_labels(benign_events, attack_events, attack_meta)
    events, labels = _assign_bookkeeping_fields(events, labels, int(cfg.attacks.num_generation_batches), rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    users.to_parquet(out_dir / "users.parquet", index=False)
    events.to_parquet(out_dir / "events.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    attack_meta.to_parquet(out_dir / "attacks.parquet", index=False)
    drift_log.to_csv(out_dir / "drift_log.csv", index=False)


def test_build_model_suite_returns_well_formed_result(graph_verification_cfg, tmp_path):
    run_dir = tmp_path / "run"
    _write_run(graph_verification_cfg, run_dir)

    suite = build_model_suite(run_dir, graph_verification_cfg, include_hf=False)

    expected_models = {"rule_based_baseline", "isolation_forest", "xgboost_none", "xgboost_class_weight", "xgboost_smote", "river_online"}
    evaluated_names = {ev.name for ev in suite.evaluations}
    assert expected_models <= evaluated_names
    assert PRIMARY_MODEL_NAME in suite.models
    assert suite.primary_model is suite.models[PRIMARY_MODEL_NAME]

    for name in expected_models:
        scores = suite.test_scores[name]
        assert "record_id" in scores.columns and "score" in scores.columns
        assert len(scores) == suite.split.n_test

    ev = suite.evaluation_for(PRIMARY_MODEL_NAME)
    assert ev.name == PRIMARY_MODEL_NAME

    assert suite.drift_eval is not None
    assert len(suite.train_features) == suite.split.n_train
    assert len(suite.test_features) == suite.split.n_test
