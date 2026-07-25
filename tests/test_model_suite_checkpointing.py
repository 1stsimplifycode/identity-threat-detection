"""Phase 4 (Scale-up resilience): verifies build_model_suite()'s
checkpointing actually resumes correctly -- a second call with the same
checkpoint_dir must (1) be much faster (it loaded from disk instead of
retraining), and (2) produce numerically identical results to the first
call, not just "doesn't crash."
"""
from __future__ import annotations

import time

from evaluation.model_suite import build_model_suite
from tests.test_model_suite import _write_run


def test_checkpointed_rerun_matches_fresh_run_and_is_faster(graph_verification_cfg, tmp_path):
    run_dir = tmp_path / "run"
    _write_run(graph_verification_cfg, run_dir)
    checkpoint_dir = tmp_path / "checkpoints"

    t0 = time.time()
    suite1 = build_model_suite(run_dir, graph_verification_cfg, include_hf=False, checkpoint_dir=checkpoint_dir)
    first_elapsed = time.time() - t0

    # Every checkpoint file expected to exist after a full fresh run.
    assert (checkpoint_dir / "features.parquet").exists()
    assert (checkpoint_dir / "features_meta.json").exists()
    for name in ["rule_based_baseline", "isolation_forest", "xgboost_none", "xgboost_class_weight", "xgboost_smote", "river_online"]:
        assert (checkpoint_dir / f"{name}_eval.pkl").exists(), f"missing eval checkpoint for {name}"
        assert (checkpoint_dir / f"{name}_scores.parquet").exists(), f"missing scores checkpoint for {name}"

    t0 = time.time()
    suite2 = build_model_suite(run_dir, graph_verification_cfg, include_hf=False, checkpoint_dir=checkpoint_dir)
    second_elapsed = time.time() - t0

    # The resumed run must be substantially faster -- it should do almost no
    # real computation, only load from disk.
    assert second_elapsed < first_elapsed / 2, (
        f"resumed run ({second_elapsed:.2f}s) was not meaningfully faster than the fresh run "
        f"({first_elapsed:.2f}s) -- checkpointing may not actually be skipping work"
    )

    # Results must match EXACTLY, not just "both ran without error."
    names1 = {ev.name: ev for ev in suite1.evaluations}
    names2 = {ev.name: ev for ev in suite2.evaluations}
    assert set(names1) == set(names2)
    for name in names1:
        ev1, ev2 = names1[name], names2[name]
        assert ev1.precision == ev2.precision
        assert ev1.recall == ev2.recall
        assert ev1.f1 == ev2.f1
        assert ev1.n_flagged == ev2.n_flagged

    for name in suite1.test_scores:
        s1 = suite1.test_scores[name].set_index("record_id")["score"].sort_index()
        s2 = suite2.test_scores[name].set_index("record_id")["score"].sort_index()
        assert (s1 == s2).all(), f"scores diverged for {name} between fresh and resumed run"

    assert suite1.feature_elapsed_seconds == suite2.feature_elapsed_seconds
    assert len(suite1.features) == len(suite2.features)


def test_partial_checkpoint_resumes_only_missing_models(graph_verification_cfg, tmp_path):
    """Simulates an interrupted run: manually remove one model's checkpoint
    after a full run, then verify a re-run only recomputes that one model
    (the others are loaded from disk) and still produces a complete result.
    """
    run_dir = tmp_path / "run"
    _write_run(graph_verification_cfg, run_dir)
    checkpoint_dir = tmp_path / "checkpoints"

    build_model_suite(run_dir, graph_verification_cfg, include_hf=False, checkpoint_dir=checkpoint_dir)

    # Simulate "the process was killed before xgboost_smote's checkpoint was written".
    (checkpoint_dir / "xgboost_smote_eval.pkl").unlink()
    (checkpoint_dir / "xgboost_smote_scores.parquet").unlink()

    suite = build_model_suite(run_dir, graph_verification_cfg, include_hf=False, checkpoint_dir=checkpoint_dir)
    names = {ev.name for ev in suite.evaluations}
    assert "xgboost_smote" in names
    assert (checkpoint_dir / "xgboost_smote_eval.pkl").exists(), "re-run should have re-saved the missing checkpoint"
