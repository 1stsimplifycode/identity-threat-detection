"""Phase 4 (Scale-up resilience): the single highest-stakes test in this
resilience effort. `compute_feature_table_resumable()` checkpoints live,
mutable feature-computation state (behavioral + graph rolling structures)
mid-stream -- a subtle bug here would silently corrupt every downstream
model's input on a resumed Scale-up run, which is worse than the slow
crashes this mechanism exists to survive. This test genuinely interrupts
computation (via an injected exception, not a graceful stop) partway
through, confirms a real partial checkpoint was left behind, resumes, and
asserts the resumed result is BYTE-IDENTICAL to an uninterrupted reference
run -- not just "both finished without error."
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

import feature_engineering.pipeline as pipeline
from feature_engineering.pipeline import compute_feature_table, compute_feature_table_resumable
from tests.test_model_suite import _write_run


def test_resumable_feature_computation_survives_interruption_and_matches_reference(graph_verification_cfg, tmp_path):
    run_dir = tmp_path / "run"
    _write_run(graph_verification_cfg, run_dir)
    users = pd.read_parquet(run_dir / "users.parquet")
    events = pd.read_parquet(run_dir / "events.parquet")

    reference = compute_feature_table(events, users, graph_verification_cfg)

    checkpoint_dir = tmp_path / "feature_checkpoints"
    checkpoint_every = 1000
    crash_after = len(events) // 2
    assert crash_after > checkpoint_every, "test dataset too small to exercise a real checkpoint before the injected crash"

    original_update = pipeline.FeaturePipelineState.update
    call_count = {"n": 0}

    def flaky_update(self, event):
        call_count["n"] += 1
        if call_count["n"] == crash_after:
            raise RuntimeError("simulated crash mid-feature-computation")
        return original_update(self, event)

    with patch.object(pipeline.FeaturePipelineState, "update", flaky_update):
        with pytest.raises(RuntimeError, match="simulated crash"):
            compute_feature_table_resumable(events, users, graph_verification_cfg, checkpoint_dir, checkpoint_every=checkpoint_every)

    state_path, rows_path, progress_path = pipeline._feature_checkpoint_paths(checkpoint_dir)
    assert state_path.exists() and rows_path.exists() and progress_path.exists(), "no checkpoint survived the simulated crash"
    n_done_at_crash = json.loads(progress_path.read_text(encoding="utf-8"))["n_done"]
    assert 0 < n_done_at_crash < crash_after, f"unexpected checkpoint progress: {n_done_at_crash}"

    resumed = compute_feature_table_resumable(events, users, graph_verification_cfg, checkpoint_dir, checkpoint_every=checkpoint_every)

    assert not state_path.exists(), "checkpoint files should be cleaned up after successful completion"
    assert not rows_path.exists()
    assert not progress_path.exists()

    ref_sorted = reference.sort_values("record_id").reset_index(drop=True)
    res_sorted = resumed.sort_values("record_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(ref_sorted, res_sorted)


def test_resumable_feature_computation_matches_reference_with_no_interruption(graph_verification_cfg, tmp_path):
    """Sanity check independent of the crash-injection machinery: the
    resumable path with checkpointing enabled but never actually
    interrupted must still match the plain (non-resumable) computation.
    """
    run_dir = tmp_path / "run"
    _write_run(graph_verification_cfg, run_dir)
    users = pd.read_parquet(run_dir / "users.parquet")
    events = pd.read_parquet(run_dir / "events.parquet")

    reference = compute_feature_table(events, users, graph_verification_cfg)
    resumable_result = compute_feature_table_resumable(
        events, users, graph_verification_cfg, tmp_path / "checkpoints", checkpoint_every=1000,
    )

    ref_sorted = reference.sort_values("record_id").reset_index(drop=True)
    res_sorted = resumable_result.sort_values("record_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(ref_sorted, res_sorted)
