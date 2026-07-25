"""Merges behavioral (3a) and graph-derived (3b) features into one feature
table, keyed by `record_id` -- the join key used to attach labels later,
never a model input itself (same discipline as every other bookkeeping
field in this project).

`FeaturePipelineState` wraps one `BehavioralFeatureState` and one
`GraphFeatureState` behind a single `update()`/`compute_batch()` interface,
so Phase 4's streaming inference loop can hold one object alive per
deployed process rather than wiring two separately.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from feature_engineering.behavioral import BehavioralFeatureState
from feature_engineering.feature_names import FEATURE_COLUMNS
from feature_engineering.graph import GraphFeatureState

# Default granularity for compute_feature_table_resumable()'s periodic
# checkpoints. At full (Scale-up) scale this stage alone has been observed
# to run 40+ minutes -- by far the single longest uninterrupted stretch in
# the whole pipeline (see docs/scale_up_report.md). Checkpointing every
# 50,000 events bounds how much work an interruption can lose to a few
# minutes, at the cost of periodically re-serializing the state and partial
# output (a few seconds each, small next to the stage's total runtime).
RESUMABLE_CHECKPOINT_INTERVAL = 50_000


class FeaturePipelineState:
    def __init__(self, cfg: DictConfig, users: pd.DataFrame) -> None:
        self.behavioral = BehavioralFeatureState(cfg, users)
        self.graph = GraphFeatureState(cfg, users)

    def update(self, event: dict[str, Any]) -> dict[str, Any]:
        features = self.behavioral.update(event)
        features.update(self.graph.update(event))
        return features

    def compute_batch(self, events: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for event in events.to_dict("records"):
            feats = self.update(event)
            feats["record_id"] = event["record_id"]
            rows.append(feats)
        return pd.DataFrame(rows, columns=["record_id"] + FEATURE_COLUMNS)


def compute_feature_table(events: pd.DataFrame, users: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Fresh-state batch computation: sorts `events` chronologically
    (defensive -- `generator/run.py`'s output is already sorted, but this
    function shouldn't silently assume its caller got that right), builds a
    new `FeaturePipelineState`, and returns the merged feature table.
    """
    events_sorted = events.sort_values("timestamp")
    state = FeaturePipelineState(cfg, users)
    return state.compute_batch(events_sorted)


def _feature_checkpoint_paths(checkpoint_dir: Path) -> tuple[Path, Path, Path]:
    return (
        checkpoint_dir / "feature_state.pkl",
        checkpoint_dir / "feature_rows_partial.parquet",
        checkpoint_dir / "feature_progress.json",
    )


def compute_feature_table_resumable(
    events: pd.DataFrame,
    users: pd.DataFrame,
    cfg: DictConfig,
    checkpoint_dir: Path,
    checkpoint_every: int = RESUMABLE_CHECKPOINT_INTERVAL,
    verbose: bool = True,
) -> pd.DataFrame:
    """Same result as `compute_feature_table()`, but resumable: every
    `checkpoint_every` events, the live `FeaturePipelineState` (every
    rolling behavioral/graph structure -- all plain dicts/deques/sets/
    NetworkX graphs, confirmed picklable) plus the rows computed so far are
    saved to `checkpoint_dir`. A re-run with the SAME `checkpoint_dir`
    resumes from the last save instead of recomputing from the first
    event. Intermediate checkpoint files are removed once this function
    returns successfully -- they represent in-progress state only; the
    caller (`evaluation/model_suite.py`) checkpoints the FINAL feature
    table separately, once this stage is genuinely complete.
    """
    state_path, rows_path, progress_path = _feature_checkpoint_paths(checkpoint_dir)
    events_sorted = events.sort_values("timestamp").reset_index(drop=True)
    total = len(events_sorted)

    if state_path.exists() and progress_path.exists() and rows_path.exists():
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        rows: list[dict[str, Any]] = pd.read_parquet(rows_path).to_dict("records")
        n_done = int(json.loads(progress_path.read_text(encoding="utf-8"))["n_done"])
        if verbose:
            print(f"[checkpoint] resuming feature computation from event {n_done}/{total}", flush=True)
    else:
        state = FeaturePipelineState(cfg, users)
        rows = []
        n_done = 0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    remaining = events_sorted.iloc[n_done:].to_dict("records")
    for offset, event in enumerate(remaining):
        i = n_done + offset
        feats = state.update(event)
        feats["record_id"] = event["record_id"]
        rows.append(feats)

        if (i + 1) % checkpoint_every == 0:
            with open(state_path, "wb") as f:
                pickle.dump(state, f)
            pd.DataFrame(rows, columns=["record_id"] + FEATURE_COLUMNS).to_parquet(rows_path, index=False)
            progress_path.write_text(json.dumps({"n_done": i + 1}), encoding="utf-8")
            if verbose:
                print(f"[checkpoint] feature computation progress: {i + 1}/{total}", flush=True)

    result = pd.DataFrame(rows, columns=["record_id"] + FEATURE_COLUMNS)

    for p in (state_path, rows_path, progress_path):
        p.unlink(missing_ok=True)

    return result
