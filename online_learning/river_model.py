"""Online/streaming learner (Phase 3) -- the one place in this project that
actually exercises the dual-mode feature design's whole point: features and
the River model are both updated ONE EVENT AT A TIME, in chronological
order, through the exact same `FeaturePipelineState.update()` Phase 4's
real-time dashboard will call. This is not "the same dataset trained with a
River model instead of sklearn" -- it is a true prequential (predict, then
learn) streaming loop.

Model: `river.forest.ARFClassifier` (Adaptive Random Forest) -- the problem
statement names "River Hoeffding Tree or Adaptive Random Forest" as
alternatives; ARF is used here based on a real finding, not a coin flip.
A single `HoeffdingTreeClassifier`, tried first (both multi-class over the
6 classes and, after that failed, binary attack/benign), NEVER SPLIT even
once across the full ~40k-row train stream (`model.summary["n_nodes"] == 1`
the entire time), confirmed not to be a delta/grace_period tuning issue by
testing much more permissive settings and a synthetic sanity check proving
the algorithm itself splits fine on trivially separable data. The real
cause: this project's attack signal needs feature INTERACTIONS (which is
exactly what XGBoost's boosted ensemble exploits, at ROC-AUC ~0.99), and a
single incremental tree only gets one shot at a root split from marginal
per-feature distributions -- with ~0.5% attack prevalence, no single
feature's marginal distribution clears the Hoeffding bound for a
statistically confident split. Switching to ARF (10 trees, online bagging)
fixed this directly: ROC-AUC 0.93 vs. the single tree's chance-level 0.50
on identical data.

Binary (attack/benign), not the full 6-class target the batch models use:
concentrates what little minority signal exists onto one decision boundary
instead of spreading it across 5 already-tiny classes, and matches River's
actual role here (fast real-time risk flagging, the same role Isolation
Forest and the rule baseline play) -- genuine multi-class attack-TYPE
prediction is already covered by XGBoost and the HF model, both trained
with full-dataset (not single-pass online) visibility.
"""
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig
from river import forest

from feature_engineering.pipeline import FEATURE_COLUMNS, FeaturePipelineState

RIVER_SCORE_COLUMNS: list[str] = ["river_anomaly_score", "river_predicted_class"]


def build_river_model(cfg: DictConfig) -> forest.ARFClassifier:
    river_cfg = cfg.models.river_online
    return forest.ARFClassifier(n_models=int(river_cfg.n_models), seed=int(river_cfg.seed))


def run_river_online(
    events: pd.DataFrame,
    users: pd.DataFrame,
    labels: pd.DataFrame,
    train_record_ids: frozenset[str],
    cfg: DictConfig,
) -> pd.DataFrame:
    """Streams every event (train AND test) through one shared
    `FeaturePipelineState` and one River model in chronological order.
    Every event is SCORED; the model only ever LEARNS from rows in
    `train_record_ids`, keeping the same train/test discipline as every
    batch model in this project despite the fundamentally different
    (incremental, one-event-at-a-time) training loop.
    """
    events_sorted = events.sort_values("timestamp")
    is_attack_by_record = labels.set_index("record_id")["is_attack"]

    pipeline_state = FeaturePipelineState(cfg, users)
    model = build_river_model(cfg)

    rows: list[dict] = []
    for event in events_sorted.to_dict("records"):
        features = pipeline_state.update(event)
        x = {col: features[col] for col in FEATURE_COLUMNS}

        proba = model.predict_proba_one(x)
        anomaly_score = float(proba.get(True, 0.0)) if proba else 0.0
        predicted_class = "brute_force" if anomaly_score >= 0.5 else "benign"
        # NOTE: River's model here is binary (attack/benign); "brute_force"
        # is used as a placeholder non-benign label purely so this column's
        # values stay within ATTACK_TYPE_CLASSES for the shared multi-class
        # report machinery in evaluation/report.py -- River does not
        # actually distinguish WHICH attack type, only anomalous-or-not.
        # This limitation is stated plainly in docs/phase_3_report.md.

        rows.append({
            "record_id": event["record_id"],
            "river_anomaly_score": anomaly_score,
            "river_predicted_class": predicted_class,
        })

        if event["record_id"] in train_record_ids:
            y = bool(is_attack_by_record.get(event["record_id"], False))
            model.learn_one(x, y)

    return pd.DataFrame(rows, columns=["record_id"] + RIVER_SCORE_COLUMNS)
