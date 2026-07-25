"""Unsupervised/statistical baseline model (constraint: "one ML model" for
Phase 2b) -- learns "normal" from the merged behavioral+graph feature table
without ever seeing labels, consistent with this project's behavioral (not
signature-based) design philosophy.
"""
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import IsolationForest

from feature_engineering.pipeline import FEATURE_COLUMNS


def train_isolation_forest(train_features: pd.DataFrame, cfg: DictConfig) -> IsolationForest:
    if_cfg = cfg.models.isolation_forest
    model = IsolationForest(
        n_estimators=int(if_cfg.n_estimators),
        contamination=float(if_cfg.contamination),
        random_state=int(if_cfg.random_state),
    )
    model.fit(train_features[FEATURE_COLUMNS].to_numpy())
    return model


def score_isolation_forest(model: IsolationForest, features: pd.DataFrame) -> pd.DataFrame:
    """Higher `iforest_anomaly_score` = more anomalous (sklearn's own
    `decision_function` runs the opposite way -- higher = more normal --
    so it's negated here to keep score direction consistent with
    `models/baseline.py`'s `rule_risk_score`).
    """
    scores = -model.decision_function(features[FEATURE_COLUMNS].to_numpy())
    return pd.DataFrame({"record_id": features["record_id"].to_numpy(), "iforest_anomaly_score": scores})
