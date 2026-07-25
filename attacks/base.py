"""Shared helpers for attack generators: MITRE mapping loading and severity
sampling.

Every attack generator in this package returns
`(list[dict] events, dict attack_metadata)` where each event dict has
exactly the same keys as a benign event (see generator/events.py's
EVENTS_COLUMN_ORDER) plus two temporary keys, `_tmp_attack_id` and
`_tmp_attack_type`, which `generator/run.py` extracts into the separate
labels table and then drops -- the events table itself never contains an
is_attack/attack_id column, so a model reading the events table has no
label sitting in its feature space.
"""
from __future__ import annotations

import yaml
import numpy as np
from omegaconf import DictConfig

SEVERITY_ORDER: list[str] = ["low", "medium", "high", "critical"]


def load_mitre_mapping(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_severity(cfg: DictConfig, rng: np.random.Generator, boost: bool = False) -> str:
    """Sample a severity label from `cfg.attacks.severity_weights`.

    `boost=True` shifts probability mass toward the higher end of
    SEVERITY_ORDER (used when an attack succeeded or was unusually large).
    """
    weights = dict(cfg.attacks.severity_weights)
    labels = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    if boost:
        idx = np.array([SEVERITY_ORDER.index(label) for label in labels], dtype=float)
        probs = probs * (1 + 0.5 * idx)
    probs = probs / probs.sum()
    return str(labels[int(rng.choice(len(labels), p=probs))])
