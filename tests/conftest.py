"""Shared test fixtures: a tiny (fast) Hydra config for unit tests, separate
from configs/small_dev.yaml (which is already small, but tests use an even
tinier profile so the whole suite runs in a couple of seconds).
"""
# IMPORTANT: torch before pandas/pyarrow anywhere in this process -- see
# evaluation/run_evaluation.py's identical comment; the same Windows DLL
# conflict applies to the test process too, since test_hf_classifier.py
# imports both.
import torch  # noqa: F401

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str((Path(__file__).resolve().parent.parent / "configs"))

TINY_OVERRIDES = [
    "population.num_users=60",
    "events.num_days=3",
]

# Larger than tiny_cfg specifically for the leakage audit: with only ~51
# attack events (tiny_cfg's scale), a 30%-holdout test fold has so few
# positives that ROC-AUC sampling noise alone can exceed epsilon on an
# honest, leak-free dataset (observed directly during Phase 1 development).
# This fixture gives the audit enough attack events to be statistically
# meaningful while still running in a couple of seconds.
AUDIT_OVERRIDES = [
    "population.num_users=2000",
    "events.num_days=10",
]


@pytest.fixture
def tiny_cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="small_dev", overrides=TINY_OVERRIDES)
    return cfg


@pytest.fixture
def audit_cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="small_dev", overrides=AUDIT_OVERRIDES)
    return cfg


# Higher imbalance_ratio than the real small_dev profile (0.005): at that
# rate, brute_force's large per-campaign event count can satisfy the whole
# attack-event quota within 1-2 campaigns via the injector's round-robin
# dispatch, starving the other 4 types of a fair sample within one run. This
# fixture exists purely so Phase 2b's graph-feature verification tests get
# enough of every attack type for a statistically meaningful comparison --
# it is NOT meant to imply 5% is a realistic imbalance level.
GRAPH_VERIFICATION_OVERRIDES = [
    "population.num_users=2000",
    "events.num_days=10",
    "attacks.imbalance_ratio=0.05",
]


@pytest.fixture
def graph_verification_cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="small_dev", overrides=GRAPH_VERIFICATION_OVERRIDES)
    return cfg
