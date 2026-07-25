"""Determinism: the same seed must produce byte-identical output, and a
different seed must produce different output (otherwise the seed isn't
actually doing anything).
"""
from __future__ import annotations

import pandas as pd
from omegaconf import OmegaConf

from generator.events import generate_login_events
from generator.population import generate_population
from generator.seeding import set_global_seed


def _generate(cfg):
    rng = set_global_seed(int(cfg.seed))
    users = generate_population(cfg, rng)
    events = generate_login_events(users, cfg, rng)
    return users, events


def test_same_seed_produces_identical_output(tiny_cfg):
    users_a, events_a = _generate(tiny_cfg)
    users_b, events_b = _generate(tiny_cfg)

    pd.testing.assert_frame_equal(users_a, users_b)
    pd.testing.assert_frame_equal(events_a, events_b)


def test_different_seed_produces_different_output(tiny_cfg):
    cfg_alt = OmegaConf.merge(tiny_cfg, {"seed": 999})

    users_a, events_a = _generate(tiny_cfg)
    users_b, events_b = _generate(cfg_alt)

    assert not users_a["department"].equals(users_b["department"]) or not events_a["timestamp"].equals(events_b["timestamp"])
