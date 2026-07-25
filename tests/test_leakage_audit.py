"""Leakage audit (constraint #2): the generated dataset must pass, and --
critically -- the audit itself must be able to catch a deliberately leaky
dataset (a negative control proving the audit has teeth, not just that our
generator happens to pass it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from attacks.injector import inject_attacks
from evaluation.leakage_audit import run_leakage_audit
from generator.events import generate_login_events
from generator.population import generate_population
from generator.run import PROJECT_ROOT, _assemble_events_and_labels, _assign_bookkeeping_fields, load_mitre_mapping
from generator.seeding import set_global_seed


def test_generated_data_passes_leakage_audit(audit_cfg):
    # Uses audit_cfg (2,000 users / 10 days), not tiny_cfg (60 users / 3
    # days): tiny_cfg produces so few attack events (~51) that a 30%-holdout
    # test fold has too few positives for ROC-AUC to be a stable estimate,
    # and an honest, leak-free dataset can occasionally exceed epsilon by
    # sampling noise alone. See evaluation/leakage_audit.py's module
    # docstring and tests/conftest.py's AUDIT_OVERRIDES comment.
    rng = set_global_seed(int(audit_cfg.seed))
    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(audit_cfg, rng)
    benign_events = generate_login_events(users, audit_cfg, rng)
    attack_events, attack_meta = inject_attacks(users, benign_events, audit_cfg, mitre_mapping, rng)
    events, labels = _assemble_events_and_labels(benign_events, attack_events, attack_meta)
    events, labels = _assign_bookkeeping_fields(events, labels, int(audit_cfg.attacks.num_generation_batches), rng)

    result = run_leakage_audit(events, labels, epsilon=float(audit_cfg.audit.leakage_epsilon))
    assert result.passed, result.report()


def test_audit_detects_a_deliberately_leaky_dataset():
    """Negative control: without this test, a leakage_audit that always
    returns passed=True would look identical to a correct one.
    """
    n = 2000
    rng = np.random.default_rng(0)
    is_attack = rng.random(n) < 0.05
    record_id = [f"r{i}" for i in range(n)]

    # Deliberately leaky: attack rows are clustered at the end of
    # insertion_order, and generation_batch encodes the label outright --
    # exactly what constraint #2 forbids the real generator from doing.
    order = np.argsort(is_attack.astype(int))
    insertion_order = np.empty(n, dtype=int)
    insertion_order[order] = np.arange(n)
    generation_batch = is_attack.astype(int) * 10

    events = pd.DataFrame({
        "record_id": record_id,
        "insertion_order": insertion_order,
        "generation_batch": generation_batch,
    })
    labels = pd.DataFrame({"record_id": record_id, "is_attack": is_attack})

    result = run_leakage_audit(events, labels, epsilon=0.05)

    assert not result.passed, "leakage audit failed to catch a deliberately leaky dataset"
    assert result.roc_auc > 0.9
