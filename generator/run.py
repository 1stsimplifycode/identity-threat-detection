"""Hydra entrypoint: generate a full run (users, events, labels, attacks) and
write it under `cfg.run.output_dir`.

Usage (from the project root, with the venv active):
    python -m generator.run --config-name small_dev
    python -m generator.run --config-name config          # full scale

IMPORTANT: run as a module (`python -m generator.run ...`), not as a script
(`python generator/run.py`) -- the latter does not put the project root on
sys.path, which breaks the `from generator...` / `from attacks...` imports.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from attacks.injector import inject_attacks
from generator.constants import random_hex
from generator.drift import resolve_drift_schedule
from generator.events import generate_login_events
from generator.population import generate_population
from generator.seeding import set_global_seed
from preprocessing.schema import ATTACKS_SCHEMA, EVENTS_SCHEMA, LABELS_SCHEMA, USERS_SCHEMA, validate_dataframe

# Columns that must stay boolean after pd.concat -- pandas can silently
# upcast a bool column to object dtype during concat depending on the other
# frame's dtype for that column (notably when the other frame is empty and
# lacks the column outright), so this is re-asserted explicitly rather than
# trusted to concat's dtype inference.
BOOL_COLUMNS: list[str] = ["mfa_used", "is_weekend", "is_holiday", "is_off_hours"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DISCLAIMER = (
    "Synthetic data. Not derived from or validated against real organizational "
    "logs. For benchmarking detection methods only."
)


def load_mitre_mapping(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _assemble_events_and_labels(benign_events: pd.DataFrame, attack_events: pd.DataFrame, attack_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine benign and attack events into one events table, and build the
    separate labels table.

    Deliberately avoids `pd.concat`-ing a column that is 100%-NaN on one
    side (e.g. an all-None `_tmp_attack_id` for every benign row) against a
    populated column on the other: besides triggering pandas' "concatenation
    with empty or all-NA entries" dtype-inference warning, that pattern is
    exactly the kind of implicit, easy-to-get-wrong bookkeeping this project
    wants to avoid near the label. The attack tags are instead carried as
    plain Python lists, aligned by explicit benign/attack row counts, and
    only ever assembled into a DataFrame once -- for the `labels` table.
    """
    n_benign = len(benign_events)
    event_columns = list(benign_events.columns)

    if len(attack_events) > 0:
        # At small scale, a nullable column (e.g. failure_reason,
        # resource_accessed) can legitimately end up 100%-null on one side
        # purely by chance -- e.g. no benign failures happened to occur in
        # a tiny run. That's expected sparsity, not a bug, but pandas warns
        # about a future dtype-resolution change in exactly this case; the
        # current, correct behavior is what we want, so it's suppressed
        # narrowly around this one call rather than left to alarm every run.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*empty or all-NA entries.*")
            events = pd.concat([benign_events, attack_events[event_columns]], ignore_index=True, sort=False)
        attack_ids = [None] * n_benign + list(attack_events["_tmp_attack_id"])
        attack_types = [None] * n_benign + list(attack_events["_tmp_attack_type"])
    else:
        events = benign_events.reset_index(drop=True).copy()
        attack_ids = [None] * n_benign
        attack_types = [None] * n_benign

    events = events.reset_index(drop=True)
    events[BOOL_COLUMNS] = events[BOOL_COLUMNS].astype(bool)

    labels = pd.DataFrame({"attack_id": attack_ids, "attack_type": attack_types})
    labels["is_attack"] = labels["attack_id"].notna()

    meta_lookup = attack_meta.set_index("attack_id")["mitre_technique_ids"].to_dict() if len(attack_meta) else {}
    labels["mitre_technique_ids"] = labels["attack_id"].map(meta_lookup)
    labels = labels.reset_index(drop=True)

    return events, labels


def _assign_bookkeeping_fields(events: pd.DataFrame, labels: pd.DataFrame, num_generation_batches: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign record_id / insertion_order / generation_batch independent of
    label and time, THEN sort by timestamp -- so the final on-disk row order
    (chronological, like a real log) is decoupled from these bookkeeping
    fields' values. See evaluation/leakage_audit.py.
    """
    n = len(events)
    record_ids = [random_hex(rng, 16) for _ in range(n)]
    perm = rng.permutation(n)
    insertion_order = np.empty(n, dtype=np.int64)
    insertion_order[perm] = np.arange(n)
    generation_batch = rng.integers(0, num_generation_batches, size=n)

    events = events.reset_index(drop=True)
    events["record_id"] = record_ids
    events["insertion_order"] = insertion_order
    events["generation_batch"] = generation_batch

    labels = labels.reset_index(drop=True)
    labels["record_id"] = record_ids

    events = events.sort_values("timestamp").reset_index(drop=True)
    labels = labels.set_index("record_id").loc[events["record_id"]].reset_index()
    return events, labels


@hydra.main(version_base=None, config_path="../configs", config_name="small_dev")
def main(cfg: DictConfig) -> None:
    rng = set_global_seed(int(cfg.seed))

    mitre_mapping = load_mitre_mapping(PROJECT_ROOT / "configs" / "mitre_mapping.yaml")

    users = generate_population(cfg, rng)
    drift, drift_log = resolve_drift_schedule(users, cfg, rng)
    benign_events = generate_login_events(users, cfg, rng, drift=drift)
    attack_events, attack_meta = inject_attacks(users, benign_events, cfg, mitre_mapping, rng)

    events, labels = _assemble_events_and_labels(benign_events, attack_events, attack_meta)
    events, labels = _assign_bookkeeping_fields(events, labels, int(cfg.attacks.num_generation_batches), rng)

    validate_dataframe(users, USERS_SCHEMA, "users")
    validate_dataframe(events, EVENTS_SCHEMA, "events")
    validate_dataframe(labels, LABELS_SCHEMA, "labels")
    if len(attack_meta):
        validate_dataframe(attack_meta, ATTACKS_SCHEMA, "attacks")

    out_dir = PROJECT_ROOT / cfg.run.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    users.to_parquet(out_dir / "users.parquet", index=False)
    events.to_parquet(out_dir / "events.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    attack_meta.to_parquet(out_dir / "attacks.parquet", index=False)

    # Ground-truth drift log (Phase 3): empty (with correct headers) if
    # cfg.events.drift.enabled is false, one row per scheduled drift event
    # otherwise -- see generator/drift.py.
    drift_log.to_csv(out_dir / "drift_log.csv", index=False)

    config_yaml = OmegaConf.to_yaml(cfg)
    config_hash = hashlib.sha256(config_yaml.encode()).hexdigest()[:16]

    run_metadata = {
        "disclaimer": DISCLAIMER,
        "config_hash": config_hash,
        "seed": int(cfg.seed),
        "num_users": int(len(users)),
        "num_events": int(len(events)),
        "num_attack_events": int(labels["is_attack"].sum()),
        "attack_ratio_actual": float(labels["is_attack"].mean()) if len(labels) else 0.0,
        "num_drift_events": int(len(drift_log)),
        "total_drift_affected_users": int(drift_log["affected_user_count"].sum()) if len(drift_log) else 0,
        "num_attack_campaigns": int(len(attack_meta)),
        "mitre_mapping_version": int(mitre_mapping["schema_version"]),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, default=str)
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        f.write(config_yaml)
    with open(out_dir / "DISCLAIMER.txt", "w", encoding="utf-8") as f:
        f.write(DISCLAIMER + "\n")

    print(
        f"Wrote {len(events)} events "
        f"({run_metadata['num_attack_events']} attack-labeled, "
        f"ratio={run_metadata['attack_ratio_actual']:.4%}) to {out_dir}"
    )
    print(f"Config hash: {config_hash}")


if __name__ == "__main__":
    main()
