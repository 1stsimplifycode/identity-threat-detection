"""Attack injection orchestrator.

Generates attack campaigns (alternating across the configured
`enabled_attacks`) until the requested `imbalance_ratio` (fraction of all
events that are attack-labeled) is reached, and returns the combined attack
events and attack-metadata tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from attacks.brute_force import generate_brute_force_attack
from attacks.credential_misuse import generate_credential_misuse_attack
from attacks.device_spoofing import build_device_index, generate_device_spoofing_attack
from attacks.impossible_travel import generate_impossible_travel_attack
from attacks.lateral_movement import generate_lateral_movement_attack
from generator.events import EVENTS_COLUMN_ORDER
from preprocessing.schema import ATTACKS_SCHEMA

EVENTS_TEMP_COLUMNS: list[str] = ["_tmp_attack_id", "_tmp_attack_type"]


def inject_attacks(
    users: pd.DataFrame,
    benign_events: pd.DataFrame,
    cfg: DictConfig,
    mitre_mapping: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_benign = len(benign_events)
    imbalance_ratio = float(cfg.attacks.imbalance_ratio)
    if n_benign == 0 or imbalance_ratio <= 0:
        return pd.DataFrame(), pd.DataFrame()

    target_attack_events = int(round(imbalance_ratio * n_benign / (1 - imbalance_ratio)))
    # A nonzero imbalance ratio should always yield at least one attack
    # campaign, even at small dev scale where the raw computation can round
    # down to zero -- otherwise labels can silently end up single-class.
    target_attack_events = max(target_attack_events, 1)

    holidays = {pd.Timestamp(d).date() for d in cfg.events.holidays}
    active_user_ids = users.loc[users["employment_status"] != "terminated", "user_id"].tolist()
    successful_events_by_user = benign_events.loc[benign_events["auth_result"] == "success"].groupby("user_id")
    # Built once (not per-campaign): device_spoofing needs the org-wide
    # device -> {signature, users, last event} index, which is only cheap to
    # build a single time from the full benign_events table.
    device_index = build_device_index(benign_events)

    enabled_attacks = list(cfg.attacks.enabled_attacks)
    if not enabled_attacks:
        return pd.DataFrame(), pd.DataFrame()

    all_events: list[dict] = []
    all_meta: list[dict] = []
    attack_counter = 0
    idx = 0
    generated = 0
    max_iterations = target_attack_events * 20 + 100  # safety valve against pathological configs

    iterations = 0
    while generated < target_attack_events and iterations < max_iterations:
        iterations += 1
        attack_type = enabled_attacks[idx % len(enabled_attacks)]
        idx += 1
        attack_counter += 1
        attack_id = f"ATK-{attack_counter:06d}"

        if attack_type == "brute_force":
            events, meta = generate_brute_force_attack(attack_id, active_user_ids, cfg, mitre_mapping, rng, holidays)
        elif attack_type == "impossible_travel":
            events, meta = generate_impossible_travel_attack(attack_id, users, successful_events_by_user, cfg, mitre_mapping, rng, holidays)
        elif attack_type == "credential_misuse":
            events, meta = generate_credential_misuse_attack(attack_id, users, cfg, mitre_mapping, rng, holidays)
        elif attack_type == "lateral_movement":
            events, meta = generate_lateral_movement_attack(attack_id, users, successful_events_by_user, cfg, mitre_mapping, rng, holidays)
        elif attack_type == "device_spoofing":
            events, meta = generate_device_spoofing_attack(attack_id, users, device_index, cfg, mitre_mapping, rng, holidays)
        else:
            raise ValueError(f"Unknown attack type in cfg.attacks.enabled_attacks: {attack_type!r}")

        if events is None:
            continue  # this attempt couldn't find valid parameters (e.g. no travel candidate); try the next type

        all_events.extend(events)
        all_meta.append(meta)
        generated += len(events)

    # Even in the (now rare, since target is floored at 1 above) empty case,
    # give the fallback frames the full expected column set -- an empty
    # DataFrame missing columns entirely causes KeyErrors downstream and
    # dtype-promotion surprises when concatenated with a populated frame.
    if all_events:
        attack_events_df = pd.DataFrame(all_events)
    else:
        attack_events_df = pd.DataFrame(columns=EVENTS_COLUMN_ORDER + EVENTS_TEMP_COLUMNS)

    if all_meta:
        attack_meta_df = pd.DataFrame(all_meta)
    else:
        attack_meta_df = pd.DataFrame(columns=list(ATTACKS_SCHEMA.keys()))

    return attack_events_df, attack_meta_df
