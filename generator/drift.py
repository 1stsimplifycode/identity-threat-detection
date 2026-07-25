"""Concept-drift simulation (Phase 3): a config-driven schedule of
behavioral shifts applied to a cohort of users, rolled out gradually (each
affected user's individual transition day is jittered within
`ramp_days` of the scheduled day, rather than everyone changing on the same
instant), with the exact ground truth logged to
`data/runs/<run>/drift_log.csv`. That log is what `drift_detection/`
evaluates ADWIN's detected-drift timing against.

Two change types (extensible only alongside their own generator/events.py
support -- this project's discipline of never hand-waving a behavior before
its consumer exists, same as attacks/ and configs/mitre_mapping.yaml):
    remote_work_shift -- work_mode becomes "remote"; home location moves to
                          a new city (simulating relocation)
    schedule_shift     -- shift_start_hour/shift_end_hour resample to a new
                          daily pattern

Each change type touches a disjoint set of attributes, so a user can
independently be affected by more than one drift event over the simulation
(e.g. relocating AND later changing shift) without conflict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from generator.population import sample_shift
from preprocessing.geo_utils import HOME_CITY_POOL

CHANGE_TYPES: tuple[str, ...] = ("remote_work_shift", "schedule_shift")

DRIFT_LOG_COLUMNS: list[str] = ["day", "change_type", "description", "affected_user_count"]


@dataclass
class ResolvedDrift:
    """Per-change-type, per-user rollout day and the attribute overrides
    that take effect from that day forward.
    """
    drift_days: dict[str, dict[str, int]] = field(default_factory=dict)
    drifted_values: dict[str, dict[str, dict]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.drift_days)

    def effective_overrides(self, user_id: str, day_index: int) -> dict:
        """All attribute overrides active for `user_id` on `day_index`,
        merged across every change type whose rollout day has passed for
        this user (later change types checked in dict-insertion order, so
        if two ever touched the same attribute the later-scheduled one
        would win -- moot in practice since the two types' attributes don't
        overlap).
        """
        overrides: dict = {}
        for change_type, days in self.drift_days.items():
            drift_day = days.get(user_id)
            if drift_day is not None and day_index >= drift_day:
                overrides.update(self.drifted_values[change_type][user_id])
        return overrides


def resolve_drift_schedule(users: pd.DataFrame, cfg: DictConfig, rng: np.random.Generator) -> tuple[ResolvedDrift, pd.DataFrame]:
    """Resolve `cfg.events.drift` into a `ResolvedDrift` (for
    `generator/events.py` to apply) and the ground-truth `drift_log.csv`
    DataFrame (one row per scheduled drift event, not per affected user).
    """
    resolved = ResolvedDrift()
    drift_cfg = cfg.events.get("drift", None)
    if drift_cfg is None or not bool(drift_cfg.get("enabled", False)):
        return resolved, pd.DataFrame(columns=DRIFT_LOG_COLUMNS)

    ramp_days = int(drift_cfg.get("ramp_days", 0))
    active_user_ids = users.loc[users["employment_status"] != "terminated", "user_id"].tolist()
    users_indexed = users.set_index("user_id")
    countries = list(cfg.population.home_countries)

    log_rows: list[dict] = []

    for item in drift_cfg.get("schedule", []):
        day = int(item["day"])
        change_type = str(item["change_type"])
        if change_type not in CHANGE_TYPES:
            raise ValueError(
                f"Unknown drift change_type {change_type!r} in configs/events drift schedule "
                f"-- must be one of {CHANGE_TYPES} (add generator/events.py support for a new "
                f"type before adding it here)."
            )
        fraction = float(item["affected_fraction"])
        description = str(item.get("description", change_type))

        n_affected = max(1, int(round(len(active_user_ids) * fraction)))
        n_affected = min(n_affected, len(active_user_ids))
        affected = rng.choice(active_user_ids, size=n_affected, replace=False)

        days_map: dict[str, int] = {}
        values_map: dict[str, dict] = {}
        for user_id in affected:
            individual_day = day + (int(rng.integers(0, ramp_days + 1)) if ramp_days > 0 else 0)
            days_map[str(user_id)] = individual_day

            if change_type == "remote_work_shift":
                new_country = str(rng.choice(countries))
                options = HOME_CITY_POOL.get(new_country, HOME_CITY_POOL["US"])
                city, lat, lon = options[int(rng.integers(0, len(options)))]
                values_map[str(user_id)] = {
                    "work_mode": "remote",
                    "home_country": new_country,
                    "home_city": city,
                    "home_lat": lat + rng.normal(0, 0.05),
                    "home_lon": lon + rng.normal(0, 0.05),
                }
            elif change_type == "schedule_shift":
                department = users_indexed.loc[user_id, "department"]
                new_start, new_end = sample_shift(rng, department)
                values_map[str(user_id)] = {"shift_start_hour": new_start, "shift_end_hour": new_end}

        resolved.drift_days[change_type] = days_map
        resolved.drifted_values[change_type] = values_map
        log_rows.append({
            "day": day,
            "change_type": change_type,
            "description": description,
            "affected_user_count": len(affected),
        })

    return resolved, pd.DataFrame(log_rows, columns=DRIFT_LOG_COLUMNS)
