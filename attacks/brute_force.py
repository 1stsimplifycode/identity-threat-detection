"""Brute-force attack generator.

Simulates a burst of failed password-authentication attempts against a
single target account from a device/IP not in that account's known pool,
within a short configurable time window, optionally ending in a successful
login (account compromise).

Leakage-safety notes (see evaluation/leakage_audit.py and constraint #2):
 - Attack start time is drawn uniformly over the *entire* simulation window
   (not restricted to off-hours), so `is_off_hours` is NOT a deterministic
   giveaway of an attack -- some brute-force bursts land during business
   hours, matching how benign traffic also has off-hours activity.
 - New devices/IPs are not attack-exclusive: benign traffic also generates
   new devices at `new_device_probability`, so "new device" alone is not a
   trivial separator either.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from attacks.base import load_mitre_mapping, sample_severity
from generator.constants import OS_BY_DEVICE_TYPE, fake_ip, random_hex, weighted_choice
from preprocessing.geo_utils import GLOBAL_CITY_POOL

BROWSERS = ["Chrome", "Firefox", "Edge", "Safari", "Other"]


def generate_brute_force_attack(
    attack_id: str,
    active_user_ids: list[str],
    cfg: DictConfig,
    mitre_mapping: dict,
    rng: np.random.Generator,
    holidays: set,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bf_cfg = cfg.attacks.brute_force
    ev_cfg = cfg.events

    target_user = str(rng.choice(active_user_ids))
    n_attempts = int(rng.integers(bf_cfg.min_attempts, bf_cfg.max_attempts + 1))
    window_minutes = float(rng.uniform(bf_cfg.window_minutes[0], bf_cfg.window_minutes[1]))

    start_date = pd.Timestamp(ev_cfg.start_date)
    total_span_minutes = int(ev_cfg.num_days) * 24 * 60
    start_offset = float(rng.uniform(0, max(total_span_minutes - window_minutes, 1)))
    attack_start = start_date + pd.Timedelta(minutes=start_offset)

    succeeds = bool(rng.random() < bf_cfg.success_probability)

    device_type_weights = dict(bf_cfg.attacker_device_type_weights)
    device_type = weighted_choice(rng, device_type_weights)
    os_name = str(rng.choice(OS_BY_DEVICE_TYPE[device_type]))
    device_id = f"DEV-ATTACKER-{random_hex(rng, 4)}"

    city, lat, lon, country = GLOBAL_CITY_POOL[int(rng.integers(0, len(GLOBAL_CITY_POOL)))]
    ip = fake_ip(rng)
    asn = f"AS{int(rng.integers(1000, 65000))}"
    session_id = f"SESS-{random_hex(rng, 6)}"

    # Inter-arrival gaps that sum to the attack window (a Dirichlet split).
    gaps_minutes = rng.dirichlet(np.ones(n_attempts)) * window_minutes

    events: list[dict[str, Any]] = []
    ts = attack_start
    for i, gap in enumerate(gaps_minutes):
        ts = ts + pd.Timedelta(minutes=float(gap))
        is_last = i == n_attempts - 1
        result = "success" if (is_last and succeeds) else "failure"
        failure_reason = None if result == "success" else "bad_password"

        events.append({
            "user_id": target_user,
            "session_id": session_id,
            "timestamp": ts,
            "event_type": "login_attempt",
            "auth_result": result,
            "auth_method": "password",
            "mfa_used": False,
            "failure_reason": failure_reason,
            "device_id": device_id,
            "device_type": device_type,
            "os": os_name,
            "browser": str(rng.choice(BROWSERS)),
            "ip_address": ip,
            "asn": asn,
            "isp_name": f"ISP-{country}-{int(rng.integers(1, 20))}",
            "geo_country": country,
            "geo_city": city,
            "geo_lat": lat + rng.normal(0, 0.01),
            "geo_lon": lon + rng.normal(0, 0.01),
            "network_type": str(rng.choice(["public_wifi", "mobile_carrier"])),
            "session_duration_seconds": float(rng.lognormal(6.0, 0.5)) if result == "success" else None,
            "resource_accessed": None,
            "action_count": int(rng.poisson(2)) if result == "success" else 0,
            "is_weekend": ts.weekday() >= 5,
            "is_holiday": ts.date() in holidays,
            "is_off_hours": not (7 <= ts.hour <= 19),
            "_tmp_attack_id": attack_id,
            "_tmp_attack_type": "brute_force",
        })

    boost = succeeds or n_attempts > (bf_cfg.max_attempts * 0.7)
    severity = sample_severity(cfg, rng, boost=boost)
    technique_ids = mitre_mapping["mappings"]["brute_force"]["technique_ids"]

    rationale = (
        f"{n_attempts} failed password attempts against a single account from an unfamiliar device "
        f"within {window_minutes:.1f} minutes"
        + (", ending in a successful login." if succeeds else ", with no successful login.")
    )

    metadata = {
        "attack_id": attack_id,
        "attack_type": "brute_force",
        "start_time": events[0]["timestamp"],
        "end_time": events[-1]["timestamp"],
        "severity": severity,
        "target_user_id": target_user,
        "affected_assets": "authentication_service",
        "rationale": rationale,
        "mitre_technique_ids": ",".join(technique_ids),
        "mitre_tactic": mitre_mapping["mappings"]["brute_force"]["tactic"],
        "num_events_generated": len(events),
        "parameters_json": json.dumps({
            "num_attempts": n_attempts,
            "window_minutes": round(window_minutes, 2),
            "succeeded": succeeds,
            "source_ip": ip,
            "source_country": country,
        }),
        "mitre_mapping_version": int(mitre_mapping["schema_version"]),
    }
    return events, metadata
