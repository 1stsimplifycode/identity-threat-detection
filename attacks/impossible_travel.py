"""Impossible-travel attack generator.

Anchors on a real prior successful benign login for the target user, then
synthesizes a second successful login shortly after at a location whose
distance/elapsed-time ratio implies a physically implausible travel speed.

Leakage-safety note: candidate destination cities are drawn from the same
GLOBAL_CITY_POOL used for legitimate business travel in generator/events.py,
so "a login from an unfamiliar city" alone is not attack-exclusive -- only
the distance/time combination (an implied speed above threshold) is.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from attacks.base import load_mitre_mapping, sample_severity
from generator.constants import OS_BY_DEVICE_TYPE, RESOURCE_TYPES_BY_DEPT, fake_ip, random_hex, weighted_choice
from preprocessing.geo_utils import GLOBAL_CITY_POOL, haversine_km

BROWSERS = ["Chrome", "Firefox", "Edge", "Safari", "Other"]


def generate_impossible_travel_attack(
    attack_id: str,
    users: pd.DataFrame,
    successful_events_by_user,  # pandas GroupBy of benign events with auth_result == "success", grouped by user_id
    cfg: DictConfig,
    mitre_mapping: dict,
    rng: np.random.Generator,
    holidays: set,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    it_cfg = cfg.attacks.impossible_travel

    user_ids_with_history = list(successful_events_by_user.groups.keys())
    if not user_ids_with_history:
        return None, None

    target_user = str(rng.choice(user_ids_with_history))
    anchor_group = successful_events_by_user.get_group(target_user)
    anchor = anchor_group.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]

    distance_km = None
    city = lat = lon = country = None
    for _ in range(20):
        candidate_city, candidate_lat, candidate_lon, candidate_country = GLOBAL_CITY_POOL[int(rng.integers(0, len(GLOBAL_CITY_POOL)))]
        d = haversine_km(anchor["geo_lat"], anchor["geo_lon"], candidate_lat, candidate_lon)
        if it_cfg.min_distance_km <= d <= it_cfg.max_distance_km:
            distance_km, city, lat, lon, country = d, candidate_city, candidate_lat, candidate_lon, candidate_country
            break
    if distance_km is None:
        return None, None  # no candidate city satisfied the configured distance band

    max_time_hours = distance_km / it_cfg.min_implied_speed_kmh
    max_time_minutes = min(max_time_hours * 60, it_cfg.time_window_minutes[1])
    min_time_minutes = it_cfg.time_window_minutes[0]
    if max_time_minutes <= min_time_minutes:
        max_time_minutes = min_time_minutes + 1
    delta_minutes = float(rng.uniform(min_time_minutes, max_time_minutes))
    ts = anchor["timestamp"] + pd.Timedelta(minutes=delta_minutes)
    implied_speed_kmh = distance_km / (delta_minutes / 60)

    device_type_weights = dict(it_cfg.attacker_device_type_weights)
    device_type = weighted_choice(rng, device_type_weights)
    os_name = str(rng.choice(OS_BY_DEVICE_TYPE[device_type]))
    device_id = f"DEV-ATTACKER-{random_hex(rng, 4)}"

    department = users.set_index("user_id").loc[target_user, "department"]
    resource_options = RESOURCE_TYPES_BY_DEPT.get(department, RESOURCE_TYPES_BY_DEPT["_default"])
    session_id = f"SESS-{random_hex(rng, 6)}"

    event = {
        "user_id": target_user,
        "session_id": session_id,
        "timestamp": ts,
        "event_type": "login_attempt",
        "auth_result": "success",
        "auth_method": str(rng.choice(["password", "sso"])),
        "mfa_used": False,
        "failure_reason": None,
        "device_id": device_id,
        "device_type": device_type,
        "os": os_name,
        "browser": str(rng.choice(BROWSERS)),
        "ip_address": fake_ip(rng),
        "asn": f"AS{int(rng.integers(1000, 65000))}",
        "isp_name": f"ISP-{country}-{int(rng.integers(1, 20))}",
        "geo_country": country,
        "geo_city": city,
        "geo_lat": lat,
        "geo_lon": lon,
        "network_type": str(rng.choice(["public_wifi", "mobile_carrier"])),
        "session_duration_seconds": float(rng.lognormal(6.5, 0.7)),
        "resource_accessed": str(rng.choice(resource_options)),
        "action_count": int(rng.poisson(3) + 1),
        "is_weekend": ts.weekday() >= 5,
        "is_holiday": ts.date() in holidays,
        "is_off_hours": not (7 <= ts.hour <= 19),
        "_tmp_attack_id": attack_id,
        "_tmp_attack_type": "impossible_travel",
    }

    boost = implied_speed_kmh > it_cfg.min_implied_speed_kmh * 3
    severity = sample_severity(cfg, rng, boost=boost)
    technique_ids = mitre_mapping["mappings"]["impossible_travel"]["technique_ids"]

    rationale = (
        f"Successful login from {country} occurred {delta_minutes:.1f} minutes after this account's previous "
        f"successful login from {anchor['geo_country']}, {distance_km:.0f} km away -- an implied travel speed of "
        f"{implied_speed_kmh:.0f} km/h, physically impossible for legitimate travel."
    )

    metadata = {
        "attack_id": attack_id,
        "attack_type": "impossible_travel",
        "start_time": anchor["timestamp"],
        "end_time": ts,
        "severity": severity,
        "target_user_id": target_user,
        "affected_assets": "authentication_service,application_access",
        "rationale": rationale,
        "mitre_technique_ids": ",".join(technique_ids),
        "mitre_tactic": mitre_mapping["mappings"]["impossible_travel"]["tactic"],
        "num_events_generated": 1,
        "parameters_json": json.dumps({
            "distance_km": round(distance_km, 1),
            "delta_minutes": round(delta_minutes, 2),
            "implied_speed_kmh": round(implied_speed_kmh, 1),
            "anchor_country": str(anchor["geo_country"]),
            "destination_country": country,
        }),
        "mitre_mapping_version": int(mitre_mapping["schema_version"]),
    }
    return [event], metadata
