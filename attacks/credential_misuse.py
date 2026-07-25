"""Credential-misuse attack generator.

Simulates a valid credential being used by someone other than its
legitimate owner: a single successful login (no failure burst, unlike
brute_force) followed by a resource_access whose target is BOTH outside the
account's normal department AND above its normal privilege scope --
a conjunction, not a single field, which is what keeps this from being
trivially separable from the ordinary cross-department access
generator/events.py already produces at a low rate (which stays within
privilege scope).

Leakage-safety notes:
 - Device/geo for the compromised session look like ordinary (if
   unfamiliar-device) traffic -- no burst pattern, no impossible-travel
   physics -- so this attack's tell is purely the resource/privilege
   mismatch, not device or geo, keeping the five attack types'
   signatures distinguishable from one another rather than overlapping.
 - The privilege violation is graded (how far above the account's actual
   privilege the accessed resource sits), not a fixed "reached admin_console"
   rule, so severity varies realistically rather than being a constant tell.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from attacks.base import load_mitre_mapping, sample_severity
from generator.constants import MIN_PRIVILEGE_BY_RESOURCE, OS_BY_DEVICE_TYPE, RESOURCE_TYPES_BY_DEPT, fake_ip, random_hex, weighted_choice
from preprocessing.constants import PRIVILEGE_LEVELS, RESOURCE_TYPES

BROWSERS = ["Chrome", "Firefox", "Edge", "Safari", "Other"]


def _pick_overreach_resource(rng: np.random.Generator, department: str, privilege_level: str) -> str | None:
    """Pick a resource outside `department`'s typical set AND above
    `privilege_level`'s scope. Returns None if no such resource exists for
    this user (e.g. a domain_admin has no resource left to "overreach" to),
    in which case the caller should try a different target user.
    """
    dept_resources = RESOURCE_TYPES_BY_DEPT.get(department, RESOURCE_TYPES_BY_DEPT["_default"])
    priv_rank = PRIVILEGE_LEVELS.index(privilege_level)
    candidates = [
        r for r in RESOURCE_TYPES
        if r not in dept_resources and PRIVILEGE_LEVELS.index(MIN_PRIVILEGE_BY_RESOURCE[r]) > priv_rank
    ]
    if not candidates:
        return None
    # Bias toward the most-overreaching candidate (higher required privilege)
    # so severity has real variance to key off, but keep some randomness.
    candidates.sort(key=lambda r: PRIVILEGE_LEVELS.index(MIN_PRIVILEGE_BY_RESOURCE[r]))
    if rng.random() < 0.6:
        return candidates[-1]
    return str(rng.choice(candidates))


def generate_credential_misuse_attack(
    attack_id: str,
    users: pd.DataFrame,
    cfg: DictConfig,
    mitre_mapping: dict,
    rng: np.random.Generator,
    holidays: set,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    ev_cfg = cfg.events
    users_indexed = users.set_index("user_id")
    active_user_ids = users.loc[users["employment_status"] != "terminated", "user_id"].tolist()

    target_user = None
    overreach_resource = None
    for _ in range(20):  # a handful of tries; most users have SOME resource above their privilege
        candidate = str(rng.choice(active_user_ids))
        row = users_indexed.loc[candidate]
        resource = _pick_overreach_resource(rng, row["department"], row["privilege_level"])
        if resource is not None:
            target_user, overreach_resource = candidate, resource
            break
    if target_user is None:
        return None, None

    row = users_indexed.loc[target_user]
    normal_resource = str(rng.choice(RESOURCE_TYPES_BY_DEPT.get(row["department"], RESOURCE_TYPES_BY_DEPT["_default"])))
    start_date = pd.Timestamp(ev_cfg.start_date)
    total_span_minutes = int(ev_cfg.num_days) * 24 * 60
    start_offset = float(rng.uniform(0, total_span_minutes))
    ts = start_date + pd.Timedelta(minutes=start_offset)
    session_id = f"SESS-{random_hex(rng, 6)}"

    device_type = str(rng.choice(["desktop", "laptop", "mobile"], p=[0.4, 0.4, 0.2]))
    os_name = str(rng.choice(OS_BY_DEVICE_TYPE[device_type]))
    device_id = f"DEV-ATTACKER-{random_hex(rng, 4)}"
    auth_method = str(rng.choice(["password", "sso"]))

    login_event = {
        "user_id": target_user,
        "session_id": session_id,
        "timestamp": ts,
        "event_type": "login_attempt",
        "auth_result": "success",
        "auth_method": auth_method,
        "mfa_used": False,
        "failure_reason": None,
        "device_id": device_id,
        "device_type": device_type,
        "os": os_name,
        "browser": str(rng.choice(BROWSERS)),
        "ip_address": fake_ip(rng),
        "asn": f"AS{int(rng.integers(1000, 65000))}",
        "isp_name": f"ISP-{row['home_country']}-{int(rng.integers(1, 20))}",
        "geo_country": row["home_country"],
        "geo_city": row["home_city"],
        "geo_lat": float(row["home_lat"]) + rng.normal(0, 0.05),
        "geo_lon": float(row["home_lon"]) + rng.normal(0, 0.05),
        "network_type": str(rng.choice(["home_isp", "public_wifi"])),
        "session_duration_seconds": float(rng.lognormal(6.8, 0.6)),
        "resource_accessed": normal_resource,
        "action_count": 1,
        "is_weekend": ts.weekday() >= 5,
        "is_holiday": ts.date() in holidays,
        "is_off_hours": not (7 <= ts.hour <= 19),
        "_tmp_attack_id": attack_id,
        "_tmp_attack_type": "credential_misuse",
    }

    resource_ts = ts + pd.Timedelta(seconds=int(rng.integers(10, 120)))
    resource_event = {
        "user_id": target_user,
        "session_id": session_id,
        "timestamp": resource_ts,
        "event_type": "resource_access",
        "auth_result": None,
        "auth_method": None,
        "mfa_used": False,
        "failure_reason": None,
        "device_id": device_id,
        "device_type": device_type,
        "os": os_name,
        "browser": login_event["browser"],
        "ip_address": login_event["ip_address"],
        "asn": login_event["asn"],
        "isp_name": login_event["isp_name"],
        "geo_country": login_event["geo_country"],
        "geo_city": login_event["geo_city"],
        "geo_lat": login_event["geo_lat"],
        "geo_lon": login_event["geo_lon"],
        "network_type": login_event["network_type"],
        "session_duration_seconds": None,
        "resource_accessed": overreach_resource,
        "action_count": int(rng.poisson(3) + 1),
        "is_weekend": resource_ts.weekday() >= 5,
        "is_holiday": resource_ts.date() in holidays,
        "is_off_hours": login_event["is_off_hours"],
        "_tmp_attack_id": attack_id,
        "_tmp_attack_type": "credential_misuse",
    }

    events = [login_event, resource_event]

    priv_gap = PRIVILEGE_LEVELS.index(MIN_PRIVILEGE_BY_RESOURCE[overreach_resource]) - PRIVILEGE_LEVELS.index(row["privilege_level"])
    severity = sample_severity(cfg, rng, boost=(priv_gap >= 2))
    technique_ids = mitre_mapping["mappings"]["credential_misuse"]["technique_ids"]

    rationale = (
        f"{row['privilege_level']}-privilege account in {row['department']} successfully accessed "
        f"{overreach_resource}, a resource outside its department and above its normal privilege scope, "
        f"from a device not previously associated with this account."
    )

    metadata = {
        "attack_id": attack_id,
        "attack_type": "credential_misuse",
        "start_time": ts,
        "end_time": resource_ts,
        "severity": severity,
        "target_user_id": target_user,
        "affected_assets": f"authentication_service,{overreach_resource}",
        "rationale": rationale,
        "mitre_technique_ids": ",".join(technique_ids),
        "mitre_tactic": mitre_mapping["mappings"]["credential_misuse"]["tactic"],
        "num_events_generated": len(events),
        "parameters_json": json.dumps({
            "overreach_resource": overreach_resource,
            "account_privilege_level": row["privilege_level"],
            "account_department": row["department"],
            "source_ip": login_event["ip_address"],
        }),
        "mitre_mapping_version": int(mitre_mapping["schema_version"]),
    }
    return events, metadata
