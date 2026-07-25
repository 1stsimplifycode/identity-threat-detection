"""Device-spoofing attack generator.

Two variants, chosen at random per campaign:
 (a) fingerprint mismatch -- a device_id with an established, consistent
     (device_type, os) signature in the account's own history suddenly
     appears with a DIFFERENT device_type/os (the underlying hardware/OS
     changed but the device_id claims to be the same device).
 (b) cross-user device reuse -- a device_id normally scoped to one user is
     used by a DIFFERENT user shortly after, spiking that device's fan-in.

Leakage-safety note: benign traffic already has a low-probability "shared
kiosk device" pattern (generator/events.py's shared_kiosk_device_probability,
scoped to SHARED_DEVICE_DEPARTMENTS) so ">1 user on one device" is not
100% attack-exclusive -- variant (b) specifically targets a device that is
NOT already a known shared/kiosk device (device_index excludes
"DEV-KIOSK-*" ids), so the tell is a PREVIOUSLY single-user device suddenly
gaining a second user, not device-sharing in general.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from attacks.base import load_mitre_mapping, sample_severity
from generator.constants import OS_BY_DEVICE_TYPE, RESOURCE_TYPES_BY_DEPT, random_hex


def build_device_index(benign_events: pd.DataFrame) -> dict[str, dict]:
    """One entry per device_id seen in benign traffic: its established
    (device_type, os) signature, the set of users who have used it, and its
    most recent event (for anchoring a spoofing attack shortly after).
    Built once in attacks/injector.py and passed into this module.
    """
    index: dict[str, dict] = {}
    for device_id, group in benign_events.groupby("device_id"):
        last_row = group.loc[group["timestamp"].idxmax()]
        index[device_id] = {
            "device_type": group["device_type"].iloc[0],
            "os": group["os"].iloc[0],
            "user_ids": set(group["user_id"].unique()),
            "last_event": last_row,
        }
    return index


def _swap_signature(rng: np.random.Generator, device_type: str, os_name: str) -> tuple[str, str]:
    other_types = [t for t in OS_BY_DEVICE_TYPE if t != device_type]
    new_type = str(rng.choice(other_types))
    other_os_options = [o for o in OS_BY_DEVICE_TYPE[new_type] if o != os_name]
    new_os = str(rng.choice(other_os_options)) if other_os_options else str(rng.choice(OS_BY_DEVICE_TYPE[new_type]))
    return new_type, new_os


def _build_event(user_id, session_id, ts, device_id, device_type, os_name, resource, holidays, attack_id) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": ts,
        "event_type": "login_attempt",
        "auth_result": "success",
        "auth_method": "sso",
        "mfa_used": False,
        "failure_reason": None,
        "device_id": device_id,
        "device_type": device_type,
        "os": os_name,
        "browser": None,
        "ip_address": None,  # filled by caller
        "asn": None,
        "isp_name": None,
        "geo_country": None,
        "geo_city": None,
        "geo_lat": None,
        "geo_lon": None,
        "network_type": None,
        "session_duration_seconds": None,
        "resource_accessed": resource,
        "action_count": 1,
        "is_weekend": ts.weekday() >= 5,
        "is_holiday": ts.date() in holidays,
        "is_off_hours": not (7 <= ts.hour <= 19),
        "_tmp_attack_id": attack_id,
        "_tmp_attack_type": "device_spoofing",
    }


def generate_device_spoofing_attack(
    attack_id: str,
    users: pd.DataFrame,
    device_index: dict[str, dict],
    cfg: DictConfig,
    mitre_mapping: dict,
    rng: np.random.Generator,
    holidays: set,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    ds_cfg = cfg.attacks.device_spoofing
    users_indexed = users.set_index("user_id")

    # Exclude already-legitimately-shared kiosk devices (see module docstring).
    candidates = {d: info for d, info in device_index.items() if not d.startswith("DEV-KIOSK-") and len(info["user_ids"]) == 1}
    if not candidates:
        return None, None

    variant = "fingerprint_mismatch" if rng.random() < ds_cfg.fingerprint_mismatch_probability else "cross_user_reuse"
    device_id = str(rng.choice(list(candidates.keys())))
    info = candidates[device_id]
    anchor = info["last_event"]
    owner_user_id = next(iter(info["user_ids"]))

    ts = anchor["timestamp"] + pd.Timedelta(minutes=float(rng.uniform(*ds_cfg.reappearance_minutes)))
    ip_geo_fields = ["ip_address", "asn", "isp_name", "geo_country", "geo_city", "geo_lat", "geo_lon", "network_type", "browser"]

    if variant == "fingerprint_mismatch":
        target_user = owner_user_id
        new_type, new_os = _swap_signature(rng, info["device_type"], info["os"])
        session_id = f"SESS-{random_hex(rng, 6)}"
        owner_department = users_indexed.loc[target_user, "department"]
        normal_resource = str(rng.choice(RESOURCE_TYPES_BY_DEPT.get(owner_department, RESOURCE_TYPES_BY_DEPT["_default"])))
        event = _build_event(target_user, session_id, ts, device_id, new_type, new_os, normal_resource, holidays, attack_id)
        for field in ip_geo_fields:
            event[field] = anchor[field]  # same network context; only the device signature changed
        rationale = (
            f"Device {device_id}, previously seen consistently as {info['device_type']}/{info['os']} for this "
            f"account, reappeared reporting a different signature ({new_type}/{new_os}) -- consistent with a "
            f"cloned or spoofed device fingerprint."
        )
        params = {"variant": variant, "device_id": device_id, "original_signature": f"{info['device_type']}/{info['os']}", "reported_signature": f"{new_type}/{new_os}"}
    else:
        other_user_ids = [u for u in users["user_id"] if u != owner_user_id and u in users_indexed.index]
        if not other_user_ids:
            return None, None
        target_user = str(rng.choice(other_user_ids))
        session_id = f"SESS-{random_hex(rng, 6)}"
        target_department = users_indexed.loc[target_user, "department"]
        normal_resource = str(rng.choice(RESOURCE_TYPES_BY_DEPT.get(target_department, RESOURCE_TYPES_BY_DEPT["_default"])))
        event = _build_event(target_user, session_id, ts, device_id, info["device_type"], info["os"], normal_resource, holidays, attack_id)
        for field in ip_geo_fields:
            event[field] = anchor[field]  # device claims the SAME network context as its established owner
        rationale = (
            f"Device {device_id}, previously associated only with {owner_user_id}, was used by a different "
            f"account ({target_user}) within {int((ts - anchor['timestamp']).total_seconds() / 60)} minutes of "
            f"its established owner's last use -- consistent with a spoofed or cloned device identifier."
        )
        params = {"variant": variant, "device_id": device_id, "original_owner": owner_user_id, "new_user": target_user}

    events = [event]
    severity = sample_severity(cfg, rng, boost=(variant == "cross_user_reuse"))
    technique_ids = mitre_mapping["mappings"]["device_spoofing"]["technique_ids"]

    metadata = {
        "attack_id": attack_id,
        "attack_type": "device_spoofing",
        "start_time": ts,
        "end_time": ts,
        "severity": severity,
        "target_user_id": target_user,
        "affected_assets": "authentication_service",
        "rationale": rationale,
        "mitre_technique_ids": ",".join(technique_ids),
        "mitre_tactic": mitre_mapping["mappings"]["device_spoofing"]["tactic"],
        "num_events_generated": len(events),
        "parameters_json": json.dumps(params),
        "mitre_mapping_version": int(mitre_mapping["schema_version"]),
    }
    return events, metadata
