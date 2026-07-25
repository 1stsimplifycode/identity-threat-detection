"""Lateral-movement attack generator.

Simulates an account accessing a chain of resources that crosses multiple
departments' normal footprints in a single, fast session, generally trending
toward higher-value systems (see RESOURCE_VALUE_TIER) -- the multi-hop,
cross-domain escalation pattern is what distinguishes this from
credential_misuse's single atypical-resource touch.

Leakage-safety design: the entry login itself is deliberately unremarkable
-- it reuses a real prior benign event's device/geo/network (the same
"anchor on real history" approach impossible_travel.py already uses), so
nothing about the login looks suspicious. The tell is *only* the resource
sequence that follows: crossing multiple department resource-sets quickly,
which generator/events.py's own benign resource_chain_probability /
cross_department_resource_probability already produce a milder version of
(single cross-department hops, longer gaps) -- lateral_movement's chain is
distinguished by degree (multiple domains, faster hops, value-escalating),
not by a feature no benign session could ever have.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from attacks.base import load_mitre_mapping, sample_severity
from generator.constants import RESOURCE_TYPES_BY_DEPT, RESOURCE_VALUE_TIER, random_hex
from preprocessing.constants import RESOURCE_TYPES


def _pick_escalating_chain(rng: np.random.Generator, home_department: str, n_hops: int) -> list[str]:
    """A resource sequence of individually-foreign resource *types*
    (genuinely outside `home_department`'s own typical set -- not just
    "belonging to a different department's list," since department resource
    sets overlap heavily, e.g. Executive's own set already includes
    hr_system and crm), trending toward higher RESOURCE_VALUE_TIER.
    """
    home_resources = set(RESOURCE_TYPES_BY_DEPT.get(home_department, RESOURCE_TYPES_BY_DEPT["_default"]))
    foreign_resources = [r for r in RESOURCE_TYPES if r not in home_resources]
    if not foreign_resources:
        return []
    n_pick = min(n_hops, len(foreign_resources))
    resources = list(rng.choice(foreign_resources, size=n_pick, replace=False))
    # sort the chain itself by value tier so it *trends* upward, not randomly
    resources.sort(key=lambda r: RESOURCE_VALUE_TIER.get(r, 0))
    return resources


def generate_lateral_movement_attack(
    attack_id: str,
    users: pd.DataFrame,
    successful_events_by_user,  # pandas GroupBy of benign success events, grouped by user_id (see impossible_travel.py)
    cfg: DictConfig,
    mitre_mapping: dict,
    rng: np.random.Generator,
    holidays: set,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    lm_cfg = cfg.attacks.lateral_movement
    users_indexed = users.set_index("user_id")

    user_ids_with_history = list(successful_events_by_user.groups.keys())
    if not user_ids_with_history:
        return None, None

    target_user = str(rng.choice(user_ids_with_history))
    anchor_group = successful_events_by_user.get_group(target_user)
    anchor = anchor_group.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
    department = users_indexed.loc[target_user, "department"]

    n_hops = int(rng.integers(lm_cfg.min_hops, lm_cfg.max_hops + 1))
    chain_resources = _pick_escalating_chain(rng, department, n_hops)
    if not chain_resources:
        return None, None  # this user's department has no "other departments" to cross (shouldn't happen, but be safe)

    session_id = f"SESS-{random_hex(rng, 6)}"
    login_ts = anchor["timestamp"] + pd.Timedelta(minutes=float(rng.uniform(30, 24 * 60)))
    normal_resource = str(rng.choice(RESOURCE_TYPES_BY_DEPT.get(department, RESOURCE_TYPES_BY_DEPT["_default"])))

    shared_context = {
        "device_id": anchor["device_id"], "device_type": anchor["device_type"], "os": anchor["os"],
        "browser": anchor["browser"], "ip_address": anchor["ip_address"], "asn": anchor["asn"],
        "isp_name": anchor["isp_name"], "geo_country": anchor["geo_country"], "geo_city": anchor["geo_city"],
        "geo_lat": anchor["geo_lat"], "geo_lon": anchor["geo_lon"], "network_type": anchor["network_type"],
    }

    login_event = {
        "user_id": target_user,
        "session_id": session_id,
        "timestamp": login_ts,
        "event_type": "login_attempt",
        "auth_result": "success",
        "auth_method": str(rng.choice(["password", "sso"])),
        "mfa_used": bool(rng.random() < 0.5),
        "failure_reason": None,
        **shared_context,
        "session_duration_seconds": float(rng.lognormal(7.0, 0.6)),
        "resource_accessed": normal_resource,
        "action_count": 1,
        "is_weekend": login_ts.weekday() >= 5,
        "is_holiday": login_ts.date() in holidays,
        "is_off_hours": not (7 <= login_ts.hour <= 19),
        "_tmp_attack_id": attack_id,
        "_tmp_attack_type": "lateral_movement",
    }

    events = [login_event]
    ts = login_ts
    hop_gap_seconds = lm_cfg.hop_gap_seconds  # [min, max], deliberately shorter than benign resource_chain_hops gaps
    for resource in chain_resources:
        ts = ts + pd.Timedelta(seconds=int(rng.integers(hop_gap_seconds[0], hop_gap_seconds[1] + 1)))
        events.append({
            "user_id": target_user,
            "session_id": session_id,
            "timestamp": ts,
            "event_type": "resource_access",
            "auth_result": None,
            "auth_method": None,
            "mfa_used": False,
            "failure_reason": None,
            **shared_context,
            "session_duration_seconds": None,
            "resource_accessed": resource,
            "action_count": int(rng.poisson(2) + 1),
            "is_weekend": ts.weekday() >= 5,
            "is_holiday": ts.date() in holidays,
            "is_off_hours": login_event["is_off_hours"],
            "_tmp_attack_id": attack_id,
            "_tmp_attack_type": "lateral_movement",
        })

    max_tier = max(RESOURCE_VALUE_TIER.get(r, 0) for r in chain_resources)
    severity = sample_severity(cfg, rng, boost=(max_tier >= 2))
    technique_ids = mitre_mapping["mappings"]["lateral_movement"]["technique_ids"]

    rationale = (
        f"Account in {department} accessed {len(chain_resources)} resources outside its department "
        f"({', '.join(chain_resources)}) within a single fast session, trending toward higher-sensitivity "
        f"systems -- inconsistent with its normal single-department access footprint."
    )

    metadata = {
        "attack_id": attack_id,
        "attack_type": "lateral_movement",
        "start_time": login_ts,
        "end_time": ts,
        "severity": severity,
        "target_user_id": target_user,
        "affected_assets": ",".join(chain_resources),
        "rationale": rationale,
        "mitre_technique_ids": ",".join(technique_ids),
        "mitre_tactic": mitre_mapping["mappings"]["lateral_movement"]["tactic"],
        "num_events_generated": len(events),
        "parameters_json": json.dumps({
            "home_department": department,
            "chain_resources": chain_resources,
            "num_hops": len(chain_resources),
            "max_resource_value_tier": max_tier,
        }),
        "mitre_mapping_version": int(mitre_mapping["schema_version"]),
    }
    return events, metadata
