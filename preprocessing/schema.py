"""Schema definitions and validation for every table this pipeline produces.

Each schema is a dict: column_name -> spec, where spec has keys:
    dtype     : "str" | "category" | "int" | "float" | "bool" | "datetime"
    nullable  : bool
    values    : list[str]        (only for dtype == "category")
    range     : [min, max]       (only for dtype in {"int", "float"})

validate_dataframe() checks a DataFrame against a schema and raises a single
ValueError listing *every* violation found (not just the first), so a failed
run tells you everything wrong in one shot.
"""
from __future__ import annotations

import pandas as pd

from preprocessing.constants import (
    ATTACK_TYPES,
    AUTH_METHODS,
    BROWSERS,
    DEPARTMENTS,
    DEVICE_TYPES,
    EMPLOYMENT_STATUSES,
    EVENT_TYPES,
    FAILURE_REASONS,
    NETWORK_TYPES,
    PRIVILEGE_LEVELS,
    RESOURCE_TYPES,
    SEVERITIES,
    WORK_MODES,
)

USERS_SCHEMA: dict[str, dict] = {
    "user_id": {"dtype": "str", "nullable": False},
    "department": {"dtype": "category", "nullable": False, "values": DEPARTMENTS},
    "role": {"dtype": "str", "nullable": False},
    "privilege_level": {"dtype": "category", "nullable": False, "values": PRIVILEGE_LEVELS},
    "manager_id": {"dtype": "str", "nullable": True},
    "join_date": {"dtype": "datetime", "nullable": False},
    "employment_status": {"dtype": "category", "nullable": False, "values": EMPLOYMENT_STATUSES},
    "work_mode": {"dtype": "category", "nullable": False, "values": WORK_MODES},
    "home_country": {"dtype": "str", "nullable": False},
    "home_city": {"dtype": "str", "nullable": False},
    "home_lat": {"dtype": "float", "nullable": False, "range": [-90, 90]},
    "home_lon": {"dtype": "float", "nullable": False, "range": [-180, 180]},
    "shift_start_hour": {"dtype": "int", "nullable": False, "range": [0, 23]},
    "shift_end_hour": {"dtype": "int", "nullable": False, "range": [0, 23]},
    "typical_devices": {"dtype": "int", "nullable": False, "range": [1, 10]},
}

EVENTS_SCHEMA: dict[str, dict] = {
    # --- bookkeeping fields: MUST be independent of label/time (see
    # evaluation/leakage_audit.py) ---
    "record_id": {"dtype": "str", "nullable": False},
    "insertion_order": {"dtype": "int", "nullable": False, "range": [0, 10**9]},
    "generation_batch": {"dtype": "int", "nullable": False, "range": [0, 10**6]},
    # --- identity / timing ---
    "user_id": {"dtype": "str", "nullable": False},
    "session_id": {"dtype": "str", "nullable": False},
    "timestamp": {"dtype": "datetime", "nullable": False},
    "event_type": {"dtype": "category", "nullable": False, "values": EVENT_TYPES},
    # --- auth details (not applicable to resource_access rows -- see
    # generator/events.py, which leaves these fields null for that event type) ---
    "auth_result": {"dtype": "category", "nullable": True, "values": ["success", "failure"]},
    "auth_method": {"dtype": "category", "nullable": True, "values": AUTH_METHODS},
    # mfa_used stays non-nullable: on resource_access rows (where auth_result/
    # auth_method are null, since no authentication happens on that row) it is
    # simply False, a sentinel rather than a true nullable bool -- this avoids
    # needing pandas' nullable "boolean" extension dtype throughout the pipeline
    # for one field.
    "mfa_used": {"dtype": "bool", "nullable": False},
    "failure_reason": {"dtype": "category", "nullable": True, "values": FAILURE_REASONS},
    # --- device info ---
    "device_id": {"dtype": "str", "nullable": False},
    "device_type": {"dtype": "category", "nullable": False, "values": DEVICE_TYPES},
    "os": {"dtype": "str", "nullable": True},
    "browser": {"dtype": "category", "nullable": True, "values": BROWSERS},
    # --- network info ---
    "ip_address": {"dtype": "str", "nullable": False},
    "asn": {"dtype": "str", "nullable": False},
    "isp_name": {"dtype": "str", "nullable": False},
    "geo_country": {"dtype": "str", "nullable": False},
    "geo_city": {"dtype": "str", "nullable": False},
    "geo_lat": {"dtype": "float", "nullable": False, "range": [-90, 90]},
    "geo_lon": {"dtype": "float", "nullable": False, "range": [-180, 180]},
    "network_type": {"dtype": "category", "nullable": False, "values": NETWORK_TYPES},
    # --- behavior fields ---
    "session_duration_seconds": {"dtype": "float", "nullable": True, "range": [0, 24 * 3600]},
    "resource_accessed": {"dtype": "category", "nullable": True, "values": RESOURCE_TYPES},
    "action_count": {"dtype": "int", "nullable": False, "range": [0, 10000]},
    "is_weekend": {"dtype": "bool", "nullable": False},
    "is_holiday": {"dtype": "bool", "nullable": False},
    "is_off_hours": {"dtype": "bool", "nullable": False},
}

# Fields whose only purpose is generation bookkeeping; the leakage audit
# trains exclusively on these (joined to the label) and must find near-chance
# separability. Kept here so tests/evaluation reference one canonical list.
EVENTS_METADATA_ONLY_FIELDS: list[str] = ["record_id", "insertion_order", "generation_batch"]

LABELS_SCHEMA: dict[str, dict] = {
    "record_id": {"dtype": "str", "nullable": False},
    "is_attack": {"dtype": "bool", "nullable": False},
    "attack_id": {"dtype": "str", "nullable": True},
    "attack_type": {"dtype": "category", "nullable": True, "values": ATTACK_TYPES},
    "mitre_technique_ids": {"dtype": "str", "nullable": True},
}

ATTACKS_SCHEMA: dict[str, dict] = {
    "attack_id": {"dtype": "str", "nullable": False},
    "attack_type": {"dtype": "category", "nullable": False, "values": ATTACK_TYPES},
    "start_time": {"dtype": "datetime", "nullable": False},
    "end_time": {"dtype": "datetime", "nullable": False},
    "severity": {"dtype": "category", "nullable": False, "values": SEVERITIES},
    "target_user_id": {"dtype": "str", "nullable": False},
    "affected_assets": {"dtype": "str", "nullable": False},
    # Plain-English, per-campaign explanation generated from the attack's own
    # parameters (e.g. "60 failed password attempts against a single account
    # from an unfamiliar device within 2.3 minutes, ending in a successful
    # login"). This is the primary human-readable field the problem statement
    # requires; MITRE tags below are supplementary metadata, not a substitute.
    "rationale": {"dtype": "str", "nullable": False},
    "mitre_technique_ids": {"dtype": "str", "nullable": False},
    "mitre_tactic": {"dtype": "str", "nullable": False},
    "num_events_generated": {"dtype": "int", "nullable": False, "range": [1, 100000]},
    "parameters_json": {"dtype": "str", "nullable": False},
    "mitre_mapping_version": {"dtype": "int", "nullable": False, "range": [1, 1000]},
}


def validate_dataframe(df: pd.DataFrame, schema: dict[str, dict], name: str) -> None:
    """Validate `df` against `schema`, raising ValueError with *all* issues.

    Checks: required columns present, nullability, categorical membership,
    dtype family (datetime/numeric/bool), and numeric range.
    """
    errors: list[str] = []

    for col, spec in schema.items():
        if col not in df.columns:
            errors.append(f"[{name}] missing required column: {col}")
            continue

        series = df[col]
        if not spec.get("nullable", False) and series.isna().any():
            n_null = int(series.isna().sum())
            errors.append(f"[{name}.{col}] contains {n_null} null value(s) but is not nullable")

        non_null = series.dropna()
        dtype = spec["dtype"]

        if dtype == "category" and spec.get("values") is not None:
            bad = set(non_null.unique()) - set(spec["values"])
            if bad:
                errors.append(f"[{name}.{col}] unexpected categorical value(s): {sorted(map(str, bad))[:5]}")
        elif dtype == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(series):
                errors.append(f"[{name}.{col}] expected datetime dtype, got {series.dtype}")
        elif dtype in ("int", "float"):
            if not pd.api.types.is_numeric_dtype(series):
                errors.append(f"[{name}.{col}] expected numeric dtype, got {series.dtype}")
            elif "range" in spec and len(non_null):
                lo, hi = spec["range"]
                if (non_null < lo).any() or (non_null > hi).any():
                    errors.append(f"[{name}.{col}] value(s) outside expected range [{lo}, {hi}]")
        elif dtype == "bool":
            if not pd.api.types.is_bool_dtype(series):
                errors.append(f"[{name}.{col}] expected bool dtype, got {series.dtype}")
        # dtype == "str": object dtype is acceptable, nothing further to check.

    if errors:
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Schema validation failed for '{name}' with {len(errors)} issue(s):\n{bullet_list}")
