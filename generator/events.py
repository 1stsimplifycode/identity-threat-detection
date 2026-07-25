"""Synthetic login/authentication event generator.

Produces realistic-*shaped* (statistically plausible, not real-log-validated)
benign login traffic respecting shift patterns, weekends, holidays, and
remote-vs-onsite work mode. Attack events are generated separately by
`attacks/` and merged in by `generator/run.py`.

Phase 2a note: `event_type` now includes `resource_access` alongside
`login_attempt` -- a successful login occasionally (config-driven
probability) fans out into a short chain of resource_access events sharing
its `session_id`, so that the lateral-movement attack's resource-chain
signature (attacks/lateral_movement.py) is not the *only* source of
multi-resource sessions in the dataset, and so the access-chain graph
feature (feature_engineering, Phase 2b) has real benign traffic to contrast
against. Occasional cross-department resource access (still within the
user's privilege scope) and, for a couple of departments, a shared "kiosk"
device are also generated here for the same reason: to keep
attacks/credential_misuse.py's and attacks/device_spoofing.py's tells from
being 100%-exclusive to attack traffic. See docs/phase_2a_report.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from generator.constants import (
    MIN_PRIVILEGE_BY_RESOURCE,
    OS_BY_DEVICE_TYPE,
    RESOURCE_TYPES_BY_DEPT,
    SHARED_DEVICE_DEPARTMENTS,
    fake_ip,
    random_hex,
    weighted_choice,
)
from generator.drift import ResolvedDrift
from preprocessing.constants import PRIVILEGE_LEVELS, RESOURCE_TYPES
from preprocessing.geo_utils import GLOBAL_CITY_POOL

EVENTS_COLUMN_ORDER: list[str] = [
    "user_id", "session_id", "timestamp", "event_type", "auth_result", "auth_method", "mfa_used",
    "failure_reason", "device_id", "device_type", "os", "browser", "ip_address", "asn",
    "isp_name", "geo_country", "geo_city", "geo_lat", "geo_lon", "network_type",
    "session_duration_seconds", "resource_accessed", "action_count",
    "is_weekend", "is_holiday", "is_off_hours",
]

# Shared-context fields copied verbatim from a login_attempt row onto every
# resource_access row it spawns (same device, same location, same network --
# it's the same authenticated session, just a later resource touch).
_SESSION_CONTEXT_FIELDS: list[str] = [
    "user_id", "session_id", "device_id", "device_type", "os", "browser",
    "ip_address", "asn", "isp_name", "geo_country", "geo_city", "geo_lat",
    "geo_lon", "network_type",
]


def _is_hour_off_shift(hour: int, shift_start: int, shift_end: int) -> bool:
    if shift_start <= shift_end:
        return not (shift_start <= hour <= shift_end)
    return not (hour >= shift_start or hour <= shift_end)  # overnight shift wraps past midnight


def _build_device_pool(rng: np.random.Generator, user_id: str, n_devices: int, device_type_weights: dict[str, float]) -> list[dict]:
    pool = []
    for i in range(n_devices):
        device_type = weighted_choice(rng, device_type_weights)
        os_name = str(rng.choice(OS_BY_DEVICE_TYPE[device_type]))
        pool.append({"device_id": f"DEV-{user_id}-{i:02d}", "device_type": device_type, "os": os_name})
    return pool


def _build_kiosk_device_pools(rng: np.random.Generator, device_type_weights: dict[str, float], devices_per_dept: int = 2) -> dict[str, list[dict]]:
    """Shared devices for departments in SHARED_DEVICE_DEPARTMENTS (built
    once, outside the per-user loop, since they're shared ACROSS users).
    """
    pools: dict[str, list[dict]] = {}
    for dept in SHARED_DEVICE_DEPARTMENTS:
        dept_slug = dept.replace(" ", "")
        pool = []
        for i in range(devices_per_dept):
            device_type = weighted_choice(rng, device_type_weights)
            os_name = str(rng.choice(OS_BY_DEVICE_TYPE[device_type]))
            pool.append({"device_id": f"DEV-KIOSK-{dept_slug}-{i:02d}", "device_type": device_type, "os": os_name})
        pools[dept] = pool
    return pools


def _sample_event_hour(rng: np.random.Generator, shift_start: int, shift_end: int, off_hours_prob: float) -> int:
    if rng.random() < off_hours_prob:
        return int(rng.integers(0, 24))
    if shift_start <= shift_end:
        return int(rng.integers(shift_start, shift_end + 1))
    span = list(range(shift_start, 24)) + list(range(0, shift_end + 1))
    return int(rng.choice(span))


def _pick_resource(rng: np.random.Generator, department: str, privilege_level: str, cross_dept_prob: float) -> str:
    """Pick a resource_accessed value. With `cross_dept_prob` probability,
    pick a resource outside the department's typical set but still within
    the user's own privilege scope -- ordinary, non-suspicious cross-team
    access (e.g. an engineer checking a shared file_share doc), which keeps
    "accessed a resource outside my department" from being attack-exclusive
    to credential_misuse (whose tell additionally requires exceeding
    privilege scope -- a conjunction, not this single condition).
    """
    dept_resources = RESOURCE_TYPES_BY_DEPT.get(department, RESOURCE_TYPES_BY_DEPT["_default"])
    if rng.random() < cross_dept_prob:
        priv_rank = PRIVILEGE_LEVELS.index(privilege_level)
        allowed_outside = [
            r for r in RESOURCE_TYPES
            if r not in dept_resources and PRIVILEGE_LEVELS.index(MIN_PRIVILEGE_BY_RESOURCE[r]) <= priv_rank
        ]
        if allowed_outside:
            return str(rng.choice(allowed_outside))
    return str(rng.choice(dept_resources))


def _build_resource_chain(
    rng: np.random.Generator,
    department: str,
    privilege_level: str,
    session_context: dict[str, Any],
    base_ts: pd.Timestamp,
    n_hops: int,
    cross_dept_prob: float,
    holidays: set,
) -> list[dict[str, Any]]:
    """Benign resource_access events following a successful login, sharing
    the login's session_id and device/geo/network context.
    """
    events: list[dict[str, Any]] = []
    ts = base_ts
    for _ in range(n_hops):
        ts = ts + pd.Timedelta(seconds=int(rng.integers(5, 180)))
        resource = _pick_resource(rng, department, privilege_level, cross_dept_prob)
        event = {field: session_context[field] for field in _SESSION_CONTEXT_FIELDS}
        event.update({
            "timestamp": ts,
            "event_type": "resource_access",
            "auth_result": None,
            "auth_method": None,
            "mfa_used": False,
            "failure_reason": None,
            "session_duration_seconds": None,
            "resource_accessed": resource,
            "action_count": int(rng.poisson(2) + 1),
            "is_weekend": ts.weekday() >= 5,
            "is_holiday": ts.date() in holidays,
            "is_off_hours": session_context["is_off_hours"],  # same session, same shift context
        })
        events.append(event)
    return events


def generate_login_events(
    users: pd.DataFrame,
    cfg: DictConfig,
    rng: np.random.Generator,
    drift: ResolvedDrift | None = None,
) -> pd.DataFrame:
    """Generate benign login events (plus any resulting resource_access
    chains) for every non-terminated user over the simulation window
    defined by `cfg.events`.

    `drift` (Phase 3), if given, is applied per (user, day_index): once a
    user's individual rollout day for a scheduled drift event has passed,
    their effective work_mode/home-location/shift attributes are overridden
    for that day's generation onward -- see `generator/drift.py`.
    """
    ev_cfg = cfg.events
    start_date = pd.Timestamp(ev_cfg.start_date)
    num_days = int(ev_cfg.num_days)
    holidays = {pd.Timestamp(d).date() for d in ev_cfg.holidays}
    dates = [start_date + pd.Timedelta(days=d) for d in range(num_days)]

    device_type_weights = dict(ev_cfg.device_type_weights)
    auth_method_weights = dict(ev_cfg.auth_method_weights)
    failure_reason_weights = dict(ev_cfg.failure_reason_weights)
    browsers = ["Chrome", "Firefox", "Edge", "Safari", "Other"]
    kiosk_device_pools = _build_kiosk_device_pools(rng, device_type_weights)

    resource_chain_probability = float(ev_cfg.get("resource_chain_probability", 0.0))
    resource_chain_hops = list(ev_cfg.get("resource_chain_hops", [1, 4]))
    cross_department_resource_probability = float(ev_cfg.get("cross_department_resource_probability", 0.0))
    shared_kiosk_device_probability = float(ev_cfg.get("shared_kiosk_device_probability", 0.0))

    rows: list[dict[str, Any]] = []

    for _, user in users.iterrows():
        if user["employment_status"] == "terminated":
            # Documented Phase 1 simplification: terminated users generate no
            # post-termination activity tail. See docs/phase_1_report.md.
            continue

        user_id = user["user_id"]
        device_pool = _build_device_pool(rng, user_id, int(user["typical_devices"]), device_type_weights)
        department = user["department"]

        for day_index, date in enumerate(dates):
            if date.date() < user["join_date"].date():
                continue

            # -- resolve this user's EFFECTIVE attributes for this day,
            # applying any drift overrides whose rollout day has passed
            # (Phase 3) -- everything below this point uses these, not the
            # user's static population-table attributes, so a drifted user's
            # behavior actually changes from their individual rollout day
            # forward.
            overrides = drift.effective_overrides(user_id, day_index) if drift is not None else {}
            work_mode = overrides.get("work_mode", user["work_mode"])
            home_country = overrides.get("home_country", user["home_country"])
            home_city = overrides.get("home_city", user["home_city"])
            home_lat = overrides.get("home_lat", user["home_lat"])
            home_lon = overrides.get("home_lon", user["home_lon"])
            shift_start = int(overrides.get("shift_start_hour", user["shift_start_hour"]))
            shift_end = int(overrides.get("shift_end_hour", user["shift_end_hour"]))
            off_hours_base_prob = min(0.15 * (ev_cfg.remote_offhours_multiplier if work_mode == "remote" else 1.0), 0.5)

            multiplier = 1.0
            is_weekend = date.weekday() >= 5
            is_holiday = date.date() in holidays
            if is_weekend:
                multiplier *= ev_cfg.weekend_activity_multiplier
            if is_holiday:
                multiplier *= ev_cfg.holiday_activity_multiplier
            if user["employment_status"] == "on_leave":
                multiplier *= ev_cfg.on_leave_activity_multiplier

            lam = max(float(ev_cfg.avg_events_per_user_per_day) * multiplier, 0.001)
            n_events = int(rng.poisson(lam))
            if n_events == 0:
                continue

            for _ in range(n_events):
                hour = _sample_event_hour(rng, shift_start, shift_end, off_hours_base_prob)
                ts = date + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))
                is_off_hours = _is_hour_off_shift(hour, shift_start, shift_end)
                session_id = f"SESS-{random_hex(rng, 6)}"

                auth_result = "failure" if rng.random() < ev_cfg.failure_rate_baseline else "success"
                auth_method = weighted_choice(rng, auth_method_weights)
                if auth_method in ("mfa_push", "mfa_otp"):
                    mfa_used = bool(rng.random() < ev_cfg.mfa_adoption_rate)
                elif auth_method == "sso":
                    mfa_used = bool(rng.random() < 0.3)
                else:
                    mfa_used = False
                failure_reason = weighted_choice(rng, failure_reason_weights) if auth_result == "failure" else None

                if department in kiosk_device_pools and rng.random() < shared_kiosk_device_probability:
                    pool = kiosk_device_pools[department]
                    dev = pool[int(rng.integers(0, len(pool)))]
                elif rng.random() < ev_cfg.new_device_probability:
                    new_device_type = weighted_choice(rng, device_type_weights)
                    dev = {
                        "device_id": f"DEV-{user['user_id']}-NEW-{random_hex(rng, 3)}",
                        "device_type": new_device_type,
                        "os": str(rng.choice(OS_BY_DEVICE_TYPE[new_device_type])),
                    }
                else:
                    dev = device_pool[int(rng.integers(0, len(device_pool)))]

                browser = str(rng.choice(browsers)) if dev["device_type"] in ("desktop", "laptop", "mobile") and auth_method in ("password", "sso") else None

                is_travel = rng.random() < ev_cfg.travel_event_probability
                if is_travel:
                    city, lat, lon, country = GLOBAL_CITY_POOL[int(rng.integers(0, len(GLOBAL_CITY_POOL)))]
                    network_type = str(rng.choice(["public_wifi", "mobile_carrier"]))
                else:
                    lat = float(home_lat) + rng.normal(0, 0.02)
                    lon = float(home_lon) + rng.normal(0, 0.02)
                    city, country = home_city, home_country
                    if work_mode in ("onsite", "hybrid") and not is_off_hours:
                        network_type = "corporate_vpn"
                    else:
                        network_type = str(rng.choice(["corporate_vpn", "home_isp"], p=[0.4, 0.6]))

                session_duration = None
                resource = None
                action_count = 0
                if auth_result == "success":
                    session_duration = float(min(rng.lognormal(7.2, 0.9), 6 * 3600))
                    resource = _pick_resource(rng, department, user["privilege_level"], cross_department_resource_probability)
                    action_count = int(rng.poisson(4) + 1)

                login_row = {
                    "user_id": user["user_id"],
                    "session_id": session_id,
                    "timestamp": ts,
                    "event_type": "login_attempt",
                    "auth_result": auth_result,
                    "auth_method": auth_method,
                    "mfa_used": mfa_used,
                    "failure_reason": failure_reason,
                    "device_id": dev["device_id"],
                    "device_type": dev["device_type"],
                    "os": dev["os"],
                    "browser": browser,
                    "ip_address": fake_ip(rng),
                    "asn": f"AS{int(rng.integers(1000, 65000))}",
                    "isp_name": f"ISP-{country}-{int(rng.integers(1, 20))}",
                    "geo_country": country,
                    "geo_city": city,
                    "geo_lat": lat,
                    "geo_lon": lon,
                    "network_type": network_type,
                    "session_duration_seconds": session_duration,
                    "resource_accessed": resource,
                    "action_count": action_count,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "is_off_hours": is_off_hours,
                }
                rows.append(login_row)

                if auth_result == "success" and rng.random() < resource_chain_probability:
                    n_hops = int(rng.integers(resource_chain_hops[0], resource_chain_hops[1] + 1))
                    rows.extend(_build_resource_chain(
                        rng, department, user["privilege_level"], login_row, ts,
                        n_hops, cross_department_resource_probability, holidays,
                    ))

    if not rows:
        return pd.DataFrame(columns=EVENTS_COLUMN_ORDER)

    events = pd.DataFrame(rows)
    return events[EVENTS_COLUMN_ORDER]
