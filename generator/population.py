"""Synthetic user population generator.

Produces a DataFrame of users with department, role, privilege level, a
simplified two-tier manager hierarchy, join date, and employment status --
plus the home-location and shift-timing attributes the event generator
needs to produce realistic-shaped (not real-log-validated) login behavior.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from generator.constants import MANAGER_ROLE_KEYWORDS, ROLES_BY_DEPARTMENT
from preprocessing.geo_utils import HOME_CITY_POOL

USERS_COLUMN_ORDER: list[str] = [
    "user_id", "department", "role", "privilege_level", "manager_id", "join_date",
    "employment_status", "work_mode", "home_country", "home_city", "home_lat", "home_lon",
    "shift_start_hour", "shift_end_hour", "typical_devices",
]


def _sample_privilege(rng: np.random.Generator, role: str, base_dist: dict[str, float]) -> str:
    labels = list(base_dist.keys())
    probs = np.array(list(base_dist.values()), dtype=float)
    if any(k in role for k in ("Chief", "VP", "Director")):
        probs = probs * np.array([0.3, 0.9, 1.6, 2.0])
    elif "Manager" in role or "Administrator" in role:
        probs = probs * np.array([0.7, 1.3, 1.4, 1.2])
    probs = probs / probs.sum()
    return str(labels[int(rng.choice(len(labels), p=probs))])


def sample_shift(rng: np.random.Generator, department: str) -> tuple[int, int]:
    """Public (used by generator/drift.py too, for schedule_shift drift
    events, which need the same shift-pattern distribution a newly-hired
    user in that department would get)."""
    if department in ("Customer Support", "Operations", "IT"):
        pattern = rng.choice(["day", "evening", "night"], p=[0.55, 0.30, 0.15])
    else:
        pattern = "day"

    if pattern == "day":
        start = int(rng.integers(7, 10))
        end = min(start + 9, 23)
    elif pattern == "evening":
        start = int(rng.integers(14, 17))
        end = (start + 9) % 24
    else:
        start = int(rng.integers(21, 24)) % 24
        end = (start + 9) % 24
    return start, end


def generate_population(cfg: DictConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the user population per `cfg.population` and `cfg.events.start_date`."""
    n = int(cfg.population.num_users)
    sim_start = pd.Timestamp(cfg.events.start_date)

    dept_weights = dict(cfg.population.department_weights)
    departments = list(dept_weights.keys())
    dept_probs = np.array(list(dept_weights.values()), dtype=float)
    dept_probs = dept_probs / dept_probs.sum()
    dept_choice = [str(d) for d in rng.choice(departments, size=n, p=dept_probs)]

    roles = [str(rng.choice(ROLES_BY_DEPARTMENT[d])) for d in dept_choice]

    priv_dist = dict(cfg.population.privilege_distribution)
    privilege = [_sample_privilege(rng, r, priv_dist) for r in roles]

    work_dist = dict(cfg.population.work_mode_distribution)
    work_labels = list(work_dist.keys())
    work_probs = np.array(list(work_dist.values()), dtype=float)
    work_probs = work_probs / work_probs.sum()
    work_mode = [str(w) for w in rng.choice(work_labels, size=n, p=work_probs)]

    termination_rate = float(cfg.population.termination_rate)
    status_draw = rng.random(n)
    employment_status = np.where(
        status_draw < termination_rate, "terminated",
        np.where(status_draw < termination_rate + 0.01, "on_leave", "active"),
    )

    countries = list(cfg.population.home_countries)
    country_choice = [str(c) for c in rng.choice(countries, size=n)]

    home_city, home_lat, home_lon = [], [], []
    for country in country_choice:
        options = HOME_CITY_POOL.get(country, HOME_CITY_POOL["US"])
        city, lat, lon = options[int(rng.integers(0, len(options)))]
        home_city.append(city)
        home_lat.append(lat + rng.normal(0, 0.05))
        home_lon.append(lon + rng.normal(0, 0.05))

    # Join dates: ~90% tenured (60 days - 5 years before sim start), ~10%
    # recent (< 60 days before sim start). Phase 2 will tag the recent
    # cohort as "cold start" for filtered evaluation; Phase 1 only needs the
    # distribution shape to exist.
    join_date = []
    for _ in range(n):
        if rng.random() < 0.10:
            days_ago = int(rng.integers(1, 60))
        else:
            days_ago = int(rng.integers(60, 5 * 365))
        join_date.append(sim_start - pd.Timedelta(days=days_ago))

    shifts = [sample_shift(rng, d) for d in dept_choice]
    shift_start = [s[0] for s in shifts]
    shift_end = [s[1] for s in shifts]

    typical_devices = rng.integers(1, 5, size=n)

    user_id = [f"U{i:06d}" for i in range(1, n + 1)]

    users = pd.DataFrame({
        "user_id": user_id,
        "department": dept_choice,
        "role": roles,
        "privilege_level": privilege,
        "join_date": pd.to_datetime(join_date),
        "employment_status": employment_status,
        "work_mode": work_mode,
        "home_country": country_choice,
        "home_city": home_city,
        "home_lat": home_lat,
        "home_lon": home_lon,
        "shift_start_hour": shift_start,
        "shift_end_hour": shift_end,
        "typical_devices": typical_devices,
    })

    users["manager_id"] = _assign_managers(users, departments, rng)

    return users[USERS_COLUMN_ORDER]


def _assign_managers(users: pd.DataFrame, departments: list[str], rng: np.random.Generator) -> list[str | None]:
    """Simplified two-tier hierarchy: ICs report to a manager-eligible user in
    their own department; manager-eligible users report to an Executive, or
    to nobody. This deliberately avoids arbitrary-depth org charts (and the
    cycle-handling they'd require) -- see docs/phase_1_report.md, Deferrals.
    """
    manager_eligible = users["role"].str.contains("|".join(MANAGER_ROLE_KEYWORDS))

    eligible_ids_by_dept = {
        d: users.loc[manager_eligible & (users["department"] == d), "user_id"].tolist()
        for d in departments
    }
    all_eligible_ids = users.loc[manager_eligible, "user_id"].tolist()
    exec_ids = eligible_ids_by_dept.get("Executive", [])

    manager_ids: list[str | None] = []
    for idx, row in users.iterrows():
        if row["department"] == "Executive" and rng.random() < 0.5:
            manager_ids.append(None)
            continue
        if manager_eligible.loc[idx]:
            if exec_ids and rng.random() < 0.7:
                manager_ids.append(str(rng.choice(exec_ids)))
            else:
                manager_ids.append(None)
            continue
        pool = [m for m in eligible_ids_by_dept.get(row["department"], []) if m != row["user_id"]]
        if not pool:
            pool = [m for m in all_eligible_ids if m != row["user_id"]]
        manager_ids.append(str(rng.choice(pool)) if pool else None)

    return manager_ids
