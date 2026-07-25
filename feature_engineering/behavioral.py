"""Behavioral (per-event, time-based) feature engineering -- constraint 3a.

One stateful class, `BehavioralFeatureState`, is the single source of truth
for these features: `update(event)` mutates per-user/per-department rolling
state and returns that event's feature row, computed from state as it
existed BEFORE this event (never after, so a feature never leaks its own
event's contribution into itself). Batch mode (`compute_batch`) simply
replays `update()` over a chronologically-sorted DataFrame from empty
state; streaming mode (Phase 4) keeps the same state object alive across
live per-event calls. There is exactly one implementation of the feature
logic, not two that can silently drift apart.

Features produced, per event:
    velocity_kmh                 -- implied speed from the previous event's
                                     location to this one
    geo_distance_from_home_km    -- distance from the user's home location
    login_location_entropy       -- Shannon entropy of this user's recent
                                     city distribution (history only, not
                                     including the current event)
    device_switch_rate           -- fraction of recent consecutive event
                                     pairs where the device changed
    failed_login_ratio           -- fraction of recent login_attempt rows
                                     that failed
    peer_group_deviation         -- |this user's off-hours-activity EMA -
                                     their department's off-hours-activity EMA|
    ema_failure_rate             -- this user's own rolling EMA failure rate
                                     (a baseline in its own right, and the
                                     input peer_group_deviation is compared
                                     against at the department level)

All are plain, named, human-readable columns -- no learned embeddings.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from feature_engineering.feature_names import BEHAVIORAL_FEATURE_COLUMNS
from preprocessing.geo_utils import haversine_km

_MIN_ELAPSED_HOURS = 1.0 / 60.0  # 1 minute floor, avoids division-by-near-zero velocity spikes


class BehavioralFeatureState:
    """Rolling per-user and per-department state for the 3a behavioral
    features. Construct once per run (batch) or once per deployed process
    (streaming), then call `update()` per event in chronological order.
    """

    def __init__(self, cfg: DictConfig, users: pd.DataFrame) -> None:
        be_cfg = cfg.feature_engineering.behavioral
        self.window = pd.Timedelta(days=float(cfg.feature_engineering.window_days))
        self.ema_gamma = float(be_cfg.ema_gamma)
        self.peer_group_min_size = int(be_cfg.peer_group_min_size)

        users_indexed = users.set_index("user_id")
        self._home_lat: dict[str, float] = users_indexed["home_lat"].to_dict()
        self._home_lon: dict[str, float] = users_indexed["home_lon"].to_dict()
        self._department: dict[str, str] = users_indexed["department"].to_dict()

        self._history: dict[str, deque] = defaultdict(deque)
        self._user_offhours_ema: dict[str, float] = {}
        self._user_failure_ema: dict[str, float] = {}
        self._dept_offhours_ema: dict[str, float] = {}
        self._dept_members_seen: dict[str, set[str]] = defaultdict(set)

    def _prune(self, history: deque, now: pd.Timestamp) -> None:
        while history and (now - history[0]["timestamp"]) > self.window:
            history.popleft()

    def _velocity_kmh(self, history: deque, event: dict[str, Any]) -> float:
        if not history:
            return 0.0
        prev = history[-1]
        distance_km = haversine_km(prev["geo_lat"], prev["geo_lon"], event["geo_lat"], event["geo_lon"])
        elapsed_hours = max((event["timestamp"] - prev["timestamp"]).total_seconds() / 3600.0, _MIN_ELAPSED_HOURS)
        return distance_km / elapsed_hours

    def _location_entropy(self, history: deque) -> float:
        if not history:
            return 0.0
        counts = Counter(h["geo_city"] for h in history)
        total = sum(counts.values())
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    def _device_switch_rate(self, history: deque) -> float:
        if len(history) < 2:
            return 0.0
        switches = sum(1 for a, b in zip(history, list(history)[1:]) if a["device_id"] != b["device_id"])
        return switches / (len(history) - 1)

    def _failed_login_ratio(self, history: deque) -> float:
        logins = [h for h in history if h["is_login"]]
        if not logins:
            return 0.0
        failures = sum(1 for h in logins if h["auth_result"] == "failure")
        return failures / len(logins)

    def update(self, event: dict[str, Any]) -> dict[str, Any]:
        user_id = event["user_id"]
        ts = event["timestamp"]
        history = self._history[user_id]
        self._prune(history, ts)

        velocity_kmh = self._velocity_kmh(history, event)
        home_lat = self._home_lat.get(user_id)
        geo_distance_from_home_km = (
            haversine_km(home_lat, self._home_lon[user_id], event["geo_lat"], event["geo_lon"])
            if home_lat is not None else 0.0
        )
        login_location_entropy = self._location_entropy(history)
        device_switch_rate = self._device_switch_rate(history)
        failed_login_ratio = self._failed_login_ratio(history)

        ema_failure_rate = self._user_failure_ema.get(user_id, 0.0)
        user_offhours_ema = self._user_offhours_ema.get(user_id, 0.0)
        department = self._department.get(user_id)
        dept_members = self._dept_members_seen[department] if department else set()
        if department is not None and len(dept_members) >= self.peer_group_min_size:
            dept_offhours_ema = self._dept_offhours_ema.get(department, 0.0)
            peer_group_deviation = abs(user_offhours_ema - dept_offhours_ema)
        else:
            peer_group_deviation = 0.0  # not enough peers yet for a stable comparison

        features = {
            "velocity_kmh": velocity_kmh,
            "geo_distance_from_home_km": geo_distance_from_home_km,
            "login_location_entropy": login_location_entropy,
            "device_switch_rate": device_switch_rate,
            "failed_login_ratio": failed_login_ratio,
            "peer_group_deviation": peer_group_deviation,
            "ema_failure_rate": ema_failure_rate,
        }

        # -- state updates, using THIS event, happen after computing the
        # above (so the returned features reflect history strictly before
        # this event) --
        is_login = event["event_type"] == "login_attempt"
        off_hours_val = 1.0 if event["is_off_hours"] else 0.0
        gamma = self.ema_gamma
        self._user_offhours_ema[user_id] = gamma * user_offhours_ema + (1 - gamma) * off_hours_val
        if department is not None:
            dept_members.add(user_id)
            self._dept_offhours_ema[department] = (
                gamma * self._dept_offhours_ema.get(department, 0.0) + (1 - gamma) * off_hours_val
            )
        if is_login and event.get("auth_result") is not None:
            failure_val = 1.0 if event["auth_result"] == "failure" else 0.0
            self._user_failure_ema[user_id] = gamma * ema_failure_rate + (1 - gamma) * failure_val

        history.append({
            "timestamp": ts,
            "geo_lat": event["geo_lat"],
            "geo_lon": event["geo_lon"],
            "geo_city": event["geo_city"],
            "device_id": event["device_id"],
            "is_login": is_login,
            "auth_result": event.get("auth_result"),
        })

        return features

    def compute_batch(self, events: pd.DataFrame) -> pd.DataFrame:
        """Replay `update()` over `events` (must be chronologically sorted,
        as `generator/run.py`'s output already is) from this state's current
        (typically empty, for a fresh instance) starting point.
        """
        rows: list[dict[str, Any]] = []
        for event in events.to_dict("records"):
            feats = self.update(event)
            feats["record_id"] = event["record_id"]
            rows.append(feats)
        return pd.DataFrame(rows, columns=["record_id"] + BEHAVIORAL_FEATURE_COLUMNS)
