"""Canonical feature-column-name lists -- deliberately zero heavy
dependencies (no pandas/networkx/omegaconf), so lightweight consumers like
`dashboard/app.py` (which only needs the NAMES, never computes features
itself) can import them without pulling in the full feature-computation
stack. `feature_engineering/behavioral.py`, `graph.py`, and `pipeline.py`
all import their column list FROM here, not the reverse -- one source of
truth, consistent with this project's existing dual-mode design discipline.
"""
from __future__ import annotations

BEHAVIORAL_FEATURE_COLUMNS: list[str] = [
    "velocity_kmh",
    "geo_distance_from_home_km",
    "login_location_entropy",
    "device_switch_rate",
    "failed_login_ratio",
    "peer_group_deviation",
    "ema_failure_rate",
]

GRAPH_FEATURE_COLUMNS: list[str] = [
    "device_fan_in",
    "user_device_set_delta",
    "is_new_edge",
    "access_chain_distance",
    "peer_community_deviation",
    "device_fingerprint_mismatch",
    "session_foreign_resource_count",
    "session_hop_seconds",
]

FEATURE_COLUMNS: list[str] = BEHAVIORAL_FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS
