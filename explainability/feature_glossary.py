"""Human-readable descriptions of every model feature -- one shared source
for both SHAP's plain-language explanation strings (below) and the
dashboard's hover tooltips (Phase 4's "hover tooltips explaining metrics"
requirement), so the two never drift out of sync with different wording for
the same column.
"""
from __future__ import annotations

FEATURE_GLOSSARY: dict[str, str] = {
    # Behavioral (3a)
    "velocity_kmh": "Implied travel speed from the user's previous event location to this one.",
    "geo_distance_from_home_km": "Distance between this event's location and the user's home location.",
    "login_location_entropy": "Shannon entropy of the user's recent login-city distribution -- higher means more scattered locations.",
    "device_switch_rate": "Fraction of the user's recent consecutive events where the device changed.",
    "failed_login_ratio": "Fraction of the user's recent login attempts that failed.",
    "peer_group_deviation": "How far this user's off-hours activity level sits from their department's typical level.",
    "ema_failure_rate": "This user's own rolling exponential-moving-average login failure rate.",
    # Graph-derived (3b)
    "device_fan_in": "Number of distinct users who have historically used this device.",
    "user_device_set_delta": "How far this device sits from the user's own typical device-usage pattern.",
    "is_new_edge": "How unfamiliar this (user, device) pairing is -- 0 if well-established, higher the more novel.",
    "access_chain_distance": "Cost of moving from the user's last-visited resource to this one, within their own department's normal transition patterns.",
    "peer_community_deviation": "Fraction of the user's peer community (by device-sharing pattern) who have NOT also used this device.",
    "device_fingerprint_mismatch": "Whether this device's claimed type/OS differs from the first signature ever recorded for this device ID.",
    "session_foreign_resource_count": "Number of distinct resource types accessed so far this session that fall outside this user's own department's typical resources.",
    "session_hop_seconds": "Seconds since the previous event in this same session -- a large value means no recent activity in this session (or its first event).",
}


def describe(feature: str) -> str:
    return FEATURE_GLOSSARY.get(feature, feature.replace("_", " "))
