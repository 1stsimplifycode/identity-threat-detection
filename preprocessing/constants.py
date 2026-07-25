"""Canonical categorical vocabularies shared across the pipeline.

These lists are the single source of truth for "what values are allowed in
this categorical column" -- used directly by preprocessing.schema for
validation, and imported by generator/ and attacks/ so that generation code
and validation code can never silently drift apart.

Generation *ratios* (how often each value occurs) are NOT here -- those are
config-driven (see configs/population, configs/events, configs/attacks).
This module only fixes the *vocabulary*, analogous to how MITRE technique
names are fixed reference data in configs/mitre_mapping.yaml.
"""
from __future__ import annotations

DEPARTMENTS: list[str] = [
    "Engineering",
    "Sales",
    "Finance",
    "HR",
    "IT",
    "Legal",
    "Operations",
    "Marketing",
    "Executive",
    "Customer Support",
]

PRIVILEGE_LEVELS: list[str] = ["standard", "elevated", "admin", "domain_admin"]

WORK_MODES: list[str] = ["onsite", "hybrid", "remote"]

EMPLOYMENT_STATUSES: list[str] = ["active", "on_leave", "terminated"]

DEVICE_TYPES: list[str] = ["desktop", "laptop", "mobile", "server", "unknown"]

NETWORK_TYPES: list[str] = ["corporate_vpn", "home_isp", "public_wifi", "mobile_carrier"]

AUTH_METHODS: list[str] = ["password", "sso", "mfa_push", "mfa_otp", "api_token"]

FAILURE_REASONS: list[str] = ["bad_password", "mfa_denied", "account_locked", "unknown_user"]

RESOURCE_TYPES: list[str] = [
    "email", "vpn", "crm", "hr_system", "code_repo", "file_share", "admin_console",
]

BROWSERS: list[str] = ["Chrome", "Firefox", "Edge", "Safari", "Other"]

SEVERITIES: list[str] = ["low", "medium", "high", "critical"]

# event_type vocabulary. "resource_access" (added Phase 2a) represents a
# resource-to-resource transition within an already-authenticated session --
# it shares session_id with the login_attempt that started the session, and
# is what makes the access-chain graph (feature_engineering, Phase 2b) and
# lateral-movement detection possible at all.
EVENT_TYPES: list[str] = ["login_attempt", "resource_access"]

# The fixed, complete attack scope per the authoritative problem statement --
# exactly these 5, no more, no fewer. (An earlier draft of this project also
# planned "insider_threat" as a 6th type; that was dropped when the
# authoritative spec fixed the list to these 5.)
ATTACK_TYPES: list[str] = [
    "brute_force",
    "impossible_travel",
    "credential_misuse",
    "lateral_movement",
    "device_spoofing",
]
