"""Generation-only reference vocabulary: role lists, weighting helpers, and
small utility functions used by both the population and event generators.

Ratios/probabilities that are genuine experiment knobs (imbalance level,
window size, event rate, ...) live in Hydra configs, per the project's
config-driven design constraint -- what's here is vocabulary (job titles,
which resources exist) and structural helpers, analogous to how MITRE
technique names are fixed reference data rather than a tunable.
"""
from __future__ import annotations

import numpy as np

ROLES_BY_DEPARTMENT: dict[str, list[str]] = {
    "Engineering": [
        "Software Engineer", "Senior Software Engineer", "Engineering Manager",
        "QA Engineer", "DevOps Engineer", "Data Engineer",
    ],
    "Sales": [
        "Sales Representative", "Account Executive", "Sales Manager", "Sales Engineer",
    ],
    "Finance": [
        "Financial Analyst", "Accountant", "Finance Manager", "Controller",
    ],
    "HR": [
        "HR Generalist", "Recruiter", "HR Manager", "Compensation Analyst",
    ],
    "IT": [
        "IT Support Specialist", "System Administrator", "Network Engineer", "IT Manager",
    ],
    "Legal": [
        "Legal Counsel", "Paralegal", "Compliance Officer",
    ],
    "Operations": [
        "Operations Analyst", "Operations Manager", "Logistics Coordinator",
    ],
    "Marketing": [
        "Marketing Specialist", "Content Strategist", "Marketing Manager", "SEO Analyst",
    ],
    "Executive": [
        "VP", "Director", "Chief Officer",
    ],
    "Customer Support": [
        "Support Agent", "Support Team Lead", "Customer Success Manager",
    ],
}

MANAGER_ROLE_KEYWORDS: tuple[str, ...] = (
    "Manager", "Director", "VP", "Chief", "Lead", "Controller",
)

OS_BY_DEVICE_TYPE: dict[str, list[str]] = {
    "desktop": ["Windows", "macOS", "Linux"],
    "laptop": ["Windows", "macOS", "Linux"],
    "mobile": ["iOS", "Android"],
    "server": ["Linux", "Windows"],
    "unknown": ["Windows"],
}

RESOURCE_TYPES_BY_DEPT: dict[str, list[str]] = {
    "Engineering": ["code_repo", "email", "vpn", "file_share"],
    "Sales": ["crm", "email", "vpn"],
    "Finance": ["file_share", "email", "vpn", "hr_system"],
    "HR": ["hr_system", "email", "file_share"],
    "IT": ["admin_console", "vpn", "email", "file_share"],
    "Legal": ["file_share", "email", "vpn"],
    "Operations": ["file_share", "email", "vpn"],
    "Marketing": ["email", "file_share", "crm"],
    "Executive": ["email", "file_share", "hr_system", "crm"],
    "Customer Support": ["crm", "email", "vpn"],
    "_default": ["email", "vpn"],
}

# Minimum privilege_level (see PRIVILEGE_LEVELS' ascending order in
# preprocessing/constants.py: standard < elevated < admin < domain_admin)
# normally required to reach each resource, independent of department. Used
# by attacks/credential_misuse.py: a successful login reaching a resource
# that is BOTH outside the user's department's typical set (RESOURCE_TYPES_BY_DEPT)
# AND above their privilege_level is the attack's tell -- a conjunction, not
# a single field, so it isn't trivially separable from ordinary cross-department
# access (which benign traffic also does at a low rate -- see
# `cross_department_resource_probability` in configs/events/*.yaml).
MIN_PRIVILEGE_BY_RESOURCE: dict[str, str] = {
    "email": "standard",
    "vpn": "standard",
    "file_share": "standard",
    "crm": "standard",
    "code_repo": "standard",
    "hr_system": "elevated",
    "admin_console": "admin",
}

# Coarse sensitivity ranking of resources (higher = more sensitive/valuable),
# used by attacks/lateral_movement.py to build a resource_access chain that
# trends toward higher-value systems -- the multi-hop, cross-domain escalation
# pattern is what distinguishes lateral_movement from credential_misuse's
# single atypical-resource touch. This is a generation-time heuristic, not
# the real detection signal: feature_engineering's actual access-chain graph
# (Phase 2b) is what a detector learns from, built from real event frequency,
# not this table.
RESOURCE_VALUE_TIER: dict[str, int] = {
    "email": 0,
    "vpn": 0,
    "crm": 1,
    "code_repo": 1,
    "file_share": 1,
    "hr_system": 2,
    "admin_console": 3,
}

# Departments where a small, low-probability "shared kiosk/service-terminal"
# device benignly appears across multiple users (e.g. a shared workstation at
# a support desk or ops floor). Exists so that ">1 user on one device" is not
# 100% attack-exclusive for attacks/device_spoofing.py's cross-user device
# reuse variant -- see generator/events.py's shared_kiosk_device_probability.
SHARED_DEVICE_DEPARTMENTS: tuple[str, ...] = ("Operations", "Customer Support")


def fake_ip(rng: np.random.Generator) -> str:
    """A well-formed, but not necessarily routable, synthetic IPv4 address."""
    return f"{rng.integers(1, 224)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def random_hex(rng: np.random.Generator, n_bytes: int = 8) -> str:
    """A random-looking hex id derived from the seeded Generator.

    Used everywhere a `uuid.uuid4()`-style id is needed (new device
    suffixes, attacker device ids, record ids): uuid4() draws from the OS
    entropy pool and is NOT reproducible even with every other source of
    randomness seeded, which would silently break the "same seed -> same
    output" determinism guarantee. This does the same job deterministically.
    """
    return rng.bytes(n_bytes).hex()


def weighted_choice(rng: np.random.Generator, weights: dict[str, float]):
    """Sample one key from a {label: weight} mapping, normalizing weights."""
    labels = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs = probs / probs.sum()
    return labels[int(rng.choice(len(labels), p=probs))]
