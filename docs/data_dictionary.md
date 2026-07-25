# Data Dictionary

> **Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.**

Covers the four tables written by `generator/run.py` under
`data/runs/<run.name>/`. Schemas are defined once, in code, at
`preprocessing/schema.py` -- this document is the human-readable mirror of
that file; if the two ever disagree, `preprocessing/schema.py` is correct
and this file needs updating.

---

## `users.parquet`

One row per synthetic employee.

| Field | Type | Nullable | Valid range / values | Notes |
|---|---|---|---|---|
| `user_id` | str | no | `U000001`-style, unique | |
| `department` | category | no | Engineering, Sales, Finance, HR, IT, Legal, Operations, Marketing, Executive, Customer Support | |
| `role` | str | no | department-specific job title | see `generator/constants.py:ROLES_BY_DEPARTMENT` |
| `privilege_level` | category | no | standard, elevated, admin, domain_admin | correlated with role keywords (Manager/Director/VP/Chief) |
| `manager_id` | str | yes | references another `user_id` | simplified two-tier hierarchy (see Phase 1 report, Deferrals) |
| `join_date` | datetime | no | | ~90% tenured (60d-5y before sim start), ~10% recent (<60d) |
| `employment_status` | category | no | active, on_leave, terminated | |
| `work_mode` | category | no | onsite, hybrid, remote | |
| `home_country` | str | no | ISO-ish country code | |
| `home_city` | str | no | | |
| `home_lat` / `home_lon` | float | no | [-90,90] / [-180,180] | jittered around a real reference city |
| `shift_start_hour` / `shift_end_hour` | int | no | [0,23] | day/evening/night patterns |
| `typical_devices` | int | no | [1,10] | size of this user's "known device" pool |

## `events.parquet`

One row per login attempt or resource access. **Chronologically sorted** on
disk. *(Phase 2a)* a successful login occasionally fans out into a short
chain of `resource_access` rows sharing its `session_id` -- see the
`event_type`/`session_id` notes below.

| Field | Type | Nullable | Valid range / values | Notes |
|---|---|---|---|---|
| `record_id` | str | no | random hex, unique | **bookkeeping only** -- see Leakage safety below |
| `insertion_order` | int | no | [0, N) | **bookkeeping only** |
| `generation_batch` | int | no | [0, num_generation_batches) | **bookkeeping only** |
| `user_id` | str | no | FK -> users | |
| `session_id` | str | no | `SESS-xxxxxxxxxxxx`-style | *(Phase 2a)* groups a login_attempt with every resource_access row it spawned |
| `timestamp` | datetime | no | | UTC |
| `event_type` | category | no | `login_attempt`, `resource_access` | *(Phase 2a adds `resource_access`)* |
| `auth_result` | category | yes | success, failure | null on `resource_access` rows (not an auth event) |
| `auth_method` | category | yes | password, sso, mfa_push, mfa_otp, api_token | null on `resource_access` rows |
| `mfa_used` | bool | no | | `False` sentinel (not null) on `resource_access` rows -- see schema.py's comment on why this stays non-nullable |
| `failure_reason` | category | yes | bad_password, mfa_denied, account_locked, unknown_user | only set when `auth_result == failure` |
| `device_id` | str | no | | see note on `is_known_device` below |
| `device_type` | category | no | desktop, laptop, mobile, server, unknown | |
| `os` | str | yes | | |
| `browser` | category | yes | Chrome, Firefox, Edge, Safari, Other | null for non-browser auth |
| `ip_address` | str | no | synthetic, well-formed IPv4 | |
| `asn` | str | no | synthetic `AS####` | |
| `isp_name` | str | no | synthetic | |
| `geo_country` / `geo_city` | str | no | | |
| `geo_lat` / `geo_lon` | float | no | [-90,90] / [-180,180] | |
| `network_type` | category | no | corporate_vpn, home_isp, public_wifi, mobile_carrier | |
| `session_duration_seconds` | float | yes | [0, 86400] | only set on a successful `login_attempt` row |
| `resource_accessed` | category | yes | email, vpn, crm, hr_system, code_repo, file_share, admin_console | set on every successful `login_attempt` row AND every `resource_access` row (never null when `auth_result` is success or absent-because-resource_access -- a null resource on a successful session would itself be a leakage-adjacent tell) |
| `action_count` | int | no | [0, 10000] | |
| `is_weekend` / `is_holiday` / `is_off_hours` | bool | no | | `is_off_hours` relative to the user's own shift; `resource_access` rows inherit their parent login's value |

**Note on `is_known_device`:** deliberately *not* a column here. The
generator tracks each user's "known device pool" internally to decide
realistic behavior, but does not expose a ground-truth
familiarity flag directly -- doing so would let a model use generation
metadata instead of learning device familiarity from history. Phase 2's
`feature_engineering/` will compute device recency/familiarity empirically
from the event history, the way a real detector would have to.

## `labels.parquet`

One row per event (row-aligned with `events.parquet` via `record_id`).
**Never merged into `events.parquet` itself** -- a model reading the events
table has no label sitting in its own feature space.

| Field | Type | Nullable | Valid range / values | Notes |
|---|---|---|---|---|
| `record_id` | str | no | FK -> events | |
| `is_attack` | bool | no | | |
| `attack_id` | str | yes | FK -> attacks | null for benign events |
| `attack_type` | category | yes | brute_force, impossible_travel, credential_misuse, lateral_movement, device_spoofing | null for benign events; the fixed, complete 5-type scope as of Phase 2a |
| `mitre_technique_ids` | str | yes | comma-joined technique IDs | null for benign events |

## `attacks.parquet`

One row per attack *campaign*: a brute-force burst, a single
impossible-travel event pair, a credential-misuse login+overreach pair, a
lateral-movement login+resource-chain, or a single device-spoofing event.

| Field | Type | Nullable | Valid range / values | Notes |
|---|---|---|---|---|
| `attack_id` | str | no | `ATK-000001`-style | |
| `attack_type` | category | no | brute_force, impossible_travel, credential_misuse, lateral_movement, device_spoofing | |
| `start_time` / `end_time` | datetime | no | | |
| `severity` | category | no | low, medium, high, critical | |
| `target_user_id` | str | no | FK -> users | single target per campaign (still true for all 5 types) |
| `affected_assets` | str | no | comma-joined | |
| `rationale` | str | no | plain English | *(Phase 2a)* the primary human-readable explanation, generated per-campaign from its own parameters -- e.g. "60 failed password attempts against a single account from an unfamiliar device within 2.3 minutes, ending in a successful login." |
| `mitre_technique_ids` | str | no | comma-joined | from `configs/mitre_mapping.yaml`; supplementary metadata, not a substitute for `rationale` |
| `mitre_tactic` | str | no | | |
| `num_events_generated` | int | no | [1, 100000] | |
| `parameters_json` | str | no | JSON-encoded dict | attack-specific parameters (attempt count, distance, implied speed, overreach resource, chain hops, spoofing variant, ...) |
| `mitre_mapping_version` | int | no | | `schema_version` from `configs/mitre_mapping.yaml` at generation time (2, as of Phase 2a) |

## `drift_log.csv`

*(Phase 3)* One row per scheduled drift event (not per affected user) --
the ground truth `drift_detection/adwin_detector.py` evaluates ADWIN
against. Empty (headers only) when `cfg.events.drift.enabled` is false.

| Field | Type | Nullable | Valid range / values | Notes |
|---|---|---|---|---|
| `day` | int | no | [0, num_days) | the CONFIGURED day; individual affected users roll out within `ramp_days` of this, staggered (see `generator/drift.py`) |
| `change_type` | category | no | remote_work_shift, schedule_shift | |
| `description` | str | no | plain English | from `configs/events/*.yaml`'s schedule entry |
| `affected_user_count` | int | no | >=1 | actual count selected, close to `affected_fraction * active_users` |

---

## Leakage safety (constraint #2)

`record_id`, `insertion_order`, and `generation_batch` are assigned via a
random permutation applied *after* all events (benign + attack) are
concatenated -- independent of `is_attack` and of the final,
timestamp-sorted row order. `evaluation/leakage_audit.py` trains a decision
stump on exactly these three fields and confirms it cannot separate
`is_attack` beyond a small epsilon (default 0.05 on ROC-AUC vs. chance).
Run it against any generated run:

```bash
python -m evaluation.leakage_audit data/runs/small_dev
```
