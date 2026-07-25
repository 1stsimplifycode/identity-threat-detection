# Phase 2a Report: Attack Scope Completion

> **Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.**

## What was built

- **Schema expansion** (`preprocessing/schema.py`, `preprocessing/constants.py`):
  `event_type` now includes `resource_access` alongside `login_attempt`;
  every event carries a `session_id`; `auth_result`/`auth_method` are now
  nullable (not applicable to `resource_access` rows); `ATTACK_TYPES_PHASE1`
  was renamed to `ATTACK_TYPES` and now lists the fixed, complete 5-type
  scope; `ATTACKS_SCHEMA` gained a required `rationale` field.
- **`generator/events.py`**: successful logins now occasionally (config
  `resource_chain_probability`) fan out into a 1-4 hop chain of
  `resource_access` events sharing the login's `session_id`; occasional
  cross-department resource access (`cross_department_resource_probability`,
  still within the user's privilege scope); a small "shared kiosk device"
  pattern for Operations/Customer Support (`shared_kiosk_device_probability`).
- **`generator/constants.py`**: `MIN_PRIVILEGE_BY_RESOURCE`,
  `RESOURCE_VALUE_TIER`, `SHARED_DEVICE_DEPARTMENTS` -- new reference
  vocabulary the three new attack generators key off.
- **`configs/mitre_mapping.yaml`** (bumped to `schema_version: 2`):
  added `credential_misuse` (TA0004, T1078/T1078.002), `lateral_movement`
  (TA0008, T1021), `device_spoofing` (TA0005, T1036 -- explicitly flagged
  in its own `notes:` as an approximate best-effort tag, since Enterprise
  ATT&CK has no clean 1:1 "device fingerprint spoofing" technique).
- **Three new attack generators**, same `(list[dict], dict)` contract as
  the existing two:
  - `attacks/credential_misuse.py` -- a single successful login (no burst)
    followed by a resource access outside BOTH the account's department AND
    its privilege scope.
  - `attacks/lateral_movement.py` -- a resource-access chain crossing
    multiple departments' resource sets in one fast session, trending
    toward higher-sensitivity systems.
  - `attacks/device_spoofing.py` -- two variants: a known device
    reappearing with a different type/OS signature, or a device normally
    scoped to one user being used by a different user shortly after.
- **`attacks/injector.py`**: wired in all three, plus a device index
  (`build_device_index`) built once from benign traffic for
  `device_spoofing`.
- **`brute_force.py`/`impossible_travel.py` backfilled** with `session_id`
  and `rationale` to satisfy the new required schema fields.
- **5 new/updated tests** in `tests/test_attacks.py`, plus a new
  `tests/test_schema.py` test for the session/resource-access schema. All
  15 tests pass.
- **Dropped `insider_threat`** from scope entirely -- the authoritative
  problem statement's fixed attack list is exactly the 5 above.

## Key design decisions

**Every new attack's tell is a conjunction or a pattern, never a single
field** -- matching Phase 1's existing discipline. `credential_misuse`
requires BOTH outside-department AND outside-privilege (either alone is
something benign traffic now also does, at a low rate).
`lateral_movement`'s tell is the *chain* (multiple departments, fast hops,
escalating value), not any single resource touch -- ordinary cross-
department access and multi-resource sessions already exist in benign
traffic in a milder form. `device_spoofing`'s cross-user variant explicitly
excludes the already-legitimate shared-kiosk-device pattern, so ">1 user on
a device" isn't attack-exclusive either.

**A real bug caught and fixed during this phase**: the first draft of all
three new generators left `resource_accessed` null on their successful
login rows. But no *benign* successful login is ever generated with a null
resource -- `_pick_resource()` in `generator/events.py` always assigns one
on success. A null resource on an otherwise-successful row would have been
an accidental, trivial tell ("success + null resource → attack") sitting
right next to the intended signal. Fixed by giving each new attack's login
row a normal, department-appropriate resource, and keeping the actual
anomaly isolated to a separate `resource_access` row (credential_misuse,
lateral_movement) or the device signature itself (device_spoofing). This
is exactly the kind of leakage class constraint #2 is aimed at, just
introduced by new code rather than by the original bookkeeping-field
design — a useful reminder that the leakage discipline has to be
re-applied at every new attack type, not just baked in once.

**`lateral_movement` anchors on real benign history**, the same pattern
`impossible_travel.py` already used: the entry login reuses an actual
prior successful event's device/geo/network, so the login itself is
unremarkable and the entire signal is the resource chain that follows.

**Rationale is now the primary human-readable field; MITRE stays
supplementary.** Per the problem statement's "no MITRE ontology sprawl"
instruction, every attack campaign now generates a plain-English
`rationale` string from its own concrete parameters (not a templated MITRE
name), e.g.: *"40 failed password attempts against a single account from
an unfamiliar device within 2.0 minutes, with no successful login."*

## Verification results

```
python -m pytest tests/ -v          -> 15 passed
python -m generator.run --config-name small_dev
  -> Wrote 57,050 events (312 attack-labeled, ratio=0.5469%) in 27s
python -m evaluation.leakage_audit data/runs/small_dev
  -> PASSED, mean ROC-AUC 0.5007 (std 0.0025) across 15 repeated holdouts
```

Attack campaigns generated this run: brute_force (7), impossible_travel
(6), credential_misuse (6), lateral_movement (6), device_spoofing (6).

Sample rationale, one per type:

- **brute_force**: "40 failed password attempts against a single account
  from an unfamiliar device within 2.0 minutes, with no successful login."
- **impossible_travel**: "Successful login from GB occurred 83.5 minutes
  after this account's previous successful login from CA, 7571 km away --
  an implied travel speed of 5437 km/h, physically impossible for
  legitimate travel."
- **credential_misuse**: "standard-privilege account in Operations
  successfully accessed admin_console, a resource outside its department
  and above its normal privilege scope, from a device not previously
  associated with this account."
- **lateral_movement**: "Account in Operations accessed 4 resources
  outside its department (email, crm, file_share, hr_system) within a
  single fast session, trending toward higher-sensitivity systems --
  inconsistent with its normal single-department access footprint."
- **device_spoofing**: "Device DEV-U001830-01, previously associated only
  with U001830, was used by a different account (U003440) within 137
  minutes of its established owner's last use -- consistent with a
  spoofed or cloned device identifier."

## Known limitations / deliberate deferrals

- **`RESOURCE_VALUE_TIER` and the department-crossing heuristic in
  `lateral_movement.py` are generation-time heuristics, not the real
  detection signal.** Phase 2b's actual `access_chain_distance` graph
  feature, computed from real event frequency via NetworkX, is what a
  detector will actually learn from -- this phase only needed the
  generator to produce a chain that a proper graph feature *should* flag,
  not to precompute the graph itself.
- **Single target user per campaign**, still true for all 5 types (matches
  Phase 1's existing simplification).
- **`device_spoofing`'s MITRE tag (T1036, Masquerading) is a best-effort
  approximation**, explicitly flagged as such in the mapping file itself --
  there is no clean 1:1 Enterprise ATT&CK technique for device-fingerprint
  spoofing specifically.
- **No cold-start tagging yet** (Phase 2b/3 per the updated plan).
- **Terminated users still generate no post-termination activity tail**
  (unchanged from Phase 1).

## Synthetic-data disclaimer

Every generated run still carries `DISCLAIMER.txt`, and `run_metadata.json`
includes the same text verbatim:

> Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.
