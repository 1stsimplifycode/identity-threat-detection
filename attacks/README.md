# attacks/

Attack generators and the injection orchestrator.

- **`base.py`** -- MITRE mapping loader and severity sampling shared by every
  attack type.
- **`brute_force.py`** -- burst of failed password attempts against one
  account from an unfamiliar device/IP, optional final compromise.
- **`impossible_travel.py`** -- a second successful login shortly after a
  real prior one, at a distance/time implying impossible travel speed.
- **`credential_misuse.py`** -- a single successful login (no burst) followed
  by a resource access outside both the account's department and privilege
  scope.
- **`lateral_movement.py`** -- a resource-access chain crossing multiple
  departments' resource sets in one fast session, trending toward
  higher-sensitivity systems.
- **`device_spoofing.py`** -- a known device reappearing with a different
  type/OS signature, or a device normally scoped to one user being used by a
  different user shortly after.
- **`injector.py`** -- alternates across `cfg.attacks.enabled_attacks` until
  the configured `imbalance_ratio` is reached; returns attack events (with
  temporary `_tmp_attack_id`/`_tmp_attack_type` tags) and attack metadata.

## Leakage safety (constraint #2)

Every attack generator is written so that no field combination, ID range, or
timestamp quantization is attack-exclusive by construction:
- Attack start times are drawn from across the *entire* simulation window,
  not restricted to off-hours.
- New devices/IPs and unfamiliar cities also occur in benign traffic (see
  `generator/events.py`'s `new_device_probability` / `travel_event_probability`),
  so neither is a trivial giveaway on its own.
- The `_tmp_attack_id`/`_tmp_attack_type` tags never reach the events table
  written to disk -- `generator/run.py` moves them into the separate
  `labels` table and drops them from `events`.
- `record_id`, `insertion_order`, and `generation_batch` (the events table's
  bookkeeping fields) are assigned by `generator/run.py` via a random
  permutation applied *after* all events (benign + attack) are concatenated,
  independent of label and of the final timestamp-sorted row order.

`evaluation/leakage_audit.py` verifies this empirically on every generated
run.

## Phase status

Phase 2a: all 5 attack types complete (brute_force, impossible_travel,
credential_misuse, lateral_movement, device_spoofing) -- this is the fixed,
complete attack scope per the authoritative problem statement; no further
attack types are planned.
