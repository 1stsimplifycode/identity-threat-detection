# tests/

Run with:

```bash
python -m pytest tests/ -v
```

- **`conftest.py`** -- `tiny_cfg` fixture (60 users, 3 days) so the whole
  suite runs in seconds, independent of `configs/small_dev.yaml`'s own
  5-minute dev-loop target.
- **`test_generator_determinism.py`** -- same seed -> identical output;
  different seed -> different output.
- **`test_schema.py`** -- every generated table satisfies its schema; events
  and labels stay row-aligned; the events table never carries the label
  directly; events are chronologically sorted.
- **`test_leakage_audit.py`** -- the real generated dataset passes the
  leakage audit, **and** a deliberately leaky synthetic dataset is correctly
  flagged as failing (negative control -- proves the audit has teeth).
- **`test_attacks.py`** -- MITRE tagging, time ordering, severity vocabulary,
  and attack-type-specific parameter bounds (brute-force attempt counts,
  impossible-travel implied speed), plus all 5 attack types' sanity checks
  (Phase 2a).
- **`test_graph_features.py`** (Phase 2b) -- explicit, non-eyeballed proof
  that `access_chain_distance` spikes on `lateral_movement` and
  `is_new_edge` spikes on `device_spoofing`'s `cross_user_reuse` variant,
  plus a dual-mode determinism check.
- **`test_drift.py`** (Phase 3) -- the ground-truth `drift_log.csv` matches
  the configured schedule, and drifted users actually show the changed
  behavior in generated events (not just a metadata entry with no effect).
- **`test_cold_start.py`** (Phase 3) -- cold-start priors change only
  cold-start rows' eligible columns; established rows are untouched.
- **`test_xgboost_classifier.py`** (Phase 3) -- all 3 imbalance-handling
  conditions train and score without error, including on classes with
  very few train examples.
- **`test_river_online.py`** (Phase 3) -- the streaming loop scores every
  event with well-formed output.
- **`test_adwin_detector.py`** (Phase 3) -- ADWIN actually detects the
  configured drift event on real generated data, at a reasonable lag.

Fixtures beyond `tiny_cfg`: `audit_cfg` (2,000 users/10 days -- enough
attack events for statistically meaningful checks) and
`graph_verification_cfg` (audit_cfg scale + a higher imbalance_ratio, so
every attack type gets a fair sample within one run despite the injector's
round-robin dispatch).
