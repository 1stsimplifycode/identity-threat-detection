# generator/

Synthetic user population and login-event generation.

- **`seeding.py`** -- `set_global_seed()`, the single reproducibility entrypoint.
- **`constants.py`** -- generation-only vocabulary (role lists, geo/device
  helpers) that isn't a tunable ratio (those live in `configs/`).
- **`population.py`** -- `generate_population()`: users with department,
  role, privilege level, a simplified two-tier manager hierarchy, join date,
  employment status, home location, and shift timing.
- **`events.py`** -- `generate_login_events()`: benign login/auth events
  respecting shift patterns, weekends, holidays, and remote-vs-onsite work
  mode. Drift-aware (Phase 3): accepts an optional `ResolvedDrift` and
  applies each user's effective (possibly overridden) attributes per day.
- **`drift.py`** (Phase 3) -- `resolve_drift_schedule()`: resolves
  `cfg.events.drift`'s config-driven schedule into per-user rollout days
  and attribute overrides, gradually staggered within `ramp_days`, plus the
  ground-truth `drift_log.csv` row(s) `drift_detection/` evaluates ADWIN
  against.
- **`run.py`** -- Hydra entrypoint tying population + drift + events +
  attack injection together, validating schemas, and writing
  `data/runs/<run.name>/{users,events,labels,attacks}.parquet`,
  `drift_log.csv`, `run_metadata.json`, `config_resolved.yaml`, and
  `DISCLAIMER.txt`.

## Run it

```bash
python -m generator.run --config-name small_dev
```

## Scope notes

- `event_type` covers `login_attempt` and (as of Phase 2a) `resource_access`
  -- a successful login occasionally fans out into a short resource-access
  chain sharing its `session_id`. No logout events.
- Terminated users generate no post-termination activity tail (documented
  simplification, not a bug).
- No engineered behavioral features here (velocity, entropy, peer-group
  deviation, ...) -- those are `feature_engineering/`'s job, starting Phase 2b.
  This package only produces raw fields.

See `docs/phase_1_report.md` and `docs/phase_2a_report.md` for the full list
of deferrals and design decisions.
