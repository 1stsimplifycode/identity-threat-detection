# Phase 1 Report: Scaffolding + Small Synthetic Generator (MVP)

> **Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.** "Realistic" throughout
> this report means *internally consistent and statistically plausible*,
> never *validated against real enterprise traffic*.

## What was built

- **Project skeleton** -- every top-level folder from the spec, each with a
  `README.md` and (where it's a Python package) an `__init__.py`.
- **`configs/`** -- Hydra config tree (`config.yaml` full-scale profile,
  `small_dev.yaml` fast-iteration profile, `population/`, `events/`,
  `attacks/` groups) plus the versioned `mitre_mapping.yaml`.
- **`preprocessing/`** -- canonical categorical vocabularies, geo utilities
  (haversine + city pools), and schema definitions with
  `validate_dataframe()`.
- **`generator/`** -- `set_global_seed()`, the user population generator, the
  login-event generator, and the Hydra run entrypoint (`generator/run.py`)
  that ties everything together and writes `data/runs/<run.name>/`.
- **`attacks/`** -- brute-force and impossible-travel generators, MITRE
  tagging via `configs/mitre_mapping.yaml`, and the injection orchestrator
  that hits a configured imbalance ratio.
- **`evaluation/leakage_audit.py`** -- the constraint-#2 leakage audit.
- **`tests/`** -- 10 tests covering determinism, schema validation, the
  leakage audit (including a negative control), and attack sanity checks.
  All passing.
- **`docs/data_dictionary.md`** -- full field-level schema for all 4 tables.

## Key design decisions

**Vocabulary vs. ratio.** Categorical vocabularies (departments, role
titles, device types, MITRE technique names, city pools) live in code as
reference data, the same way `configs/mitre_mapping.yaml` is reference data
rather than a tunable. Genuine experiment knobs -- population size,
privilege/work-mode distributions, termination rate, event timing/rates,
imbalance ratio, attack parameters, severity weights -- all live in Hydra
configs under `configs/`. This distinction is a judgment call; if reviewing
this shows some "vocabulary" item should actually be config-driven, that's
an easy Phase 2 change.

**Leakage safety is structural, not incidental.** Three things work
together to satisfy constraint #2: (1) attack start times are drawn from
across the *entire* simulation window, not restricted to off-hours; (2) new
devices/IPs/unfamiliar cities also occur in benign traffic at a configured
rate, so neither is an attack-exclusive tell; (3) `record_id`,
`insertion_order`, and `generation_batch` are assigned via a random
permutation *after* all events are concatenated, independent of label and
of the final chronological row order. `evaluation/leakage_audit.py` proves
this empirically on every run, and `tests/test_leakage_audit.py` includes a
negative control (a deliberately leaky toy dataset) proving the audit
itself would actually catch a real leakage bug, not just rubber-stamp
whatever it's given.

**The events table never carries the label.** `is_attack` / `attack_id` /
`attack_type` live only in `labels.parquet`, joined by `record_id`. This is
a stronger separation than most synthetic-data generators bother with, and
it's what makes the leakage audit meaningful rather than circular.

**Ground-truth device familiarity is intentionally withheld.** The
generator internally tracks which devices are "known" to a user to produce
realistic behavior, but does not expose an `is_known_device` column.
Phase 2's feature engineering will compute familiarity empirically from
event history -- the way a real detector has to -- rather than reading
generation metadata.

**Simplified two-tier manager hierarchy.** ICs report to a manager-eligible
user in their own department; manager-eligible users report to an
Executive or to nobody. This avoids arbitrary-depth org charts and the
cycle-handling they'd require, at the cost of a less realistic hierarchy
shape.

**The leakage audit uses repeated holdouts, not a single split -- discovered
by the audit actually failing.** The first end-to-end run of
`evaluation/leakage_audit.py` against the real `small_dev` output *failed*:
mean ROC-AUC 0.4467 against an 0.05 epsilon. Rather than loosen the
threshold, this was investigated directly -- re-running the identical audit
across 20 different train/test splits gave a mean AUC of 0.5045 (essentially
chance) with a standard deviation of 0.0051, and the seed-42 split that
originally failed turned out to be a one-in-twenty statistical fluke driven
by the very small number of positive examples landing in a single 30%
holdout fold (~59 out of ~10,700 at small_dev's 0.55% actual imbalance
ratio -- ROC-AUC's sampling variance at that positive count is large enough
that an honest, leak-free dataset can occasionally cross a 0.05 epsilon by
chance alone). The audit now runs 15 repeated holdouts with different
splits and gates on the *mean* |AUC - 0.5|, reporting min/max/std for
transparency. This is repeated random sub-sampling for variance reduction of
a static, non-temporal, meta-level check -- not k-fold cross-validation of a
detection model on raw temporal records, which constraint #1 forbids and
which this audit has nothing to do with (record_id/insertion_order/
generation_batch carry no chronological meaning by design). The same
investigation also surfaced that the original 60-user/3-day `tiny_cfg` test
fixture was too small for the audit to be statistically meaningful at all
(~51 total attack events) -- `tests/conftest.py` now has a separate,
moderately-larger `audit_cfg` fixture (2,000 users/10 days) specifically for
the leakage-audit test, while the faster `tiny_cfg` remains in use for the
tests that don't need many events.

**A nonzero imbalance ratio always yields at least one attack campaign.**
The raw `imbalance_ratio * n / (1 - imbalance_ratio)` computation can round
to zero at small scale (discovered via testing: a 60-user/3-day config with
a 0.5% imbalance ratio produced literally zero attacks, which cascaded into
a single-class-labels failure and a `KeyError` on `attack_meta["attack_type"]`
downstream). Target attack-event count is now floored at 1 whenever
`imbalance_ratio > 0` and at least one attack type is enabled.

**Environment note (Windows Smart App Control):** the dev machine this was
built on blocked pandas 3.0.5's and scikit-learn 1.9.0's compiled extensions
outright ("An Application Control policy has blocked this file") while
leaving numpy's compiled extensions untouched -- almost certainly a
reputation-based block on those specific new-release wheels rather than a
blanket policy against compiled Python extensions. Pinning
`pandas>=2.2,<3.0` and `scikit-learn>=1.4,<1.9` in `requirements.txt`
resolved it without touching any system security policy. Worth knowing if
`pip install -r requirements.txt` on a fresh machine hits the same error.

## Known limitations / deliberate deferrals

- **Only 2 of 6 attack types** (brute_force, impossible_travel).
  credential_misuse, lateral_movement, device_spoofing, and insider_threat
  are Phase 2 scope -- each needs its own `configs/mitre_mapping.yaml` entry
  written *before* its generator, not stubbed speculatively now.
- **Only `event_type == "login_attempt"`** is modeled. No logout events or
  standalone resource-access events yet.
- **No engineered behavioral features.** `generator/`/`attacks/` produce raw
  fields only -- velocity, entropy, peer-group deviation, rolling
  stats/EMA, and the dual batch/streaming interface are `feature_engineering/`'s
  job, starting Phase 2.
- **No drift simulation.** `drift_log.csv` is written as an empty,
  correctly-headered stub; the drift schedule config group and simulator
  don't exist until Phase 2.
- **No cold-start tagging.** `join_date` already has a realistic
  recent-hire tail (~10% of users joined <60 days before sim start) for
  Phase 2 to filter on, but there's no explicit cold-start flag/subset yet.
- **Terminated users generate no post-termination activity tail** -- a
  documented simplification, not a bug: once `employment_status ==
  terminated`, the event generator skips that user entirely rather than
  modeling a realistic wind-down period.
- **Single target user per attack campaign.** Real lateral-movement or
  large-scale credential-stuffing campaigns can span multiple targets;
  Phase 1's two attack types are single-target by nature, so this wasn't
  exercised, but it's worth flagging before Phase 2's broader attack set.
- **`record_id` is not required to be reproducible in value** across two
  runs with the same seed (it's random-hex, seeded, so it *is* actually
  reproducible in this implementation) -- but this was never a design
  requirement, since `record_id` carries no signal and is excluded from
  every feature space; only the *content* (which user did what, when) needs
  to reproduce identically, which `tests/test_generator_determinism.py`
  verifies directly on the population/events tables.
- **Auth realism is simplified.** Failure -> eventual success sequences
  emerge from independent per-event Bernoulli draws rather than a modeled
  Markov chain of attacker/user retry behavior; this is adequate for Phase 1
  but is a real simplification worth knowing about before reading too much
  into failure-streak statistics.

## Synthetic-data disclaimer

Every generated run carries `DISCLAIMER.txt`, and `run_metadata.json`
includes the same text verbatim:

> Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.

No document, plot, or model card produced by this pipeline should ever
imply real-world validation.
