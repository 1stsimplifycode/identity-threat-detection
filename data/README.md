# data/

Generated output lives here, under `data/runs/<run.name>/` (e.g.
`data/runs/small_dev/`), one subfolder per Hydra run:

```
data/runs/<run.name>/
├── users.parquet
├── events.parquet
├── labels.parquet
├── attacks.parquet
├── drift_log.csv          # empty stub until Phase 2's drift simulator exists
├── run_metadata.json      # config hash, seed, counts, disclaimer
├── config_resolved.yaml   # fully-resolved Hydra config for this run
└── DISCLAIMER.txt
```

**Nothing under `data/runs/` is committed to git** (see `.gitignore`) --
runs are reproducible from their logged seed + config hash, not stored as
repo artifacts.

> Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.
