# Identity Threat Detection Benchmark Pipeline

> **Synthetic data. Not derived from or validated against real organizational logs.
> For benchmarking detection methods only.**
>
> Everything in this repository — generated logs, reports, dashboards, model
> cards — exists to compare detection *methods* against each other on
> internally-consistent synthetic traffic. "Realistic" here means
> **statistically plausible**, not **validated against real enterprise logs**
> (we have none). No artifact in this repo should ever claim real-world
> generalization.

A research-grade, reproducible pipeline that:

1. **Generates** synthetic enterprise identity/authentication logs — a user
   population plus login events with realistic-shaped normal behavior and
   injected, MITRE ATT&CK-mapped attacks.
2. **Detects** identity-based threats on that data, benchmarking methods from
   a dumb static-threshold baseline up through supervised, unsupervised, and
   online learners, under strictly chronological evaluation.

## Status

Built in phases, each reviewed before the next starts. See `docs/phase_1_report.md`
and `docs/phase_2a_report.md` for what exists today and what's deliberately deferred.

- [x] **Phase 1** — scaffolding, small-scale generator (2 attack types), leakage audit, schema validation
- [x] **Phase 2a** — full 5-type attack coverage (brute_force, impossible_travel,
      credential_misuse, lateral_movement, device_spoofing), `resource_access`/
      `session_id` schema expansion, per-campaign plain-English `rationale`
- [x] **Phase 2b** — behavioral (3a) + graph-derived (3b) feature engineering
      (dual-mode `update()`/`compute_batch()`), rule baseline + Isolation
      Forest, chronological split, six-criteria evaluation report, and
      pytest-verified graph-feature spikes on `lateral_movement`/`device_spoofing`
- [x] **Phase 3** — concept-drift simulation + ground-truth `drift_log.csv`,
      cold-start department priors, XGBoost multi-class classifier (3
      imbalance-handling conditions: none/class_weight/SMOTE), River
      Adaptive Random Forest online learner (true streaming, reusing the
      dual-mode feature interface), ADWIN drift detection evaluated
      against ground truth, fine-tuned HF `bert-tiny` detector -- full
      model set in one six-criteria comparison table with MCC and MTTD.
      **Best model: XGBoost+SMOTE** (F1 0.646, ROC-AUC 0.993, PR-AUC 0.880)
- [x] **Phase 4** — offline exact SHAP (`shap.TreeExplainer` on `xgboost_smote`)
      + a lightweight, explicitly-labeled streaming approximation, both shown
      side-by-side in a Streamlit analyst dashboard (live filters, a
      detection-threshold slider with a live precision/recall curve, a
      sortable flagged-events table with click-to-expand SHAP/graph-feature/
      raw-field detail, and per-user trend drill-down), an enterprise dark
      "security-ops" visual theme, and a tested `render.yaml` +
      `docs/deployment.md` for a single free Render Web Service
- [ ] Scale-up — re-run the *same, already-tested* generator config at full scale

## Non-negotiable design constraints (apply to every phase)

- **Chronological splits only.** No k-fold CV on raw temporal records anywhere.
- **No label-leakage from generation artifacts.** `record_id`, `insertion_order`,
  and `generation_batch` are deliberately randomized independent of label and
  time; `evaluation/leakage_audit.py` proves this on every generated dataset.
- **Dumb baseline always reported.** Every model comparison table includes the
  static-threshold baseline.
- **Explainability is a documented tradeoff.** Exact SHAP runs offline/batch
  only; streaming inference uses lightweight precomputed importances (Phase 4).
- **MITRE mapping is versioned config**, defined in `configs/mitre_mapping.yaml`
  before any attack data is generated.
- **Config-driven.** Every ratio, window size, imbalance level, drift schedule,
  and user count lives in Hydra configs under `configs/`.
- **Reproducible.** Fixed seeds (`generator/seeding.py`), config hash logged
  with every run.

## Quick start (Phase 1)

```bash
# from the project root, with .venv activated
python -m generator.run --config-name small_dev
python -m pytest tests/ -v
```

This generates ~5,000 users / ~2 weeks / ~50k events under
`data/runs/small_dev/` in well under 5 minutes on a laptop.

## Repository layout

Each top-level folder has its own `README.md` explaining its purpose and
current-phase scope. See `docs/data_dictionary.md` for the full field-level
schema of every generated table, and `docs/phase_1_report.md` for design
decisions and deferrals.
