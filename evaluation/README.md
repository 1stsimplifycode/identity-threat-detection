# evaluation/

- **`leakage_audit.py`** (Phase 1) -- trains a decision stump on the events
  table's metadata-only fields (`record_id`, `insertion_order`,
  `generation_batch`) across 15 repeated train/test holdouts, and confirms
  the *mean* ROC-AUC across those repeats cannot separate `is_attack` beyond
  a small epsilon (default 0.05 vs. chance). Repeated holdouts, not a single
  split, because a single split's AUC estimate is noisy enough at low
  imbalance ratios to occasionally cross epsilon on an honest, leak-free
  dataset -- see `docs/phase_1_report.md` for the investigation that found
  this. Run standalone:

  ```bash
  python -m evaluation.leakage_audit data/runs/small_dev
  ```

  Exits non-zero and prints a failure report if the audit fails -- this is
  meant to be wired into CI once a CI config exists.

- **`chronological_split.py`** (Phase 2b) -- time-respecting train/test
  split (constraint #1): trains on an earlier window, tests on a later one,
  with an explicit, loggable split timestamp. `apply_split()` filters any
  `record_id`-keyed table (features, labels) to one side without
  re-deriving the split.
- **`report.py` / `run_evaluation.py`** (Phase 2b) -- computes and renders
  the six-criteria comparison table (Precision/Recall/F1/ROC-AUC/PR-AUC,
  false-positives/day, per-attack-type recall, scalability), the rule
  baseline included in every row per constraint #3. Run standalone:

  ```bash
  python -m evaluation.run_evaluation --config-name small_dev
  ```

  Writes `docs/phase_2b_evaluation_report.md`.

## Phase status

Phase 2b: leakage audit, chronological split, and the six-criteria report
for the rule baseline + Isolation Forest. Phase 3 extends the same report
to the full model set (XGBoost, River, fine-tuned HF model), adds MCC and
MTTD, and ties drift detection to the ground-truth drift log.
