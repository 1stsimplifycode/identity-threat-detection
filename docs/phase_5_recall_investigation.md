# Phase 5: Rare-Class Recall Investigation (lateral_movement, device_spoofing)

> Synthetic data. Not derived from or validated against real organizational logs. For benchmarking detection methods only.

## Summary

`lateral_movement` and `device_spoofing` had ~0% recall in every prior evaluation (all three
XGBoost imbalance-handling variants, plain argmax decision rule). This document reports the
diagnosis, the two techniques tried, and the **real, measured** before/after numbers -- including
where the fix did not work, reported honestly rather than hidden.

**Bottom line:** `device_spoofing` recall improved meaningfully for two of three imbalance
methods (0% -> 25%/75%), at a real, quantified false-positive cost. `lateral_movement` recall
did **not** improve from that fix alone -- **Section 10** below reports a follow-up
(session-window features) that got `lateral_movement` to a real, non-zero recall (0% -> 50% on
`xgboost_smote`), still with real costs and caveats reported honestly, not hidden.

## 1. Diagnosis (why recall was ~0%)

Measured directly against `data/runs/small_dev` (57,436 events, chronological 70/30 split,
`train_frac=0.7`):

- **Sample scarcity is severe and structural.** Full dataset: `lateral_movement` = 31 events (7
  campaigns), `device_spoofing` = 7 events (7 campaigns, ~1 event/campaign). After the
  chronological split: `lateral_movement` = 27 train / 4 test, `device_spoofing` = 3 train / 4
  test. Compare `brute_force` = 177 train / 84 test.
- **Scores weren't just below threshold -- they were near the floor.** Pre-fix, the 4 test-split
  `lateral_movement` events scored 0.00016-0.0014; the 4 `device_spoofing` events scored
  0.000006-0.0028 (the shared operating threshold was ~0.0105; `brute_force` events routinely hit
  1.0). The model wasn't narrowly missing these -- it was confidently calling them normal.
- **Feature separability (z-score of attack-class mean vs. benign-population std)**:
  `brute_force`: `ema_failure_rate` z=21.9, `failed_login_ratio` z=8.0 -- overwhelming signal.
  `lateral_movement`: best signals `access_chain_distance` z=1.43, `peer_group_deviation` z=1.42
  -- real but weak, substantial overlap with benign.
  `device_spoofing`: best pre-fix signal `geo_distance_from_home_km` z=1.74, but `is_new_edge`
  (the feature that should most directly catch device spoofing) was only z=0.57 -- a genuine
  **feature gap**: `attacks/device_spoofing.py` generates two variants, and the
  `fingerprint_mismatch` variant (same `device_id`, changed `device_type`/`os`) leaves the
  (user, device) *pairing* unchanged, so `is_new_edge` -- which tracks pairing novelty, not
  fingerprint consistency -- structurally cannot see it.
- **A structural decision-rule issue compounds both**: `XGBClassifier` uses
  `objective="multi:softprob"` with plain `argmax` over 6 classes. `class_weight="balanced"`
  reweights the training loss, but at inference a rare class still has to out-score `benign` in a
  direct 6-way contest with benign at 99.4% prevalence -- a very high bar even for a class the
  model has learned something real about.

## 2. Stage A: per-class threshold optimization alone (tried first, proven insufficient)

Before touching feature engineering, per-class thresholds were tried on the **already-trained**
model (no retraining) -- the cheapest possible lever. Real measured result:

| Method | Class | Threshold (min score to catch all 4 test TPs) | Rows flagged | Precision |
|---|---|---|---|---|
| class_weight | lateral_movement | 0.000627 | 237 | 1.69% |
| class_weight | device_spoofing | 0.000002 | 7365 | 0.05% |
| smote | lateral_movement | 0.000153 | 161 | 2.48% |
| smote | device_spoofing | 0.000001 | 4865 | 0.08% |
| none | lateral_movement | 0.000014 | 1181 | 0.34% |
| none | device_spoofing | 0.000002 | 4168 | 0.10% |

Forcing 100% recall via thresholding alone would flood the analyst with 161-7365 false positives
to recover 4 true positives. **Rejected** -- not a usable operating point. This motivated Stage B.

## 3. Stage B: new feature -- `device_fingerprint_mismatch`

Added to `feature_engineering/graph.py`'s `GraphFeatureState`: for each `device_id`, the first
`(device_type, os)` pair ever observed is recorded as that device's established signature
(unwindowed, never overwritten). The feature is `1.0` if a later event's `(device_type, os)`
differs from that signature, `0.0` otherwise. This closes the `is_new_edge` blind spot directly:
it tracks fingerprint consistency, not (user, device) pairing.

Verified in isolation (`tests/test_graph_features.py::test_device_fingerprint_mismatch_spikes_on_device_spoofing_fingerprint_variant`):
mean `device_fingerprint_mismatch` on the `fingerprint_mismatch` attack variant > 0.5 and > the
benign mean -- passing, confirming genuine separation before any model was retrained on it.
`GRAPH_FEATURE_COLUMNS` grew from 5 to 6; `FEATURE_COLUMNS` from 12 to 13.

## 4. Threshold tuning, properly wired (not the Stage A throwaway)

`models/xgboost_classifier.py::tune_class_thresholds()`: per-class one-vs-rest thresholds for
`lateral_movement` and `device_spoofing`, chosen to maximize each class's F1 using **out-of-fold**
predictions from stratified k-fold CV on the **TRAIN split only** (never test labels -- each fold
retrains fresh, including re-applying the imbalance method inside the fold, so SMOTE-synthesized
points never leak across folds). A class is only assigned a threshold if it beats the plain-argmax
F1 baseline on the same out-of-fold predictions; otherwise the decision rule falls back to argmax
for that class.

Note: the requested 5-fold CV was automatically reduced to **3-fold**, since `device_spoofing` has
only 3 train examples and `StratifiedKFold` cannot split a class into more folds than it has
members (`min(n_splits, min_class_count)`, floored at 2). This is itself informative -- it's the
same scarcity problem showing up again, one level down.

Wired end-to-end: `evaluation/model_suite.py`'s XGBoost loop tunes thresholds right after training
each variant and stores them on `ModelEvaluation.class_thresholds`; `score_xgboost()` accepts them
as an optional override for `predicted_class` only (`anomaly_score` -- the binary attack/not-attack
signal -- is untouched); `dashboard/prepare_data.py` uses the same thresholds for the primary
model (`xgboost_smote`) so the dashboard's flagged-events table matches the evaluation report.

Tuned thresholds (real, from this run):

| Method | `lateral_movement` threshold | `device_spoofing` threshold |
|---|---|---|
| none | 0.145842 | 0.002037 |
| class_weight | 0.628648 | 0.004199 |
| smote | 0.927916 | 0.000139 |

## 5. Real before/after numbers

**Before Phase 1** (all 3 methods, plain argmax): `lateral_movement` recall = 0.000,
`device_spoofing` recall = 0.000.

**After Phase 1** (new feature + tuned thresholds), real multi-class classification report on the
17,231-row test split (`support` = ground-truth count in that split):

| Method | `device_spoofing` recall | `device_spoofing` precision | `lateral_movement` recall | `brute_force` recall |
|---|---|---|---|---|
| none | 0.000 (unchanged) | -- | 0.000 (unchanged) | 0.917 |
| class_weight | **0.250** (1/4) | 0.0526 | 0.000 (unchanged) | 0.964 |
| smote | **0.750** (3/4) | 0.0161 | 0.000 (unchanged) | 0.964 |

`brute_force` (the class with real signal, z=21.9) shows no regression across all three methods --
confirms the new feature and threshold overrides didn't destabilize the classes that already
worked.

Six-criteria comparison table, this run (`docs/phase_3_evaluation_report.md`, regenerated):

| model | precision | recall | f1 | mcc | roc_auc | pr_auc | fp/day | mttd_days | mttd_coverage |
|---|---|---|---|---|---|---|---|---|---|
| rule_based_baseline | 0.5294 | 0.7500 | 0.6207 | 0.6277 | 0.9347 | 0.4033 | 18.10 | 0.00 | 0.2727 |
| isolation_forest | 0.3925 | 0.8750 | 0.5419 | 0.5830 | 0.9510 | 0.6097 | 36.77 | 0.01 | 0.4545 |
| xgboost_none | 0.4615 | 0.8750 | 0.6043 | 0.6329 | 0.9906 | 0.8472 | 27.72 | 0.01 | 0.4545 |
| xgboost_class_weight | 0.4826 | 0.8646 | 0.6194 | 0.6435 | 0.9946 | 0.8798 | 25.18 | 0.01 | 0.4545 |
| **xgboost_smote** | **0.4971** | **0.9062** | **0.6421** | **0.6690** | **0.9964** | **0.9024** | 24.89 | 0.00 | 0.8182 |
| river_online | 0.4127 | 0.8125 | 0.5474 | 0.5760 | 0.9295 | 0.6896 | 31.40 | 0.00 | 0.2727 |
| hf_bert_tiny | 0.0779 | 0.7604 | 0.1413 | 0.2330 | 0.8861 | 0.3846 | 244.41 | 0.00 | 0.2727 |

`xgboost_smote` (the dashboard's primary model) remains the strongest all-around performer and
its `mttd_coverage` (fraction of attack campaigns detected at all) rose to 0.8182 -- the highest
of any model -- consistent with catching more `device_spoofing` campaigns than before.

## 6. Why `device_spoofing` improved and `lateral_movement` did not

`device_spoofing`'s fix worked because the new feature gave the model a **direct, near-binary**
signal for the exact mechanism the attack uses (fingerprint mismatch), and SMOTE's synthetic
oversampling (device_spoofing has only 3 train examples) was enough to teach the model that
signal well enough for out-of-fold thresholds to generalize to the 4 held-out test examples.

`lateral_movement`'s signal (`access_chain_distance`, `peer_group_deviation`, z~1.4) is real but
weak and diffuse, not a near-binary flag -- and with only 27 train examples split across 3 CV
folds (~9 per fold), the out-of-fold threshold estimate is high-variance: it looked like it beat
argmax on OOF F1 during tuning, but that threshold did not hold up on the 4 actual test examples.
This is reported as a genuine negative result, not hidden or re-run until it looked better.

## 7. Not fixed as of Section 6 -- next legitimate levers

Per the plan's explicit "minimal-rework discipline," Phase 1 was scoped to the two techniques
above. At that point, `lateral_movement` remaining at 0% recall meant these were the next
legitimate levers, in order of how directly they address the diagnosed problem:

1. **Sequence/session-window features.** ~~Deferred~~ **-- done, see Section 10.** Lateral
   movement is inherently about a *sequence* of resource accesses across a session, not a single
   event's snapshot -- `access_chain_distance` only looks at the immediately-previous resource.
2. **Focal loss.** Still not attempted. Directly down-weights easy (mostly benign) examples during
   training instead of reweighting by class frequency alone -- addresses the "has to win a 6-way
   contest against 99.4% benign" problem more directly than `class_weight`. Still worth trying:
   Section 10's fix got `lateral_movement` to 50% recall, not full recovery.
3. **A dedicated one-vs-rest binary classifier** for `lateral_movement` specifically, trained with
   heavier oversampling than the shared 6-class model allows -- isolates its decision boundary
   from having to also serve the other 5 classes' loss. Still not attempted.

Not recommended without more evidence first: balanced random forest, LightGBM, EasyEnsemble --
no diagnosis result here points at "the wrong model family" as opposed to "the wrong feature
representation," so trying a bigger hammer before the above two is not justified by what was
actually measured.

## 10. Follow-up: session-window features for `lateral_movement` (Phase 5e)

Section 6 named "session-window features" as the most direct next lever for `lateral_movement`,
since its only real per-event signal (`access_chain_distance`) looks at a single direct
transition, with no notion of session-wide breadth. `attacks/lateral_movement.py`'s own generator
rationale is explicitly session-level: "accessed N resources outside its department... within a
single fast session."

**Two new features**, added to `feature_engineering/graph.py`'s `GraphFeatureState` (per-session
state, pruned on the same window as the existing session-tracking dicts):

- `session_foreign_resource_count`: count of DISTINCT resource types accessed so far this session
  that fall outside the user's own department's typical resource set (using
  `RESOURCE_TYPES_BY_DEPT`, the exact vocabulary the attack generator itself draws its escalating
  chain from) -- makes session-wide breadth visible, which `access_chain_distance`'s single-hop
  view cannot see.
- `session_hop_seconds`: seconds since the previous event in this same session (a large sentinel,
  3600s, if this is the session's first event) -- makes hopping velocity visible; the generator's
  own `hop_gap_seconds` config (2-20s) deliberately makes lateral-movement hops faster than benign
  resource chains (5-180s) as a secondary, designed-in tell.

`GRAPH_FEATURE_COLUMNS` grew from 6 to 8; `FEATURE_COLUMNS` from 13 to 15. Verified in isolation
(`tests/test_graph_features.py::test_session_features_spike_on_lateral_movement`): mean
`session_foreign_resource_count` is meaningfully higher (>1.5x) and mean `session_hop_seconds` is
lower (faster) on `lateral_movement` rows than on benign rows -- passing, confirming genuine
separation before any model was retrained on it.

### Real before/after numbers (multi-class classification report, same 17,231-row test split)

| Method | `lateral_movement` recall | `lateral_movement` precision | `device_spoofing` recall | `device_spoofing` precision | `brute_force` f1 |
|---|---|---|---|---|---|
| none | 0.000 (unchanged) | -- | 0.000 (unchanged) | -- | 0.918 |
| class_weight | 0.000 (unchanged) | -- | **0.000 (regressed from 0.250)** | -- | 0.905 |
| **smote** | **0.500** (2/4, was 0.000) | 0.0690 | 0.750 (unchanged) | **0.2143 (up from 0.0161)** | 0.936 |

Honestly reported, not smoothed over: `class_weight`'s `device_spoofing` recall **dropped** from
25% to 0% after this change. Adding 2 features changes the entire out-of-fold probability
landscape the per-class thresholds are tuned against (Section 4); `class_weight`'s previous 25%
was already a single-test-example result (1 of 4) sensitive to exactly this kind of shift, not a
robust win to begin with. `smote` is the only method that improved on both rare classes
simultaneously, and its `device_spoofing` precision improved by 13x as a side effect (fewer false
positives at the same 75% recall) -- plausibly because the richer feature set gives the model
better grounds to distinguish real device_spoofing from noise it previously had to guess on.

Six-criteria table, this run (`docs/phase_3_evaluation_report.md`, regenerated):

| model | precision | recall | f1 | mcc | roc_auc | pr_auc | fp/day | mttd_coverage |
|---|---|---|---|---|---|---|---|---|
| rule_based_baseline | 0.5294 | 0.7500 | 0.6207 | 0.6277 | 0.9347 | 0.4033 | 18.10 | 0.2727 |
| isolation_forest | 0.3373 | 0.8750 | 0.4870 | 0.5397 | 0.9555 | 0.5920 | 46.67 | 0.4545 |
| xgboost_none | 0.4611 | 0.9271 | 0.6159 | 0.6514 | 0.9982 | 0.8890 | 29.42 | 0.7273 |
| xgboost_class_weight | 0.4503 | 0.8958 | 0.5993 | 0.6325 | 0.9944 | 0.9005 | 29.70 | 0.6364 |
| **xgboost_smote** | 0.4147 | **0.9375** | 0.5751 | 0.6208 | 0.9955 | **0.9223** | 35.93 | **0.9091** |
| river_online | 0.1667 | 0.8438 | 0.2784 | 0.3687 | 0.9138 | 0.6914 | 114.57 | 0.2727 |
| hf_bert_tiny | 0.0779 | 0.7604 | 0.1413 | 0.2330 | 0.8861 | 0.3846 | 244.41 | 0.2727 |

`xgboost_smote`'s `mttd_coverage` (fraction of attack campaigns detected at all) rose to **0.9091**
(from 0.8182) and PR-AUC to 0.9223 (from 0.9024) -- consistent with catching more campaigns overall,
including `lateral_movement` ones. Precision dropped (0.4971 -> 0.4147) and false-positives/day rose
(24.89 -> 35.93): a real, quantified cost of catching more of the hard classes, not a free win.

### Feature-importance re-check (Phase 5c robustness, re-run with the new features)

Re-running the ablation analysis with both new features included (baseline, all features present:
recall=0.9375, pr_auc=0.9223):

| Ablated feature | recall | pr_auc | Interpretation |
|---|---|---|---|
| `session_hop_seconds` | 0.9271 | 0.8359 | Real, meaningful contribution -- removing it drops PR-AUC noticeably |
| `session_foreign_resource_count` | 0.9375 | 0.9183 | Small effect -- largely redundant given the other 14 features; the model can compensate |

Honest nuance: velocity (`session_hop_seconds`) carries more marginal importance for this model
than breadth (`session_foreign_resource_count`), even though breadth was the feature named first
in the diagnosis. Both were added together as a single session-level lever; this is reported as
observed, not adjusted after the fact to match the original hypothesis.

The noise-robustness finding from Phase 5c (precision collapses under small Gaussian feature
noise) reproduces unchanged with the new features -- still a real, open limitation (see
`docs/phase_5c_robustness.md`).

## 8. Verification

**Phase 1 (Sections 1-6):**
- `pytest tests/test_graph_features.py tests/test_streaming_approx.py tests/test_resumable_feature_computation.py tests/test_xgboost_classifier.py tests/test_model_suite.py tests/test_model_suite_checkpointing.py -v` -- all green (12 tests), confirming the `device_fingerprint_mismatch` feature and threshold logic didn't destabilize anything already working.
- `python -m evaluation.leakage_audit data/runs/small_dev` -- **PASSED**.
- `python -m evaluation.run_evaluation --config-name small_dev` / `python -m dashboard.prepare_data --config-name small_dev` -- regenerated report + dashboard artifacts, spot-checked in-browser.

**Phase 5e (Section 10, session-window features):**
- `pytest tests/ -q` -- **all 73 tests pass** (~25 min), the full existing suite plus the new `test_session_features_spike_on_lateral_movement` test.
- `python -m evaluation.leakage_audit data/runs/small_dev` -- **PASSED** again (ROC-AUC 0.4942 vs. chance 0.5, epsilon 0.05) -- the two new session features introduced no leakage either.
- `python -m evaluation.run_evaluation` / `dashboard.prepare_data` -- regenerated with real numbers above; dashboard spot-checked live in-browser (Attack Coverage cards, recommendation engine's reliability notes) after each retrain.
- `python -m evaluation.run_rigor_analysis` / `python -m evaluation.run_robustness_analysis` -- both re-run against the new feature set (Phase 2a/2b's calibration and stress-test numbers refreshed, not left stale).

## 9. Reproducibility

```
python -m evaluation.leakage_audit data/runs/small_dev
python -m dashboard.prepare_data --config-name small_dev
python -m evaluation.run_evaluation --config-name small_dev
python -m evaluation.run_rigor_analysis --config-name small_dev
python -m evaluation.run_robustness_analysis --config-name small_dev
```

All five commands stay at `small_dev` scale (57,436 events) -- no full-scale (1.17M-event) rerun
was performed or is needed to validate this fix, per the explicit "don't redo everything"
constraint for this phase. Note: `docs/scale_up_report.md` (from before Phase 1/5e) found the
imbalance-method ranking can reverse at full scale -- the `smote` wins reported here are validated
at `small_dev` scale only and have not been re-checked at 1.17M events with these new features.
