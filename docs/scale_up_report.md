# Scale-Up Report

> **Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.**

## What this stage did

Every phase through Phase 4 was built and validated at `small_dev` scale
(5,000 users, 14 simulated days, ~57K events -- "full dev loop under 5
minutes on a laptop," per `configs/small_dev.yaml`). This stage re-ran the
identical pipeline -- same generator, same feature engineering, same six
model configurations, same leakage audit, same ADWIN drift detection -- at
`configs/config.yaml`'s full-scale profile: **8,000 users, 120 simulated
days, ~1.0 events/user/day**, producing **1,167,750 events** and a
**350,325-row chronological test split**. Nothing was skipped: all 7
models (rule baseline, isolation forest, 3 XGBoost imbalance variants,
River online learner, fine-tuned `hf_bert_tiny`) and ADWIN drift detection
ran to completion. Full results: `docs/phase_3_evaluation_report.md`.

This report's job is to answer two questions honestly: **did anything
change at 20x the data**, and **does the deployed dashboard need to carry
the full-scale dataset** (the latter is answered in
`docs/deployment.md`'s "Scale-up decision" section; this report supplies
the evidence for it).

## Headline comparison: small_dev vs. full-scale

| model | small_dev PR-AUC | full-scale PR-AUC | small_dev F1 | full-scale F1 | small_dev MCC | full-scale MCC |
|---|---|---|---|---|---|---|
| rule_based_baseline | 0.403 | 0.2065 | 0.621 | 0.0738 | 0.628 | 0.1677 |
| isolation_forest | 0.658 | 0.5177 | 0.482 | 0.4968 | 0.538 | 0.5136 |
| xgboost_none | 0.847 | 0.8993 | 0.604 | 0.6499 | 0.633 | 0.6747 |
| xgboost_class_weight | 0.875 | 0.8828 | 0.604 | 0.6822 | 0.631 | 0.6987 |
| **xgboost_smote** (small_dev winner) | **0.880** | 0.8503 | **0.646** | 0.5865 | **0.667** | 0.6164 |
| river_online | 0.736 | 0.6754 | 0.409 | 0.578 | 0.467 | 0.5888 |
| hf_bert_tiny | 0.388 | 0.1089 | 0.139 | 0.2122 | 0.233 | 0.2242 |

(small_dev figures from `docs/phase_3_report.md`'s own evaluation table,
n_test=17,231; full-scale figures from `docs/phase_3_evaluation_report.md`,
n_test=350,325. Same code path both times -- `evaluation/model_suite.py`
-- so these are directly comparable, not two different pipelines.)

## Finding 1: the "best" imbalance-handling method reverses at scale

At `small_dev` scale, **SMOTE was the clear winner** among the three
XGBoost imbalance variants -- highest PR-AUC (0.880), F1 (0.646), and MCC
(0.667) of any model in the table, and this was reported as such in
`docs/phase_3_report.md`. At full scale, that ranking **reverses**:
`xgboost_class_weight` now has the best F1 (0.6822) and MCC (0.6987), and
`xgboost_none` (no resampling at all) has the best PR-AUC (0.8993) and
ROC-AUC (0.9774) of the three -- while `xgboost_smote` drops to the
**worst** of the three XGBoost variants on nearly every metric (F1 0.5865,
MCC 0.6164, PR-AUC 0.8503, fp/day 52.96 -- the highest of the three, and
worst MTTD coverage at 0.6119 vs. 0.7413 for `xgboost_none`).

**A plausible explanation, stated as a hypothesis, not a proven cause:**
this project's attack-injection rate is configured as a fixed *ratio*
(`configs/attacks/default.yaml`'s `imbalance_ratio`), so the full-scale
run's ~20x larger test set carries roughly 20x more genuine positive
examples too (1,761 real attack-labeled test rows at full scale, vs. an
estimated ~90 at small_dev scale, from the classification-report supports
in each evaluation report). SMOTE's benefit is synthesizing plausible
minority-class examples by interpolating between real ones -- valuable
when there are only a handful of real positives to learn a decision
boundary from, and a small_dev train split likely has well under 100
genuine attack rows for XGBoost to fit against directly. At full scale,
with roughly 20x more *real* minority examples available, the marginal
value of synthetic interpolation shrinks, and the added synthetic points
themselves may introduce boundary noise that a straightforward
class-reweighting (`class_weight`) or even no resampling at all doesn't.
This is exactly the kind of finding constraint #3 ("sophistication has to
earn its keep") exists to surface -- and here, at scale, the more elaborate
resampling technique doesn't. Confirming the causal mechanism would need a
dedicated ablation (e.g. re-running SMOTE at several intermediate scales)
that is out of scope for this report; the reversal itself is directly
measured, not speculative.

**Practical consequence:** `evaluation/model_suite.py`'s `PRIMARY_MODEL_NAME`
(used by the dashboard's primary-model score column, SHAP explanations,
and now the zero-knowledge proof's `model_name` binding -- see
`docs/zero_knowledge_proofs.md`) is currently `xgboost_smote`, chosen
based on the small_dev-scale comparison. This scale-up run's evidence
suggests `xgboost_class_weight` would be the stronger choice for the
full-scale regime specifically. Changing `PRIMARY_MODEL_NAME` itself is
**out of scope for this report** (it would ripple into the SHAP
explainer's `TreeExplainer` target, the dashboard's threshold slider
defaults, and every downstream artifact) -- flagged here as a concrete,
evidence-backed recommendation for a future review gate, not silently
acted on.

## Finding 2: `hf_bert_tiny` degrades further at scale, and the reason is visible in its own config

`hf_bert_tiny` was already the weakest model at small_dev scale (PR-AUC
0.388, the second-lowest in the table). At full scale it gets
meaningfully worse (PR-AUC 0.1089, recall drops from 0.771 to 0.3453).
This is not a mystery: `configs/models/default.yaml`'s
`hf_classifier.max_benign_train_examples: 6000` caps its training sample
**regardless of dataset scale**, by explicit, documented design (see that
config's own comment and `docs/phase_3_report.md`'s discussion of CPU-time
tradeoffs). At small_dev scale, the actual training sample (~6,224 rows,
per `docs/phase_3_report.md`) was already a large fraction (~15%) of the
40,205-row train split -- the model saw a reasonably representative
slice. At full scale, the same ~6,000-row cap against an 817,425-row train
split is under 1% of the data -- the model is trained on a *much* thinner
relative slice of the full-scale distribution, while every other model in
the comparison (rule baseline, isolation forest, all three XGBoost
variants, River) trains on the entire available data. This was a known,
explicitly documented tradeoff going in (not a new discovery), but this
run is the first direct measurement of its cost at scale: HF's relative
disadvantage against the other six models widens considerably as the
dataset grows, because its effective training-data fraction shrinks while
everyone else's stays constant.

## Finding 3: drift detection remains sound, but the schedule differs by design

`configs/config.yaml`'s longer 120-day simulation window has room for two
scheduled drift events (day 40 `remote_work_shift`, day 80
`schedule_shift`) vs. `small_dev`'s single event (day 7
`remote_work_shift`, since its 14-day window can't fit two well-separated
events). ADWIN detected both full-scale events:

| day | change_type | detected | detection_lag_days |
|---|---|---|---|
| 40 | remote_work_shift | True | 0.64 |
| 80 | schedule_shift | True | 14.42 |

The `remote_work_shift` detection lag (0.64 days) is consistent with the
small_dev run's single-event result (0.49 days) -- this drift type remains
fast to detect at either scale. `schedule_shift`'s much longer lag (14.42
days) is a new data point this scale-up run provides that small_dev's
single-event schedule never exercised; it suggests `schedule_shift` is an
inherently subtler behavioral change for ADWIN's `geo_distance_from_home_km`-based
monitoring (per `drift_detection/adwin_detector.py`'s module docstring) to
pick up than a remote-work shift is, independent of dataset scale.

## Finding 4: feature computation throughput and wall-clock cost

Feature computation processed the full 1,167,750-event dataset at
**444 events/sec**, taking **2,628.2 seconds (~43.8 minutes)** wall-clock
on the development machine. This is the single most expensive stage of
the full-scale pipeline by a wide margin (model training/scoring for all
7 models combined took less time). This throughput figure is a useful
planning number for any future scale target: a hypothetical 10M-event run
would need roughly 6.3 hours of feature computation alone at this same
per-event rate, before any model training.

## What did NOT change at scale

- **The leakage audit still passes** (task #48) -- the generator's
  record-independence guarantees hold at 20x the volume, not just at
  small_dev's smaller sample.
- **The model ranking's broad shape is stable**: tree-based models
  (the three XGBoost variants) remain the strongest detectors by a wide
  margin at both scales; `hf_bert_tiny` remains the weakest; the rule
  baseline remains a genuinely strong precision floor relative to
  Isolation Forest specifically. Only the *fine-grained* ordering among
  the three XGBoost imbalance variants reverses (Finding 1) -- the
  higher-level conclusion "use a tree-based model with some
  imbalance-handling" is unchanged.
- **No new leakage, schema, or generator defects surfaced.** The same
  generation code, unmodified, produced a dataset 20x larger with no
  reported anomalies beyond the two findings above, both of which are
  properties of the *learning problem at scale*, not of the data
  generator or pipeline correctness.

## Relationship to the deployment-scale decision

This report's evidence directly informed the decision recorded in
`docs/deployment.md`'s "Scale-up decision" section: the deployed Render
dashboard continues shipping `small_dev`-scale data, not this run's
full-scale dataset. In short -- the full-scale run's raw output (301 MB)
and the ~20x larger dashboard artifacts it implies are a poor fit for a
free-tier, git-based deploy, and this report's own findings (Findings 1-3)
don't change *what* the dashboard needs to demonstrate, only refine which
model variant would ideally be primary -- a separate, deliberately
deferred decision (see Finding 1's "Practical consequence").
