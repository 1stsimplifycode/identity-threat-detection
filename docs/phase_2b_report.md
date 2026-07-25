# Phase 2b Report: Feature Engineering + Baseline + One ML Model

> **Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.**

## What was built

- **`feature_engineering/behavioral.py`** (3a) -- `BehavioralFeatureState`:
  velocity, geo-distance from home, login-location entropy, device-switch
  rate, failed-login ratio, peer-group deviation, EMA failure-rate baseline.
  Rolling per-user/per-department state on a config-driven window.
- **`feature_engineering/graph.py`** (3b) -- `GraphFeatureState`: an
  in-memory bipartite user-device graph, per-department resource-transition
  graphs, and a periodically-refreshed Louvain user-community partition
  (plain NetworkX algorithms, no GNN). Produces the 5 required named
  columns: `device_fan_in`, `user_device_set_delta`, `is_new_edge`,
  `access_chain_distance`, `peer_community_deviation`.
- **`feature_engineering/pipeline.py`**: `FeaturePipelineState` /
  `compute_feature_table()` merge both families into one table keyed by
  `record_id`.
- **`evaluation/chronological_split.py`**: time-respecting train/test split
  (constraint #1) with an explicit, loggable split timestamp.
- **`models/baseline.py`**: the dumb, rule-based detector (failed-login
  count, new-country flag, off-hours flag) -- deliberately reads only raw
  event fields, no engineered features at all.
- **`models/isolation_forest.py`**: the unsupervised "one ML model" for
  this phase, trained on the merged feature table.
- **`evaluation/report.py` + `evaluation/run_evaluation.py`**: computes and
  renders the six-criteria comparison table (`docs/phase_2b_evaluation_report.md`),
  rule baseline included in every row per constraint #3.
- **`tests/test_graph_features.py`**: explicit, non-eyeballed verification
  that `access_chain_distance` spikes on `lateral_movement` and
  `is_new_edge` spikes on `device_spoofing`'s `cross_user_reuse` variant,
  plus a determinism sanity check on the dual-mode feature interface.
- **3 new config groups** (`configs/feature_engineering/`, `configs/models/`,
  `configs/evaluation/`), wired into both `config.yaml` and `small_dev.yaml`.

## Two real bugs found during the explicit verification step, and fixed

Constraint #2's leakage-audit precedent held here too: the plan required
*explicit pytest assertions* that the graph features actually spike on the
attacks they're meant to catch, "before moving on" -- and the first attempt
at that check failed outright. Rather than loosen the assertions, both root
causes were found and fixed.

**Bug 1 -- the attack generator itself.** `lateral_movement.py`'s
`_pick_escalating_chain` picked whole *department names* other than the
target's home department, then used that department's full resource list.
But department resource sets overlap heavily (Executive's own set already
includes `hr_system` and `crm`), so the "escalation" chain could
accidentally include resources that were already completely normal for
that specific user. Fixed by filtering directly to resource *types* not in
the user's own department's set (`RESOURCE_TYPES_BY_DEPT`), regardless of
which other department(s) happen to also use them.

**Bug 2 -- the feature design itself.** `access_chain_distance` was
originally a single, shared, org-wide resource-transition graph, with
distance computed via full shortest-path search. With only ~7 resource
types and hub resources (`email`/`vpn`/`file_share`) present in nearly
every department's normal set, two things went wrong simultaneously:
(a) any two resources end up 1-2 cheap hops apart via those hubs regardless
of the specific user asking, and (b) a transition can be "normal" for the
org as a whole (e.g. `vpn->crm` is routine for Sales) even when it's
genuinely foreign to a *different* department's user. The result, caught
directly by the new test: `lateral_movement`'s own escalating hops scored
a *lower* mean distance (0.018) than ordinary benign traffic (1.26) --
backwards. Fixed two ways together: (1) the resource-transition graph is
now scoped **per department** (comparing to "is this common among my
peers," matching `peer_community_deviation`'s own peer-relative framing),
and (2) distance is now the **direct edge cost** from the user's most
recently visited resource, not a multi-hop shortest-path search (which
would reintroduce the hub-shortcut problem even within a smaller,
department-scoped graph). After both fixes: `lateral_movement` mean 3.97
vs. benign mean 1.47 (N=73, `imbalance_ratio=0.05` verification config) --
the correct direction, with a real margin.

Both bugs are documented in code comments at the exact functions that
changed (`attacks/lateral_movement.py:_pick_escalating_chain`,
`feature_engineering/graph.py:_access_chain_distance`), not just here.

## Key design decisions

**Dual-mode is structural, not aspirational.** Both `BehavioralFeatureState`
and `GraphFeatureState` expose exactly one `update(event) -> dict` method
as the single source of feature logic; `compute_batch()` is nothing more
than "replay `update()` over a sorted DataFrame from empty state." Phase
4's streaming loop will call the same `update()` on a live-kept object --
there is no second implementation to drift out of sync with the first.

**Rolling-window pruning is O(1)-amortized everywhere**, via
timestamp-ordered deques of insertions (bipartite edges, resource-graph
edges, per-user resource history) -- never a full-graph rebuild per event.
Louvain community detection is refreshed every `louvain_refresh_events`
events (config, default 500), not per-event, since per-event community
reassignment would be both wasteful and run-to-run unstable.

**`peer_community_deviation` needs scale/density to be informative.** At
very small test scale (a couple hundred events) most Louvain "communities"
are singletons and deviation stays at 0 almost everywhere; at real
`small_dev` scale (57k events) it becomes meaningfully non-zero for ~1.7%
of rows. This is a genuine scale-sensitivity of the feature, not a bug --
flagged in `feature_engineering/README.md`.

**The rule baseline never touches engineered features at all**, by design
-- it exists specifically to answer "does the sophistication of
`feature_engineering/` actually buy anything," and can only do that
honestly if it's built from independent, simpler machinery
(`models/baseline.py`'s own rolling failed-login counter, not
`behavioral.py`'s EMA-smoothed one).

**The operating threshold is chosen from the TRAIN score distribution only**
(a configurable percentile, default 99th), never from test labels, per
constraint #1 -- applied identically to both the rule baseline and
Isolation Forest for a fair, leakage-free comparison.

## Evaluation results (small_dev scale, chronological 70/30 split)

Full report: `docs/phase_2b_evaluation_report.md`. Headline numbers:

| model | precision | recall | f1 | roc_auc | pr_auc | fp/day |
|---|---|---|---|---|---|---|
| rule_based_baseline | 0.000 | 0.000 | 0.000 | 0.847 | 0.011 | 6.4 |
| isolation_forest | 0.207 | 0.600 | 0.308 | 0.975 | 0.296 | 19.9 |

**Read honestly, not oversold:** Isolation Forest clearly separates
classes better by rank (ROC-AUC 0.975 vs. 0.847, PR-AUC 0.296 vs. 0.011),
and its per-attack-type breakdown shows it catches 86% of `brute_force`
events at this threshold. But **neither model reliably catches
`credential_misuse`, `device_spoofing`, or `impossible_travel` at the
chosen 99th-percentile threshold** in this run, and the rule baseline's
precision/recall are literally 0.0 at that specific threshold despite a
respectable underlying ROC-AUC -- a direct consequence of its `rule_risk_score`
being a coarse 4-value integer (0-3), where the 99th percentile can land on
a threshold that happens to catch zero true positives in a given test
split. This is a real, reportable limitation of Phase 2b's model set, not
a bug to paper over: `brute_force`'s extreme, unmistakable footprint (tens
of failed logins in minutes) is easy for both a crude rule and an
unsupervised anomaly score to catch; the subtler attack types' signatures
are exactly the kind of thing Phase 3's supervised XGBoost classifier
(trained directly on labels, not just "is this unusual") is expected to do
much better on -- that comparison is the point of building the full model
set before drawing conclusions.

## Known limitations / deliberate deferrals

- **No cold-start handling yet** (Phase 3 per the updated plan).
- **No drift detection or SMOTE/class-weighting comparison yet** (Phase 3).
- **No SHAP or per-event explanation for Isolation Forest yet** (Phase 4,
  per constraint #4's explicit offline/streaming tradeoff).
- **`peer_community_deviation`'s scale-sensitivity** (see above) means its
  contribution to Isolation Forest's score should be re-examined once
  full-scale data is available (Scale-up phase).
- **Per-attack-type "recall" in this phase's classification criterion is
  a proxy, not real classification** -- Phase 3's supervised model is what
  actually predicts attack *type*, not just anomaly/not-anomaly.
- **The direct-edge-cost fix to `access_chain_distance` is deliberately
  narrow** (looks only at the single most recent hop, not the full
  historical footprint the problem statement's wording suggests) --
  a considered tradeoff given the resource vocabulary's small size (~7
  types), documented in the function's own comment, and worth revisiting
  if the resource vocabulary ever grows substantially.

## Synthetic-data disclaimer

Every generated run still carries `DISCLAIMER.txt`; the evaluation report
carries the same text verbatim. No result in this report should be read as
validated against real enterprise traffic.
