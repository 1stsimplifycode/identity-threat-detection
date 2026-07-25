# Six-Criteria Evaluation Report (Phase 2b, small_dev scale)

> Synthetic data. Not derived from or validated against real organizational logs. For benchmarking detection methods only.

**Split:** Chronological split: train_frac=0.70, split_timestamp=2026-01-11 13:03:15, n_train=39922, n_test=17110

## 1. Detection accuracy (+ 2. false positives)

| model               |   precision |   recall |     f1 |   roc_auc |   pr_auc (headline) |   false_positives_per_day |   n_flagged |   n_test |
|:--------------------|------------:|---------:|-------:|----------:|--------------------:|--------------------------:|------------:|---------:|
| rule_based_baseline |      0      |      0   | 0      |    0.8467 |              0.0109 |                      6.36 |          22 |    17110 |
| isolation_forest    |      0.2069 |      0.6 | 0.3077 |    0.9754 |              0.2963 |                     19.94 |          87 |    17110 |

PR-AUC is the headline metric given class imbalance (per the problem statement).

## 5. Classification (per-attack-type detection recall)

Not true multi-class classification yet (Phase 3's supervised XGBoost adds that) -- this is the closest honest proxy available from a binary anomaly score: of events belonging to each attack type, what fraction did each model flag.

| model               |   brute_force |   credential_misuse |   device_spoofing |   impossible_travel |
|:--------------------|--------------:|--------------------:|------------------:|--------------------:|
| rule_based_baseline |        0      |                   0 |                 0 |                   0 |
| isolation_forest    |        0.8571 |                   0 |                 0 |                   0 |

## 4. Scalability

- Feature computation throughput: 1948 events/sec
- Feature computation wall-clock time for this run: 29.3s over 57032 events

## 3. Explainability

The rule-based baseline is inherently explainable: `rule_risk_score` is a sum of 3 named, human-readable flags (failed-login count, new-country, off-hours) -- the explanation *is* the score. Isolation Forest's engineered-feature score has no per-event explanation yet; offline SHAP (exact, batch-only) and a lightweight streaming approximation are Phase 4 scope, per constraint #4 -- not silently skipped, explicitly deferred.

## 6. Design

See `docs/phase_2b_report.md` for architecture, modularity, and reproducibility notes.
