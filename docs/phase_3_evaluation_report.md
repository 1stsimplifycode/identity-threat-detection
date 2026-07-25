# Six-Criteria Evaluation Report (Phase 3, small_dev scale)

> Synthetic data. Not derived from or validated against real organizational logs. For benchmarking detection methods only.

**Split:** Chronological split: train_frac=0.70, split_timestamp=2026-01-11 11:13:02, n_train=40205, n_test=17231

## 1. Detection accuracy (+ 2. false positives, MTTD)

PR-AUC is the headline metric given class imbalance (per the problem statement). `mttd_days`/`mttd_coverage`: mean days from an attack campaign's true start to its first flagged event, and the fraction of campaigns detected at all (a model can have a great mean lag on the few campaigns it catches while missing most of them -- both numbers are reported so that isn't hidden).

| model                |   precision |   recall |     f1 |    mcc |   roc_auc |   pr_auc (headline) |   false_positives_per_day |   mttd_days |   mttd_coverage |   n_flagged |   n_test |
|:---------------------|------------:|---------:|-------:|-------:|----------:|--------------------:|--------------------------:|------------:|----------------:|------------:|---------:|
| rule_based_baseline  |      0.5294 |   0.75   | 0.6207 | 0.6277 |    0.9347 |              0.4033 |                     18.1  |        0    |          0.2727 |         136 |    17231 |
| isolation_forest     |      0.3373 |   0.875  | 0.487  | 0.5397 |    0.9555 |              0.592  |                     46.67 |        0.01 |          0.4545 |         249 |    17231 |
| xgboost_none         |      0.4611 |   0.9271 | 0.6159 | 0.6514 |    0.9982 |              0.889  |                     29.42 |        0    |          0.7273 |         193 |    17231 |
| xgboost_class_weight |      0.4503 |   0.8958 | 0.5993 | 0.6325 |    0.9944 |              0.9005 |                     29.7  |        0    |          0.6364 |         191 |    17231 |
| xgboost_smote        |      0.4147 |   0.9375 | 0.5751 | 0.6208 |    0.9955 |              0.9223 |                     35.93 |        0    |          0.9091 |         217 |    17231 |
| river_online         |      0.1667 |   0.8438 | 0.2784 | 0.3687 |    0.9138 |              0.6914 |                    114.57 |        0    |          0.2727 |         486 |    17231 |
| hf_bert_tiny         |      0.0779 |   0.7604 | 0.1413 | 0.233  |    0.8861 |              0.3846 |                    244.41 |        0    |          0.2727 |         937 |    17231 |

## 5. Classification

### Per-attack-type detection recall (all models -- a proxy for models without genuine multi-class output)

| model                |   brute_force |   credential_misuse |   device_spoofing |   impossible_travel |   lateral_movement |
|:---------------------|--------------:|--------------------:|------------------:|--------------------:|-------------------:|
| rule_based_baseline  |        0.8571 |                 0   |              0    |                   0 |               0    |
| isolation_forest     |        0.9762 |                 0   |              0    |                   1 |               0    |
| xgboost_none         |        0.9762 |                 0.5 |              0.25 |                   1 |               0.75 |
| xgboost_class_weight |        0.9643 |                 0.5 |              0    |                   1 |               0.5  |
| xgboost_smote        |        0.9643 |                 0.5 |              0.75 |                   1 |               0.75 |
| river_online         |        0.9643 |                 0   |              0    |                   0 |               0    |
| hf_bert_tiny         |        0.869  |                 0   |              0    |                   0 |               0    |

### Real multi-class classification report (models with genuine class predictions)

**xgboost_none**

|                   |   precision |   recall |   f1-score |    support |
|:------------------|------------:|---------:|-----------:|-----------:|
| benign            |      0.999  |   0.9984 |     0.9987 | 17135      |
| brute_force       |      0.907  |   0.9286 |     0.9176 |    84      |
| credential_misuse |      0      |   0      |     0      |     2      |
| device_spoofing   |      0      |   0      |     0      |     4      |
| impossible_travel |      0      |   0      |     0      |     2      |
| lateral_movement  |      0      |   0      |     0      |     4      |
| accuracy          |      0.9974 |   0.9974 |     0.9974 |     0.9974 |
| macro avg         |      0.3177 |   0.3212 |     0.3194 | 17231      |
| weighted avg      |      0.9979 |   0.9974 |     0.9976 | 17231      |

**xgboost_class_weight**

|                   |   precision |   recall |   f1-score |   support |
|:------------------|------------:|---------:|-----------:|----------:|
| benign            |      0.9993 |   0.9987 |     0.999  | 17135     |
| brute_force       |      0.8526 |   0.9643 |     0.905  |    84     |
| credential_misuse |      1      |   0.5    |     0.6667 |     2     |
| device_spoofing   |      0      |   0      |     0      |     4     |
| impossible_travel |      0.6667 |   1      |     0.8    |     2     |
| lateral_movement  |      0      |   0      |     0      |     4     |
| accuracy          |      0.998  |   0.998  |     0.998  |     0.998 |
| macro avg         |      0.5864 |   0.5772 |     0.5618 | 17231     |
| weighted avg      |      0.9981 |   0.998  |     0.998  | 17231     |

**xgboost_smote**

|                   |   precision |   recall |   f1-score |    support |
|:------------------|------------:|---------:|-----------:|-----------:|
| benign            |      0.9996 |   0.9971 |     0.9984 | 17135      |
| brute_force       |      0.9101 |   0.9643 |     0.9364 |    84      |
| credential_misuse |      0.3333 |   0.5    |     0.4    |     2      |
| device_spoofing   |      0.2143 |   0.75   |     0.3333 |     4      |
| impossible_travel |      0.6667 |   1      |     0.8    |     2      |
| lateral_movement  |      0.069  |   0.5    |     0.1212 |     4      |
| accuracy          |      0.9968 |   0.9968 |     0.9968 |     0.9968 |
| macro avg         |      0.5322 |   0.7852 |     0.5982 | 17231      |
| weighted avg      |      0.9986 |   0.9968 |     0.9976 | 17231      |

**river_online**

|                   |   precision |   recall |   f1-score |    support |
|:------------------|------------:|---------:|-----------:|-----------:|
| benign            |      0.9944 |   1      |     0.9972 | 17135      |
| brute_force       |      0      |   0      |     0      |    84      |
| credential_misuse |      0      |   0      |     0      |     2      |
| device_spoofing   |      0      |   0      |     0      |     4      |
| impossible_travel |      0      |   0      |     0      |     2      |
| lateral_movement  |      0      |   0      |     0      |     4      |
| accuracy          |      0.9944 |   0.9944 |     0.9944 |     0.9944 |
| macro avg         |      0.1657 |   0.1667 |     0.1662 | 17231      |
| weighted avg      |      0.9889 |   0.9944 |     0.9917 | 17231      |

**hf_bert_tiny**

|                   |   precision |   recall |   f1-score |    support |
|:------------------|------------:|---------:|-----------:|-----------:|
| benign            |      0.9985 |   0.9883 |     0.9933 | 17135      |
| brute_force       |      0.2583 |   0.8333 |     0.3944 |    84      |
| credential_misuse |      0      |   0      |     0      |     2      |
| device_spoofing   |      0      |   0      |     0      |     4      |
| impossible_travel |      0      |   0      |     0      |     2      |
| lateral_movement  |      0      |   0      |     0      |     4      |
| accuracy          |      0.9868 |   0.9868 |     0.9868 |     0.9868 |
| macro avg         |      0.2095 |   0.3036 |     0.2313 | 17231      |
| weighted avg      |      0.9942 |   0.9868 |     0.9897 | 17231      |

### Per-class decision thresholds (rare-class recall fix, see docs/phase_5_recall_investigation.md)

Tuned via out-of-fold CV on the TRAIN split only (never test labels); a class only gets a threshold here if it beat plain argmax on held-out F1 during tuning. Rows with no entries fell back to plain argmax -- honestly, not silently.

- **xgboost_none**: lateral_movement=0.120058, device_spoofing=0.001381
- **xgboost_class_weight**: lateral_movement=0.858734, device_spoofing=0.011137
- **xgboost_smote**: lateral_movement=0.168303, device_spoofing=0.003555

## 4. Scalability

- Feature computation throughput: 2007 events/sec
- Feature computation wall-clock time for this run: 28.6s over 57436 events

## 3. Explainability

The rule-based baseline is inherently explainable: `rule_risk_score` is a sum of 3 named, human-readable flags -- the explanation *is* the score. Every other model's engineered-feature score has no per-event explanation yet; offline SHAP (exact, batch-only) and a lightweight streaming approximation are Phase 4 scope, per constraint #4 -- not silently skipped, explicitly deferred.

## 6. Design

See `docs/phase_3_report.md` for architecture, modularity, and reproducibility notes.

## Drift detection (ADWIN vs. ground-truth drift_log.csv)

|   day | change_type       | detected   |   detection_lag_days |
|------:|:------------------|:-----------|---------------------:|
|     7 | remote_work_shift | True       |                 0.49 |
