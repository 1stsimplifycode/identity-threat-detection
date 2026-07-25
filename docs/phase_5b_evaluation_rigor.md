# Phase 5b: Evaluation Rigor (calibration, confidence intervals, significance)

> Synthetic data. Not derived from or validated against real organizational logs. For benchmarking detection methods only.

Post-hoc statistics computed directly over the model suite's existing test-set checkpoints -- no retraining. Every bootstrap uses a fixed seed (42); re-running this exact command reproduces these numbers exactly.

## Balanced accuracy + calibration

`balanced_accuracy` is reported for every model (valid regardless of score type). Brier score and Expected Calibration Error (ECE) are reported only for models whose score is a genuine [0, 1] probability -- `rule_based_baseline` (a 0-3 flag count) and `isolation_forest` (an unbounded signed anomaly score) are excluded, not silently given a misleading number.

| model                |   balanced_accuracy |   brier_score |      ece |
|:---------------------|--------------------:|--------------:|---------:|
| rule_based_baseline  |              0.8731 |      nan      | nan      |
| isolation_forest     |              0.9327 |      nan      | nan      |
| xgboost_none         |              0.9605 |        0.0014 |   0.0012 |
| xgboost_class_weight |              0.9449 |        0.0016 |   0.0013 |
| xgboost_smote        |              0.965  |        0.0017 |   0.0017 |
| river_online         |              0.9101 |        0.0031 |   0.0049 |
| hf_bert_tiny         |              0.855  |        0.012  |   0.0331 |

## Bootstrap 95% confidence intervals (300 resamples, seed=42)

Nonparametric percentile bootstrap over the test-set indices -- how much each metric would plausibly vary on a different sample from the same underlying population, not just a single point estimate presented as exact.

| model                | metric    |   estimate |   ci_95_lo |   ci_95_hi |
|:---------------------|:----------|-----------:|-----------:|-----------:|
| rule_based_baseline  | precision |     0.5294 |     0.4541 |     0.6094 |
| rule_based_baseline  | recall    |     0.75   |     0.657  |     0.8258 |
| rule_based_baseline  | f1        |     0.6207 |     0.5541 |     0.6906 |
| rule_based_baseline  | roc_auc   |     0.9347 |     0.9028 |     0.9626 |
| rule_based_baseline  | pr_auc    |     0.4033 |     0.3282 |     0.4912 |
| isolation_forest     | precision |     0.3373 |     0.2856 |     0.3967 |
| isolation_forest     | recall    |     0.875  |     0.8082 |     0.9346 |
| isolation_forest     | f1        |     0.487  |     0.4301 |     0.552  |
| isolation_forest     | roc_auc   |     0.9555 |     0.9178 |     0.9859 |
| isolation_forest     | pr_auc    |     0.592  |     0.4951 |     0.7049 |
| xgboost_none         | precision |     0.4611 |     0.403  |     0.5317 |
| xgboost_none         | recall    |     0.9271 |     0.8743 |     0.9777 |
| xgboost_none         | f1        |     0.6159 |     0.5561 |     0.6803 |
| xgboost_none         | roc_auc   |     0.9982 |     0.9964 |     0.9994 |
| xgboost_none         | pr_auc    |     0.889  |     0.8238 |     0.9459 |
| xgboost_class_weight | precision |     0.4503 |     0.3821 |     0.5196 |
| xgboost_class_weight | recall    |     0.8958 |     0.8279 |     0.9509 |
| xgboost_class_weight | f1        |     0.5993 |     0.5329 |     0.6612 |
| xgboost_class_weight | roc_auc   |     0.9944 |     0.9891 |     0.9982 |
| xgboost_class_weight | pr_auc    |     0.9005 |     0.844  |     0.9464 |
| xgboost_smote        | precision |     0.4147 |     0.3544 |     0.4811 |
| xgboost_smote        | recall    |     0.9375 |     0.8906 |     0.9794 |
| xgboost_smote        | f1        |     0.5751 |     0.5134 |     0.6387 |
| xgboost_smote        | roc_auc   |     0.9955 |     0.9903 |     0.9991 |
| xgboost_smote        | pr_auc    |     0.9223 |     0.8775 |     0.9665 |
| river_online         | precision |     0.1667 |     0.1361 |     0.2012 |
| river_online         | recall    |     0.8438 |     0.7715 |     0.9081 |
| river_online         | f1        |     0.2784 |     0.2325 |     0.3264 |
| river_online         | roc_auc   |     0.9138 |     0.8622 |     0.953  |
| river_online         | pr_auc    |     0.6914 |     0.5919 |     0.792  |
| hf_bert_tiny         | precision |     0.0779 |     0.0621 |     0.094  |
| hf_bert_tiny         | recall    |     0.7604 |     0.6756 |     0.8334 |
| hf_bert_tiny         | f1        |     0.1413 |     0.1146 |     0.1687 |
| hf_bert_tiny         | roc_auc   |     0.8861 |     0.8346 |     0.9292 |
| hf_bert_tiny         | pr_auc    |     0.3846 |     0.307  |     0.4912 |

## Significance: are the 3 XGBoost imbalance methods really different? (PR-AUC, paired bootstrap, 300 resamples)

Paired bootstrap on PR-AUC (the problem statement's headline metric under imbalance) -- both models scored on the SAME resampled test rows each iteration. p < 0.05 means the gap is unlikely to be bootstrap noise; a 95% CI on the difference that excludes 0 says the same thing from the interval side.

| model_a              | model_b              |   pr_auc_a |   pr_auc_b |    diff | diff_95_ci        |   p_value |
|:---------------------|:---------------------|-----------:|-----------:|--------:|:------------------|----------:|
| xgboost_none         | xgboost_class_weight |     0.889  |     0.9005 | -0.0115 | [-0.0736, 0.0304] |    0.8467 |
| xgboost_none         | xgboost_smote        |     0.889  |     0.9223 | -0.0333 | [-0.0867, 0.0115] |    0.24   |
| xgboost_class_weight | xgboost_smote        |     0.9005 |     0.9223 | -0.0218 | [-0.0558, 0.0014] |    0.0867 |
