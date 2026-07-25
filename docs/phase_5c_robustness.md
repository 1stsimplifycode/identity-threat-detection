# Phase 5c: Robustness / Stress Testing (xgboost_smote)

> Synthetic data. Not derived from or validated against real organizational logs. For benchmarking detection methods only.

Stress tests of the ALREADY-TRAINED `xgboost_smote` model's FIXED decision boundary (operating threshold chosen once, on the TRAIN split, never touched again) against perturbed test inputs -- no retraining per scenario, only re-scoring. Every random perturbation uses a fixed seed (42); re-running reproduces these numbers exactly.

**Baseline (no perturbation):** precision=0.4147, recall=0.9375, f1=0.5751, pr_auc=0.9223, operating threshold=0.010707

## 1. Noise injection (additive Gaussian noise, scaled to each feature's own TRAIN-split std)

Simulates measurement noise / sensor jitter in real-world telemetry -- every one of the 13 engineered features gets independent noise at the stated fraction of its own std.

|   noise_fraction_of_std |   precision |   recall |     f1 |   pr_auc |   brier_score |    ece |
|------------------------:|------------:|---------:|-------:|---------:|--------------:|-------:|
|                    0    |      0.4147 |   0.9375 | 0.5751 |   0.9223 |        0.0017 | 0.0017 |
|                    0.05 |      0.0452 |   0.9062 | 0.0861 |   0.4107 |        0.0209 | 0.0281 |
|                    0.1  |      0.0359 |   0.8854 | 0.0691 |   0.2697 |        0.0258 | 0.0344 |
|                    0.25 |      0.0272 |   0.9062 | 0.0527 |   0.1493 |        0.0312 | 0.0431 |
|                    0.5  |      0.0193 |   0.8958 | 0.0379 |   0.0434 |        0.0708 | 0.0892 |
|                    1    |      0.0127 |   0.8854 | 0.0251 |   0.0158 |        0.1635 | 0.1971 |

## 2. Missing-feature ablation (replaced with TRAIN-split median -- a realistic "unavailable at inference, imputed" scenario, not zeroing)

| ablated                        |   precision |   recall |     f1 |   pr_auc |
|:-------------------------------|------------:|---------:|-------:|---------:|
| behavioral (7 features)        |      0.625  |   0.0521 | 0.0962 |   0.1235 |
| graph (6 features)             |      0.042  |   0.9062 | 0.0803 |   0.7842 |
| ema_failure_rate               |      0.3056 |   0.3438 | 0.3235 |   0.2826 |
| failed_login_ratio             |      0.4712 |   0.9375 | 0.6272 |   0.9292 |
| access_chain_distance          |      0.4574 |   0.8958 | 0.6056 |   0.8934 |
| peer_group_deviation           |      0.411  |   0.9375 | 0.5714 |   0.911  |
| device_fingerprint_mismatch    |      0.4065 |   0.9062 | 0.5613 |   0.9031 |
| geo_distance_from_home_km      |      0.5664 |   0.8438 | 0.6778 |   0.4714 |
| session_foreign_resource_count |      0.4147 |   0.9375 | 0.5751 |   0.9183 |
| session_hop_seconds            |      0.073  |   0.9271 | 0.1354 |   0.8359 |
