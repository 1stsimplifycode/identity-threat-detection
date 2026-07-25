"""Phase 5c (robustness / stress testing) entrypoint: how much does
`xgboost_smote`'s (the dashboard's primary model) real detection
performance degrade under noisy or missing feature inputs?

Retrains ONE model (xgboost_smote, small_dev scale -- a few seconds, not
the full model suite) because the trained model OBJECT is needed for
repeated re-scoring and is not checkpointed (see
evaluation/model_suite.py's module docstring). Reuses the checkpointed
feature table (`checkpoints/features.parquet`, already includes
`device_fingerprint_mismatch`) so feature computation itself is not
redone. Every stress-test scenario below only re-SCORES the same fixed
model against perturbed test features -- no retraining per scenario.

Usage:
    python -m evaluation.run_robustness_analysis --config-name small_dev
"""
from __future__ import annotations

import torch  # noqa: F401  -- see evaluation/model_suite.py's identical comment

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from evaluation.calibration import compute_calibration
from evaluation.chronological_split import apply_split, chronological_split
from evaluation.robustness import ablate_features, inject_gaussian_noise
from feature_engineering.feature_names import BEHAVIORAL_FEATURE_COLUMNS, FEATURE_COLUMNS, GRAPH_FEATURE_COLUMNS
from models.xgboost_classifier import score_xgboost, train_xgboost

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NOISE_FRACTIONS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]

# Individually-ablated features: the ones Phase 5's own diagnosis (see
# docs/phase_5_recall_investigation.md) found actually carry signal for
# SOME class, so ablating them one at a time answers "how much does the
# model rely on each specific real signal" rather than an arbitrary subset.
KEY_FEATURES_TO_ABLATE = [
    "ema_failure_rate", "failed_login_ratio", "access_chain_distance",
    "peer_group_deviation", "device_fingerprint_mismatch", "geo_distance_from_home_km",
    "session_foreign_resource_count", "session_hop_seconds",
]


def _binary_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="small_dev")
def main(cfg: DictConfig) -> None:
    run_dir = PROJECT_ROOT / cfg.run.output_dir
    checkpoint_dir = run_dir / "checkpoints"
    features_path = checkpoint_dir / "features.parquet"
    if not features_path.exists():
        raise SystemExit(f"{features_path} not found -- run evaluation.run_evaluation or dashboard.prepare_data first.")

    print("Loading checkpointed features (no recomputation)...")
    features = pd.read_parquet(features_path)
    events = pd.read_parquet(run_dir / "events.parquet")
    labels = pd.read_parquet(run_dir / "labels.parquet")

    split = chronological_split(events, train_frac=float(cfg.evaluation.train_frac))
    labels_slim = labels[["record_id", "is_attack", "attack_type", "attack_id"]]
    train_features = apply_split(features, split, "train")
    test_features = apply_split(features, split, "test")
    train_labels = apply_split(labels, split, "train")[["record_id", "attack_type"]]
    test_labels = apply_split(labels_slim, split, "test")

    print("Training xgboost_smote (single model, not the full suite)...")
    model = train_xgboost(train_features, train_labels, "smote", cfg)

    baseline_scores = score_xgboost(model, test_features)
    train_scores = score_xgboost(model, train_features)["xgb_anomaly_score"]
    threshold = float(np.percentile(train_scores.to_numpy(), float(cfg.evaluation.operating_threshold_percentile)))
    print(f"Operating threshold (train-split {cfg.evaluation.operating_threshold_percentile}th percentile): {threshold:.6f}")

    y_true = test_labels.merge(baseline_scores, on="record_id")["is_attack"].astype(int).to_numpy()

    feature_stds = train_features[FEATURE_COLUMNS].std()
    feature_medians = train_features[FEATURE_COLUMNS].median()

    # -- 1. noise injection: additive Gaussian noise scaled to each feature's
    # own TRAIN-split std, increasing fraction --
    noise_rows = []
    for frac in NOISE_FRACTIONS:
        noisy = inject_gaussian_noise(test_features, FEATURE_COLUMNS, feature_stds, frac, seed=42)
        scores_df = score_xgboost(model, noisy)
        y_score = scores_df["xgb_anomaly_score"].to_numpy()
        metrics = _binary_scores(y_true, y_score, threshold)
        calib = compute_calibration(f"noise_{frac}", y_true, y_score)
        noise_rows.append({
            "noise_fraction_of_std": frac, **{k: round(v, 4) for k, v in metrics.items()},
            "brier_score": round(calib.brier_score, 4), "ece": round(calib.ece, 4),
        })
        print(f"noise_fraction={frac}: recall={metrics['recall']:.4f} pr_auc={metrics['pr_auc']:.4f} ece={calib.ece:.4f}")

    # -- 2. group-level ablation (behavioral vs. graph feature family) --
    ablation_rows = []
    for group_name, cols in [("behavioral (7 features)", BEHAVIORAL_FEATURE_COLUMNS), ("graph (6 features)", GRAPH_FEATURE_COLUMNS)]:
        ablated = ablate_features(test_features, cols, feature_medians)
        scores_df = score_xgboost(model, ablated)
        metrics = _binary_scores(y_true, scores_df["xgb_anomaly_score"].to_numpy(), threshold)
        ablation_rows.append({"ablated": group_name, **{k: round(v, 4) for k, v in metrics.items()}})
        print(f"ablated={group_name}: recall={metrics['recall']:.4f} pr_auc={metrics['pr_auc']:.4f}")

    # -- 3. individual key-feature ablation --
    for col in KEY_FEATURES_TO_ABLATE:
        ablated = ablate_features(test_features, [col], feature_medians)
        scores_df = score_xgboost(model, ablated)
        metrics = _binary_scores(y_true, scores_df["xgb_anomaly_score"].to_numpy(), threshold)
        ablation_rows.append({"ablated": col, **{k: round(v, 4) for k, v in metrics.items()}})
        print(f"ablated={col}: recall={metrics['recall']:.4f} pr_auc={metrics['pr_auc']:.4f}")

    baseline_metrics = _binary_scores(y_true, baseline_scores["xgb_anomaly_score"].to_numpy(), threshold)

    lines = [
        "# Phase 5c: Robustness / Stress Testing (xgboost_smote)",
        "",
        "> Synthetic data. Not derived from or validated against real organizational logs. "
        "For benchmarking detection methods only.",
        "",
        "Stress tests of the ALREADY-TRAINED `xgboost_smote` model's FIXED decision boundary "
        "(operating threshold chosen once, on the TRAIN split, never touched again) against "
        "perturbed test inputs -- no retraining per scenario, only re-scoring. Every random "
        "perturbation uses a fixed seed (42); re-running reproduces these numbers exactly.",
        "",
        f"**Baseline (no perturbation):** precision={baseline_metrics['precision']:.4f}, "
        f"recall={baseline_metrics['recall']:.4f}, f1={baseline_metrics['f1']:.4f}, "
        f"pr_auc={baseline_metrics['pr_auc']:.4f}, operating threshold={threshold:.6f}",
        "",
        "## 1. Noise injection (additive Gaussian noise, scaled to each feature's own TRAIN-split std)",
        "",
        "Simulates measurement noise / sensor jitter in real-world telemetry -- every one of the "
        "13 engineered features gets independent noise at the stated fraction of its own std.",
        "",
        pd.DataFrame(noise_rows).to_markdown(index=False),
        "",
        "## 2. Missing-feature ablation (replaced with TRAIN-split median -- a realistic "
        "\"unavailable at inference, imputed\" scenario, not zeroing)",
        "",
        pd.DataFrame(ablation_rows).to_markdown(index=False),
        "",
    ]

    out_path = PROJECT_ROOT / "docs" / "phase_5c_robustness.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
