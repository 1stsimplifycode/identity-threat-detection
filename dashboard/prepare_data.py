"""Phase 4 offline dashboard data-precomputation script.

Render's free tier has no GPU, limited CPU/RAM, and EPHEMERAL DISK -- and
this project's design philosophy says "train models OFFLINE and ship
trained artifacts." This script does every expensive step exactly once,
locally: builds the full model suite (via `evaluation/model_suite.py`, so
these numbers can never drift from the Phase 3 evaluation report's own),
computes exact SHAP + the lightweight streaming approximation over a
sample of flagged test events, and assembles a per-user feature-history
table for trend drill-down -- then writes small, cheap-to-load Parquet/JSON
artifacts under `dashboard/data/<run_name>/`. `dashboard/app.py` only ever
READS these files at request time; the one exception (an explicit
"Regenerate" button that re-invokes this script) is documented in
docs/deployment.md.

Usage:
    python -m dashboard.prepare_data --config-name small_dev
"""
from __future__ import annotations

# IMPORTANT: torch before pandas/pyarrow -- see evaluation/model_suite.py's
# identical comment (genuine Windows DLL conflict on this dev environment).
import torch  # noqa: F401

import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from evaluation.chronological_split import apply_split
from evaluation.model_suite import PRIMARY_MODEL_NAME, build_model_suite
from evaluation.report import attack_type_recall_table, comparison_table
from explainability.shap_batch import (
    compute_global_feature_weights,
    explain_flagged_events,
    explanations_to_json_column,
    select_flagged_sample,
)
from explainability.streaming_approx import StreamingApproxExplainer, compute_feature_baseline, explain_stream_batch
from feature_engineering.pipeline import FEATURE_COLUMNS
from models.xgboost_classifier import score_xgboost
from preprocessing.constants import ATTACK_TYPES, DEPARTMENTS, SEVERITIES
from zkp.detection_proof import generate_detection_proof

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_EVENT_FIELDS: list[str] = [
    "record_id", "user_id", "timestamp", "event_type", "auth_result", "auth_method",
    "mfa_used", "failure_reason", "device_id", "device_type", "os", "browser",
    "ip_address", "geo_country", "geo_city", "network_type", "resource_accessed",
    "is_off_hours", "is_weekend",
]

DISCLAIMER = (
    "Synthetic data. Not derived from or validated against real organizational "
    "logs. For benchmarking detection methods only."
)


def _out_dir(cfg: DictConfig) -> Path:
    out = PROJECT_ROOT / "dashboard" / "data" / str(cfg.run.name)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_test_detail(suite, xgb_test_scores: pd.DataFrame) -> pd.DataFrame:
    """One denormalized row per TEST event: raw fields, department/role,
    attack ground truth (+ severity/rationale when it's a real attack),
    every engineered feature, and the primary model's score + predicted
    class. This single table backs the flagged-events table, the live
    threshold slider (recompute precision/recall by re-thresholding
    `score`), and per-user trend drill-down filtering.
    """
    test_events_raw = apply_split(suite.events, suite.split, "test")[RAW_EVENT_FIELDS]
    detail = test_events_raw.merge(
        suite.labels[["record_id", "is_attack", "attack_type", "attack_id"]], on="record_id", how="left",
    )
    detail = detail.merge(
        suite.users[["user_id", "department", "role", "privilege_level"]], on="user_id", how="left",
    )
    detail = detail.merge(
        suite.attack_meta[["attack_id", "severity", "rationale", "mitre_technique_ids", "mitre_tactic"]],
        on="attack_id", how="left",
    )
    detail = detail.merge(suite.test_features, on="record_id", how="left")
    detail = detail.merge(
        xgb_test_scores.rename(columns={"xgb_anomaly_score": "score", "xgb_predicted_class": "predicted_class"}),
        on="record_id", how="left",
    )
    return detail


@hydra.main(version_base=None, config_path="../configs", config_name="small_dev")
def main(cfg: DictConfig) -> None:
    run_dir = PROJECT_ROOT / cfg.run.output_dir
    out_dir = _out_dir(cfg)
    dash_cfg = cfg.dashboard
    shap_cfg = cfg.models.shap_explainability

    include_hf = bool(cfg.evaluation.get("include_hf", True))
    # Same checkpoint directory evaluation/run_evaluation.py uses -- a prior
    # `run_evaluation` run's completed stages are reused here too, and vice
    # versa. See evaluation/model_suite.py's module docstring.
    checkpoint_dir = run_dir / "checkpoints"
    print("Building model suite (reuses evaluation/model_suite.py -- identical numbers to the Phase 3 report)...")
    suite = build_model_suite(run_dir, cfg, include_hf=include_hf, checkpoint_dir=checkpoint_dir)

    # -- 1. model comparison + attack-type recall (six-criteria panel) --
    comparison_table(suite.evaluations).to_parquet(out_dir / "model_comparison.parquet", index=False)
    attack_type_recall_table(suite.evaluations).to_parquet(out_dir / "attack_type_recall.parquet", index=False)

    # -- 1b. primary model's REAL per-class classification report (precision/
    # recall/f1/support per attack type, from evaluate_model()'s genuine
    # multi-class metrics) -- lets the recommendation engine ground its
    # confidence language in this model's actual backtested performance for
    # the predicted class, instead of a fabricated per-event confidence
    # score. See dashboard/app.py's render_recommendation().
    primary_classification_report = suite.evaluation_for(PRIMARY_MODEL_NAME).multiclass_report
    if primary_classification_report is not None:
        primary_classification_report.reset_index(names="class_name").to_parquet(
            out_dir / "classification_report.parquet", index=False,
        )

    # -- 2. denormalized per-test-event detail table --
    # Per-class thresholds (tuned in model_suite.py's XGBoost loop, out-of-
    # fold on the TRAIN split only) so the dashboard's predicted_class
    # matches the evaluation report's, not a plain-argmax version of it.
    primary_class_thresholds = suite.evaluation_for(PRIMARY_MODEL_NAME).class_thresholds
    xgb_test_scores = score_xgboost(suite.primary_model, suite.test_features, class_thresholds=primary_class_thresholds)
    detail = _build_test_detail(suite, xgb_test_scores)

    # -- 3. threshold-slider bounds, from the TRAIN-split score distribution
    # only (never test labels) -- the same discipline evaluate_model() uses
    # for the report's own single operating threshold, which is this
    # slider's default position. --
    xgb_train_scores = score_xgboost(suite.primary_model, suite.train_features)["xgb_anomaly_score"]
    slider_min = float(np.percentile(xgb_train_scores, float(dash_cfg.threshold_slider_min_percentile)))
    slider_max = float(np.percentile(xgb_train_scores, float(dash_cfg.threshold_slider_max_percentile)))
    default_threshold = suite.evaluation_for(PRIMARY_MODEL_NAME).threshold
    print(f"Threshold slider range: [{slider_min:.4f}, {slider_max:.4f}], default={default_threshold:.4f}")

    # -- 4. exact SHAP (offline batch, over a sample of flagged events) --
    flagged_ids = select_flagged_sample(
        detail, "score", slider_min, int(shap_cfg.max_flagged_samples) or None, int(shap_cfg.sample_seed),
    )
    print(f"Explaining {len(flagged_ids)} flagged test events (score >= slider_min, capped by config)...")
    shap_explanations = explain_flagged_events(
        suite.primary_model, suite.test_features, xgb_test_scores["xgb_predicted_class"],
        flagged_ids, top_k=int(shap_cfg.top_k_features),
    )
    shap_explanations = explanations_to_json_column(shap_explanations)
    shap_explanations = shap_explanations.rename(columns={
        "top_features": "shap_top_features_json", "explanation": "shap_explanation",
    }).drop(columns=["predicted_class"])

    # -- 5. lightweight streaming approximation, over the SAME flagged
    # sample -- shown alongside exact SHAP in the dashboard so the
    # documented tradeoff (constraint #4) is visible, not hidden. --
    global_weights = compute_global_feature_weights(suite.primary_model, suite.test_features, flagged_ids)
    baseline = compute_feature_baseline(suite.train_features)
    approx_explainer = StreamingApproxExplainer(global_weights, baseline)
    approx_explanations = explain_stream_batch(approx_explainer, suite.test_features, flagged_ids, top_k=int(shap_cfg.top_k_features))
    approx_explanations["top_features"] = approx_explanations["top_features"].apply(json.dumps)
    approx_explanations = approx_explanations.rename(columns={
        "top_features": "streaming_approx_top_features_json", "explanation": "streaming_approx_explanation",
    })

    detail = detail.merge(shap_explanations, on="record_id", how="left")
    detail = detail.merge(approx_explanations, on="record_id", how="left")
    detail.to_parquet(out_dir / "test_events_detail.parquet", index=False)

    # -- 5b. zero-knowledge threshold proofs, over a small SAMPLE of the
    # flagged events (see configs/models/default.yaml's zkp: block for why
    # this is a sample, not every flagged row -- each proof takes a few
    # seconds of hand-rolled 2048-bit-group crypto). Proves "this record's
    # score >= the operating threshold" without the proof itself revealing
    # the score; see zkp/detection_proof.py and docs/zero_knowledge_proofs.md. --
    zkp_cfg = cfg.models.get("zkp")
    if zkp_cfg is not None and bool(zkp_cfg.enabled):
        # NOTE: intentionally re-selects from `detail` at `default_threshold`
        # rather than reusing `flagged_ids` (which was selected at
        # `slider_min`, a much looser cutoff for the SHAP sample) -- a proof
        # claims "score >= default_threshold", so only records that actually
        # clear that exact threshold can have one constructed at all.
        truly_flagged = detail.loc[detail["score"] >= default_threshold, "record_id"]
        n_available = len(truly_flagged)
        n_sample = min(int(zkp_cfg.max_proof_samples), n_available) if zkp_cfg.max_proof_samples else n_available
        sample_ids = (
            truly_flagged.sample(n=n_sample, random_state=int(zkp_cfg.sample_seed)).tolist()
            if n_sample < n_available
            else truly_flagged.tolist()
        )
        print(f"Generating {len(sample_ids)} zero-knowledge threshold proofs (of {n_available} flagged at threshold, sampled)...")
        scores_by_id = detail.set_index("record_id")["score"]
        proofs = {}
        for record_id in sample_ids:
            proof = generate_detection_proof(
                score=float(scores_by_id.loc[record_id]),
                threshold=default_threshold,
                record_id=str(record_id),
                model_name=PRIMARY_MODEL_NAME,
                scale=int(zkp_cfg.scale),
                n_bits=int(zkp_cfg.n_bits),
            )
            proofs[str(record_id)] = proof.to_dict()
        zkp_artifact = {
            "model_name": PRIMARY_MODEL_NAME,
            "threshold": default_threshold,
            "scale": int(zkp_cfg.scale),
            "n_bits": int(zkp_cfg.n_bits),
            "n_flagged_total": n_available,
            "n_proofs": len(proofs),
            "proofs": proofs,
        }
        (out_dir / "zkp_proofs.json").write_text(json.dumps(zkp_artifact), encoding="utf-8")

    # -- 6. per-user feature history (train + test, chronological) for the
    # drill-down trend chart -- doesn't care about the split, only order. --
    history = suite.events[["record_id", "user_id", "timestamp"]].merge(suite.features, on="record_id")
    history = history.merge(suite.users[["user_id", "department"]], on="user_id", how="left")
    history = history.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    history.to_parquet(out_dir / "feature_history.parquet", index=False)

    # -- 7. run summary + filter vocabularies --
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "run_name": str(cfg.run.name),
        "n_users": int(len(suite.users)),
        "n_events": int(len(suite.events)),
        "n_train": suite.split.n_train,
        "n_test": suite.split.n_test,
        "split_summary": suite.split.summary(),
        "primary_model": PRIMARY_MODEL_NAME,
        "threshold_slider_min": slider_min,
        "threshold_slider_max": slider_max,
        "threshold_default": default_threshold,
        "n_flagged_explained": len(flagged_ids),
        "feature_columns": FEATURE_COLUMNS,
        "departments": DEPARTMENTS,
        "attack_types": ATTACK_TYPES,
        "severities": SEVERITIES,
        "drift_eval": suite.drift_eval.to_dict(orient="records"),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote dashboard data to {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
