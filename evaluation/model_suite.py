"""Shared model-training-and-scoring suite -- the single place that trains
and scores every model in this project's comparison table. Both
`evaluation/run_evaluation.py` (the six-criteria report) and
`dashboard/prepare_data.py` (the dashboard's precomputed artifacts) call
`build_model_suite()` rather than each re-implementing model training, so
the dashboard's numbers and the evaluation report's numbers can never
silently drift apart -- the same discipline `feature_engineering/pipeline.py`
already applies to features (one `update()`, not two implementations).

`ModelSuiteResult.test_scores` normalizes every model's differently-named
score/prediction columns (`rule_risk_score`, `xgb_anomaly_score`,
`xgb_predicted_class`, ...) into one consistent shape per model --
`record_id`, `score`, and optionally `predicted_class` -- keyed by
`record_id` (not row position), so downstream joins are safe regardless of
any internal reordering a model's own scoring function does.

Checkpointing (`checkpoint_dir`, optional): at full (Scale-up) scale, a
single run can take well over an hour, and this dev environment has shown
it can kill a long-running background process for reasons outside this
code's control (observed directly during the Scale-up stage -- see
docs/scale_up_report.md). When `checkpoint_dir` is given, the feature
table (the single most expensive stage) and each model's
(ModelEvaluation, test_scores) pair are written to disk immediately after
being computed; a re-run of `build_model_suite()` with the SAME
`checkpoint_dir` loads whatever already exists instead of recomputing it.
This does not skip or shorten any model's real training/scoring on a
clean run -- every model still runs in full exactly once -- it only makes
an INTERRUPTED run resumable without redoing already-finished work.
Trained model OBJECTS are not checkpointed (only their evaluation output
is); a resumed model's entry in `ModelSuiteResult.models` will be absent
-- fine for `run_evaluation.py`, which never reads `.models`, but a real
limitation for any future caller (e.g. `dashboard.prepare_data`) that
needs the live model object for a checkpoint-skipped stage.
"""
from __future__ import annotations

# IMPORTANT: torch must be imported before pandas/pyarrow anywhere in this
# process -- see models/README.md / evaluation/run_evaluation.py's
# identical comment (a genuine Windows DLL conflict between pyarrow's
# bundled Arrow C++ runtime and torch's bundled libraries).
import torch  # noqa: F401

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from transformers import BertForSequenceClassification, BertTokenizerFast

from drift_detection.adwin_detector import bin_by_event_count, evaluate_against_drift_log, run_adwin
from evaluation.chronological_split import ChronologicalSplit, apply_split, chronological_split
from evaluation.report import ModelEvaluation, evaluate_model
from feature_engineering.cold_start import apply_cold_start_priors, compute_department_priors
from feature_engineering.pipeline import compute_feature_table, compute_feature_table_resumable
from models.baseline import compute_rule_based_scores
from models.hf_classifier import build_sequences, score_hf_classifier, subsample_for_training, train_hf_classifier
from models.isolation_forest import score_isolation_forest, train_isolation_forest
from models.xgboost_classifier import score_xgboost, train_xgboost, tune_class_thresholds
from online_learning.river_model import run_river_online

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts" / "hf_bert_tiny"

# The model whose exact SHAP explanations power the dashboard's flagged-
# events table -- the best all-around performer per docs/phase_3_report.md
# (highest F1/MCC/ROC-AUC/PR-AUC/MTTD-coverage of the full set).
PRIMARY_MODEL_NAME = "xgboost_smote"


@dataclass
class ModelSuiteResult:
    users: pd.DataFrame
    events: pd.DataFrame
    labels: pd.DataFrame
    attack_meta: pd.DataFrame
    drift_log: pd.DataFrame
    features: pd.DataFrame
    split: ChronologicalSplit
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    base_test: pd.DataFrame
    models: dict[str, object] = field(default_factory=dict)
    test_scores: dict[str, pd.DataFrame] = field(default_factory=dict)
    evaluations: list[ModelEvaluation] = field(default_factory=list)
    multiclass_reports: dict[str, pd.DataFrame] = field(default_factory=dict)
    feature_elapsed_seconds: float = 0.0
    drift_eval: pd.DataFrame | None = None

    @property
    def primary_model(self):
        return self.models[PRIMARY_MODEL_NAME]

    def evaluation_for(self, name: str) -> ModelEvaluation:
        return next(ev for ev in self.evaluations if ev.name == name)


def _base_frame(events_slim: pd.DataFrame, labels_slim: pd.DataFrame, split: ChronologicalSplit, part: str) -> pd.DataFrame:
    return apply_split(events_slim, split, part).merge(labels_slim, on="record_id", how="left")


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def _load_features_checkpoint(checkpoint_dir: Path | None) -> tuple[pd.DataFrame | None, float | None]:
    if checkpoint_dir is None:
        return None, None
    parquet_path = checkpoint_dir / "features.parquet"
    meta_path = checkpoint_dir / "features_meta.json"
    if parquet_path.exists() and meta_path.exists():
        features = pd.read_parquet(parquet_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return features, float(meta["feature_elapsed_seconds"])
    return None, None


def _save_features_checkpoint(checkpoint_dir: Path | None, features: pd.DataFrame, feature_elapsed: float) -> None:
    if checkpoint_dir is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(checkpoint_dir / "features.parquet", index=False)
    (checkpoint_dir / "features_meta.json").write_text(
        json.dumps({"feature_elapsed_seconds": feature_elapsed}), encoding="utf-8",
    )


def _load_model_checkpoint(checkpoint_dir: Path | None, name: str) -> tuple[ModelEvaluation | None, pd.DataFrame | None]:
    if checkpoint_dir is None:
        return None, None
    eval_path = checkpoint_dir / f"{name}_eval.pkl"
    scores_path = checkpoint_dir / f"{name}_scores.parquet"
    if eval_path.exists() and scores_path.exists():
        with open(eval_path, "rb") as f:
            evaluation = pickle.load(f)
        scores = pd.read_parquet(scores_path)
        return evaluation, scores
    return None, None


def _save_model_checkpoint(checkpoint_dir: Path | None, name: str, evaluation: ModelEvaluation, scores: pd.DataFrame) -> None:
    if checkpoint_dir is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / f"{name}_eval.pkl", "wb") as f:
        pickle.dump(evaluation, f)
    scores.to_parquet(checkpoint_dir / f"{name}_scores.parquet", index=False)


def build_model_suite(
    run_dir: Path,
    cfg: DictConfig,
    include_hf: bool = True,
    checkpoint_dir: Path | None = None,
    verbose: bool = True,
) -> ModelSuiteResult:
    users = pd.read_parquet(run_dir / "users.parquet")
    events = pd.read_parquet(run_dir / "events.parquet")
    labels = pd.read_parquet(run_dir / "labels.parquet")
    attack_meta = pd.read_parquet(run_dir / "attacks.parquet")
    drift_log = pd.read_csv(run_dir / "drift_log.csv")

    features, feature_elapsed = _load_features_checkpoint(checkpoint_dir)
    if features is not None:
        _log(f"[checkpoint] loaded features ({len(features)} rows) -- skipping feature computation", verbose)
    else:
        t0 = time.time()
        if checkpoint_dir is not None:
            # Resumable: an interruption mid-computation (the single
            # longest uninterrupted stage at full scale) loses at most
            # RESUMABLE_CHECKPOINT_INTERVAL events, not the whole stage --
            # see feature_engineering/pipeline.py's module-level constant
            # and docs/scale_up_report.md.
            features = compute_feature_table_resumable(events, users, cfg, checkpoint_dir, verbose=verbose)
        else:
            features = compute_feature_table(events, users, cfg)
        feature_elapsed = time.time() - t0
        _log(f"computed {len(features)} feature rows in {feature_elapsed:.1f}s", verbose)

        split_for_priors = chronological_split(events, train_frac=float(cfg.evaluation.train_frac))
        train_features_raw = apply_split(features, split_for_priors, "train")
        train_events_raw = apply_split(events, split_for_priors, "train")
        priors = compute_department_priors(train_features_raw, train_events_raw, users)
        features = apply_cold_start_priors(features, events, users, priors, cfg)
        _save_features_checkpoint(checkpoint_dir, features, feature_elapsed)

    # Split is cheap and purely deterministic from `events` -- always
    # recomputed fresh (never checkpointed), even when features were loaded
    # from a checkpoint.
    split = chronological_split(events, train_frac=float(cfg.evaluation.train_frac))

    labels_slim = labels[["record_id", "is_attack", "attack_type", "attack_id"]]
    events_slim = events[["record_id", "timestamp"]]
    base_train = _base_frame(events_slim, labels_slim, split, "train")
    base_test = _base_frame(events_slim, labels_slim, split, "test")

    train_features = apply_split(features, split, "train")
    test_features = apply_split(features, split, "test")

    result = ModelSuiteResult(
        users=users, events=events, labels=labels, attack_meta=attack_meta, drift_log=drift_log,
        features=features, split=split, train_features=train_features, test_features=test_features,
        base_test=base_test, feature_elapsed_seconds=feature_elapsed,
    )

    threshold_pct = float(cfg.evaluation.operating_threshold_percentile)

    # -- rule-based baseline --
    cached_eval, cached_scores = _load_model_checkpoint(checkpoint_dir, "rule_based_baseline")
    if cached_eval is not None:
        result.evaluations.append(cached_eval)
        result.test_scores["rule_based_baseline"] = cached_scores
        _log("[checkpoint] loaded rule_based_baseline", verbose)
    else:
        rule_scores = compute_rule_based_scores(events, cfg)
        rule_train_scores = base_train.merge(rule_scores, on="record_id", how="left")["rule_risk_score"]
        rule_test_frame = base_test.merge(rule_scores, on="record_id", how="left")
        ev = evaluate_model(
            "rule_based_baseline", rule_train_scores, rule_test_frame["rule_risk_score"],
            rule_test_frame[["is_attack", "attack_type", "attack_id"]], rule_test_frame["timestamp"],
            threshold_pct, attack_meta=attack_meta,
        )
        ts = pd.DataFrame({"record_id": rule_test_frame["record_id"], "score": rule_test_frame["rule_risk_score"]})
        result.evaluations.append(ev)
        result.test_scores["rule_based_baseline"] = ts
        _save_model_checkpoint(checkpoint_dir, "rule_based_baseline", ev, ts)
        _log("evaluated: rule_based_baseline", verbose)

    # -- Isolation Forest --
    cached_eval, cached_scores = _load_model_checkpoint(checkpoint_dir, "isolation_forest")
    if cached_eval is not None:
        result.evaluations.append(cached_eval)
        result.test_scores["isolation_forest"] = cached_scores
        _log("[checkpoint] loaded isolation_forest", verbose)
    else:
        iforest_model = train_isolation_forest(train_features, cfg)
        result.models["isolation_forest"] = iforest_model
        iforest_train_scores = base_train.merge(score_isolation_forest(iforest_model, train_features), on="record_id", how="left")["iforest_anomaly_score"]
        iforest_test_frame = base_test.merge(score_isolation_forest(iforest_model, test_features), on="record_id", how="left")
        ev = evaluate_model(
            "isolation_forest", iforest_train_scores, iforest_test_frame["iforest_anomaly_score"],
            iforest_test_frame[["is_attack", "attack_type", "attack_id"]], iforest_test_frame["timestamp"],
            threshold_pct, attack_meta=attack_meta,
        )
        ts = pd.DataFrame({"record_id": iforest_test_frame["record_id"], "score": iforest_test_frame["iforest_anomaly_score"]})
        result.evaluations.append(ev)
        result.test_scores["isolation_forest"] = ts
        _save_model_checkpoint(checkpoint_dir, "isolation_forest", ev, ts)
        _log("evaluated: isolation_forest", verbose)

    # -- XGBoost, 3 imbalance-handling conditions --
    for method in cfg.models.xgboost.imbalance_methods:
        name = f"xgboost_{method}"
        cached_eval, cached_scores = _load_model_checkpoint(checkpoint_dir, name)
        if cached_eval is not None:
            result.evaluations.append(cached_eval)
            result.test_scores[name] = cached_scores
            result.multiclass_reports[name] = cached_eval.multiclass_report
            _log(f"[checkpoint] loaded {name}", verbose)
            continue

        model = train_xgboost(train_features, base_train[["record_id", "attack_type"]], str(method), cfg)
        result.models[name] = model

        # Per-class decision thresholds for the rare classes (lateral_movement,
        # device_spoofing) that plain argmax structurally under-detects at low
        # prevalence -- tuned via out-of-fold CV on the TRAIN split only, so
        # this never touches test labels. See models/xgboost_classifier.py's
        # tune_class_thresholds() and docs/phase_5_recall_investigation.md.
        class_thresholds = tune_class_thresholds(
            train_features, base_train[["record_id", "attack_type"]], str(method), cfg,
        )
        _log(f"  tuned class thresholds for {name}: {class_thresholds}", verbose)

        train_scores_df = score_xgboost(model, train_features)
        test_scores_df = score_xgboost(model, test_features, class_thresholds=class_thresholds)
        train_scores = base_train.merge(train_scores_df, on="record_id", how="left")["xgb_anomaly_score"]
        test_frame = base_test.merge(test_scores_df, on="record_id", how="left")
        ev = evaluate_model(
            name, train_scores, test_frame["xgb_anomaly_score"],
            test_frame[["is_attack", "attack_type", "attack_id"]], test_frame["timestamp"],
            threshold_pct, attack_meta=attack_meta, predicted_class=test_frame["xgb_predicted_class"],
        )
        ev.class_thresholds = class_thresholds
        ts = pd.DataFrame({
            "record_id": test_frame["record_id"], "score": test_frame["xgb_anomaly_score"],
            "predicted_class": test_frame["xgb_predicted_class"],
        })
        result.evaluations.append(ev)
        result.multiclass_reports[name] = ev.multiclass_report
        result.test_scores[name] = ts
        _save_model_checkpoint(checkpoint_dir, name, ev, ts)
        _log(f"evaluated: {name}", verbose)

    # -- River online learner --
    cached_eval, cached_scores = _load_model_checkpoint(checkpoint_dir, "river_online")
    if cached_eval is not None:
        result.evaluations.append(cached_eval)
        result.test_scores["river_online"] = cached_scores
        result.multiclass_reports["river_online"] = cached_eval.multiclass_report
        _log("[checkpoint] loaded river_online", verbose)
    else:
        river_scores = run_river_online(events, users, labels, split.train_record_ids, cfg)
        river_train_scores = base_train.merge(river_scores, on="record_id", how="left")["river_anomaly_score"]
        river_test_frame = base_test.merge(river_scores, on="record_id", how="left")
        ev = evaluate_model(
            "river_online", river_train_scores, river_test_frame["river_anomaly_score"],
            river_test_frame[["is_attack", "attack_type", "attack_id"]], river_test_frame["timestamp"],
            threshold_pct, attack_meta=attack_meta, predicted_class=river_test_frame["river_predicted_class"],
        )
        ts = pd.DataFrame({
            "record_id": river_test_frame["record_id"], "score": river_test_frame["river_anomaly_score"],
            "predicted_class": river_test_frame["river_predicted_class"],
        })
        result.evaluations.append(ev)
        result.multiclass_reports["river_online"] = ev.multiclass_report
        result.test_scores["river_online"] = ts
        _save_model_checkpoint(checkpoint_dir, "river_online", ev, ts)
        _log("evaluated: river_online", verbose)

    # -- fine-tuned HF bert-tiny (optional -- skippable for fast dashboard-
    # data iteration; loads the saved artifact rather than retraining) --
    if include_hf:
        cached_eval, cached_scores = _load_model_checkpoint(checkpoint_dir, "hf_bert_tiny")
        if cached_eval is not None:
            result.evaluations.append(cached_eval)
            result.test_scores["hf_bert_tiny"] = cached_scores
            result.multiclass_reports["hf_bert_tiny"] = cached_eval.multiclass_report
            _log("[checkpoint] loaded hf_bert_tiny", verbose)
        else:
            hf_sequences = build_sequences(features, events, int(cfg.models.hf_classifier.sequence_length_k))
            if HF_ARTIFACT_DIR.exists():
                hf_model = BertForSequenceClassification.from_pretrained(str(HF_ARTIFACT_DIR))
                hf_tokenizer = BertTokenizerFast.from_pretrained(str(HF_ARTIFACT_DIR))
                _log("loaded HF model artifact from disk", verbose)
            else:
                train_labels_full = base_train[["record_id", "attack_type"]]
                sub_seqs, sub_labels = subsample_for_training(hf_sequences, train_labels_full, cfg, seed=int(cfg.seed))
                hf_model, hf_tokenizer = train_hf_classifier(sub_seqs, sub_labels, cfg)
                HF_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                hf_model.save_pretrained(str(HF_ARTIFACT_DIR))
                hf_tokenizer.save_pretrained(str(HF_ARTIFACT_DIR))
                _log("trained and saved HF model artifact", verbose)
            result.models["hf_bert_tiny"] = hf_model

            train_seqs_for_scoring = hf_sequences[hf_sequences["record_id"].isin(split.train_record_ids)]
            test_seqs_for_scoring = hf_sequences[hf_sequences["record_id"].isin(split.test_record_ids)]
            hf_train_scores_df = score_hf_classifier(hf_model, hf_tokenizer, train_seqs_for_scoring, cfg)
            hf_test_scores_df = score_hf_classifier(hf_model, hf_tokenizer, test_seqs_for_scoring, cfg)
            hf_train_scores = base_train.merge(hf_train_scores_df, on="record_id", how="left")["hf_anomaly_score"]
            hf_test_frame = base_test.merge(hf_test_scores_df, on="record_id", how="left")
            ev = evaluate_model(
                "hf_bert_tiny", hf_train_scores, hf_test_frame["hf_anomaly_score"],
                hf_test_frame[["is_attack", "attack_type", "attack_id"]], hf_test_frame["timestamp"],
                threshold_pct, attack_meta=attack_meta, predicted_class=hf_test_frame["hf_predicted_class"],
            )
            ts = pd.DataFrame({
                "record_id": hf_test_frame["record_id"], "score": hf_test_frame["hf_anomaly_score"],
                "predicted_class": hf_test_frame["hf_predicted_class"],
            })
            result.evaluations.append(ev)
            result.multiclass_reports["hf_bert_tiny"] = ev.multiclass_report
            result.test_scores["hf_bert_tiny"] = ts
            _save_model_checkpoint(checkpoint_dir, "hf_bert_tiny", ev, ts)
            _log("evaluated: hf_bert_tiny", verbose)

    # -- ADWIN drift detection vs. ground truth --
    events_sorted = events.sort_values("timestamp").reset_index(drop=True)
    geo_merged = events_sorted[["record_id", "timestamp"]].merge(features[["record_id", "geo_distance_from_home_km"]], on="record_id")
    binned = bin_by_event_count(geo_merged["geo_distance_from_home_km"], geo_merged["timestamp"], int(cfg.models.adwin.bin_size))
    detected = run_adwin(binned, cfg)
    start_date = pd.Timestamp(cfg.events.start_date)
    result.drift_eval = evaluate_against_drift_log(detected, drift_log, start_date)
    _log("computed: ADWIN drift evaluation", verbose)

    return result
