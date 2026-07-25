"""Six-criteria evaluation report (per the problem statement's evaluation
section): detection accuracy, false positives, explainability, scalability,
classification, design. The rule-based baseline appears in every table
alongside every other model, per constraint #3.

This module computes the numeric criteria (accuracy, false-positives/day,
MTTD, scalability, classification) for any model exposing a
`record_id -> anomaly_score` scoring table (and, for models that produce
one, a `record_id -> predicted_class` table for real multi-class metrics);
explainability and design are narrative criteria documented in the phase
reports rather than numbers here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelEvaluation:
    name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    mcc: float
    roc_auc: float
    pr_auc: float
    false_positives_per_day: float
    mttd_days: float | None
    mttd_coverage: float
    n_flagged: int
    n_test: int
    attack_type_recall: dict[str, float] = field(default_factory=dict)
    multiclass_report: pd.DataFrame | None = None
    class_thresholds: dict[str, float] = field(default_factory=dict)


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    has_both_classes = len(set(y_true)) > 1
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if has_both_classes else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_score) if has_both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, y_score) if has_both_classes else float("nan"),
    }


def _false_positives_per_day(y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.Series) -> float:
    n_fp = int(((y_pred == 1) & (y_true == 0)).sum())
    span_days = (timestamps.max() - timestamps.min()).total_seconds() / 86400.0
    return n_fp / span_days if span_days > 0 else float("nan")


def _attack_type_recall(y_pred: np.ndarray, attack_types: pd.Series) -> dict[str, float]:
    """Per-attack-type detection recall: of events belonging to each attack
    type, what fraction did this model flag as anomalous. Useful even for
    models WITHOUT genuine multi-class output (the rule baseline,
    Isolation Forest) -- for models that DO have one (XGBoost, HF, River),
    `multiclass_report` below is the real classification metric.
    """
    out: dict[str, float] = {}
    df = pd.DataFrame({"pred": y_pred, "attack_type": attack_types})
    for attack_type, group in df.dropna(subset=["attack_type"]).groupby("attack_type"):
        out[str(attack_type)] = float(group["pred"].mean())
    return out


def _mean_time_to_detection(
    y_pred: np.ndarray, timestamps: pd.Series, attack_ids: pd.Series, attack_meta: pd.DataFrame,
) -> tuple[float | None, float]:
    """Groups flagged events by attack CAMPAIGN (not individual event):
    for each campaign with at least one test-split event, finds the first
    flagged event's lag (in days) after the campaign's ground-truth
    `start_time`. Returns (mean lag over DETECTED campaigns, coverage =
    fraction of campaigns with a test-split event that were detected at
    all) -- both reported, since a model could have a great mean lag on
    the few campaigns it catches while missing most of them entirely.
    """
    df = pd.DataFrame({"pred": y_pred, "timestamp": timestamps.to_numpy(), "attack_id": attack_ids.to_numpy()})
    df = df.dropna(subset=["attack_id"])
    if len(df) == 0 or attack_meta is None or len(attack_meta) == 0:
        return None, float("nan")

    campaign_start = attack_meta.set_index("attack_id")["start_time"]
    lags: list[float] = []
    n_campaigns = 0
    n_detected = 0
    for attack_id, group in df.groupby("attack_id"):
        n_campaigns += 1
        flagged = group[group["pred"] == 1]
        if len(flagged) == 0:
            continue
        n_detected += 1
        start = campaign_start.get(attack_id)
        if start is None:
            continue
        lag_days = (flagged["timestamp"].min() - start).total_seconds() / 86400.0
        lags.append(max(lag_days, 0.0))

    mean_lag = float(np.mean(lags)) if lags else None
    coverage = n_detected / n_campaigns if n_campaigns > 0 else float("nan")
    return mean_lag, coverage


def build_multiclass_report(y_true_class: pd.Series, y_pred_class: pd.Series) -> pd.DataFrame:
    """Real per-class precision/recall/F1 + confusion counts (via
    `support`), for models that produce genuine multi-class predictions
    (XGBoost, HF, River) -- NOT available for the rule baseline / Isolation
    Forest, which only ever output a binary anomaly flag.
    """
    report = classification_report(y_true_class, y_pred_class, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.round(4)


def evaluate_model(
    name: str,
    train_scores: pd.Series,
    test_scores: pd.Series,
    test_labels: pd.DataFrame,
    test_timestamps: pd.Series,
    operating_threshold_percentile: float,
    attack_meta: pd.DataFrame | None = None,
    predicted_class: pd.Series | None = None,
) -> ModelEvaluation:
    """`train_scores`/`test_scores` are anomaly scores (higher = more
    anomalous) aligned to `test_labels`/`test_timestamps` (which must
    include `is_attack`, `attack_type`, `attack_id`) by position. The
    operating threshold is chosen from the TRAIN score distribution only
    (never test labels), per constraint #1.

    `attack_meta` (attack_id -> start_time, ...) enables MTTD; omit to skip
    it. `predicted_class` (aligned to test_labels) enables a real
    multi-class classification report; omit for models without one.
    """
    threshold = float(np.percentile(train_scores.to_numpy(), operating_threshold_percentile))
    y_true = test_labels["is_attack"].astype(int).to_numpy()
    y_score = test_scores.to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    metrics = _binary_metrics(y_true, y_score, y_pred)
    fp_per_day = _false_positives_per_day(y_true, y_pred, test_timestamps)
    attack_type_recall = _attack_type_recall(y_pred, test_labels["attack_type"])
    mttd_days, mttd_coverage = _mean_time_to_detection(
        y_pred, test_timestamps, test_labels["attack_id"] if "attack_id" in test_labels.columns else pd.Series([None] * len(test_labels)), attack_meta,
    )

    mc_report = None
    if predicted_class is not None:
        y_true_class = test_labels["attack_type"].fillna("benign")
        mc_report = build_multiclass_report(y_true_class, predicted_class)

    return ModelEvaluation(
        name=name,
        threshold=threshold,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        mcc=metrics["mcc"],
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        false_positives_per_day=fp_per_day,
        mttd_days=mttd_days,
        mttd_coverage=mttd_coverage,
        n_flagged=int(y_pred.sum()),
        n_test=len(y_true),
        attack_type_recall=attack_type_recall,
        multiclass_report=mc_report,
    )


def comparison_table(evaluations: list[ModelEvaluation]) -> pd.DataFrame:
    rows = []
    for ev in evaluations:
        rows.append({
            "model": ev.name,
            "precision": round(ev.precision, 4),
            "recall": round(ev.recall, 4),
            "f1": round(ev.f1, 4),
            "mcc": round(ev.mcc, 4) if not np.isnan(ev.mcc) else None,
            "roc_auc": round(ev.roc_auc, 4) if not np.isnan(ev.roc_auc) else None,
            "pr_auc (headline)": round(ev.pr_auc, 4) if not np.isnan(ev.pr_auc) else None,
            "false_positives_per_day": round(ev.false_positives_per_day, 2),
            "mttd_days": round(ev.mttd_days, 2) if ev.mttd_days is not None else None,
            "mttd_coverage": round(ev.mttd_coverage, 4) if not np.isnan(ev.mttd_coverage) else None,
            "n_flagged": ev.n_flagged,
            "n_test": ev.n_test,
        })
    return pd.DataFrame(rows)


def attack_type_recall_table(evaluations: list[ModelEvaluation]) -> pd.DataFrame:
    all_types = sorted({t for ev in evaluations for t in ev.attack_type_recall})
    rows = []
    for ev in evaluations:
        row = {"model": ev.name}
        for t in all_types:
            row[t] = round(ev.attack_type_recall.get(t, float("nan")), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def render_markdown_report(
    comparison_df: pd.DataFrame,
    attack_type_df: pd.DataFrame,
    scalability: dict[str, float],
    split_summary: str,
    multiclass_reports: dict[str, pd.DataFrame] | None = None,
    class_thresholds: dict[str, dict[str, float]] | None = None,
) -> str:
    lines = [
        "# Six-Criteria Evaluation Report (Phase 3, small_dev scale)",
        "",
        "> Synthetic data. Not derived from or validated against real organizational logs. For benchmarking detection methods only.",
        "",
        f"**Split:** {split_summary}",
        "",
        "## 1. Detection accuracy (+ 2. false positives, MTTD)",
        "",
        "PR-AUC is the headline metric given class imbalance (per the problem statement). "
        "`mttd_days`/`mttd_coverage`: mean days from an attack campaign's true start to its "
        "first flagged event, and the fraction of campaigns detected at all (a model can have "
        "a great mean lag on the few campaigns it catches while missing most of them -- both "
        "numbers are reported so that isn't hidden).",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## 5. Classification",
        "",
        "### Per-attack-type detection recall (all models -- a proxy for models without genuine multi-class output)",
        "",
        attack_type_df.to_markdown(index=False),
        "",
    ]
    if multiclass_reports:
        lines.append("### Real multi-class classification report (models with genuine class predictions)")
        lines.append("")
        for name, report_df in multiclass_reports.items():
            lines.append(f"**{name}**")
            lines.append("")
            lines.append(report_df.to_markdown())
            lines.append("")
    if class_thresholds and any(class_thresholds.values()):
        lines.append(
            "### Per-class decision thresholds (rare-class recall fix, see "
            "docs/phase_5_recall_investigation.md)"
        )
        lines.append("")
        lines.append(
            "Tuned via out-of-fold CV on the TRAIN split only (never test labels); a class "
            "only gets a threshold here if it beat plain argmax on held-out F1 during tuning. "
            "Rows with no entries fell back to plain argmax -- honestly, not silently."
        )
        lines.append("")
        for name, thresholds in class_thresholds.items():
            if thresholds:
                pretty = ", ".join(f"{cls}={t:.6f}" for cls, t in thresholds.items())
                lines.append(f"- **{name}**: {pretty}")
        lines.append("")
    lines.extend([
        "## 4. Scalability",
        "",
        f"- Feature computation throughput: {scalability.get('events_per_sec', float('nan')):.0f} events/sec",
        f"- Feature computation wall-clock time for this run: {scalability.get('feature_compute_seconds', float('nan')):.1f}s "
        f"over {scalability.get('n_events', 0)} events",
        "",
        "## 3. Explainability",
        "",
        "The rule-based baseline is inherently explainable: `rule_risk_score` is a sum of "
        "3 named, human-readable flags -- the explanation *is* the score. Every other model's "
        "engineered-feature score has no per-event explanation yet; offline SHAP (exact, "
        "batch-only) and a lightweight streaming approximation are Phase 4 scope, per "
        "constraint #4 -- not silently skipped, explicitly deferred.",
        "",
        "## 6. Design",
        "",
        "See `docs/phase_3_report.md` for architecture, modularity, and reproducibility notes.",
        "",
    ])
    return "\n".join(lines)
