"""Offline, exact SHAP explainability -- constraint #4's "exact SHAP is run
only in an offline batch explainability job over a sample of flagged
events," never in the streaming inference path (see
`explainability/streaming_approx.py` for that side of the tradeoff).

Explains the project's best-performing model, `xgboost_smote`
(docs/phase_3_report.md), via `shap.TreeExplainer` -- exact for tree
ensembles, not a model-agnostic approximation. For each flagged test event,
attributes the model's predicted class to its top contributing features
and renders a short plain-language explanation string, so every anomaly
score shipped to the dashboard comes with a reason, not just a number.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
from omegaconf import DictConfig
from xgboost import XGBClassifier

from explainability.feature_glossary import describe
from feature_engineering.pipeline import FEATURE_COLUMNS
from models.xgboost_classifier import ATTACK_TYPE_CLASSES


def select_flagged_sample(
    scored_test_frame: pd.DataFrame,
    score_col: str,
    threshold: float,
    max_samples: int | None,
    seed: int,
) -> list[str]:
    """`scored_test_frame` must have `record_id` and `score_col`. Returns
    the `record_id`s of every event at or above `threshold`, capped to
    `max_samples` via reproducible random sampling (not "first N" --
    avoids silently biasing the explained sample toward the start of the
    test window). `max_samples=None` explains every flagged event.
    """
    flagged = scored_test_frame.loc[scored_test_frame[score_col] >= threshold, "record_id"]
    if max_samples is not None and len(flagged) > max_samples:
        flagged = flagged.sample(n=max_samples, random_state=seed)
    return flagged.tolist()


def _shap_values_by_class(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Normalizes shap's multi-class TreeExplainer output, which varies by
    shap version, to one consistent shape: (n_samples, n_features, n_classes).
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        # Legacy API: one (n_samples, n_features) array per class.
        return np.stack(raw, axis=-1)
    values = np.asarray(raw)
    if values.ndim == 2:
        # Single-output model (shouldn't happen for this multi:softprob
        # classifier, but handled rather than assumed away).
        return values[:, :, None]
    return values


def _format_contributor(feature: str, shap_value: float, feature_value: float) -> str:
    return f"{feature}={feature_value:.3g} ({shap_value:+.3f})"


def _build_explanation(predicted_class: str, contributors: list[tuple[str, float, float]]) -> str:
    positive = [c for c in contributors if c[1] > 0]
    negative = [c for c in contributors if c[1] < 0]
    parts = [f"Flagged as {predicted_class}."]
    if positive:
        parts.append("Risk driven up by: " + ", ".join(_format_contributor(*c) for c in positive) + ".")
    if negative:
        parts.append("Offset by: " + ", ".join(_format_contributor(*c) for c in negative) + ".")
    if not positive and not negative:
        parts.append("No individual feature stood out; risk reflects a diffuse combination of small signals.")
    return " ".join(parts)


def explain_flagged_events(
    model: XGBClassifier,
    features: pd.DataFrame,
    predicted_class: pd.Series,
    record_ids: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """`features` must contain `record_id` + `FEATURE_COLUMNS` for at least
    every id in `record_ids`; `predicted_class` must be aligned to
    `features` by position (as `models.xgboost_classifier.score_xgboost`
    returns it). Returns one row per `record_id` with the predicted class,
    the top-`top_k` SHAP contributors (feature, value, shap_value,
    description), and a rendered plain-language `explanation` string.
    """
    features_reset = features.reset_index(drop=True)
    mask = features_reset["record_id"].isin(record_ids)
    subset = features_reset.loc[mask].reset_index(drop=True)
    pred_for_subset = predicted_class.reset_index(drop=True).loc[mask].reset_index(drop=True)

    X = subset[FEATURE_COLUMNS].to_numpy()
    shap_by_class = _shap_values_by_class(model, X)  # (n, n_features, n_classes)

    rows = []
    for i in range(len(subset)):
        pred_class = str(pred_for_subset.iloc[i])
        class_idx = ATTACK_TYPE_CLASSES.index(pred_class) if pred_class in ATTACK_TYPE_CLASSES else 0
        class_idx = min(class_idx, shap_by_class.shape[-1] - 1)
        row_shap = shap_by_class[i, :, class_idx]
        row_values = X[i]

        order = np.argsort(-np.abs(row_shap))[:top_k]
        contributors = [(FEATURE_COLUMNS[j], float(row_shap[j]), float(row_values[j])) for j in order]

        rows.append({
            "record_id": subset.loc[i, "record_id"],
            "predicted_class": pred_class,
            "top_features": [
                {
                    "feature": feat,
                    "shap_value": round(val, 6),
                    "feature_value": round(fval, 6),
                    "description": describe(feat),
                }
                for feat, val, fval in contributors
            ],
            "explanation": _build_explanation(pred_class, contributors),
        })

    return pd.DataFrame(rows, columns=["record_id", "predicted_class", "top_features", "explanation"])


def compute_global_feature_weights(
    model: XGBClassifier, features: pd.DataFrame, record_ids: list[str],
) -> dict[str, float]:
    """Precomputed ONCE, offline, from this same exact-SHAP batch job --
    the mean absolute SHAP magnitude per feature across the explained
    sample (all classes pooled, not just each row's predicted class, since
    this is meant as a general "how much does this feature usually matter"
    weight, not a per-class one), normalized to sum to 1.

    This is the bridge artifact `explainability/streaming_approx.py`
    consumes: the lightweight streaming approximation's feature weights
    come directly from real exact-SHAP output, not a separately-invented
    importance metric.
    """
    subset = features[features["record_id"].isin(record_ids)]
    X = subset[FEATURE_COLUMNS].to_numpy()
    shap_by_class = _shap_values_by_class(model, X)  # (n, n_features, n_classes)
    mean_abs = np.abs(shap_by_class).mean(axis=(0, 2))  # (n_features,)
    total = mean_abs.sum()
    if total <= 0:
        return {feat: 1.0 / len(FEATURE_COLUMNS) for feat in FEATURE_COLUMNS}
    return {feat: float(val / total) for feat, val in zip(FEATURE_COLUMNS, mean_abs)}


def explanations_to_json_column(explanations: pd.DataFrame) -> pd.DataFrame:
    """Serializes `top_features` (a list of dicts) to a JSON string, for
    writing to parquet/CSV artifacts that dashboard/prepare_data.py loads.
    """
    out = explanations.copy()
    out["top_features"] = out["top_features"].apply(json.dumps)
    return out
