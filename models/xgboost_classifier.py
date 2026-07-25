"""Supervised multi-class attack-type classifier (Phase 3) -- the first
model in this project that predicts WHICH attack type, not just
anomaly/not-anomaly, satisfying the "classification" evaluation criterion
directly rather than via a recall proxy (Phase 2b's stopgap).

Implements the three imbalance-handling conditions the problem statement
requires be compared against each other AND against no resampling:
"none", "class_weight" (sample weights via sklearn's balanced formula), and
"smote" (imbalanced-learn, applied strictly to the TRAIN split only -- SMOTE
is never applied to test data, which would leak synthetic neighbors of test
points into evaluation).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from feature_engineering.pipeline import FEATURE_COLUMNS

ATTACK_TYPE_CLASSES: list[str] = [
    "benign", "brute_force", "impossible_travel", "credential_misuse", "lateral_movement", "device_spoofing",
]
_CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(ATTACK_TYPE_CLASSES)}


def multiclass_labels(labels: pd.DataFrame) -> pd.Series:
    """`attack_type` with benign rows (NaN) filled to the literal 'benign'
    class, for multi-class targets.
    """
    return labels["attack_type"].fillna("benign")


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.map(_CLASS_TO_INDEX).to_numpy()


def decode_labels(y: np.ndarray) -> list[str]:
    return [ATTACK_TYPE_CLASSES[i] for i in y]


def _apply_smote(X: np.ndarray, y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE needs at least 2 samples of a class to interpolate between,
    and its k_neighbors must be < the smallest class count. Both are real
    risks at this project's small imbalance ratio (some attack types can
    have only a handful of train examples) -- handled by shrinking
    k_neighbors automatically, and falling back to plain random oversampling
    (not SMOTE's synthetic interpolation) for the pathological case of a
    class with fewer than 2 samples, rather than crashing.
    """
    counts = Counter(y)
    min_count = min(counts.values())
    if min_count < 2:
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=random_state)
    else:
        k_neighbors = max(1, min(5, min_count - 1))
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    return sampler.fit_resample(X, y)


def train_xgboost(train_features: pd.DataFrame, train_labels: pd.DataFrame, imbalance_method: str, cfg: DictConfig) -> XGBClassifier:
    xgb_cfg = cfg.models.xgboost
    if imbalance_method not in xgb_cfg.imbalance_methods:
        raise ValueError(f"Unknown imbalance_method {imbalance_method!r}, expected one of {list(xgb_cfg.imbalance_methods)}")

    X = train_features[FEATURE_COLUMNS].to_numpy()
    y = encode_labels(multiclass_labels(train_labels))
    sample_weight = None

    if imbalance_method == "smote":
        X, y = _apply_smote(X, y, int(xgb_cfg.random_state))
    elif imbalance_method == "class_weight":
        sample_weight = compute_sample_weight("balanced", y)
    # "none": no resampling, no weighting -- the direct comparison point.

    model = XGBClassifier(
        n_estimators=int(xgb_cfg.n_estimators),
        max_depth=int(xgb_cfg.max_depth),
        learning_rate=float(xgb_cfg.learning_rate),
        random_state=int(xgb_cfg.random_state),
        objective="multi:softprob",
        num_class=len(ATTACK_TYPE_CLASSES),
        eval_metric="mlogloss",
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _train_one_fold(X: np.ndarray, y: np.ndarray, imbalance_method: str, xgb_cfg: DictConfig) -> XGBClassifier:
    sample_weight = None
    if imbalance_method == "smote":
        X, y = _apply_smote(X, y, int(xgb_cfg.random_state))
    elif imbalance_method == "class_weight":
        sample_weight = compute_sample_weight("balanced", y)
    model = XGBClassifier(
        n_estimators=int(xgb_cfg.n_estimators),
        max_depth=int(xgb_cfg.max_depth),
        learning_rate=float(xgb_cfg.learning_rate),
        random_state=int(xgb_cfg.random_state),
        objective="multi:softprob",
        num_class=len(ATTACK_TYPE_CLASSES),
        eval_metric="mlogloss",
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _oof_probabilities(
    train_features: pd.DataFrame, train_labels: pd.DataFrame, imbalance_method: str, cfg: DictConfig, n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predicted probabilities on the TRAIN split only, via
    stratified k-fold CV -- lets threshold tuning below see genuinely
    held-out predictions without ever touching test labels (constraint #1:
    the operating decision is chosen without looking at test data). Each
    fold retrains fresh, including re-applying the imbalance method inside
    the fold only, so SMOTE-synthesized points from one fold never leak
    into another fold's held-out predictions.
    """
    X_all = train_features[FEATURE_COLUMNS].to_numpy()
    y_all = encode_labels(multiclass_labels(train_labels))
    xgb_cfg = cfg.models.xgboost

    class_counts = Counter(y_all)
    min_class_count = min(class_counts.values())
    # A class with fewer than n_splits train examples can't be stratified
    # into that many folds -- shrink n_splits to what's actually possible
    # rather than crashing; still at least 2 folds (a 1-fold "split" isn't
    # a held-out estimate at all).
    n_splits = max(2, min(n_splits, min_class_count))

    oof = np.zeros((len(y_all), len(ATTACK_TYPE_CLASSES)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(xgb_cfg.random_state))
    for fold_train_idx, fold_val_idx in skf.split(X_all, y_all):
        model = _train_one_fold(X_all[fold_train_idx], y_all[fold_train_idx], imbalance_method, xgb_cfg)
        oof[fold_val_idx] = model.predict_proba(X_all[fold_val_idx])
    return oof, y_all


def tune_class_thresholds(
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    imbalance_method: str,
    cfg: DictConfig,
    target_classes: tuple[str, ...] = ("lateral_movement", "device_spoofing"),
    n_splits: int = 5,
) -> dict[str, float]:
    """Per-class one-vs-rest probability thresholds that OVERRIDE plain
    argmax for rare classes whose raw probability can legitimately be low
    (competing against benign at 99%+ prevalence in a direct softmax
    contest) while still being far above that class's own noise floor --
    see docs/phase_5_recall_investigation.md for the measurement that
    motivated this.

    Tuned to maximize each target class's F1 (one-vs-rest) using
    out-of-fold predictions on the TRAIN split only -- never test labels.
    A class is only included in the returned dict if the best threshold
    found actually beats the plain-argmax F1 baseline on the same
    out-of-fold predictions; classes where no threshold helps are omitted
    so the decision rule honestly falls back to argmax for them rather
    than forcing a threshold that doesn't earn its keep.
    """
    oof, y_all = _oof_probabilities(train_features, train_labels, imbalance_method, cfg, n_splits)
    argmax_pred = oof.argmax(axis=1)

    thresholds: dict[str, float] = {}
    for cls in target_classes:
        idx = _CLASS_TO_INDEX[cls]
        y_binary = (y_all == idx).astype(int)
        if y_binary.sum() == 0:
            continue

        scores = oof[:, idx]
        baseline_f1 = f1_score(y_binary, (argmax_pred == idx).astype(int), zero_division=0)

        best_f1 = baseline_f1
        best_threshold = None
        for t in np.unique(scores):
            pred = (scores >= t).astype(int)
            f1 = f1_score(y_binary, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        if best_threshold is not None:
            thresholds[cls] = best_threshold
    return thresholds


def decode_labels_with_thresholds(proba: np.ndarray, class_thresholds: dict[str, float]) -> list[str]:
    """Plain argmax, then overridden per row for any class in
    `class_thresholds` whose own probability clears its own tuned bar --
    only classes that showed a real out-of-fold F1 improvement in
    `tune_class_thresholds` are ever passed in here. If more than one
    target class clears its threshold on the same row (rare, given how
    infrequent these classes are), the one with the higher probability
    wins.
    """
    predicted = decode_labels(proba.argmax(axis=1))
    if not class_thresholds:
        return predicted
    for cls, t in class_thresholds.items():
        idx = _CLASS_TO_INDEX[cls]
        triggered = np.nonzero(proba[:, idx] >= t)[0]
        for i in triggered:
            current = predicted[i]
            if current in class_thresholds and proba[i, _CLASS_TO_INDEX[current]] > proba[i, idx]:
                continue
            predicted[i] = cls
    return predicted


def score_xgboost(
    model: XGBClassifier, features: pd.DataFrame, class_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Returns both a binary-style anomaly score (P(not benign), for the
    same comparison table every other model appears in) and the model's
    predicted class (for genuine multi-class classification metrics).

    `class_thresholds` (optional, from `tune_class_thresholds`) overrides
    plain argmax for the predicted class only -- `anomaly_score` is always
    P(not benign) regardless, since it feeds the separate binary
    attack/not-attack metrics, not classification.
    """
    X = features[FEATURE_COLUMNS].to_numpy()
    proba = model.predict_proba(X)
    benign_index = _CLASS_TO_INDEX["benign"]
    anomaly_score = 1.0 - proba[:, benign_index]
    if class_thresholds:
        predicted_class = decode_labels_with_thresholds(proba, class_thresholds)
    else:
        predicted_class = decode_labels(proba.argmax(axis=1))
    return pd.DataFrame({
        "record_id": features["record_id"].to_numpy(),
        "xgb_anomaly_score": anomaly_score,
        "xgb_predicted_class": predicted_class,
    })
