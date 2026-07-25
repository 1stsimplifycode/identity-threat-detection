"""Cold-start handling (Phase 3): a new user with little or no history
must still get a reasonable initial risk baseline -- not a default
"anomalous" label (e.g. `access_chain_distance`'s cold-start sentinel,
returned for ANY never-before-seen resource regardless of how ordinary it
is for that user's department) and not a default "normal" one either (an
EMA baseline that simply starts at 0 before any of the user's own events
have accumulated, which can look artificially clean).

Applied as a post-processing step over an already-computed feature table,
using priors derived from a REFERENCE set of established rows -- by
convention the caller's TRAIN split (see evaluation/run_evaluation.py),
never test rows, to keep the same leakage discipline as the
operating-threshold selection in evaluation/report.py.
"""
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

# Features worth replacing during the cold-start window -- deliberately NOT
# every feature: is_new_edge/device_fan_in reading as "maximally new" on a
# user's genuinely first device is correct behavior to preserve, not a flaw
# to paper over.
COLD_START_ELIGIBLE_COLUMNS: list[str] = [
    "access_chain_distance",
    "peer_group_deviation",
    "ema_failure_rate",
]


def compute_department_priors(reference_features: pd.DataFrame, reference_events: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Median of each cold-start-eligible feature, per department, computed
    from `reference_features`/`reference_events` (established rows only --
    the caller's TRAIN split).
    """
    merged = reference_events[["record_id", "user_id"]].merge(reference_features, on="record_id")
    merged = merged.merge(users[["user_id", "department"]], on="user_id")
    return merged.groupby("department")[COLD_START_ELIGIBLE_COLUMNS].median()


def apply_cold_start_priors(
    features: pd.DataFrame,
    events: pd.DataFrame,
    users: pd.DataFrame,
    priors: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Return a copy of `features` with COLD_START_ELIGIBLE_COLUMNS replaced
    by the user's department prior, for rows within
    `cfg.feature_engineering.cold_start.window_days` of the user's
    `join_date`. Rows for a department absent from `priors` (e.g. a
    department with no established rows in the reference set) fall back to
    their original computed value rather than a NaN.
    """
    window = pd.Timedelta(days=float(cfg.feature_engineering.cold_start.window_days))
    context = events[["record_id", "user_id", "timestamp"]].merge(
        users[["user_id", "department", "join_date"]], on="user_id", how="left"
    )
    context["is_cold_start"] = (context["timestamp"] - context["join_date"]) < window

    result = features.merge(context[["record_id", "department", "is_cold_start"]], on="record_id", how="left")
    is_cold_start = result["is_cold_start"].fillna(False)

    for col in COLD_START_ELIGIBLE_COLUMNS:
        prior_values = result["department"].map(priors[col].to_dict()) if col in priors.columns else pd.Series(index=result.index, dtype=float)
        result[col] = result[col].where(~is_cold_start, prior_values.fillna(result[col]))

    return result.drop(columns=["department", "is_cold_start"])
