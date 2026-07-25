"""ADWIN drift detector (Phase 3), tied to the ground-truth `drift_log.csv`
so its detection timing can be evaluated against a known injection point --
not just run and eyeballed.

Monitors `geo_distance_from_home_km` (`feature_engineering.behavioral`):
that feature is computed against each user's ORIGINAL population-table home
coordinates, which never update when `generator/drift.py`'s
`remote_work_shift` relocates a user -- so a real relocation shows up as a
genuine, sustained distribution shift in this specific feature, exactly the
"a baseline needs refreshing" concept drift a production system has to
detect. This is a deliberate, documented choice of monitoring signal, not
the only one that could show drift.

Signal preparation matters a lot here, found by direct testing against the
real small_dev drift log:
  - Feeding raw per-event values (or a rolling mean over raw event order)
    into ADWIN fired hundreds of times across the ENTIRE stream, including
    well before the configured drift day. Root cause: a stream of raw
    events mixes many different users, each with their own small constant
    jitter around their (different) home coordinates plus occasional
    ordinary travel spikes -- a rolling window's mean fluctuates heavily
    just from WHICH users happen to appear in it, independent of any real
    population-level change, swamping the one genuine sustained shift.
  - A clean daily mean shows an unmistakable step change exactly at the
    drift day (confirming the injection itself works correctly) but
    small_dev's 14-day window gives ADWIN too few points (~15) to build
    statistical confidence before the stream ends.
  - Aggregating by a FIXED EVENT COUNT per bin (not calendar day) gives
    both a clean, smoothed signal AND enough bins for ADWIN to work with,
    and is what this module actually uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from river import drift as river_drift


def bin_by_event_count(values: pd.Series, timestamps: pd.Series, bin_size: int) -> pd.DataFrame:
    """Groups `values`/`timestamps` (already in chronological order) into
    fixed-size bins of `bin_size` consecutive events, returning one row per
    bin: the bin's mean value and its last (latest) timestamp -- the
    monitoring granularity ADWIN actually operates at in this module.
    """
    bin_index = np.arange(len(values)) // bin_size
    frame = pd.DataFrame({"value": values.to_numpy(), "timestamp": timestamps.to_numpy(), "bin": bin_index})
    return frame.groupby("bin").agg(mean_value=("value", "mean"), timestamp=("timestamp", "max"))


def run_adwin(binned: pd.DataFrame, cfg: DictConfig) -> pd.Series:
    """Feeds `binned["mean_value"]` through River's ADWIN one bin at a
    time. Returns the `timestamp` values (from `binned`) of every bin at
    which a change was signalled.
    """
    adwin_cfg = cfg.models.adwin
    detector = river_drift.ADWIN(delta=float(adwin_cfg.delta))
    detected_timestamps = []
    for row in binned.itertuples():
        detector.update(float(row.mean_value))
        if detector.drift_detected:
            detected_timestamps.append(row.timestamp)
    return pd.Series(detected_timestamps, dtype="datetime64[ns]")


def evaluate_against_drift_log(
    detected_timestamps: pd.Series,
    drift_log: pd.DataFrame,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """For each ground-truth drift event, finds the earliest ADWIN
    detection at or after the event's scheduled day, reporting detection
    lag in days. A drift event with no matching detection is reported as
    `detected=False`, not silently dropped.
    """
    rows = []
    for _, event in drift_log.iterrows():
        event_start = start_date + pd.Timedelta(days=int(event["day"]))
        candidates = detected_timestamps[detected_timestamps >= event_start]
        if len(candidates) == 0:
            rows.append({
                "day": int(event["day"]), "change_type": event["change_type"],
                "detected": False, "detection_lag_days": None,
            })
        else:
            first_detection = candidates.min()
            lag_days = (first_detection - event_start).total_seconds() / 86400.0
            rows.append({
                "day": int(event["day"]), "change_type": event["change_type"],
                "detected": True, "detection_lag_days": round(lag_days, 2),
            })
    return pd.DataFrame(rows)
