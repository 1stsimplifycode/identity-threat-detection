"""Time-respecting train/test split (constraint #1): train on an earlier
window, test on a later one. No shuffling, no k-fold CV on raw temporal
records anywhere in this pipeline.

The split point is computed once from `events` (by row count at a given
`train_frac`, after sorting by timestamp) and expressed as a concrete,
loggable `split_timestamp` plus the exact `record_id` sets on each side --
so any table keyed by `record_id` (features, labels) can be partitioned
identically without re-deriving the split.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ChronologicalSplit:
    train_record_ids: frozenset[str]
    test_record_ids: frozenset[str]
    split_timestamp: pd.Timestamp
    train_frac: float
    n_train: int
    n_test: int

    def summary(self) -> str:
        return (
            f"Chronological split: train_frac={self.train_frac:.2f}, "
            f"split_timestamp={self.split_timestamp}, "
            f"n_train={self.n_train}, n_test={self.n_test}"
        )


def chronological_split(events: pd.DataFrame, train_frac: float = 0.7) -> ChronologicalSplit:
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    events_sorted = events.sort_values("timestamp")
    split_idx = int(len(events_sorted) * train_frac)
    split_idx = max(1, min(split_idx, len(events_sorted) - 1))  # keep both sides non-empty
    split_timestamp = events_sorted.iloc[split_idx]["timestamp"]

    train_ids = frozenset(events_sorted.iloc[:split_idx]["record_id"])
    test_ids = frozenset(events_sorted.iloc[split_idx:]["record_id"])

    return ChronologicalSplit(
        train_record_ids=train_ids,
        test_record_ids=test_ids,
        split_timestamp=split_timestamp,
        train_frac=train_frac,
        n_train=len(train_ids),
        n_test=len(test_ids),
    )


def apply_split(df: pd.DataFrame, split: ChronologicalSplit, part: str) -> pd.DataFrame:
    """Filter any `record_id`-keyed DataFrame (features, labels, ...) to the
    train or test side of a previously-computed split.
    """
    if part not in ("train", "test"):
        raise ValueError(f"part must be 'train' or 'test', got {part!r}")
    ids = split.train_record_ids if part == "train" else split.test_record_ids
    return df[df["record_id"].isin(ids)]
