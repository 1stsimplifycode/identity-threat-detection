"""Dual-mode (batch pandas + streaming per-event) feature engineering.

- `behavioral.py`   -- 3a: per-event, time-based behavioral features.
- `graph.py`        -- 3b: NetworkX-derived relational features.
- `pipeline.py`      -- merges both into one feature table keyed by record_id.
"""
