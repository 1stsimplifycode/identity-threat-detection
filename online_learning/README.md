# online_learning/

- **`river_model.py`** (Phase 3) -- `river.tree.HoeffdingTreeClassifier`,
  streamed one event at a time through the exact same
  `FeaturePipelineState.update()` interface `feature_engineering/` exposes
  for Phase 4's real-time dashboard. A true prequential loop: every event
  is scored, but the model only ever *learns* from train-split events,
  keeping the same train/test discipline as every batch model despite the
  fundamentally different incremental training lifecycle.

Evaluated identically to the batch models in `models/` (same chronological
split, same comparison table), since it is one of the detection approaches
constraint #3 asks for -- not a Phase 4 add-on.
