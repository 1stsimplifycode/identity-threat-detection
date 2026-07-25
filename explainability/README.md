# explainability/

Explainability is a documented tradeoff (constraint #4), not something
silently simplified:

- **`shap_batch.py`** -- exact SHAP (`shap.TreeExplainer`), run only as an
  **offline batch job** over a sample of flagged events, on `xgboost_smote`
  (the best-performing model per `docs/phase_3_report.md`). Never runs in
  the streaming inference path. Also exposes
  `compute_global_feature_weights()`, the bridge artifact `streaming_approx.py`
  consumes.
- **`streaming_approx.py`** -- lightweight, precomputed feature-importance
  for the streaming path: global per-feature weights (from `shap_batch.py`'s
  real exact-SHAP output, not a separately invented metric) applied to a
  live event's z-score deviation from a TRAIN-split baseline. Cheap
  (O(n_features) arithmetic, no model inference), a real approximation, and
  labeled as such in every explanation string it produces
  (`"Approximate explanation (not exact SHAP): ..."`).
- **`feature_glossary.py`** -- one plain-English description per feature
  column, shared by both explanation layers above and the dashboard's
  hover tooltips, so wording can't drift between them.

See `docs/phase_4_report.md` for what was built and verified.
