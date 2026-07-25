# Phase 4 Report: Explainability + Analyst Dashboard + Deployment

> **Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.**

## What was built

- **`explainability/shap_batch.py`**: offline, exact SHAP (`shap.TreeExplainer`)
  over `xgboost_smote` (the best-performing model per `docs/phase_3_report.md`),
  run only over a sample of flagged test events -- never in the streaming
  path, per constraint #4. For each flagged event, attributes the model's
  *predicted* class to its top-`k` contributing features and renders a
  plain-language `explanation` string (e.g. `"Flagged as brute_force. Risk
  driven up by: ema_failure_rate=0.344 (+7.309), ..."`). Also exposes
  `compute_global_feature_weights()` -- mean-|SHAP| feature importances
  pooled across the sampled explanations -- which is the bridge artifact
  the streaming approximation below reuses, so its weights come from real
  exact-SHAP output, not a separately invented metric.
- **`explainability/streaming_approx.py`**: the other half of the
  documented tradeoff. Reuses the global weights above plus a per-feature
  TRAIN-split baseline mean/std (`compute_feature_baseline`), and scores a
  live event's approximate "contribution" per feature as `weight *
  |z-score|` -- O(n_features) arithmetic, no model inference, safe to call
  per-event in a real streaming loop (this is what a deployed
  `online_learning/river_model.py` process would call live; here it's
  applied offline to populate dashboard artifacts, standing in for that
  same call). Every explanation string it produces is prefixed
  `"Approximate explanation (not exact SHAP): ..."` -- the tradeoff is
  shown to the analyst side-by-side with the exact version, never hidden.
- **`explainability/feature_glossary.py`**: one plain-English description
  per feature column, shared by both SHAP's explanation text and the
  dashboard's hover tooltips -- so the two can't drift out of sync with
  different wording for the same column.
- **`evaluation/model_suite.py`** (new, refactored out of
  `evaluation/run_evaluation.py`): trains and scores every model in the
  comparison table exactly once. Both `run_evaluation.py` (the six-criteria
  report) and `dashboard/prepare_data.py` now call `build_model_suite()`
  rather than each re-implementing model training -- the same "one
  implementation, not two that can drift apart" discipline
  `feature_engineering/pipeline.py` already applied to features. Verified
  directly: re-running `run_evaluation.py` after the refactor reproduced
  **exactly** the same metrics (precision/recall/F1/MCC/ROC-AUC/PR-AUC/
  FP-per-day/MTTD/ADWIN lag, to 4 decimal places) as before it.
- **`feature_engineering/feature_names.py`** (new): the 12 feature-column
  name lists, pulled out into a zero-dependency module (no pandas/
  networkx/omegaconf) so lightweight consumers can import just the names.
  `behavioral.py`, `graph.py`, and `pipeline.py` all import their column
  list FROM here now, not the reverse.
- **`dashboard/prepare_data.py`**: the offline precomputation entrypoint.
  Builds the full model suite, then writes small, cheap-to-load artifacts
  to `dashboard/data/<run_name>/`: `model_comparison.parquet`,
  `attack_type_recall.parquet`, one denormalized `test_events_detail.parquet`
  (raw event fields + department/role + attack ground truth/severity/
  rationale + all 12 engineered features + the primary model's score/
  predicted class + both explanation layers, joined on `record_id`),
  `feature_history.parquet` (full train+test per-user feature timeline, for
  trend drill-down), and `run_summary.json` (dataset stats, slider bounds,
  drift-eval results, filter vocabularies).
- **`dashboard/app.py`**: the Streamlit analyst dashboard --
  **fully read-only**, loads only the artifacts above, does no model
  inference or feature computation itself. Sidebar filters (department,
  attack type, severity, time range, user-ID search) combine with a live
  detection-threshold slider whose KPI row and precision/recall-vs-threshold
  curve recompute in-memory on every drag (cheap re-thresholding of an
  already-loaded score array, not a rerun of any model). A sortable,
  click-to-select flagged-events table opens an event-detail panel showing
  raw fields, exact SHAP (with a bar chart), the streaming approximation
  (with its own bar chart, directly comparable), and every engineered
  feature's value + plain-English meaning -- plus a per-user historical
  trend chart (Plotly, flagged event marked with a vertical line) for
  drill-down. No chatbot/Q&A interface anywhere, per constraint #5.
- **`dashboard/theme.py`** + **`.streamlit/config.toml`**: the enterprise/
  "security-ops" visual identity -- dark neutral ground (`#0B0E14`/
  `#141920`), IBM Plex Sans/Mono typography, blue (`#3B82F6`) as the single
  accent, and red/amber/green used *only* for severity chips (never
  decoratively elsewhere). Status is always a text+color chip, never a
  bare color or an emoji glyph.
- **`requirements-dashboard.txt`** + **`render.yaml`** + **`docs/deployment.md`**:
  see "Key design decisions" below for why the deployed service uses a
  separate, much smaller requirement set than the offline pipeline.

## Two real issues found and fixed during this phase

**1. `numba` (a transitive SHAP dependency) hit the same Windows Smart App
Control compiled-extension block every other native package in this
project has hit** (pandas, scikit-learn, pyarrow all pinned for the same
reason in earlier phases) -- `numba==0.66.0`'s `_devicearray` extension was
blocked; pinned to `0.60.0`, which installs cleanly. This pin pulled numpy
down to `2.0.2` on this environment, which trips a pip dependency warning
against `river>=0.25`'s declared `numpy>=2.3.4` floor. Verified directly
rather than assumed safe: the full stack (torch, pandas, sklearn, xgboost,
river, shap, streamlit, plotly all imported together in one process) and
the complete pytest suite (including `test_river_online.py`, the test most
likely to catch a real river/numpy incompatibility) both pass cleanly
under numpy 2.0.2 -- a stated-constraint mismatch in river's own metadata,
not an actual break.

**2. `dashboard/app.py` was transitively importing `networkx` and
`omegaconf`** through `feature_engineering.pipeline.FEATURE_COLUMNS` --
harmless for local testing (both were already installed), but a real
problem for the deployed service's slim dependency set, and a violation of
the "the dashboard does no feature engineering" design intent even before
deployment was considered. Caught by explicitly checking `sys.modules`
after importing `dashboard.app` in a fresh process, not assumed clean.
Fixed by extracting the plain column-name lists into the new
zero-dependency `feature_engineering/feature_names.py`, which
`behavioral.py`/`graph.py`/`pipeline.py` now import FROM (one source of
truth preserved) and which `dashboard/app.py` imports directly (bypassing
`pipeline.py`'s heavier import chain entirely). Re-verified after the fix:
`dashboard.app` loads with zero of {torch, transformers, xgboost, shap,
river, networkx, imblearn} in `sys.modules`.

## Key design decisions

**The deployed Streamlit service uses `requirements-dashboard.txt`
(pandas, numpy, pyarrow, scikit-learn, streamlit, plotly), not the full
`requirements.txt`.** `dashboard/app.py` never imports torch, transformers,
xgboost, shap, river, or networkx -- verified directly (see issue #2
above), not assumed. Installing the ~1GB torch wheel and the rest of the
offline-only stack on every Render free-tier build for dependencies the
running service never touches would burn build minutes and image size for
nothing. `docs/deployment.md` documents the offline-regenerate-then-commit
workflow this implies.

**No live "Regenerate" button in the deployed app**, despite the original
plan leaving that door open. A live regenerate would require installing
the full heavy stack in the deployed service after all (defeating the
decision above) and takes several minutes locally (HF fine-tuning alone
was ~10 minutes in Phase 3) -- a real risk of exceeding Render's request
timeout even if the dependencies were installed. Regeneration is instead a
documented local workflow (`generator.run` -> `evaluation.run_evaluation`
-> `dashboard.prepare_data` -> commit -> push, which auto-redeploys). A
deliberate, stated scope decision, not a silently dropped requirement.

**Render's ephemeral disk is a non-issue for this deployment**, not a
limitation worked around -- the deployed service is fully read-only and
writes nothing at runtime, so every cold start already has everything it
needs from the committed repo. No Postgres, no SQLite-on-disk.

**`data/runs/` is gitignored except `data/runs/small_dev/`** (see
`.gitignore`'s own comment) -- the small_dev-scale dataset is committed as
the shipped static file the hosting constraints require; the eventual
full-scale Scale-up run is not, being too large for a git repo on a free
tier, with that tradeoff deferred to the Scale-up stage's own review gate.

**`evaluation/model_suite.py` is a genuine refactor, not just new code for
Phase 4.** Both the six-criteria report and the dashboard's data now come
from the identical training/scoring path -- verified by re-running the
evaluation after the refactor and diffing against the pre-refactor numbers
(exact match).

**The exact-SHAP sample size is config-bounded
(`models.shap_explainability.max_flagged_samples`, default 2000)**, and at
`small_dev` scale, 1,465 of the 17,231 test events cleared the sample's
threshold (the slider's own configured floor) -- under the cap, so no
random subsampling was actually needed this run. The cap exists for the
Scale-up stage's much larger flagged-event count, where it will matter.

**Row selection in Streamlit's `st.dataframe`** (used for the flagged-
events table's click-to-expand behavior) needed the click to land
precisely on the per-row selection checkbox, which only renders on hover
in this Streamlit version -- found and confirmed by direct browser
testing (a plain click on visible cell text only produces the grid's own
cell-focus highlight, not a `selection.rows` event). Verified working
end-to-end afterward: selecting a row correctly renders its SHAP
explanation, streaming approximation, raw fields, and per-user trend chart.

## Verification

- `streamlit run dashboard/app.py` was started locally and driven through
  a real browser (Chrome automation): KPI row, precision/recall-vs-
  threshold curve, six-criteria model comparison table, flagged-events
  table with working row selection, event-detail panel (raw fields +
  exact SHAP bar chart + streaming-approximation bar chart + feature
  glossary table), and the per-user historical trend chart (correctly
  showing a `geo_distance_from_home_km` spike leading into a flagged
  `brute_force` event) were all visually confirmed rendering correctly,
  with matching numbers to `docs/phase_3_evaluation_report.md`.
- New pytest coverage: `test_shap_batch.py`, `test_streaming_approx.py`,
  `test_model_suite.py`, `test_prepare_data_pipeline.py` (a full offline
  data-prep pipeline run against a real generated dataset), and
  `test_dashboard_app.py` (pure-function unit tests for the live
  threshold-slider math and sidebar filter logic -- `_threshold_metrics`
  checked against hand-computed precision/recall/FP counts).
- Full suite (`pytest tests/ -v`): see the run accompanying this report;
  all tests green, including every pre-existing Phase 1-3 test, confirming
  the `feature_names.py` and `model_suite.py` refactors introduced no
  regressions.

## Known limitations / deliberate deferrals

- **The streaming approximation is genuinely approximate, stated
  everywhere it's shown.** It has no notion of feature interactions or a
  specific event's actual model decision path -- only "this feature
  usually matters (by mean |SHAP|), and today's value is unusually far
  from baseline." This is the documented tradeoff constraint #4 asks for,
  not a shortfall to fix later.
- **The dashboard shows ground-truth attack labels** (`true_attack_type`,
  severity) alongside model predictions -- appropriate for this project's
  stated purpose (benchmarking detection methods on synthetic data with
  known labels), but explicitly NOT how a real deployment would look
  (a real SOC dashboard has no ground truth to show). The column is
  labeled "ground truth" in its own tooltip so this isn't ambiguous.
- **No live streaming demo in the dashboard itself** -- `online_learning/
  river_model.py`'s true event-at-a-time loop is exercised in Phase 3's
  evaluation and its own tests, not re-enacted live in the UI (the
  dashboard is a static analyst tool over a fixed test-split snapshot, per
  the "not a full observability stack" instruction).
- **Per-user trend charts show 4 of the 12 engineered features** (chosen
  for relevance to the flagged event types this project covers), not all
  12 -- a deliberate density/readability tradeoff for a line chart, not a
  data limitation (`feature_history.parquet` itself has all 12).

## Synthetic-data disclaimer

Every dashboard page carries the same synthetic-data notice verbatim as
every other artifact in this project. No result shown in the dashboard --
model scores, SHAP explanations, or trend charts -- should be read as
validated against real enterprise traffic.
