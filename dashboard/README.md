# dashboard/

A minimal, **fully read-only** Streamlit analyst dashboard -- an
enterprise/industrial "security-ops" visual theme (dark neutral ground,
blue accent, red/amber/green reserved for severity), interactive by direct
manipulation (live filters, a detection-threshold slider, a
sortable/searchable flagged-events table, per-user trend drill-down), not
conversation. No chatbot/Q&A interface.

- **`prepare_data.py`** -- the offline precomputation entrypoint. Builds
  the full model suite (`evaluation/model_suite.py`), computes exact SHAP
  + the streaming approximation over flagged test events, and writes small
  Parquet/JSON artifacts to `data/<run_name>/`. Run this (or the local
  regenerate workflow in `docs/deployment.md`) whenever the underlying
  dataset or models change -- `app.py` never recomputes anything itself.
- **`app.py`** -- the Streamlit app. Only reads `data/<run_name>/`; imports
  none of torch/transformers/xgboost/shap/river/networkx (verified
  directly via `sys.modules`), which is why `requirements-dashboard.txt`
  at the repo root -- not the full `requirements.txt` -- is what actually
  gets installed on Render.
- **`theme.py`** -- shared CSS/color/typography constants and small HTML
  chip/card helpers, imported by `app.py`.
- **`data/<run_name>/`** -- committed precomputed artifacts (small enough
  to ship in git at `small_dev` scale).

Run locally: `streamlit run dashboard/app.py` (after `prepare_data.py` has
been run at least once for the target run). See `docs/deployment.md` for
the Render deployment and `docs/phase_4_report.md` for what was built.
