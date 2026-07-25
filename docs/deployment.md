# Deployment (Render free tier)

> **Synthetic data. Not derived from or validated against real organizational
> logs. For benchmarking detection methods only.**

## What actually gets deployed

One Render **Web Service**, free plan, no database, no background workers,
no GPU. `render.yaml` at the repo root is the tested Blueprint -- `render
blueprint launch` (or "New +" -> "Blueprint" in the Render dashboard,
pointed at this repo) reads it directly.

The deployed process is `dashboard/app.py`, a Streamlit app that is
**fully read-only**: it loads Parquet/JSON artifacts from
`dashboard/data/<run_name>/` and renders them. It never trains a model,
never computes SHAP, never calls out to an external API, and never writes
to disk. Those artifacts are produced **offline, once, locally** by
`dashboard/prepare_data.py` (which itself calls `evaluation/model_suite.py`
-- the same code path the six-criteria evaluation report uses, so the
dashboard's numbers can never silently drift from `docs/phase_3_report.md`'s)
and are committed to the repo alongside the raw generated dataset
(`data/runs/small_dev/`) and the fine-tuned HF checkpoint
(`models/artifacts/hf_bert_tiny/`). This matches the project's own hosting
constraint directly: "train models OFFLINE and ship trained artifacts,"
"dataset generated once offline and shipped as static file."

## Why the deployed app installs a SEPARATE, smaller requirements file

`requirements.txt` (torch, transformers, xgboost, shap, river, networkx,
imbalanced-learn, ...) is what the OFFLINE pipeline needs to generate data,
train models, and precompute explanations. `dashboard/app.py` imports none
of that -- it only needs `streamlit`, `pandas`, `pyarrow`, `plotly`,
`numpy`, and `scikit-learn` (for one `precision_recall_curve` call).
`render.yaml`'s build command installs `requirements-dashboard.txt`, not
`requirements.txt`, deliberately: `torch` alone is roughly 1GB, and
installing the full offline stack on every free-tier build would burn
build minutes and image size for dependencies the running service never
touches. This was verified directly, not assumed --
`dashboard/app.py` was import-checked with `sys.modules` to confirm torch/
transformers/xgboost/shap/river/networkx are never loaded by it.

## Environment variables

| Variable | Set by | Purpose |
|---|---|---|
| `PORT` | Render (automatic) | Render assigns the listen port at runtime; the start command binds to it via `--server.port $PORT`. |
| `PYTHON_VERSION` | `render.yaml` | Pins the build's Python to 3.11.9, matching the dev environment every phase was built and tested against. |

No API keys, no secrets, no database URL -- per the project's own "zero
paid dependencies, prefer no external API at all" constraint, there is
nothing else to configure.

## Build & start commands

```
buildCommand: pip install -r requirements-dashboard.txt
startCommand: streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

`.streamlit/config.toml` (committed) sets the enterprise dark theme,
`headless = true`, and `enableXsrfProtection = true`; it does not set a
port (Render's `$PORT` varies per deploy, so that stays a runtime flag,
never hardcoded).

## Cold-start behavior

Render's free tier spins a web service down after a period of no traffic
and cold-starts it on the next request (usually a low tens-of-seconds
delay while the container boots and `pip`-installed dependencies load).
Because the slim dashboard requirement set has no torch/xgboost/etc. to
import, Python process startup itself is fast; the dominant cold-start
cost is Render's own container boot, not this app's import time. There is
no model loading or inference on the request path at all -- every page
view just reads already-small Parquet files (a few MB total for
`small_dev` scale) into memory once per Streamlit session.

Free-tier RAM/CPU are limited and there is no GPU. This is a non-issue for
the deployed service specifically because it does no training or batch
inference -- the only "compute" per interaction is re-thresholding an
already-loaded NumPy array (the live precision/recall slider) and rendering
Plotly charts, both cheap.

## Ephemeral disk -- why it doesn't matter here

Render free-tier services have ephemeral disk: anything written at runtime
is lost on redeploy or restart. This project's deployed service **writes
nothing at runtime**, so that limitation never bites. Every cold start (or
redeploy) pulls a fresh container from the same git commit, which already
contains every artifact the app reads. No SQLite-on-disk, no Postgres, and
no in-app "Regenerate" button are used, by design -- see the next section
for why, and how regeneration actually works instead.

## How to regenerate / retrain (there is no live "Regenerate" button)

An earlier design option considered a dashboard button that re-invokes
`dashboard/prepare_data.py` live, in the deployed process. It was
deliberately **not** built: that would require installing the full,
heavy `requirements.txt` (torch, xgboost, shap, ...) in the deployed
service after all -- defeating the slim-requirements decision above --
and a full regenerate-and-retrain pass takes several minutes locally
(see `docs/phase_3_report.md`'s HF fine-tuning timing), which risks
Render's request timeout even if the dependencies were installed. This is
a real, deliberate scope decision, not a silently dropped requirement.

Regeneration is instead a **local workflow, followed by a redeploy**:

```bash
# 1. Regenerate the synthetic dataset (same tested config each time --
#    fixed seeds, so this is reproducible, not "a new random dataset").
python -m generator.run --config-name small_dev

# 2. Retrain every model and refresh docs/phase_3_evaluation_report.md.
python -m evaluation.run_evaluation --config-name small_dev

# 3. Recompute dashboard artifacts (model comparison, exact SHAP sample,
#    streaming approximation, feature history) from the retrained models.
python -m dashboard.prepare_data --config-name small_dev

# 4. Commit the refreshed data/runs/small_dev/, dashboard/data/small_dev/,
#    and (if step 2 retrained it) models/artifacts/hf_bert_tiny/, then
#    push -- Render redeploys automatically on a push to the configured
#    branch (or trigger a manual deploy from the Render dashboard).
git add data/runs/small_dev dashboard/data/small_dev models/artifacts/hf_bert_tiny
git commit -m "Regenerate small_dev data and dashboard artifacts"
git push
```

Steps 1-3 need the full `requirements.txt` locally (`pip install -r
requirements.txt`, plus the CPU-only torch wheel -- see the repo README).

**At full (Scale-up) scale, step 2 is slow, and HF is most of why.**
`hf_bert_tiny`'s training cost is capped regardless of dataset size
(`models.hf_classifier.max_benign_train_examples`), but its inference-
scoring pass runs over the *entire* train+test split -- at ~1M+ events,
this becomes one of the single most expensive stages, for a model
`docs/phase_3_report.md` already found to be the weakest performer. Pass
`evaluation.include_hf=false` to skip it for a materially faster local
re-run:

```bash
python -m evaluation.run_evaluation --config-name config evaluation.include_hf=false
```

The comparison table simply omits `hf_bert_tiny` for that run -- reported
explicitly in the console output, never silently dropped.

## Scale-up decision: the deployed dashboard stays on `small_dev`

The Scale-up stage has now run (`configs/config.yaml`, ~1.17M events; full
results in `docs/phase_3_evaluation_report.md` and the deeper discussion in
`docs/scale_up_report.md`). **Decision: the deployed Render dashboard
continues to ship the curated `small_dev`-scale dataset, not the full-scale
one.** Measured, not estimated:

| | `small_dev` | full-scale (`default`) | ratio |
|---|---|---|---|
| Raw generated run (`data/runs/<name>/`) | 15 MB | 301 MB | ~20x |
| `n_events` | 57,436 | 1,167,750 | ~20x |
| `n_test` (chronological split) | 17,231 | 350,325 | ~20x |
| Dashboard artifacts (`dashboard/data/<name>/`) | 6.2 MB | ~120 MB (projected, linear in `n_test`) | ~20x |

Three independent reasons, any one of which would be sufficient on its own:

1. **Git repo / free-tier size budget.** `data/runs/` is gitignored except
   `data/runs/small_dev/` specifically (see `.gitignore`'s comment) for
   exactly this reason. A 301 MB raw run plus a ~120 MB dashboard-artifact
   directory is a poor fit for a git-based deploy on a free hosting tier --
   Render's free plan has no persistent volume for a separately-staged
   artifact store, so "committed to the repo" is the only delivery
   mechanism this architecture has (see "What actually gets deployed"
   above).
2. **Cold-start latency.** Every cold start reads the full dashboard
   artifact set into memory once per Streamlit session (see "Cold-start
   behavior" above). At ~20x the Parquet volume, that read -- and the
   in-memory pandas footprint for the whole session -- grows
   proportionally, working against the free tier's already-tight
   CPU/RAM budget on the exact code path every visitor hits first.
3. **The six-criteria numbers don't change the recommendation.**
   `docs/phase_3_evaluation_report.md`'s full-scale results confirm the
   same model ranking `small_dev` already showed (tree-based models,
   `xgboost_class_weight`/`xgboost_none` in particular, remain the
   strongest detectors; `hf_bert_tiny` remains the weakest). Nothing about
   the full-scale run's *findings* requires the deployed demo to carry the
   full-scale *data* -- the dashboard's job is to demonstrate the detection
   pipeline and let an analyst explore representative flagged events, not
   to be the system of record for every one of 1.17M events.

**What this means concretely:** `dashboard/data/small_dev/` (built from
`data/runs/small_dev/`, 5,000 users / 57,436 events) remains what
`render.yaml` deploys. The full-scale run's own report
(`docs/scale_up_report.md`) is committed as documentation of the
validation exercise, but its underlying `data/runs/default/` directory is
**not** committed, per the existing `.gitignore` policy, and
`dashboard/data/default/` (or equivalent full-scale dashboard artifacts)
is deliberately never generated for deployment.

This does not preclude a future, differently-architected deployment (e.g.
paid tier with a real database and pagination instead of "load the whole
Parquet file into memory") from shipping full-scale data -- that's out of
scope for the free-tier single-Web-Service architecture this project
targets, and would be its own review gate if pursued.

## Single-service limit

Everything above is one Render free Web Service. No Postgres, no Redis, no
Kubernetes, no Kafka, no Neo4j-as-a-service -- consistent with the
project's hosting constraints and its "NetworkX only, not a graph
database" architectural decision.
