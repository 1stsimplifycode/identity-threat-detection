# models/

In order of sophistication (every later model is compared against the
earlier ones in every results table, per constraint #3):

1. **`baseline.py`** (Phase 2b) -- dumb rule-based detector: static
   thresholds on failed-login count, new-country flag, off-hours flag.
   Deliberately reads only raw event fields, never the engineered feature
   table, so it stays a genuine independent floor to beat.
2. **`isolation_forest.py`** (Phase 2b) -- unsupervised/statistical,
   trained on the merged `feature_engineering` table.
3. **`xgboost_classifier.py`** (Phase 3) -- supervised, genuine multi-class
   attack-type classification (benign + 5 attack types), with the 3
   imbalance-handling conditions (`none`, `class_weight`, `smote`) compared
   identically -- SMOTE applied strictly to the TRAIN split only, with
   automatic k_neighbors shrinking (and a RandomOverSampler fallback) for
   attack types with very few train examples.
4. **`hf_classifier.py`** (Phase 3) -- fine-tuned `prajjwal1/bert-tiny`
   (~4.4M params), trained on serialized last-k-event pseudo-text
   sequences (see the module's own docstring for the honest framing: a
   tabular tree model has the right inductive bias here, a sequence
   encoder's plausible edge is short-term event ORDER, verified
   empirically not assumed). CPU-only fine-tuning; the trained checkpoint
   is saved to `models/artifacts/hf_bert_tiny/` for Phase 4's Render
   deployment to load directly, no training at request time.

The online learner (River Hoeffding Tree) lives in `online_learning/`, not
here, since it has a fundamentally different (incremental,
one-event-at-a-time) fit/update lifecycle -- see that package's README.

## A real environment issue, documented for anyone re-running this

On this project's Windows dev environment, importing `pandas` (or
`pyarrow`) before `torch` in the same process crashes torch's DLL loading
(`WinError 1114`) -- a genuine conflict between pyarrow's bundled Arrow
C++ runtime and torch's bundled `c10` library, not a code bug. Any
entrypoint that will use `hf_classifier.py` must `import torch` before
`import pandas`; see `evaluation/run_evaluation.py`'s and
`tests/conftest.py`'s top-of-file comments.
