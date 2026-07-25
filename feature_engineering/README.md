# feature_engineering/

Two feature families, both feeding one merged model-input table.

- **`behavioral.py`** (3a) -- `BehavioralFeatureState`: velocity, geo-distance
  from home, login-location entropy, device-switching rate, failed-login
  ratio, peer-group deviation, EMA baselines. Per-user and per-department
  rolling state, pruned on a config-driven window.
- **`graph.py`** (3b) -- `GraphFeatureState`: an in-memory bipartite
  user-device graph, a per-user + aggregate resource-transition graph, and
  a periodically-refreshed Louvain user-community partition (all plain
  NetworkX algorithms -- no GNN, no learned embeddings). Produces 8 named
  columns: `device_fan_in`, `user_device_set_delta`, `is_new_edge`,
  `access_chain_distance`, `peer_community_deviation`,
  `device_fingerprint_mismatch`, `session_foreign_resource_count`,
  `session_hop_seconds`.
- **`pipeline.py`** -- `FeaturePipelineState` / `compute_feature_table()`:
  merges both into one feature table keyed by `record_id` (a join key only,
  never fed to a model itself).

Both `BehavioralFeatureState` and `GraphFeatureState` share one design: a
stateful `update(event) -> dict` method is the single source of truth for
the feature logic, computed from state as it existed *before* the current
event. `compute_batch()` (Phase 2b) replays `update()` over a
chronologically-sorted DataFrame from empty state; a Phase 4 streaming loop
will keep the same state object alive across live per-event calls -- one
implementation, not two that could silently drift apart.

## Verified against injected attacks (Phase 2b, extended Phase 5)

`tests/test_graph_features.py` asserts, with real pytest checks (not
eyeballed): `access_chain_distance` is significantly higher on
`lateral_movement`-labeled rows than on benign rows, and
`device_fan_in`/`is_new_edge` are significantly higher on
`device_spoofing`-labeled rows than on benign rows -- **for the
`cross_user_reuse` variant specifically.** `device_spoofing`'s OTHER
variant, `fingerprint_mismatch` (same user, same device_id, but the
device suddenly claims a different `device_type`/`os`), was a documented
blind spot for all 5 original graph features -- confirmed by direct
measurement in `docs/phase_5_recall_investigation.md` (z=0.57 separation
on `is_new_edge`, statistically indistinguishable from noise) -- because
none of them check device *fingerprint* consistency, only (user, device)
*pairing* topology. `device_fingerprint_mismatch` (added in that same
investigation) closes this gap directly and is verified against the
`fingerprint_mismatch` variant specifically in
`tests/test_graph_features.py`.

`lateral_movement` remained at 0% recall even after that fix, because its
only real per-event signal, `access_chain_distance`, looks at a single
direct resource-to-resource transition -- it has no notion of session-wide
breadth. `session_foreign_resource_count` and `session_hop_seconds`
(Phase 5e) directly target `attacks/lateral_movement.py`'s own stated
mechanism (multiple cross-department resources touched fast, within one
session) and are verified to spike on `lateral_movement` rows in
`tests/test_graph_features.py`.

## Known scale-sensitivity

`peer_community_deviation` needs enough event density before Louvain
communities are non-trivial (mostly-singleton "communities" produce zero
deviation almost everywhere) -- meaningful at `small_dev` scale, closer to
uninformative at very small (few-hundred-event) test configs. See
`docs/phase_2b_report.md`.
