"""Graph-derived relational features -- constraint 3b.

Plain, named, SHAP-attributable columns computed from in-memory NetworkX
graph algorithms -- explicitly NOT a GNN and NOT learned embeddings (see
README.md and the project's out-of-scope list). One stateful class,
`GraphFeatureState`, mirrors `feature_engineering.behavioral`'s dual-mode
design: `update(event)` mutates rolling graph state and returns that
event's feature row, computed from state as it existed BEFORE this event.
`compute_batch()` replays it over a chronologically-sorted DataFrame.

Maintains, on a rolling window (pruned via O(1)-amortized timestamp-ordered
deques, not full-history rebuilds):
    - a bipartite user<->device graph
    - a per-user "resources visited" history
    - one small resource-transition graph PER DEPARTMENT (not one shared
      org-wide graph -- see the note on access_chain_distance below), over
      the fixed ~7 resource-type vocabulary
    - a per-user EMA-decayed device-usage distribution
    - a cached Louvain user-community partition of the bipartite graph's
      user-user device-sharing projection, refreshed every
      `louvain_refresh_events` events (not per-event -- see module design
      notes in docs/phase_2b_report.md for why)

Produces exactly 8 named columns:
    device_fan_in                -- distinct users historically on this device
    user_device_set_delta        -- deviation from this user's own rolling
                                     device-usage distribution
    is_new_edge                  -- 0.0 if this (user, device) pair is known;
                                     else a graded (0, 1] score of how far this
                                     device sits from the user's known device
                                     neighborhood in the bipartite graph
    access_chain_distance        -- cost of the direct transition from the
                                     user's most recently visited resource to
                                     the resource just accessed, within THEIR
                                     OWN DEPARTMENT's resource-transition graph
    peer_community_deviation     -- 1 - (fraction of this user's Louvain
                                     community who have also used this device)
    device_fingerprint_mismatch  -- 1.0 if this device_id's claimed
                                     (device_type, os) differs from the FIRST
                                     signature ever recorded for that
                                     device_id; else 0.0. Added specifically
                                     because `is_new_edge` tracks (user,
                                     device) PAIRING novelty and cannot see a
                                     device_id that keeps its established
                                     user but changes its claimed hardware/OS
                                     signature -- exactly the pattern
                                     `attacks/device_spoofing.py`'s variant
                                     (a) ("fingerprint mismatch") produces.
                                     Confirmed via direct measurement
                                     (docs/phase_5_recall_investigation.md)
                                     that `is_new_edge` alone has almost no
                                     separation for that variant (z=0.57
                                     vs. the benign population) before this
                                     feature was added.
    session_foreign_resource_count -- count of DISTINCT resource types
                                     accessed so far THIS SESSION that fall
                                     outside the user's own department's
                                     typical resource set (RESOURCE_TYPES_BY_
                                     DEPT, the same vocabulary
                                     `attacks/lateral_movement.py` itself
                                     draws its escalating chain from).
    session_hop_seconds           -- seconds since the previous event in
                                     this same session; a large sentinel
                                     (session_hop_seconds_sentinel) if this
                                     is the session's first event.

Design note on access_chain_distance (found during Phase 2b verification):
an org-wide, department-agnostic resource-transition graph badly
under-detects lateral movement with only ~7 resource types in the
vocabulary -- almost every transition type is "normal" for SOME department
somewhere in the org (e.g. vpn->crm is routine for Sales/Customer Support),
so a shared graph learns it as cheap/common regardless of who's asking,
even though it's genuinely foreign to a user from a different department.
Scoping the transition graph per-department (i.e., "is this common among MY
PEERS," not "is this common anywhere in the org") fixes this directly and
mirrors peer_community_deviation's own peer-relative framing.

Design note on session_foreign_resource_count / session_hop_seconds (Phase
5e, docs/phase_5_recall_investigation.md's named next lever after Phase 5's
device_fingerprint_mismatch fix): `access_chain_distance` only ever looks at
the cost of the single DIRECT transition from the last-visited resource --
it has no notion of session-wide BREADTH. `attacks/lateral_movement.py`'s
own rationale is explicitly session-level: "accessed N resources outside
its department... within a single fast session." A user who touches 3
foreign resources one hop apart each looks locally unremarkable per-hop
(each transition might individually still be "not in the transition graph
yet" = sentinel-cost, same as a single foreign hop) but is structurally
different in aggregate -- these two features make that aggregate visible:
breadth (how many distinct foreign resource types this session has touched)
and velocity (how fast), the second signal the generator's own
`hop_gap_seconds` config deliberately builds in as a secondary tell.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import networkx as nx
import pandas as pd
from omegaconf import DictConfig

from feature_engineering.feature_names import GRAPH_FEATURE_COLUMNS
from generator.constants import RESOURCE_TYPES_BY_DEPT
from preprocessing.constants import RESOURCE_TYPES


def _user_node(user_id: str) -> str:
    return f"U:{user_id}"


def _device_node(device_id: str) -> str:
    return f"D:{device_id}"


def _new_device_users_dict() -> defaultdict[str, set]:
    # A named module-level function, not a lambda -- GraphFeatureState (and
    # this factory it hands to defaultdict) must stay picklable so
    # feature_engineering.pipeline.compute_feature_table_resumable() can
    # checkpoint live state mid-computation; lambdas are not picklable by
    # the standard `pickle` module, found directly by a failing test before
    # this ever ran against real Scale-up data.
    return defaultdict(set)


class GraphFeatureState:
    def __init__(self, cfg: DictConfig, users: pd.DataFrame) -> None:
        gr_cfg = cfg.feature_engineering.graph
        self.window = pd.Timedelta(days=float(cfg.feature_engineering.window_days))
        self.louvain_refresh_events = int(gr_cfg.louvain_refresh_events)
        self.device_ema_gamma = float(gr_cfg.device_ema_gamma)
        self.sentinel = float(gr_cfg.access_chain_distance_sentinel)
        self.device_distance_cutoff = int(gr_cfg.device_distance_cutoff)
        self.session_hop_sentinel = float(gr_cfg.session_hop_seconds_sentinel)
        self._department_by_user: dict[str, str] = users.set_index("user_id")["department"].to_dict()

        self.bipartite = nx.Graph()
        self._bipartite_insertions: deque[tuple[pd.Timestamp, str, str]] = deque()

        # One small resource-transition graph per department (see module
        # docstring's design note) -- built lazily as departments are seen.
        self.resource_graph_by_dept: dict[str, nx.DiGraph] = {}
        self._resource_edge_insertions: deque[tuple[pd.Timestamp, str, str, str]] = deque()
        self._last_resource_in_session: dict[str, str] = {}
        self._session_last_seen: deque[tuple[pd.Timestamp, str]] = deque()
        # Per-session state for session_foreign_resource_count / session_hop_seconds --
        # pruned on the exact same expiry as the two dicts above (see
        # _prune_resource_graph), since all three share one session's lifetime.
        self._session_foreign_resources: dict[str, set[str]] = defaultdict(set)
        self._session_last_event_ts: dict[str, pd.Timestamp] = {}

        self._user_resource_history: dict[str, deque] = defaultdict(deque)
        self._user_device_ema: dict[str, dict[str, float]] = defaultdict(dict)

        self._events_since_refresh = 0
        self._community_of: dict[str, int] = {}
        self._community_size: dict[int, int] = {}
        self._community_device_users: dict[int, dict[str, set[str]]] = {}

        # Established (device_type, os) signature per device_id, from the
        # FIRST event ever seen for that device_id -- deliberately NOT
        # windowed/pruned like the rolling state above: a device's hardware
        # identity doesn't legitimately drift on a behavioral timescale the
        # way login patterns do, so "established" means "first observed,"
        # full stop, and is never overwritten by a later (possibly spoofed)
        # value -- overwriting on mismatch would let a sustained spoof
        # silently become the new normal, defeating the point of the check.
        self._device_signature: dict[str, tuple[str | None, str | None]] = {}

    def _dept_graph(self, department: str) -> nx.DiGraph:
        graph = self.resource_graph_by_dept.get(department)
        if graph is None:
            graph = nx.DiGraph()
            graph.add_nodes_from(RESOURCE_TYPES)
            self.resource_graph_by_dept[department] = graph
        return graph

    # -- pruning -----------------------------------------------------------

    def _prune_bipartite(self, now: pd.Timestamp) -> None:
        while self._bipartite_insertions and (now - self._bipartite_insertions[0][0]) > self.window:
            _, u, d = self._bipartite_insertions.popleft()
            if self.bipartite.has_edge(u, d):
                self.bipartite.remove_edge(u, d)
                if self.bipartite.degree(u) == 0:
                    self.bipartite.remove_node(u)
                if d in self.bipartite and self.bipartite.degree(d) == 0:
                    self.bipartite.remove_node(d)

    def _prune_resource_graph(self, now: pd.Timestamp) -> None:
        while self._resource_edge_insertions and (now - self._resource_edge_insertions[0][0]) > self.window:
            _, department, src, dst = self._resource_edge_insertions.popleft()
            graph = self.resource_graph_by_dept.get(department)
            if graph is not None and graph.has_edge(src, dst):
                graph[src][dst]["count"] = max(0, graph[src][dst]["count"] - 1)
                if graph[src][dst]["count"] == 0:
                    graph.remove_edge(src, dst)
        while self._session_last_seen and (now - self._session_last_seen[0][0]) > self.window:
            _, sid = self._session_last_seen.popleft()
            self._last_resource_in_session.pop(sid, None)
            self._session_foreign_resources.pop(sid, None)
            self._session_last_event_ts.pop(sid, None)

    def _prune_user_resource_history(self, history: deque, now: pd.Timestamp) -> None:
        while history and (now - history[0][0]) > self.window:
            history.popleft()

    # -- feature computation -------------------------------------------------

    def _device_fan_in(self, device_node: str) -> int:
        return self.bipartite.degree(device_node) if device_node in self.bipartite else 0

    def _is_new_edge(self, user_node: str, device_node: str) -> float:
        if self.bipartite.has_edge(user_node, device_node):
            return 0.0
        known_devices = list(self.bipartite.neighbors(user_node)) if user_node in self.bipartite else []
        if not known_devices or device_node not in self.bipartite:
            distance = self.device_distance_cutoff
        else:
            # unweighted BFS out to `cutoff` hops from device_node, then look
            # up whether/how-far any of the user's known devices were reached
            reachable = nx.single_source_shortest_path_length(self.bipartite, device_node, cutoff=self.device_distance_cutoff)
            distances = [reachable[known] for known in known_devices if known in reachable]
            distance = min(distances) if distances else self.device_distance_cutoff
        return 1.0 - 1.0 / (1.0 + distance)

    def _user_device_set_delta(self, user_id: str, device_id: str) -> float:
        weights = self._user_device_ema[user_id]
        total = sum(weights.values())
        current = weights.get(device_id, 0.0)
        return 1.0 - (current / total if total > 0 else 0.0)

    def _access_chain_distance(self, user_id: str, department: str | None, resource: str | None) -> float:
        if resource is None:
            return 0.0  # not applicable to this row (e.g. a failed login with no resource)
        visited = self._user_resource_history[user_id]
        visited_resources = {r for _, r in visited}
        if resource in visited_resources:
            return 0.0
        if not visited:
            return self.sentinel  # cold start: no resource history yet

        # Cost of the DIRECT edge from the user's most recently visited
        # resource to this one, within THEIR OWN DEPARTMENT's
        # resource-transition graph (see module docstring's design note --
        # a shared org-wide graph under-detects lateral movement because
        # nearly every transition type is "normal" for some department
        # somewhere). Deliberately a direct-edge lookup, not a multi-hop
        # shortest-path search: with only ~7 resource nodes even a
        # department-scoped graph would let hub resources (email/vpn)
        # create artificially cheap indirect routes to anything.
        last_resource = visited[-1][1]
        graph = self.resource_graph_by_dept.get(department) if department else None
        if graph is not None and graph.has_edge(last_resource, resource):
            return graph[last_resource][resource]["cost"]
        return self.sentinel

    def _session_foreign_resource_count(self, session_id: str) -> float:
        return float(len(self._session_foreign_resources.get(session_id, ())))

    def _session_hop_seconds(self, session_id: str, ts: pd.Timestamp) -> float:
        last_ts = self._session_last_event_ts.get(session_id)
        if last_ts is None:
            return self.session_hop_sentinel
        return float((ts - last_ts).total_seconds())

    def _device_fingerprint_mismatch(self, device_id: str, device_type: str | None, os_name: str | None) -> float:
        established = self._device_signature.get(device_id)
        if established is None:
            return 0.0  # first time seeing this device_id -- nothing established yet to contradict
        return 0.0 if established == (device_type, os_name) else 1.0

    def _peer_community_deviation(self, user_node: str, device_node: str) -> float:
        comm_id = self._community_of.get(user_node)
        if comm_id is None:
            return 0.0  # no community assignment yet (before first Louvain refresh)
        size = self._community_size.get(comm_id, 0)
        if size <= 1:
            return 0.0
        users_on_device = self._community_device_users.get(comm_id, {}).get(device_node, set())
        peers_on_device = len(users_on_device - {user_node})
        return 1.0 - (peers_on_device / (size - 1))

    # -- community refresh ---------------------------------------------------

    def _refresh_communities(self) -> None:
        user_projection = nx.Graph()
        for node in self.bipartite.nodes():
            if node.startswith("U:"):
                user_projection.add_node(node)
        for node in self.bipartite.nodes():
            if not node.startswith("D:"):
                continue
            neighbors = list(self.bipartite.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, b = neighbors[i], neighbors[j]
                    if user_projection.has_edge(a, b):
                        user_projection[a][b]["weight"] += 1
                    else:
                        user_projection.add_edge(a, b, weight=1)

        self._community_of = {}
        self._community_size = {}
        self._community_device_users = defaultdict(_new_device_users_dict)
        if user_projection.number_of_nodes() == 0:
            return
        if user_projection.number_of_edges() == 0:
            communities = [{n} for n in user_projection.nodes()]
        else:
            communities = list(nx.algorithms.community.louvain_communities(user_projection, weight="weight", seed=42))

        for comm_id, members in enumerate(communities):
            self._community_size[comm_id] = len(members)
            for user_node in members:
                self._community_of[user_node] = comm_id
                if user_node in self.bipartite:
                    for device_node in self.bipartite.neighbors(user_node):
                        self._community_device_users[comm_id][device_node].add(user_node)

    # -- main entrypoint ------------------------------------------------------

    def update(self, event: dict[str, Any]) -> dict[str, Any]:
        user_id = event["user_id"]
        device_id = event["device_id"]
        session_id = event["session_id"]
        resource = event.get("resource_accessed")
        ts = event["timestamp"]
        department = self._department_by_user.get(user_id)
        user_node, device_node = _user_node(user_id), _device_node(device_id)

        self._prune_bipartite(ts)
        self._prune_resource_graph(ts)
        history = self._user_resource_history[user_id]
        self._prune_user_resource_history(history, ts)

        # -- compute features from state as it existed BEFORE this event --
        device_fan_in = self._device_fan_in(device_node)
        is_new_edge = self._is_new_edge(user_node, device_node)
        user_device_set_delta = self._user_device_set_delta(user_id, device_id)
        access_chain_distance = self._access_chain_distance(user_id, department, resource)
        device_type, os_name = event.get("device_type"), event.get("os")
        device_fingerprint_mismatch = self._device_fingerprint_mismatch(device_id, device_type, os_name)
        session_foreign_resource_count = self._session_foreign_resource_count(session_id)
        session_hop_seconds = self._session_hop_seconds(session_id, ts)
        if self._events_since_refresh >= self.louvain_refresh_events:
            self._refresh_communities()
            self._events_since_refresh = 0
        peer_community_deviation = self._peer_community_deviation(user_node, device_node)

        features = {
            "device_fan_in": float(device_fan_in),
            "user_device_set_delta": user_device_set_delta,
            "is_new_edge": is_new_edge,
            "access_chain_distance": access_chain_distance,
            "peer_community_deviation": peer_community_deviation,
            "device_fingerprint_mismatch": device_fingerprint_mismatch,
            "session_foreign_resource_count": session_foreign_resource_count,
            "session_hop_seconds": session_hop_seconds,
        }

        # -- state updates, using THIS event --
        if not self.bipartite.has_edge(user_node, device_node):
            self.bipartite.add_edge(user_node, device_node)
        self._bipartite_insertions.append((ts, user_node, device_node))

        if device_id not in self._device_signature:
            self._device_signature[device_id] = (device_type, os_name)

        weights = self._user_device_ema[user_id]
        for d in list(weights.keys()):
            weights[d] *= self.device_ema_gamma
        weights[device_id] = weights.get(device_id, 0.0) + (1 - self.device_ema_gamma)

        if resource is not None and department is not None:
            prev_resource = self._last_resource_in_session.get(session_id)
            if prev_resource is not None and prev_resource != resource:
                graph = self._dept_graph(department)
                if graph.has_edge(prev_resource, resource):
                    graph[prev_resource][resource]["count"] += 1
                else:
                    graph.add_edge(prev_resource, resource, count=1)
                graph[prev_resource][resource]["cost"] = 1.0 / graph[prev_resource][resource]["count"]
                self._resource_edge_insertions.append((ts, department, prev_resource, resource))
            self._last_resource_in_session[session_id] = resource
            self._session_last_seen.append((ts, session_id))
            history.append((ts, resource))

            home_resources = RESOURCE_TYPES_BY_DEPT.get(department, RESOURCE_TYPES_BY_DEPT["_default"])
            if resource not in home_resources:
                self._session_foreign_resources[session_id].add(resource)
            self._session_last_event_ts[session_id] = ts

        self._events_since_refresh += 1

        return features

    def compute_batch(self, events: pd.DataFrame) -> pd.DataFrame:
        """Replay `update()` over `events` (must be chronologically sorted)
        from this state's current (typically empty, for a fresh instance)
        starting point.
        """
        rows: list[dict[str, Any]] = []
        for event in events.to_dict("records"):
            feats = self.update(event)
            feats["record_id"] = event["record_id"]
            rows.append(feats)
        return pd.DataFrame(rows, columns=["record_id"] + GRAPH_FEATURE_COLUMNS)
