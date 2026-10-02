"""
Checks on NetworkX graphs produced by ``beg.build.networkx_graphs.build_nx_graph_for_model``,
either in memory or pickled per model.

Graph contract:

* ``G.graph`` has ``config_name`` and ``model_id`` (derived C2/C7 graphs also
  carry ``derived_from``, ``weather_city``, ``graph_feature_cols`` / ``_values``)
* node key = ``instance_name``; node attrs include ``node_type`` (the node table),
  ``node_id`` (0..N-1) and all non-null DataFrame columns
* the builder *drops NaN attributes*, so a node lacking a key that its siblings
  have means the feature could not be computed for that node
* edges carry ``edge_type``/``rel`` and ``is_reverse`` (1 = added for
  ``bidirectional`` relations); a node named in an edge table but absent from
  the node tables is created with ``node_type == "(missing_from_nodes)"``
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from beg.validation.core import CheckResult, examples
from beg.validation.dataframe_checks import ADJACENCY_RELS, HIERARCHICAL_RELS
from beg.validation.spec import (
    NX_MISSING_NODE_TYPE,
    NX_RESERVED_EDGE_ATTRS,
    NX_RESERVED_NODE_ATTRS,
    ONE_HOT_GROUPS,
    ConfigSpec,
    element_is_weather_exposed,
    is_identifier_column,
    rule_for_column,
)

SCOPE = "nx"
Triple = Tuple[str, str, str]

# node tables that may legitimately have zero instances in a building
OPTIONAL_NODE_TABLES = {"comp_aperture"}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_bad_number(v: Any) -> bool:
    return isinstance(v, float) and (math.isnan(v) or math.isinf(v))


def node_type_of(G: nx.Graph, n: Any) -> str:
    return str(G.nodes[n].get("node_type", "unknown"))


def edge_triples(G: nx.MultiDiGraph) -> Counter:
    c: Counter = Counter()
    for u, v, d in G.edges(data=True):
        c[(node_type_of(G, u), str(d.get("rel", d.get("edge_type", ""))), node_type_of(G, v))] += 1
    return c


# --------------------------------------------------------------------------
# single graph
# --------------------------------------------------------------------------

def check_graph_metadata(G: nx.Graph, spec: ConfigSpec, *, expected_model_id: Optional[str] = None) -> List[CheckResult]:
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", expected_model_id or "")), scope=SCOPE)
    r: List[CheckResult] = []
    for key in ("config_name", "model_id"):
        if key not in G.graph:
            r.append(CheckResult.fail("nx.graph_metadata", f"G.graph lacks '{key}'", **kw))
    if expected_model_id is not None and str(G.graph.get("model_id")) != str(expected_model_id):
        r.append(CheckResult.fail("nx.graph_model_id_matches", f"G.graph['model_id']={G.graph.get('model_id')!r} but expected {expected_model_id!r} (file name / dataset id)", **kw))
    cname = str(G.graph.get("config_name", ""))
    if cname and spec.name not in cname and spec.full_name not in cname and str(G.graph.get("derived_from", "")).find(spec.name) < 0:
        r.append(CheckResult.warn("nx.graph_config_matches", f"G.graph['config_name']={cname!r} does not mention config '{spec.name}'", **kw))
    if not isinstance(G, nx.MultiDiGraph):
        r.append(CheckResult.warn("nx.graph_class", f"expected MultiDiGraph, got {type(G).__name__}", **kw))
    if G.number_of_nodes() == 0:
        r.append(CheckResult.fail("nx.graph_nonempty", "graph has no nodes", **kw))
    return r


def check_node_types(G: nx.Graph, spec: ConfigSpec) -> List[CheckResult]:
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    counts = Counter(node_type_of(G, n) for n in G.nodes)
    expected = set(spec.node_tables)
    observed = set(counts)
    if NX_MISSING_NODE_TYPE in observed:
        r.append(CheckResult.fail("nx.no_missing_from_nodes", f"{counts[NX_MISSING_NODE_TYPE]} nodes referenced by edges but absent from node tables e.g. {examples([n for n in G.nodes if node_type_of(G, n) == NX_MISSING_NODE_TYPE])}", **kw))
    unexpected = sorted(observed - expected - {NX_MISSING_NODE_TYPE})
    if unexpected:
        r.append(CheckResult.fail("nx.node_types_declared", f"node types not in config: {unexpected}", **kw))
    missing = sorted(expected - observed)
    for t in missing:
        sev = CheckResult.warn if t in OPTIONAL_NODE_TABLES else CheckResult.fail
        r.append(sev("nx.node_types_present", f"config node type '{t}' has no nodes in this graph", **kw))
    if not missing and not unexpected:
        r.append(CheckResult.ok("nx.node_types_present", "node types: " + ", ".join(f"{t}={counts[t]}" for t in spec.node_tables), details=dict(counts), **kw))
    for t in spec.singleton_node_tables:
        if counts.get(t, 0) > 1:
            r.append(CheckResult.fail("nx.singleton_node_type", f"'{t}' must be a single node, found {counts[t]}", **kw))
    # node_id contiguity
    ids = [G.nodes[n].get("node_id") for n in G.nodes]
    if any(i is None for i in ids):
        r.append(CheckResult.fail("nx.node_id_present", f"{sum(i is None for i in ids)} nodes without node_id", **kw))
    else:
        s = sorted(int(i) for i in ids)
        if s != list(range(len(s))):
            r.append(CheckResult.fail("nx.node_id_contiguous", f"node_id values are not 0..N-1 (N={len(s)}, min={s[0]}, max={s[-1]}, unique={len(set(s))})", **kw))
    return r


def check_edge_types(G: nx.MultiDiGraph, spec: ConfigSpec) -> List[CheckResult]:
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    trip = edge_triples(G)
    expected = spec.expected_triples
    counts = Counter(node_type_of(G, n) for n in G.nodes)

    unexpected = {t: c for t, c in trip.items() if t not in expected}
    if unexpected:
        r.append(CheckResult.fail("nx.edge_types_declared", f"edge triples not derivable from config: {unexpected}", **kw))
    for e in spec.edges:
        fwd, rev = e.forward, e.reverse
        if trip.get(fwd, 0) == 0 and (not e.is_self_type or trip.get(rev, 0) == 0):
            if e.rel in ADJACENCY_RELS and counts.get(e.src_cat, 0) < 2:
                continue  # single zone / element: no adjacency possible
            if e.rel in ADJACENCY_RELS:
                r.append(CheckResult.warn("nx.edge_types_present", f"no '{e.rel}' edges although {counts.get(e.src_cat, 0)} {e.src_cat} nodes exist", **kw))
            elif e.src_cat in OPTIONAL_NODE_TABLES or e.tgt_cat in OPTIONAL_NODE_TABLES:
                if counts.get(e.src_cat, 0) and counts.get(e.tgt_cat, 0):
                    r.append(CheckResult.fail("nx.edge_types_present", f"no '{e.rel}' edges {e.src_cat}→{e.tgt_cat} although both node types exist", **kw))
            else:
                r.append(CheckResult.fail("nx.edge_types_present", f"expected relation '{e.rel}' {e.src_cat}→{e.tgt_cat} has no edges", **kw))
    for fwd in spec.one_directional_triples:
        rev = (fwd[2], fwd[1], fwd[0])
        if trip.get(rev, 0):
            r.append(CheckResult.fail("nx.one_directional_respected", f"relation {fwd} is one-directional in config but {trip[rev]} reverse edges exist", **kw))
    if not unexpected:
        r.append(CheckResult.ok("nx.edge_types_declared", f"{len(trip)} edge triples, {G.number_of_edges()} edges", details={str(k): v for k, v in trip.items()}, **kw))
    return r


def check_edge_structure(G: nx.MultiDiGraph, spec: ConfigSpec) -> List[CheckResult]:
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    loops = [(u, v) for u, v in G.edges() if u == v]
    if loops:
        r.append(CheckResult.fail("nx.no_self_loops", f"{len(loops)} self loops e.g. {examples([u for u, _ in loops])}", **kw))
    # duplicate parallel edges with identical (rel, is_reverse)
    seen: Counter = Counter()
    for u, v, d in G.edges(data=True):
        seen[(u, v, str(d.get("rel", d.get("edge_type"))), int(d.get("is_reverse", 0)))] += 1
    dups = {k: c for k, c in seen.items() if c > 1}
    if dups:
        r.append(CheckResult.fail("nx.no_duplicate_edges", f"{len(dups)} (u,v,rel,direction) combinations appear more than once e.g. {examples([f'{k[0]}→{k[1]} {k[2]}' for k in dups])}", **kw))
    # bidirectional symmetry: for rels marked bidirectional, the forward edge set
    # equals the mirrored reverse edge set
    by_rel: Dict[str, Tuple[Set[Tuple[Any, Any]], Set[Tuple[Any, Any]]]] = defaultdict(lambda: (set(), set()))
    for u, v, d in G.edges(data=True):
        rel = str(d.get("rel", d.get("edge_type")))
        by_rel[rel][int(d.get("is_reverse", 0)) == 1].add((u, v))
    bidir_rels = {e.rel for e in spec.edges if e.bidirectional}
    for rel in bidir_rels:
        fwd, rev = by_rel.get(rel, (set(), set()))
        if not fwd and not rev:
            continue
        mirrored = {(v, u) for u, v in fwd}
        if mirrored != rev:
            r.append(CheckResult.fail("nx.bidirectional_symmetric", f"'{rel}': {len(fwd)} forward vs {len(rev)} reverse edges; {len(mirrored - rev)} forward edges lack a reverse, {len(rev - mirrored)} reverse edges lack a forward", **kw))
        else:
            r.append(CheckResult.ok("nx.bidirectional_symmetric", f"'{rel}': {len(fwd)} edge pairs symmetric", **kw))
    for rel, (_, rev) in by_rel.items():
        if rel not in bidir_rels and rev:
            r.append(CheckResult.fail("nx.reverse_only_for_bidirectional", f"'{rel}' is not bidirectional in config but has {len(rev)} is_reverse=1 edges", **kw))
    # hierarchical cardinality
    for e in spec.edges:
        if e.rel not in HIERARCHICAL_RELS or e.is_self_type:
            continue
        parents: Counter = Counter()
        for u, v, d in G.edges(data=True):
            if str(d.get("rel", d.get("edge_type"))) == e.rel and int(d.get("is_reverse", 0)) == 0 and node_type_of(G, u) == e.src_cat and node_type_of(G, v) == e.tgt_cat:
                parents[u] += 1
        children = [n for n in G.nodes if node_type_of(G, n) == e.src_cat]
        no_parent = [n for n in children if parents.get(n, 0) == 0]
        multi = [n for n in children if parents.get(n, 0) > 1]
        if no_parent:
            r.append(CheckResult.fail("nx.child_has_parent", f"{len(no_parent)}/{len(children)} {e.src_cat} nodes without '{e.rel}' → {e.tgt_cat} e.g. {examples(no_parent)}", **kw))
        if multi:
            r.append(CheckResult.fail("nx.child_single_parent", f"{len(multi)} {e.src_cat} nodes with several '{e.rel}' parents e.g. {examples(multi)}", **kw))
        if children and not no_parent and not multi:
            r.append(CheckResult.ok("nx.child_has_parent", f"every {e.src_cat} has exactly one {e.rel} → {e.tgt_cat}", **kw))
    return r


def check_connectivity(G: nx.Graph, spec: ConfigSpec) -> List[CheckResult]:
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    n = G.number_of_nodes()
    if n == 0:
        return r
    only_adjacency = bool(spec.edges) and all(e.rel in ADJACENCY_RELS for e in spec.edges) and len(spec.node_tables) == 1
    isolated = [v for v, d in G.degree() if d == 0]
    if isolated:
        sev = CheckResult.warn if only_adjacency else CheckResult.fail
        r.append(sev("nx.no_isolated_nodes", f"{len(isolated)}/{n} isolated nodes ({Counter(node_type_of(G, v) for v in isolated)}) e.g. {examples(isolated)}", **kw))
    und = G.to_undirected(as_view=True) if G.is_directed() else G
    comps = list(nx.connected_components(und))
    if len(comps) > 1:
        sizes = sorted((len(c) for c in comps), reverse=True)
        sev = CheckResult.warn if only_adjacency else CheckResult.fail
        r.append(sev("nx.weakly_connected", f"graph has {len(comps)} components (sizes {sizes[:6]}); message passing cannot reach across components", **kw))
    else:
        r.append(CheckResult.ok("nx.weakly_connected", f"single component, {n} nodes", **kw))
    degs = [d for _, d in G.degree()]
    r.append(CheckResult.ok("nx.degree_stats", f"degree min={min(degs)} mean={sum(degs) / len(degs):.1f} max={max(degs)}", details={"min": min(degs), "mean": sum(degs) / len(degs), "max": max(degs)}, **kw))
    return r


def check_node_attributes(G: nx.Graph, spec: ConfigSpec) -> List[CheckResult]:
    """Attribute completeness / types / domains / one-hot per node type."""
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    by_type: Dict[str, List[Any]] = defaultdict(list)
    for n in G.nodes:
        by_type[node_type_of(G, n)].append(n)
    for t, ns in by_type.items():
        keysets = [set(G.nodes[n]) for n in ns]
        union: Set[str] = set().union(*keysets)
        # completeness: the builder drops NaNs, so partial keys = uncomputable feature values
        missing_counter: Counter = Counter()
        for ks in keysets:
            for k in union - ks:
                missing_counter[k] += 1
        if missing_counter:
            worst = missing_counter.most_common(8)
            hint = ""
            if all(k.startswith(("is_", "has_")) or k in ("sun_exposure", "wind_exposure") for k in missing_counter):
                hint = " — boolean flags should be COALESCE(...,0) in SQL, not NULL"
            elif any(k.startswith("r_") for k in missing_counter):
                hint = " — NULL from NULLIF/division (e.g. no exterior window) is silently turned into 0.0 by the tensor builder"
            r.append(CheckResult.fail("nx.node_attrs_complete", f"'{t}': {len(missing_counter)} attributes missing on some nodes (NULL in source): " + ", ".join(f"{k} ({c}/{len(ns)})" for k, c in worst) + hint, details={"missing": dict(missing_counter)}, **kw))
        else:
            r.append(CheckResult.ok("nx.node_attrs_complete", f"'{t}': all {len(ns)} nodes share the same {len(union)} attributes", **kw))
        # types & values
        bad_type: Counter = Counter()
        bad_num: Counter = Counter()
        str_attrs: Set[str] = set()
        bool_attrs: Set[str] = set()
        domain_viol: Dict[str, List[Any]] = defaultdict(list)
        onehot_viol: Counter = Counter()
        for n in ns:
            d = G.nodes[n]
            for k, v in d.items():
                if k in NX_RESERVED_NODE_ATTRS:
                    continue
                if v is None:
                    bad_type[k] += 1
                elif isinstance(v, bool):
                    bool_attrs.add(k)
                elif isinstance(v, str):
                    if not is_identifier_column(k):
                        str_attrs.add(k)
                elif _is_number(v):
                    if _is_bad_number(v):
                        bad_num[k] += 1
                    else:
                        rule = rule_for_column(k)
                        if rule is not None:
                            viol = False
                            if rule.lo is not None:
                                viol |= (v <= rule.lo) if rule.lo_exclusive else (v < rule.lo)
                            if rule.hi is not None:
                                viol |= v > rule.hi
                            if rule.integer and float(v) != math.floor(float(v)):
                                viol = True
                            if viol:
                                domain_viol[k].append(v)
                elif not is_identifier_column(k):
                    bad_type[k] += 1
            for g in ONE_HOT_GROUPS:
                cols = g.resolve(list(d.keys()))
                if len(cols) < 2:
                    continue
                s = sum(int(d.get(c) or 0) for c in cols)
                if s > 1 or (s < 1 and not g.optional):
                    onehot_viol[g.name] += 1
        if bad_type:
            r.append(CheckResult.fail("nx.node_attr_types", f"'{t}': None/non-numeric feature values: {dict(bad_type.most_common(6))}", **kw))
        if bad_num:
            r.append(CheckResult.fail("nx.node_attr_finite", f"'{t}': NaN/inf attribute values: {dict(bad_num.most_common(6))}", **kw))
        if bool_attrs:
            r.append(CheckResult.warn("nx.node_attr_bool", f"'{t}': bool-typed attributes (expected 0/1 ints): {sorted(bool_attrs)}", **kw))
        if str_attrs:
            # informational per graph; the collection check raises one WARN per node type
            r.append(CheckResult.ok("nx.node_attr_strings", f"'{t}': string attributes (dropped by feature builders): {sorted(str_attrs)}", details={"columns": sorted(str_attrs)}, **kw))
        for k, vals in domain_viol.items():
            rule = rule_for_column(k)
            mk = CheckResult.fail if rule and rule.severity == "FAIL" else CheckResult.warn
            r.append(mk("nx.node_attr_domain", f"'{t}'.{k}: {len(vals)}/{len(ns)} nodes violate '{rule.note if rule else ''}' e.g. {examples(vals)}", **kw))
        for gname, c in onehot_viol.items():
            r.append(CheckResult.fail("nx.node_one_hot_exclusive", f"'{t}': one-hot group '{gname}' not exclusive on {c}/{len(ns)} nodes", **kw))
    return r


def check_edge_attributes(G: nx.MultiDiGraph, spec: ConfigSpec) -> List[CheckResult]:
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    # group by (src_type, rel, tgt_type): the same rel name (e.g. weather_for) can come from
    # different edge tables with different attribute columns
    by_rel: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for u, v, d in G.edges(data=True):
        by_rel[(node_type_of(G, u), str(d.get("rel", d.get("edge_type"))), node_type_of(G, v))].append(d)
    for rel, ds in by_rel.items():
        keysets = [set(d) - NX_RESERVED_EDGE_ATTRS for d in ds]
        union: Set[str] = set().union(*keysets)
        missing: Counter = Counter()
        for ks in keysets:
            for k in union - ks:
                missing[k] += 1
        if missing:
            r.append(CheckResult.fail("nx.edge_attrs_complete", f"'{rel}': attributes missing on some edges (NULL in source): {dict(missing.most_common(6))} of {len(ds)}", **kw))
        bad: Counter = Counter()
        viol: Dict[str, List[Any]] = defaultdict(list)
        for d in ds:
            for k, v in d.items():
                if k in NX_RESERVED_EDGE_ATTRS or is_identifier_column(k):
                    continue
                if v is None or _is_bad_number(v):
                    bad[k] += 1
                elif _is_number(v):
                    rule = rule_for_column(k)
                    if rule is None:
                        continue
                    if (rule.lo is not None and ((v <= rule.lo) if rule.lo_exclusive else (v < rule.lo))) or (rule.hi is not None and v > rule.hi):
                        viol[k].append(v)
        if bad:
            r.append(CheckResult.fail("nx.edge_attr_finite", f"'{rel}': None/NaN/inf edge attributes {dict(bad)}", **kw))
        for k, vals in viol.items():
            rule = rule_for_column(k)
            mk = CheckResult.fail if rule and rule.severity == "FAIL" else CheckResult.warn
            r.append(mk("nx.edge_attr_domain", f"'{rel}'.{k}: {len(vals)} edges violate '{rule.note if rule else ''}' e.g. {examples(vals)}", **kw))
    # forward/reverse pairs must carry identical attributes
    fwd: Dict[Tuple[Any, Any, str], dict] = {}
    rev: Dict[Tuple[Any, Any, str], dict] = {}
    for u, v, d in G.edges(data=True):
        rel = str(d.get("rel", d.get("edge_type")))
        (rev if int(d.get("is_reverse", 0)) == 1 else fwd)[(u, v, rel)] = d
    mism = 0
    for (u, v, rel), d in rev.items():
        f = fwd.get((v, u, rel))
        if f is None:
            continue
        a = {k: x for k, x in d.items() if k not in NX_RESERVED_EDGE_ATTRS}
        b = {k: x for k, x in f.items() if k not in NX_RESERVED_EDGE_ATTRS}
        if a != b:
            mism += 1
    if mism:
        r.append(CheckResult.fail("nx.reverse_edge_attrs_equal", f"{mism} reverse edges carry different attributes than their forward edge", **kw))
    return r


def check_semantics(G: nx.MultiDiGraph, spec: ConfigSpec) -> List[CheckResult]:
    """Domain rules that need the topology (weather exposure, homogeneous unified edges)."""
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []

    # external weather: weather_for targets == exposed elements
    for e in spec.edges:
        if e.rel != "weather_for" or e.src_cat != "weather_city":
            continue
        targets = {v for u, v, d in G.edges(data=True) if str(d.get("rel", d.get("edge_type"))) == e.rel and node_type_of(G, u) == "weather_city" and node_type_of(G, v) == e.tgt_cat}
        candidates = [n for n in G.nodes if node_type_of(G, n) == e.tgt_cat]
        if not candidates:
            continue
        sample = G.nodes[candidates[0]]
        if {"is_outdoors", "is_wall"} <= set(sample):
            exposed = {n for n in candidates if element_is_weather_exposed(G.nodes[n])}
            missing, extra = exposed - targets, targets - exposed
            if missing or extra:
                r.append(CheckResult.fail("nx.weather_edges_match_exposure", f"{e.tgt_cat}: {len(missing)} exposed nodes without weather edge, {len(extra)} weather edges to non-exposed nodes e.g. {examples(missing | extra)}", **kw))
            else:
                r.append(CheckResult.ok("nx.weather_edges_match_exposure", f"weather edges cover exactly the {len(exposed)} exposed {e.tgt_cat} nodes", **kw))
        elif "is_window" in sample:
            non_win = {n for n in targets if int(G.nodes[n].get("is_window", 0) or 0) != 1}
            if non_win:
                r.append(CheckResult.fail("nx.weather_edges_match_exposure", f"{len(non_win)} weather edges to non-window apertures", **kw))
            # aperture exposed iff its parent element is exposed
            parent_of = {u: v for u, v, d in G.edges(data=True) if str(d.get("rel")) == "part_of" and int(d.get("is_reverse", 0)) == 0}
            exposed = {n for n in candidates if int(G.nodes[n].get("is_window", 0) or 0) == 1 and parent_of.get(n) in G.nodes and element_is_weather_exposed(G.nodes[parent_of[n]])}
            if exposed != targets and parent_of:
                r.append(CheckResult.fail("nx.weather_edges_match_exposure", f"{e.tgt_cat}: {len(exposed - targets)} exposed windows lack a weather edge, {len(targets - exposed)} weather edges to unexposed apertures", **kw))
        if targets and not any(x.name == "nx.weather_edges_match_exposure" and x.status.value == "FAIL" for x in r):
            pass
    # a weather node must have edges (otherwise the weather information is unreachable)
    for t in spec.weather_node_tables:
        for n in G.nodes:
            if node_type_of(G, n) == t and G.degree(n) == 0:
                r.append(CheckResult.fail("nx.weather_node_connected", f"weather node '{n}' ({t}) is isolated", **kw))

    # homogeneous unified component graph
    for e in spec.edges:
        if e.rel != "component_relationship":
            continue
        bad_same, bad_cross, bad_ap = 0, 0, 0
        for u, v, d in G.edges(data=True):
            if str(d.get("rel")) != e.rel:
                continue
            rt = str(d.get("relationship_type", ""))
            zu, zv = G.nodes[u].get("zone_name"), G.nodes[v].get("zone_name")
            if rt == "same_zone" and zu is not None and zv is not None and zu != zv:
                bad_same += 1
            if int(d.get("is_cross_zone", 0) or 0) == 1 and zu is not None and zu == zv:
                bad_cross += 1
            if rt == "aperture_to_element":
                a, b = (u, v) if int(d.get("is_reverse", 0)) == 0 else (v, u)
                if int(G.nodes[a].get("is_aperture", 0) or 0) != 1 or int(G.nodes[b].get("is_aperture", 0) or 0) == 1:
                    bad_ap += 1
        if bad_same or bad_cross or bad_ap:
            r.append(CheckResult.fail("nx.unified_edge_semantics", f"unified edges inconsistent with node attrs: same_zone across zones={bad_same}, is_cross_zone within zone={bad_cross}, aperture_to_element endpoints wrong={bad_ap}", **kw))
        else:
            r.append(CheckResult.ok("nx.unified_edge_semantics", "component_relationship edges consistent with zone/aperture attributes", **kw))
    # comp_base_element: is_aperture ⇔ any aperture kind flag
    for n in G.nodes:
        d = G.nodes[n]
        if "is_aperture" in d and any(k in d for k in ("is_window", "is_door", "is_glass_door", "is_tubular_daylight")):
            kinds = sum(int(d.get(k, 0) or 0) for k in ("is_window", "is_door", "is_glass_door", "is_tubular_daylight"))
            if bool(int(d.get("is_aperture", 0) or 0)) != bool(kinds):
                r.append(CheckResult.fail("nx.is_aperture_consistent", f"node '{n}': is_aperture={d.get('is_aperture')} but aperture-kind flags sum={kinds}", **kw))
                break
    return r


def check_graph_level_weather(G: nx.Graph, spec: ConfigSpec, *, expected_dim: Optional[int] = None) -> List[CheckResult]:
    """Derived homogeneous graph-weather variants (C2/C7) keep weather in ``G.graph``."""
    kw = dict(config=spec.name, model_id=str(G.graph.get("model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    g = G.graph
    if "graph_feature_cols" not in g and "weather_city" not in g:
        return [CheckResult.fail("nx.graph_weather_present", "derived graph-weather graph lacks G.graph['weather_city'/'graph_feature_cols']", **kw)]
    cols = list(g.get("graph_feature_cols") or [])
    vals = g.get("graph_feature_values") or {}
    if isinstance(vals, dict):
        missing_keys = [c for c in cols if c not in vals]
        if missing_keys:
            r.append(CheckResult.fail("nx.graph_weather_complete", f"{len(missing_keys)} graph_feature_cols without a value e.g. {examples(missing_keys)}", **kw))
        bad = {c: vals[c] for c in cols if c in vals and (vals[c] is None or _is_bad_number(vals[c]) or not _is_number(vals[c]))}
        if bad:
            r.append(CheckResult.fail("nx.graph_weather_finite", f"{len(bad)} graph-level weather features are None/NaN/non-numeric (they become 0.0 in the tensor): {examples(list(bad))}", details={"bad": {k: str(v) for k, v in bad.items()}}, **kw))
        else:
            r.append(CheckResult.ok("nx.graph_weather_finite", f"all {len(cols)} graph-level weather features numeric", **kw))
        for c in cols:
            v = vals.get(c)
            rule = rule_for_column(c)
            if rule is None or not _is_number(v):
                continue
            if (rule.lo is not None and v < rule.lo) or (rule.hi is not None and v > rule.hi):
                r.append((CheckResult.fail if rule.severity == "FAIL" else CheckResult.warn)("nx.graph_weather_domain", f"graph feature {c}={v} violates '{rule.note}'", **kw))
    if expected_dim is not None and len(cols) != expected_dim:
        r.append(CheckResult.fail("nx.graph_weather_dim", f"graph_feature_cols has {len(cols)} entries, expected {expected_dim}", **kw))
    if len(set(cols)) != len(cols):
        r.append(CheckResult.fail("nx.graph_weather_cols_unique", "duplicate names in graph_feature_cols", **kw))
    if spec.weather_variant == "no_weather" and "derived_from" not in g:
        r.append(CheckResult.warn("nx.graph_weather_derived_from", "graph-weather graph lacks 'derived_from' metadata", **kw))
    return r


def run_graph_checks(G: nx.MultiDiGraph, spec: ConfigSpec, *, expected_model_id: Optional[str] = None, graph_level_weather: Optional[bool] = None, graph_weather_dim: Optional[int] = None) -> List[CheckResult]:
    """Run every single-graph check.  ``spec`` is the config the graph was built from
    (for derived C2/C7 graphs pass the *base* C1/C6 spec and ``graph_level_weather=True``)."""
    if graph_level_weather is None:
        graph_level_weather = "graph_feature_cols" in G.graph or "derived_from" in G.graph
    results: List[CheckResult] = []
    results += check_graph_metadata(G, spec, expected_model_id=expected_model_id)
    results += check_node_types(G, spec)
    results += check_edge_types(G, spec)
    results += check_edge_structure(G, spec)
    results += check_connectivity(G, spec)
    results += check_node_attributes(G, spec)
    results += check_edge_attributes(G, spec)
    results += check_semantics(G, spec)
    if graph_level_weather:
        results += check_graph_level_weather(G, spec, expected_dim=graph_weather_dim)
    return results


# --------------------------------------------------------------------------
# collections of graphs (same config)
# --------------------------------------------------------------------------

def check_graph_collection(graphs: Sequence[nx.Graph], spec: ConfigSpec, *, min_graphs_for_variance: int = 20) -> List[CheckResult]:
    """Cross-graph consistency for one configuration: stable attribute schema per
    node type, unique model ids, constant attributes across the whole collection."""
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    if not graphs:
        return [CheckResult.fail("nx.collection_nonempty", "no graphs", **kw)]
    mids = [str(G.graph.get("model_id")) for G in graphs]
    dup = [m for m, c in Counter(mids).items() if c > 1]
    if dup:
        r.append(CheckResult.fail("nx.collection_model_ids_unique", f"duplicate model_ids: {examples(dup)}", **kw))
    cnames = Counter(str(G.graph.get("config_name")) for G in graphs)
    if len(cnames) > 1:
        r.append(CheckResult.fail("nx.collection_single_config", f"graphs from several configs mixed: {dict(cnames)}", **kw))

    schema: Dict[str, Counter] = defaultdict(Counter)   # node_type -> attr -> #graphs having it on ≥1 node
    values: Dict[Tuple[str, str], Set[Any]] = defaultdict(set)
    per_type_counts: Dict[str, List[int]] = defaultdict(list)
    for G in graphs:
        seen_keys: Dict[str, Set[str]] = defaultdict(set)
        cnt: Counter = Counter()
        for _, d in G.nodes(data=True):
            t = str(d.get("node_type"))
            cnt[t] += 1
            seen_keys[t].update(k for k in d if k not in NX_RESERVED_NODE_ATTRS)
            for k, v in d.items():
                if k not in NX_RESERVED_NODE_ATTRS and _is_number(v) and len(values[(t, k)]) < 3:
                    values[(t, k)].add(v)
        for t, ks in seen_keys.items():
            for k in ks:
                schema[t][k] += 1
        for t in spec.node_tables:
            per_type_counts[t].append(cnt.get(t, 0))
    n_graphs = len(graphs)
    str_attrs: Dict[str, Set[str]] = defaultdict(set)
    for G in graphs:
        for _, d in G.nodes(data=True):
            for k, v in d.items():
                if isinstance(v, str) and not is_identifier_column(k) and k not in NX_RESERVED_NODE_ATTRS:
                    str_attrs[str(d.get("node_type"))].add(k)
    for t, ks in str_attrs.items():
        r.append(CheckResult.warn("nx.collection_string_attrs", f"'{t}': string attributes are not usable as features and are dropped silently by the tensor builders: {sorted(ks)} — encode (one-hot) or drop them explicitly", **kw))
    for t, attr_counts in schema.items():
        unstable = {k: c for k, c in attr_counts.items() if c != n_graphs}
        if unstable:
            r.append(CheckResult.fail("nx.collection_attr_schema_stable", f"'{t}': {len(unstable)} attributes present in only some graphs (feature vector would change shape/meaning): {dict(sorted(unstable.items(), key=lambda kv: kv[1])[:8])} / {n_graphs} graphs", details={"unstable": unstable}, **kw))
        else:
            r.append(CheckResult.ok("nx.collection_attr_schema_stable", f"'{t}': {len(attr_counts)} attributes present in all {n_graphs} graphs", **kw))
    if n_graphs >= min_graphs_for_variance:
        const = defaultdict(list)
        for (t, k), vs in values.items():
            if len(vs) == 1 and not is_identifier_column(k):
                const[t].append(k)
        for t, ks in const.items():
            r.append(CheckResult.warn("nx.collection_constant_attrs", f"'{t}': attributes constant across all {n_graphs} graphs: {sorted(ks)[:15]}{' …' if len(ks) > 15 else ''}", **kw))
    for t, counts in per_type_counts.items():
        zeros = sum(1 for c in counts if c == 0)
        if zeros and t not in OPTIONAL_NODE_TABLES:
            r.append(CheckResult.fail("nx.collection_node_type_coverage", f"'{t}' absent in {zeros}/{n_graphs} graphs", **kw))
        r.append(CheckResult.ok("nx.collection_node_type_stats", f"'{t}': nodes/graph min={min(counts)} max={max(counts)} mean={sum(counts) / len(counts):.1f}", **kw))
    return r
