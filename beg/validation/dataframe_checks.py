"""
Checks on the graph DataFrames of one configuration (``out["data"][config]``).

Input contract (what ``beg.pipeline.build_graph_dataframes`` returns per config)::

    payload = {
        "model_ids": [...],
        "nodes": {node_table: DataFrame},  # one row per node, keyed by (model_id, instance_name)
        "edges": {edge_table: DataFrame},  # one row per edge, endpoints in config src_col/tgt_col
    }

Chunked payloads (``{"chunks": [{"batches": [payload, ...]}]}``) are flattened by
:func:`flatten_payload`.

All checks are vectorised with pandas so they scale to the full 13k-model
dataset.  Per-model failures are reported as one CheckResult per (check,
table) with counts and example ``model_id`` values in ``details``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from beg.validation.core import CheckResult, examples, pct
from beg.validation.spec import (
    ONE_HOT_GROUPS,
    ONE_HOT_LABEL_COLUMNS,
    SHADING_COLUMNS,
    ConfigSpec,
    EdgeSpec,
    element_is_weather_exposed,
    is_identifier_column,
    rule_for_column,
)

SCOPE = "dataframe"
HIERARCHICAL_RELS = {"belongs_to", "part_of"}
ADJACENCY_RELS = {"adjacent_surface", "adjacent_zone", "component_relationship"}
MODEL_ID = "model_id"
NODE_KEY = "instance_name"


# --------------------------------------------------------------------------
# payload helpers
# --------------------------------------------------------------------------

def flatten_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``{"model_ids", "nodes", "edges"}`` from a flat or chunked payload."""
    if "nodes" in payload and "edges" in payload:
        return {
            "model_ids": list(payload.get("model_ids") or []),
            "nodes": dict(payload["nodes"]),
            "edges": dict(payload["edges"]),
        }
    node_parts: Dict[str, List[pd.DataFrame]] = {}
    edge_parts: Dict[str, List[pd.DataFrame]] = {}
    model_ids: List[Any] = []
    for ch in payload.get("chunks", []):
        for b in ch.get("batches", []):
            model_ids.extend(b.get("model_ids") or [])
            for k, df in (b.get("nodes") or {}).items():
                node_parts.setdefault(k, []).append(df)
            for k, df in (b.get("edges") or {}).items():
                edge_parts.setdefault(k, []).append(df)
    return {
        "model_ids": model_ids,
        "nodes": {k: pd.concat(v, ignore_index=True) for k, v in node_parts.items()},
        "edges": {k: pd.concat(v, ignore_index=True) for k, v in edge_parts.items()},
    }


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Numeric, non-identifier columns — what the PyG builders turn into ``x``."""
    return [c for c in df.columns if not is_identifier_column(c) and pd.api.types.is_numeric_dtype(df[c])]


def _node_index(df: pd.DataFrame, key_col: str = NODE_KEY) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[[MODEL_ID, key_col]].astype(str))


def _fail_or_warn(sev: str):
    return CheckResult.fail if sev == "FAIL" else CheckResult.warn


# --------------------------------------------------------------------------
# 1. presence / schema
# --------------------------------------------------------------------------

def check_tables_present(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame], edges: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    missing_nodes = [t for t in spec.node_tables if t not in nodes]
    missing_edges = [t for t in spec.edge_tables if t not in edges]
    if missing_nodes:
        r.append(CheckResult.fail("df.node_tables_present", f"config node tables missing from payload: {missing_nodes}", **kw))
    else:
        r.append(CheckResult.ok("df.node_tables_present", f"all {len(spec.node_tables)} node tables present", **kw))
    if missing_edges:
        r.append(CheckResult.fail("df.edge_tables_present", f"config edge tables missing from payload: {missing_edges}", **kw))
    else:
        r.append(CheckResult.ok("df.edge_tables_present", f"all {len(spec.edge_tables)} edge tables present", **kw))
    extra_n = sorted(set(nodes) - set(spec.node_tables))
    extra_e = sorted(set(edges) - set(spec.edge_tables))
    if extra_n or extra_e:
        r.append(CheckResult.warn("df.tables_not_in_config", f"payload has tables not declared in config: nodes={extra_n} edges={extra_e}", **kw))
    for t, df in nodes.items():
        if df is None or df.empty:
            r.append(CheckResult.fail("df.node_table_nonempty", f"node table '{t}' is empty", **kw))
    for t, df in edges.items():
        if df is None or df.empty:
            sev = CheckResult.warn if spec.edges_by_table().get(t, EdgeSpec("", "", "", "", "", "")).rel in ADJACENCY_RELS else CheckResult.fail
            r.append(sev("df.edge_table_nonempty", f"edge table '{t}' is empty", **kw))
    return r


def check_table_schema(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame], edges: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    for t in spec.node_tables:
        df = nodes.get(t)
        if df is None:
            continue
        for col in (MODEL_ID, NODE_KEY):
            if col not in df.columns:
                r.append(CheckResult.fail("df.node_key_columns", f"node table '{t}' lacks '{col}'", **kw))
        feats = feature_columns(df)
        if not feats:
            r.append(CheckResult.fail("df.node_has_features", f"node table '{t}' has no numeric feature columns", **kw))
        strings = [c for c in df.columns if not is_identifier_column(c) and not pd.api.types.is_numeric_dtype(df[c])]
        if strings:
            r.append(CheckResult.warn("df.node_string_columns", f"'{t}' has non-numeric, non-identifier columns that will be dropped from features: {strings}", details={"columns": strings}, **kw))
        bools = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
        if bools:
            r.append(CheckResult.warn("df.node_bool_dtype", f"'{t}' has bool-typed columns (builders expect 0/1 ints): {bools}", **kw))
        id_like_numeric = [c for c in df.columns if is_identifier_column(c) and pd.api.types.is_numeric_dtype(df[c]) and c not in (MODEL_ID,)]
        if id_like_numeric:
            r.append(CheckResult.ok("df.node_identifier_columns", f"'{t}' identifier-like numeric columns excluded from features: {id_like_numeric}", **kw))
    for e in spec.edges:
        df = edges.get(e.table)
        if df is None:
            continue
        for col in (MODEL_ID, e.src_col, e.tgt_col):
            if col not in df.columns:
                r.append(CheckResult.fail("df.edge_key_columns", f"edge table '{e.table}' lacks '{col}'", **kw))
    return r


# --------------------------------------------------------------------------
# 2. model coverage
# --------------------------------------------------------------------------

def check_model_coverage(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame], edges: Dict[str, pd.DataFrame], model_ids: Sequence[Any]) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    mids = {str(m) for m in model_ids}
    prim = nodes.get(spec.primary_node_table)
    if prim is None or MODEL_ID not in prim.columns:
        return [CheckResult.fail("df.model_ids_from_primary", f"primary table '{spec.primary_node_table}' missing", **kw)]
    prim_ids = set(prim[MODEL_ID].astype(str))
    if mids and prim_ids != mids:
        r.append(CheckResult.fail("df.model_ids_from_primary", f"model_ids list ({len(mids)}) != distinct model_id in primary table ({len(prim_ids)}); only_in_list={examples(mids - prim_ids)} only_in_table={examples(prim_ids - mids)}", **kw))
    else:
        mids = mids or prim_ids
        r.append(CheckResult.ok("df.model_ids_from_primary", f"{len(mids)} models", **kw))
    if len(mids) != len(list(model_ids)) and model_ids:
        r.append(CheckResult.fail("df.model_ids_unique", f"model_ids list has duplicates ({len(list(model_ids))} entries, {len(mids)} unique)", **kw))

    for t in spec.node_tables:
        df = nodes.get(t)
        if df is None or MODEL_ID not in df.columns:
            continue
        per_model = df.groupby(df[MODEL_ID].astype(str)).size()
        missing = sorted(mids - set(per_model.index))
        if missing:
            optional = t.startswith("comp_aperture") or "aperture" in t
            sev = CheckResult.warn if optional else CheckResult.fail
            r.append(sev("df.node_table_model_coverage", f"'{t}' has no rows for {len(missing)}/{len(mids)} models ({pct(len(missing), len(mids)):.1f}%) e.g. {examples(missing)}", details={"missing_models": missing[:50]}, **kw))
        else:
            r.append(CheckResult.ok("df.node_table_model_coverage", f"'{t}' covers all {len(mids)} models", **kw))
        extra = sorted(set(per_model.index) - mids)
        if extra:
            r.append(CheckResult.warn("df.node_table_extra_models", f"'{t}' has rows for {len(extra)} models not in model_ids e.g. {examples(extra)}", **kw))
        if t in spec.singleton_node_tables:
            multi = per_model[per_model != 1]
            if len(multi):
                r.append(CheckResult.fail("df.singleton_node_table", f"'{t}' must have exactly 1 row per model; {len(multi)} models violate e.g. {examples(multi.index)} (counts {examples(multi.values)})", **kw))
            else:
                r.append(CheckResult.ok("df.singleton_node_table", f"'{t}' has exactly 1 row per model", **kw))
        else:
            r.append(CheckResult.ok("df.node_count_stats", f"'{t}': rows/model min={int(per_model.min())} median={float(per_model.median()):.0f} max={int(per_model.max())}", details={"min": int(per_model.min()), "median": float(per_model.median()), "max": int(per_model.max())}, **kw))

    for e in spec.edges:
        df = edges.get(e.table)
        if df is None or MODEL_ID not in df.columns:
            continue
        have = set(df[MODEL_ID].astype(str))
        missing = sorted(mids - have)
        if not missing:
            r.append(CheckResult.ok("df.edge_table_model_coverage", f"'{e.table}' present for all models", **kw))
            continue
        if e.rel in ADJACENCY_RELS:
            # adjacency may legitimately be empty for single-zone / single-element models
            sev = CheckResult.warn
            hint = " (adjacency can be empty for 1-zone models)"
        elif e.rel in {"weather_for"}:
            sev = CheckResult.fail
            hint = " — every building has exposed envelope, so weather edges must exist"
        else:
            sev = CheckResult.fail
            hint = ""
        r.append(sev("df.edge_table_model_coverage", f"'{e.table}' has no rows for {len(missing)}/{len(mids)} models e.g. {examples(missing)}{hint}", details={"missing_models": missing[:50]}, **kw))
    return r


# --------------------------------------------------------------------------
# 3. node uniqueness
# --------------------------------------------------------------------------

def check_node_uniqueness(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    all_keys: List[pd.DataFrame] = []
    for t in spec.node_tables:
        df = nodes.get(t)
        if df is None or NODE_KEY not in df.columns or MODEL_ID not in df.columns:
            continue
        keys = df[[MODEL_ID, NODE_KEY]].astype(str)
        dup = keys[keys.duplicated(keep=False)]
        if len(dup):
            r.append(CheckResult.fail("df.node_key_unique", f"'{t}': {len(dup)} rows share (model_id, instance_name) e.g. {examples(dup[NODE_KEY])}", **kw))
        else:
            r.append(CheckResult.ok("df.node_key_unique", f"'{t}' keys unique", **kw))
        nulls = df[NODE_KEY].isna().sum()
        if nulls:
            r.append(CheckResult.fail("df.node_key_not_null", f"'{t}': {nulls} rows with null instance_name", **kw))
        keys = keys.assign(_table=t)
        all_keys.append(keys)
    if len(all_keys) > 1:
        allk = pd.concat(all_keys, ignore_index=True)
        cross = allk[allk.duplicated([MODEL_ID, NODE_KEY], keep=False)]
        if len(cross):
            pairs = cross.groupby([MODEL_ID, NODE_KEY])["_table"].agg(lambda s: "+".join(sorted(set(s)))).value_counts()
            r.append(CheckResult.fail("df.node_key_unique_across_tables", f"{cross[[MODEL_ID, NODE_KEY]].drop_duplicates().shape[0]} instance_names appear in several node tables of the same model (NetworkX builder would merge them): {dict(pairs.head(5))}", **kw))
        else:
            r.append(CheckResult.ok("df.node_key_unique_across_tables", "instance_names unique across node tables", **kw))
    return r


# --------------------------------------------------------------------------
# 4. edge integrity
# --------------------------------------------------------------------------

def check_edge_integrity(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame], edges: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    node_idx: Dict[str, pd.MultiIndex] = {}
    for t, df in nodes.items():
        if df is not None and NODE_KEY in df.columns and MODEL_ID in df.columns:
            node_idx[t] = _node_index(df)

    touched: Dict[str, Set[Tuple[str, str]]] = {t: set() for t in spec.node_tables}

    for e in spec.edges:
        df = edges.get(e.table)
        if df is None or df.empty or not {MODEL_ID, e.src_col, e.tgt_col} <= set(df.columns):
            continue
        n = len(df)
        mid = df[MODEL_ID].astype(str)
        src = df[e.src_col].astype(str)
        tgt = df[e.tgt_col].astype(str)

        # null endpoints
        null_ends = int(df[e.src_col].isna().sum() + df[e.tgt_col].isna().sum())
        if null_ends:
            r.append(CheckResult.fail("df.edge_endpoints_not_null", f"'{e.table}': {null_ends} null endpoint values", **kw))

        # dangling references
        for side, cat, vals in (("src", e.src_cat, src), ("tgt", e.tgt_cat, tgt)):
            idx = node_idx.get(cat)
            if idx is None:
                r.append(CheckResult.skip("df.edge_endpoints_resolve", f"'{e.table}' {side}: node table '{cat}' unavailable", **kw))
                continue
            keys = pd.MultiIndex.from_arrays([mid, vals])
            ok = keys.isin(idx)
            bad = int((~ok).sum())
            if bad:
                r.append(CheckResult.fail("df.edge_endpoints_resolve", f"'{e.table}' {side}→'{cat}': {bad}/{n} ({pct(bad, n):.2f}%) endpoints do not match any node e.g. {examples(vals[~ok])}", details={"dangling": bad, "total": n, "models": examples(mid[~ok], 10)}, **kw))
            else:
                r.append(CheckResult.ok("df.edge_endpoints_resolve", f"'{e.table}' {side}→'{cat}' all {n} resolve", **kw))
            touched.setdefault(cat, set()).update(zip(mid[ok], vals[ok]))

        # self loops
        loops = int((src == tgt).sum())
        if loops:
            r.append(CheckResult.fail("df.edge_no_self_loops", f"'{e.table}': {loops} self-loop rows", **kw))

        # duplicate rows / both directions
        pairs = pd.DataFrame({"m": mid, "a": src, "b": tgt})
        dups = int(pairs.duplicated().sum())
        if dups:
            r.append(CheckResult.fail("df.edge_rows_unique", f"'{e.table}': {dups} duplicated (model_id, src, tgt) rows", **kw))
        if e.is_self_type:
            canon = pd.DataFrame({"m": mid, "a": np.minimum(src, tgt), "b": np.maximum(src, tgt)})
            both = int(canon.duplicated().sum()) - dups
            if both > 0:
                sev = CheckResult.fail if e.bidirectional else CheckResult.warn
                r.append(sev("df.edge_not_double_stored", f"'{e.table}': {both} pairs stored in both directions" + (" although config adds reverse edges (double counting)" if e.bidirectional else ""), **kw))

        # hierarchical cardinality: child (src) has exactly one parent per model
        if e.rel in HIERARCHICAL_RELS and not e.is_self_type:
            per_child = pairs.groupby(["m", "a"]).size()
            multi = per_child[per_child > 1]
            if len(multi):
                r.append(CheckResult.fail("df.edge_child_single_parent", f"'{e.table}': {len(multi)} {e.src_cat} nodes have >1 '{e.rel}' parent e.g. {examples([k[1] for k in multi.index])}", **kw))
            idx = node_idx.get(e.src_cat)
            if idx is not None:
                has_parent = idx.isin(pd.MultiIndex.from_frame(pairs[["m", "a"]]))
                orphans = int((~has_parent).sum())
                if orphans:
                    r.append(CheckResult.fail("df.edge_child_has_parent", f"'{e.table}': {orphans}/{len(idx)} {e.src_cat} nodes have no '{e.rel}' edge to {e.tgt_cat} e.g. {examples(idx[~has_parent].get_level_values(1))}", details={"models": examples(idx[~has_parent].get_level_values(0), 10)}, **kw))
                else:
                    r.append(CheckResult.ok("df.edge_child_has_parent", f"every {e.src_cat} has exactly one {e.rel} → {e.tgt_cat}", **kw))
            idx_p = node_idx.get(e.tgt_cat)
            if idx_p is not None:
                has_child = idx_p.isin(pd.MultiIndex.from_frame(pairs[["m", "b"]]))
                childless = int((~has_child).sum())
                if childless:
                    sev = CheckResult.warn if "aperture" in e.src_cat else CheckResult.fail
                    r.append(sev("df.edge_parent_has_children", f"'{e.table}': {childless}/{len(idx_p)} {e.tgt_cat} nodes have no {e.src_cat} children e.g. {examples(idx_p[~has_child].get_level_values(1))}", **kw))

        # edge attribute sanity (numeric domain of edge features)
        r.extend(_domain_checks_for_frame(df, f"edge:{e.table}", spec))

    # isolated nodes: rows of a node table never referenced by any edge table
    for t in spec.node_tables:
        idx = node_idx.get(t)
        if idx is None or not spec.edges_touching(t):
            continue
        ref = pd.MultiIndex.from_tuples(list(touched.get(t, set())), names=[MODEL_ID, NODE_KEY]) if touched.get(t) else pd.MultiIndex.from_arrays([[], []])
        isolated = ~idx.isin(ref)
        k = int(isolated.sum())
        if k:
            if any(e.rel in ADJACENCY_RELS for e in spec.edges_touching(t)) and len(spec.edges_touching(t)) == 1:
                sev = CheckResult.warn  # zone-only graphs: isolated zones can be legit (no shared boundary)
            else:
                sev = CheckResult.fail
            r.append(sev("df.node_not_isolated", f"'{t}': {k}/{len(idx)} nodes appear in no edge table e.g. {examples(idx[isolated].get_level_values(1))}", details={"models": examples(idx[isolated].get_level_values(0), 10)}, **kw))
        else:
            r.append(CheckResult.ok("df.node_not_isolated", f"'{t}': every node has ≥1 edge", **kw))
    return r


# --------------------------------------------------------------------------
# 5. missing values / uncomputable features
# --------------------------------------------------------------------------

def check_missing_values(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame], edges: Dict[str, pd.DataFrame], *, nan_fail_threshold: float = 0.0) -> List[CheckResult]:
    """NaN/None/inf in feature columns.

    The PyG builders silently ``fillna(0.0)`` — so *any* NaN in a numeric
    feature becomes a fake zero.  Columns that are 100 % NaN indicate a
    feature that could not be calculated at all.
    """
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    frames = [(f"node:{t}", nodes.get(t)) for t in spec.node_tables] + [(f"edge:{t}", edges.get(t)) for t in spec.edge_tables]
    for label, df in frames:
        if df is None or df.empty:
            continue
        n = len(df)
        all_nan, some_nan, infs = [], {}, {}
        for c in df.columns:
            if c in (MODEL_ID, NODE_KEY):
                continue
            s = df[c]
            k = int(s.isna().sum())
            if k == n:
                all_nan.append(c)
            elif k and (is_identifier_column(c) is False):
                some_nan[c] = k
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                vals = s.to_numpy(dtype="float64", na_value=np.nan)
                ni = int(np.isinf(vals).sum())
                if ni:
                    infs[c] = ni
        if all_nan:
            r.append(CheckResult.fail("df.column_all_missing", f"{label}: columns entirely NULL (feature could not be computed): {all_nan}", details={"columns": all_nan}, **kw))
        if some_nan:
            worst = sorted(some_nan.items(), key=lambda kv: -kv[1])[:8]
            frac = max(some_nan.values()) / n
            sev = CheckResult.fail if frac > nan_fail_threshold else CheckResult.warn
            r.append(sev("df.column_partially_missing", f"{label}: {len(some_nan)} columns with NULLs (builders fill 0.0 silently); worst: " + ", ".join(f"{c}={k} ({pct(k, n):.1f}%)" for c, k in worst), details={"null_counts": some_nan}, **kw))
        if infs:
            r.append(CheckResult.fail("df.column_infinite", f"{label}: infinite values in {infs}", **kw))
        if not all_nan and not some_nan and not infs:
            r.append(CheckResult.ok("df.no_missing_values", f"{label}: no NULL/inf in {df.shape[1]} columns", **kw))
    return r


# --------------------------------------------------------------------------
# 6. value domains / booleans / one-hot
# --------------------------------------------------------------------------

def _domain_checks_for_frame(df: pd.DataFrame, label: str, spec: ConfigSpec) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    n = len(df)
    checked = 0
    for c in df.columns:
        if is_identifier_column(c) or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        rule = rule_for_column(c)
        if rule is None:
            continue
        checked += 1
        s = pd.to_numeric(df[c], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        bad = pd.Series(False, index=valid.index)
        if rule.lo is not None:
            bad |= (valid < rule.lo) if not rule.lo_exclusive else (valid <= rule.lo)
        if rule.hi is not None:
            bad |= valid > rule.hi
        if rule.integer:
            bad |= (valid != np.floor(valid))
        k = int(bad.sum())
        if k:
            mk = _fail_or_warn(rule.severity)
            models = examples(df.loc[bad[bad].index, MODEL_ID], 5) if MODEL_ID in df.columns else []
            r.append(mk("df.value_domain", f"{label}.{c}: {k}/{n} ({pct(k, n):.2f}%) values violate rule '{rule.note or rule}' e.g. {examples(valid[bad].round(4).tolist())}", details={"column": c, "violations": k, "models": models, "min": float(valid.min()), "max": float(valid.max())}, **kw))
    if checked:
        r.append(CheckResult.ok("df.value_domain_coverage", f"{label}: {checked} columns matched a domain rule", **kw))
    return r


def check_value_domains(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    for t in spec.node_tables:
        df = nodes.get(t)
        if df is None or df.empty:
            continue
        label = f"node:{t}"
        r.extend(_domain_checks_for_frame(df, label, spec))

        # one-hot groups
        for g in ONE_HOT_GROUPS:
            cols = g.resolve(list(df.columns))
            if len(cols) < 2:
                continue
            sub = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            row_sum = sub.sum(axis=1)
            bad = (row_sum > 1) | ((row_sum < 1) & (not g.optional))
            k = int(bad.sum())
            if k:
                r.append(CheckResult.fail("df.one_hot_exclusive", f"{label} group '{g.name}' {cols}: {k}/{len(df)} rows with sum≠1 (sums seen: {sorted(set(row_sum[bad].astype(int).tolist()))[:5]}) e.g. models {examples(df.loc[bad, MODEL_ID]) if MODEL_ID in df.columns else ''}", details={"group": g.name, "columns": cols}, **kw))
            else:
                r.append(CheckResult.ok("df.one_hot_exclusive", f"{label} group '{g.name}' ({len(cols)} cols) is exclusive", **kw))
            never = [c for c in cols if (sub[c] != 0).sum() == 0]
            if never:
                r.append(CheckResult.warn("df.one_hot_category_unused", f"{label} group '{g.name}': categories never active in this dataset (constant zero feature): {never}", **kw))

        # label column ↔ one-hot agreement
        for label_col, prefix in ONE_HOT_LABEL_COLUMNS.items():
            if label_col not in df.columns:
                continue
            hot_cols = [c for c in df.columns if c.startswith(prefix) and c != label_col]
            if not hot_cols:
                continue
            norm = df[label_col].astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
            expected_col = prefix + norm
            hot = df[hot_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            active = hot.idxmax(axis=1).where(hot.max(axis=1) > 0, other="")
            mismatch = active != expected_col
            k = int(mismatch.sum())
            if k:
                r.append(CheckResult.fail("df.one_hot_matches_label", f"{label}: {k} rows where '{label_col}' disagrees with the active {prefix}* column e.g. {examples(list(zip(df.loc[mismatch, label_col], active[mismatch])))}", **kw))
            else:
                r.append(CheckResult.ok("df.one_hot_matches_label", f"{label}: '{label_col}' consistent with {prefix}* one-hot", **kw))
    return r


# --------------------------------------------------------------------------
# 7. feature variance / degenerate columns
# --------------------------------------------------------------------------

def check_feature_variance(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame], edges: Dict[str, pd.DataFrame], *, min_models_for_variance: int = 20) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    frames = [(f"node:{t}", nodes.get(t)) for t in spec.node_tables] + [(f"edge:{t}", edges.get(t)) for t in spec.edge_tables]
    for label, df in frames:
        if df is None or df.empty or MODEL_ID not in df.columns:
            continue
        n_models = df[MODEL_ID].nunique()
        feats = feature_columns(df)
        if not feats:
            continue
        num = df[feats].apply(pd.to_numeric, errors="coerce")
        nun = num.nunique(dropna=True)
        constant = sorted(nun[nun <= 1].index.tolist())
        if constant and n_models >= min_models_for_variance:
            vals = {c: (num[c].dropna().iloc[0] if num[c].notna().any() else None) for c in constant[:10]}
            r.append(CheckResult.warn("df.constant_feature", f"{label}: {len(constant)} constant feature columns across {n_models} models (carry no information): {vals}{' …' if len(constant) > 10 else ''}", details={"columns": constant}, **kw))
        elif n_models >= min_models_for_variance:
            r.append(CheckResult.ok("df.constant_feature", f"{label}: no constant features", **kw))
        # all-zero shading = pipeline-level red flag
        shade_cols = [c for c in SHADING_COLUMNS if c in num.columns]
        if shade_cols and n_models >= min_models_for_variance:
            nonzero = {c: int((num[c].fillna(0) != 0).sum()) for c in shade_cols}
            if all(v == 0 for v in nonzero.values()):
                r.append(CheckResult.fail("df.shading_not_all_zero", f"{label}: shading columns {shade_cols} are zero for every row — surface_shading_metrics was not populated", **kw))
            else:
                share = pct(sum(1 for _ in df.loc[num[shade_cols[0]].fillna(0) != 0, MODEL_ID].unique()), n_models)
                r.append(CheckResult.ok("df.shading_not_all_zero", f"{label}: shading nonzero rows {nonzero}; {share:.1f}% of models have shading", **kw))
        # extreme magnitudes hint at unit errors
        big = {c: float(num[c].abs().max()) for c in feats if num[c].notna().any() and float(num[c].abs().max()) > 1e7}
        if big:
            r.append(CheckResult.warn("df.extreme_magnitude", f"{label}: |value| > 1e7 in {big} (unit error?)", **kw))
    return r


# --------------------------------------------------------------------------
# 8. physical / semantic consistency
# --------------------------------------------------------------------------

def check_physical_consistency(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame], edges: Dict[str, pd.DataFrame], *, rel_tol: float = 0.02) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []

    def _num(df: pd.DataFrame, c: str) -> pd.Series:
        return pd.to_numeric(df[c], errors="coerce")

    # --- zone-level tables -------------------------------------------------
    for t in spec.node_tables:
        df = nodes.get(t)
        if df is None or df.empty:
            continue
        cols = set(df.columns)
        label = f"node:{t}"
        if {"wall_area_total", "wall_area_ext", "wall_area_int"} <= cols:
            tot, ext, inn = _num(df, "wall_area_total"), _num(df, "wall_area_ext"), _num(df, "wall_area_int")
            diff = (ext + inn - tot).abs()
            bad = diff > rel_tol * tot.clip(lower=1.0)
            k = int(bad.sum())
            r.append((CheckResult.fail if k else CheckResult.ok)("df.wall_area_partition", f"{label}: wall_area_ext + wall_area_int ≠ wall_area_total in {k}/{len(df)} rows" if k else f"{label}: wall areas partition correctly", details={"models": examples(df.loc[bad, MODEL_ID]) if k else []}, **kw))
        if {"wwr", "window_area_ext", "wall_area_ext"} <= cols:
            wwr = _num(df, "wwr")
            calc = _num(df, "window_area_ext") / _num(df, "wall_area_ext").replace(0, np.nan)
            bad = ((wwr - calc).abs() > 0.05) & calc.notna()
            k = int(bad.sum())
            if k:
                r.append(CheckResult.warn("df.wwr_consistent", f"{label}: wwr differs from window_area_ext/wall_area_ext in {k} rows", **kw))
        if {"is_ground_floor", "is_top_floor", "floor_level", MODEL_ID} <= cols:
            g = df.groupby(MODEL_ID)["floor_level"]
            lvl = _num(df, "floor_level")
            ground_expected = (lvl == g.transform("min")).astype(int)
            top_expected = (lvl == g.transform("max")).astype(int)
            bad_g = int((_num(df, "is_ground_floor").fillna(-1) != ground_expected).sum())
            bad_t = int((_num(df, "is_top_floor").fillna(-1) != top_expected).sum())
            if bad_g or bad_t:
                r.append(CheckResult.fail("df.floor_flags_consistent", f"{label}: is_ground_floor mismatches={bad_g}, is_top_floor mismatches={bad_t} versus min/max floor_level per model", **kw))
            else:
                r.append(CheckResult.ok("df.floor_flags_consistent", f"{label}: ground/top floor flags match floor_level extremes", **kw))
            lvl_sorted = df.groupby(MODEL_ID)["floor_level"].agg(lambda s: sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int))))
            gaps = lvl_sorted[lvl_sorted.apply(lambda ls: bool(ls) and ls != list(range(ls[0], ls[0] + len(ls))))]
            if len(gaps):
                r.append(CheckResult.warn("df.floor_levels_contiguous", f"{label}: {len(gaps)} models with non-contiguous floor levels e.g. {examples(gaps.index)}", **kw))
        if {"fraction_radiant", "fraction_visible"} <= cols:
            s = _num(df, "fraction_radiant") + _num(df, "fraction_visible")
            k = int((s > 1.0 + 1e-6).sum())
            if k:
                r.append(CheckResult.fail("df.fraction_sum", f"{label}: fraction_radiant + fraction_visible > 1 in {k} rows", **kw))

        # --- component-level tables -----------------------------------------
        if {"is_ground", "sun_exposure", "wind_exposure"} <= cols:
            ground = _num(df, "is_ground") == 1
            bad = ground & ((_num(df, "sun_exposure") != 0) | (_num(df, "wind_exposure") != 0))
            k = int(bad.sum())
            r.append((CheckResult.fail if k else CheckResult.ok)("df.ground_no_exposure", f"{label}: {k} ground-contact elements with sun/wind exposure" if k else f"{label}: ground elements have no sun/wind exposure", **kw))
        if {"is_outdoors", "sun_exposure", "wind_exposure"} <= cols:
            interior = _num(df, "is_outdoors") != 1
            bad = interior & ((_num(df, "sun_exposure") == 1) | (_num(df, "wind_exposure") == 1))
            k = int(bad.sum())
            if k:
                r.append(CheckResult.warn("df.interior_no_exposure", f"{label}: {k} non-outdoor elements flagged sun/wind exposed", **kw))
        if {"is_window", "solar_transmittance"} <= cols:
            win = _num(df, "is_window") == 1
            k_opaque_glass = int((win & (_num(df, "solar_transmittance") <= 0)).sum())
            if k_opaque_glass:
                r.append(CheckResult.fail("df.window_transmits", f"{label}: {k_opaque_glass} windows with solar_transmittance ≤ 0", **kw))
            else:
                r.append(CheckResult.ok("df.window_transmits", f"{label}: all windows have positive solar transmittance", **kw))
        if {"is_wall", "solar_transmittance"} <= cols and "is_window" not in cols:
            # opaque element table (comp_element): transmittance must be 0
            k = int((_num(df, "solar_transmittance") > 0).sum())
            if k:
                r.append(CheckResult.fail("df.opaque_no_transmittance", f"{label}: {k} opaque elements with solar_transmittance > 0", **kw))
        if {"is_aperture", "solar_transmittance", "is_wall"} <= cols or ({"element_type", "is_aperture"} <= cols):
            ap = _num(df, "is_aperture") == 1
            if "solar_transmittance" in cols:
                k = int((~ap & (_num(df, "solar_transmittance") > 0)).sum())
                if k:
                    r.append(CheckResult.fail("df.opaque_no_transmittance", f"{label}: {k} non-aperture elements with solar_transmittance > 0", **kw))
        if {"u_value_total", "r_value_layers"} <= cols:
            u, rv = _num(df, "u_value_total"), _num(df, "r_value_layers")
            # U ≈ 1/(R_layers + R_films); U*R_layers should be < ~1.2 for real constructions
            bad = (u * rv) > 1.5
            k = int(bad.sum())
            if k:
                r.append(CheckResult.warn("df.u_r_consistent", f"{label}: {k} rows with u_value_total × r_value_layers > 1.5 (inconsistent thermal properties)", **kw))
        if {"total_thickness", "density", "is_window"} <= cols:
            pass  # windows legitimately have thin layers; nothing to assert

    # --- aperture ⊂ parent element (needs edge table) -------------------------
    for e in spec.edges:
        if e.rel != "part_of":
            continue
        ap, el, ed = nodes.get(e.src_cat), nodes.get(e.tgt_cat), edges.get(e.table)
        if ap is None or el is None or ed is None or ed.empty or "area" not in ap.columns or "area" not in el.columns:
            continue
        m = ed[[MODEL_ID, e.src_col, e.tgt_col]].astype(str)
        a = ap[[MODEL_ID, NODE_KEY, "area"]].astype({MODEL_ID: str, NODE_KEY: str}).rename(columns={"area": "ap_area", NODE_KEY: e.src_col})
        p = el[[MODEL_ID, NODE_KEY, "area"]].astype({MODEL_ID: str, NODE_KEY: str}).rename(columns={"area": "el_area", NODE_KEY: e.tgt_col})
        j = m.merge(a, on=[MODEL_ID, e.src_col], how="left").merge(p, on=[MODEL_ID, e.tgt_col], how="left")
        per_parent = j.groupby([MODEL_ID, e.tgt_col]).agg(ap_area=("ap_area", "sum"), el_area=("el_area", "first")).reset_index()
        bad = per_parent["ap_area"] > per_parent["el_area"] * (1 + rel_tol)
        k = int(bad.sum())
        if k:
            r.append(CheckResult.fail("df.aperture_within_parent", f"edge:{e.table}: {k}/{len(per_parent)} parent elements whose summed aperture area exceeds the element area e.g. {examples(per_parent.loc[bad, e.tgt_col])}", **kw))
        else:
            r.append(CheckResult.ok("df.aperture_within_parent", f"edge:{e.table}: aperture areas fit inside parent elements", **kw))

    # --- zone floor area vs floor elements (component level) ---------------------
    zone_t = next((t for t in spec.node_tables if t in ("comp_zone",)), None)
    el_t = next((t for t in spec.node_tables if t in ("comp_element",)), None)
    if zone_t and el_t:
        z, el = nodes.get(zone_t), nodes.get(el_t)
        if z is not None and el is not None and {"floor_area"} <= set(z.columns) and {"is_floor", "area", "zone_name"} <= set(el.columns):
            floors = el[pd.to_numeric(el["is_floor"], errors="coerce") == 1]
            fa = floors.groupby([floors[MODEL_ID].astype(str), floors["zone_name"].astype(str)])["area"].sum()
            zz = z.set_index([z[MODEL_ID].astype(str), z[NODE_KEY].astype(str)])["floor_area"]
            both = pd.concat([zz.rename("zone_fa"), fa.rename("floor_el_fa")], axis=1).dropna()
            rel = (both["zone_fa"] - both["floor_el_fa"]).abs() / both["zone_fa"].clip(lower=1.0)
            k = int((rel > 0.05).sum())
            r.append((CheckResult.warn if k else CheckResult.ok)("df.zone_floor_area_matches_elements", f"{k}/{len(both)} zones where floor_area deviates >5% from the summed floor-element area" if k else f"zone floor_area matches summed floor elements for {len(both)} zones", **kw))
            no_floor = zz.index[~zz.index.isin(fa.index)]
            if len(no_floor):
                r.append(CheckResult.fail("df.zone_has_floor_element", f"{len(no_floor)} zones have no floor element e.g. {examples(no_floor.get_level_values(1))}", **kw))

    # --- weather exposure edges (external weather configs) ------------------------
    for e in spec.edges:
        if e.rel != "weather_for" or e.src_cat != "weather_city":
            continue
        tgt_df, ed = nodes.get(e.tgt_cat), edges.get(e.table)
        if tgt_df is None or ed is None:
            continue
        if {"is_outdoors", "is_wall", "is_roof", "is_ceiling"} <= set(tgt_df.columns):
            exposed = tgt_df.apply(lambda row: element_is_weather_exposed(row.to_dict()), axis=1)
            exp_idx = pd.MultiIndex.from_frame(tgt_df.loc[exposed, [MODEL_ID, NODE_KEY]].astype(str))
            got_idx = pd.MultiIndex.from_frame(ed[[MODEL_ID, e.tgt_col]].astype(str)) if not ed.empty else pd.MultiIndex.from_arrays([[], []])
            missing = int((~exp_idx.isin(got_idx)).sum())
            extra = int((~got_idx.isin(exp_idx)).sum())
            if missing or extra:
                r.append(CheckResult.fail("df.weather_edges_match_exposure", f"edge:{e.table}: {missing} exposed {e.tgt_cat} without weather edge, {extra} weather edges to non-exposed elements", **kw))
            else:
                r.append(CheckResult.ok("df.weather_edges_match_exposure", f"edge:{e.table}: weather edges exactly cover the {len(exp_idx)} exposed elements", **kw))
        elif "is_window" in tgt_df.columns:
            got = ed[e.tgt_col].astype(str) if not ed.empty else pd.Series([], dtype=str)
            non_win = tgt_df[pd.to_numeric(tgt_df["is_window"], errors="coerce") != 1]
            k = int(non_win[NODE_KEY].astype(str).isin(set(got)).sum())
            if k:
                r.append(CheckResult.fail("df.weather_edges_match_exposure", f"edge:{e.table}: {k} weather edges target non-window apertures", **kw))
        # every model with an exposed envelope must have ≥1 weather edge
        if not ed.empty and MODEL_ID in ed.columns and MODEL_ID in tgt_df.columns:
            have = set(ed[MODEL_ID].astype(str))
            need = set(tgt_df[MODEL_ID].astype(str))
            k = len(need - have)
            if k and "aperture" not in e.tgt_cat:
                r.append(CheckResult.fail("df.weather_edges_every_model", f"edge:{e.table}: {k} models have {e.tgt_cat} rows but no weather edge", **kw))
    return r


# --------------------------------------------------------------------------
# 9. weather tables
# --------------------------------------------------------------------------

def check_weather_tables(spec: ConfigSpec, nodes: Dict[str, pd.DataFrame]) -> List[CheckResult]:
    kw = dict(config=spec.name, scope=SCOPE)
    r: List[CheckResult] = []
    for t in spec.weather_node_tables:
        df = nodes.get(t)
        if df is None or df.empty:
            continue
        if "city_key" in df.columns and MODEL_ID in df.columns:
            ck = df["city_key"].astype(str).str.lower()
            mid = df[MODEL_ID].astype(str).str.lower()
            # model_id convention: <city_key>_<nnnn>
            derived = mid.str.replace(r"_\d+$", "", regex=True)
            bad = ck != derived
            k = int(bad.sum())
            if k:
                r.append(CheckResult.fail("df.weather_city_matches_model", f"node:{t}: {k}/{len(df)} rows where city_key ≠ city part of model_id e.g. {examples(list(zip(mid[bad], ck[bad])))}", **kw))
            else:
                r.append(CheckResult.ok("df.weather_city_matches_model", f"node:{t}: city_key consistent with model_id for {len(df)} rows", **kw))
            feats = feature_columns(df)
            if feats:
                per_city = df.groupby(ck)[feats].nunique(dropna=False).max(axis=1)
                incons = per_city[per_city > 1]
                if len(incons):
                    r.append(CheckResult.fail("df.weather_identical_per_city", f"node:{t}: {len(incons)} cities whose weather features differ between models e.g. {examples(incons.index)}", **kw))
                else:
                    r.append(CheckResult.ok("df.weather_identical_per_city", f"node:{t}: weather features identical within each of {ck.nunique()} cities", **kw))
                # a weather node whose features are entirely zero = missing EPW statistics
                zero_rows = int((df[feats].fillna(0).abs().sum(axis=1) == 0).sum())
                if zero_rows:
                    r.append(CheckResult.fail("df.weather_not_all_zero", f"node:{t}: {zero_rows} rows with all-zero weather features", **kw))
    return r


# --------------------------------------------------------------------------
# aggregator
# --------------------------------------------------------------------------

def run_dataframe_checks(
    spec: ConfigSpec,
    payload: Dict[str, Any],
    *,
    nan_fail_threshold: float = 0.0,
    min_models_for_variance: int = 20,
) -> List[CheckResult]:
    """Run every DataFrame-level check for one configuration."""
    flat = flatten_payload(payload)
    nodes, edges, model_ids = flat["nodes"], flat["edges"], flat["model_ids"]
    results: List[CheckResult] = []
    results += check_tables_present(spec, nodes, edges)
    results += check_table_schema(spec, nodes, edges)
    results += check_model_coverage(spec, nodes, edges, model_ids)
    results += check_node_uniqueness(spec, nodes)
    results += check_edge_integrity(spec, nodes, edges)
    results += check_missing_values(spec, nodes, edges, nan_fail_threshold=nan_fail_threshold)
    results += check_value_domains(spec, nodes)
    results += check_feature_variance(spec, nodes, edges, min_models_for_variance=min_models_for_variance)
    results += check_physical_consistency(spec, nodes, edges)
    results += check_weather_tables(spec, nodes)
    return results
