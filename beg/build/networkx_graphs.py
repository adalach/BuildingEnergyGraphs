from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import networkx as nx
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine


def choose_shared_model_ids(
    chunk_map: pd.DataFrame,
    configs_summary: pd.DataFrame,
    config_names: List[str],
    n: int,
) -> List[str]:
    """
    Picks first n model_ids that are present in ALL requested configs,
    preserving the order in chunk_map.
    """
    # Build intersection from configs_summary.model_ids lists
    cfg_to_ids = {}
    for cfg in config_names:
        row = configs_summary.loc[configs_summary["config_name"] == cfg]
        if row.empty:
            raise KeyError(f"Config not found in configs_summary: {cfg}")
        ids = row.iloc[0]["model_ids"]
        if isinstance(ids, str):
            ids = ast.literal_eval(ids)
        cfg_to_ids[cfg] = set(ids)

    shared = set.intersection(*cfg_to_ids.values())
    if not shared:
        raise RuntimeError("No shared model_ids across the requested configs.")

    # Preserve chunk_map order
    ordered = [mid for mid in chunk_map["model_id"].tolist() if mid in shared]
    return ordered[:n]


def _iter_batches_for_config(out: dict, config_name: str):
    cfg_payload = out["data"].get(config_name, {})
    for ch in cfg_payload.get("chunks", []):
        for b in ch.get("batches", []):
            yield b


def extract_model_dfs_from_out(
    out: dict,
    config_name: str,
    model_id: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Returns (nodes_dict, edges_dict) where each DF is filtered to the given model_id,
    gathered across all batches that contain that model_id.
    """
    node_parts: Dict[str, List[pd.DataFrame]] = {}
    edge_parts: Dict[str, List[pd.DataFrame]] = {}

    for b in _iter_batches_for_config(out, config_name):
        if model_id not in (b.get("model_ids") or []):
            continue

        # Nodes
        for node_type, df in (b.get("nodes") or {}).items():
            if "model_id" not in df.columns:
                continue
            sub = df[df["model_id"] == model_id]
            if not sub.empty:
                node_parts.setdefault(node_type, []).append(sub)

        # Edges
        for edge_type, df in (b.get("edges") or {}).items():
            if "model_id" not in df.columns:
                continue
            sub = df[df["model_id"] == model_id]
            if not sub.empty:
                edge_parts.setdefault(edge_type, []).append(sub)

    nodes = {k: pd.concat(v, ignore_index=True) for k, v in node_parts.items()}
    edges = {k: pd.concat(v, ignore_index=True) for k, v in edge_parts.items()}
    return nodes, edges


def _stable_sort_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic order within a node table.
    Preference: existing idx -> id -> instance_name
    """
    keys = []
    if "idx" in df.columns:
        keys.append("idx")
    if "id" in df.columns:
        keys.append("id")
    if "instance_name" in df.columns:
        keys.append("instance_name")

    if keys:
        return df.sort_values(keys, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def _edge_endpoints(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Identify src/tgt columns in an edge df.
    """
    if "src_id" in df.columns and "tgt_id" in df.columns:
        return "src_id", "tgt_id"
    if "src" in df.columns and "tgt" in df.columns:
        return "src", "tgt"
    # fallback: try common alternatives
    candidates = [c for c in df.columns if c.lower().startswith("src")]
    candidates2 = [c for c in df.columns if c.lower().startswith("tgt")]
    if candidates and candidates2:
        return candidates[0], candidates2[0]
    raise KeyError(f"Could not determine edge endpoints columns. Columns: {list(df.columns)}")


@dataclass
class BuiltGraph:
    config_name: str
    model_id: str
    G: nx.MultiDiGraph
    name_to_node_id: Dict[str, int]


def build_nx_graph_for_model(
    config_name: str,
    model_id: str,
    nodes_dict: Dict[str, pd.DataFrame],
    edges_dict: Dict[str, pd.DataFrame],
    node_table_order: List[str],
    *,
    edge_specs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    node_id_col: str = "node_id",
) -> BuiltGraph:
    """
    Build a MultiDiGraph for one model_id.
    - Adds nodes in node_table_order
    - Assigns per-graph contiguous node_id (0..N-1) while adding nodes
    - Uses instance_name as the NetworkX node key
    - Propagates (almost) all DataFrame properties to nodes/edges, excluding join keys
      like `model_id`, `instance_name` (for nodes) and `src`/`tgt` (for edges).
    """
    G = nx.MultiDiGraph(config_name=config_name, model_id=model_id)
    name_to_node_id: Dict[str, int] = {}
    next_id = 0

    def _clean_row_attrs(row: pd.Series, ignore: set[str]) -> Dict[str, object]:
        """Convert a pandas row into a dict suitable for NX attrs.
        - Drop ignored columns
        - Drop NaN/None values
        - Convert numpy/pandas scalars to native Python types
        """
        out: Dict[str, object] = {}
        for k, v in row.items():
            if k in ignore:
                continue
            # Drop NaNs/None
            if pd.isna(v):
                continue
            # Convert pandas Timestamp to ISO string; numpy types to Python
            if hasattr(v, "isoformat"):
                try:
                    v = v.isoformat()
                except Exception:
                    pass
            try:
                import numpy as _np  # local import to avoid hard dep at import-time
                if isinstance(v, (_np.generic,)):
                    v = v.item()
            except Exception:
                pass
            out[k] = v
        return out

    # Add nodes (config order) and merge attributes from all node tables rows
    for node_type in node_table_order:
        if node_type not in nodes_dict:
            continue
        df = _stable_sort_nodes(nodes_dict[node_type])
        for _, row in df.iterrows():
            name = row.get("instance_name")
            if name is None:
                raise KeyError(f"[{config_name}] node table '{node_type}' missing 'instance_name'")

            if name not in name_to_node_id:
                name_to_node_id[name] = next_id
                next_id += 1
                # Seed with reserved attrs
                G.add_node(name, node_type=node_type, **{node_id_col: name_to_node_id[name]})
            else:
                # If node already exists, ensure node_type remains the first seen type
                if "node_type" not in G.nodes[name]:
                    G.nodes[name]["node_type"] = node_type
                if node_id_col not in G.nodes[name]:
                    G.nodes[name][node_id_col] = name_to_node_id[name]

            # Merge DF properties (do not overwrite existing values)
            ignore_cols = {"model_id", "instance_name", node_id_col}
            attrs_to_add = _clean_row_attrs(row, ignore_cols)
            node_view = G.nodes[name]
            for k, v in attrs_to_add.items():
                if k not in node_view:
                    node_view[k] = v

    # Normalize edge specs to a lookup by edge_src name
    edge_spec_map: Dict[str, Mapping[str, Any]] = {}
    if edge_specs:
        if isinstance(edge_specs, dict):
            edge_spec_map = edge_specs  # assume already keyed by edge_src
        else:
            for spec in edge_specs.values() if hasattr(edge_specs, "values") else edge_specs:
                if not isinstance(spec, Mapping):
                    continue
                edge_src = spec.get("edge_src")
                if edge_src:
                    edge_spec_map[edge_src] = spec

    # Add edges with attributes from edge DataFrames
    for edge_type, df in edges_dict.items():
        if df.empty:
            continue
        spec = edge_spec_map.get(edge_type)
        if spec:
            src_col = spec.get("src_col")
            tgt_col = spec.get("tgt_col")
            if not src_col or not tgt_col:
                raise KeyError(
                    f"[{config_name}] edge spec for '{edge_type}' missing src_col/tgt_col"
                )
        else:
            src_col, tgt_col = _edge_endpoints(df)
        for _, row in df.iterrows():
            s = row.get(src_col)
            t = row.get(tgt_col)
            if s is None or t is None:
                continue

            # An edge may reference a node absent from the node tables; add it as a placeholder
            if s not in name_to_node_id:
                name_to_node_id[s] = next_id
                next_id += 1
                G.add_node(s, node_type="(missing_from_nodes)", **{node_id_col: name_to_node_id[s]})
            if t not in name_to_node_id:
                name_to_node_id[t] = next_id
                next_id += 1
                G.add_node(t, node_type="(missing_from_nodes)", **{node_id_col: name_to_node_id[t]})

            ignore_cols = {"model_id", src_col, tgt_col}
            edge_attrs = _clean_row_attrs(row, ignore_cols)
            # Ensure edge_type is present (prefer config rel if provided)
            rel = spec.get("rel") if spec else None
            if "edge_type" not in edge_attrs:
                edge_attrs["edge_type"] = rel or edge_type
            if rel and "rel" not in edge_attrs:
                edge_attrs["rel"] = rel

            edge_attrs_fwd = dict(edge_attrs)
            if "is_reverse" not in edge_attrs_fwd:
                edge_attrs_fwd["is_reverse"] = 0
            G.add_edge(s, t, **edge_attrs_fwd)

            if spec and spec.get("bidirectional", False):
                edge_attrs_rev = dict(edge_attrs)
                edge_attrs_rev["is_reverse"] = 1
                G.add_edge(t, s, **edge_attrs_rev)

    return BuiltGraph(config_name=config_name, model_id=model_id, G=G, name_to_node_id=name_to_node_id)


# -----------------------------------------------------------------------------
# Builders from explicit node/edge specs
# -----------------------------------------------------------------------------

def _q(schema: str, table: str) -> str:
    """Quote a schema + table for SQL (handles case/characters)."""
    return f'"{schema}"."{table}"'


def _node_key(ntype: str, id_cols: List[str], rec: Mapping[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
    """Canonical node key: (<type>, (<id values...>))."""
    return (ntype, tuple(rec.get(c) for c in id_cols))


def _pick_label(rec: Mapping[str, Any], label_cols: List[str]) -> str:
    """Pick a human-readable label from preferred columns, skipping empty/NaN."""
    for c in label_cols:
        v = rec.get(c)
        if v is not None and not pd.isna(v):
            s = str(v)
            if s != "":
                return s
    for v in rec.values():
        if v is not None and not pd.isna(v):
            s = str(v)
            if s != "":
                return s
    return ""


def _edge_attrs(rel: str, fk_label: str, extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Standardize edge attributes across modes/builders."""
    base = {"rel": rel, "fk_col": fk_label, "group": rel}
    if extra:
        base.update(extra)
    return base


def _add_nodes_from_df(
    G: nx.MultiGraph | nx.MultiDiGraph,
    df: pd.DataFrame,
    *,
    ntype: str,
    id_cols: List[str],
    label_cols: List[str],
) -> None:
    """Add nodes to the graph from a DataFrame (one row → one node)."""
    for attrs in df.to_dict("records"):
        nk = _node_key(ntype, id_cols, attrs)
        label = _pick_label(attrs, label_cols)
        G.add_node(nk, **attrs, label=label, node_type=ntype)


def _add_assoc_edges(
    G: nx.MultiGraph | nx.MultiDiGraph,
    *,
    df: pd.DataFrame,
    src: Mapping[str, Any],
    tgt: Mapping[str, Any],
    rel: str,
    fk_label: str,
    undirected: bool = False,
) -> None:
    """Add edges from an association DataFrame."""
    src_type = src["type"]
    tgt_type = tgt["type"]
    src_on = src["match_on"]
    tgt_on = tgt["match_on"]

    for attrs in df.to_dict("records"):
        src_key = _node_key(src_type, list(src_on.values()), attrs)
        tgt_key = _node_key(tgt_type, list(tgt_on.values()), attrs)
        extra = {k: v for k, v in attrs.items() if k not in {*src_on.keys(), *tgt_on.keys()}}

        G.add_edge(src_key, tgt_key, **_edge_attrs(rel, fk_label, extra))
        if undirected:
            G.add_edge(tgt_key, src_key, **_edge_attrs(rel, fk_label, extra))


def build_graph(
    engine: Engine,
    node_specs: List[Dict[str, Any]],
    edge_specs: List[Dict[str, Any]],
    *,
    model_id: int,
    directed: bool = True,
    verbose: bool = True,
) -> Tuple[nx.MultiDiGraph | nx.MultiGraph, Dict[str, pd.DataFrame]]:
    """Build a graph directly from explicit node/edge specs."""
    G = nx.MultiDiGraph(model_id=model_id) if directed else nx.MultiGraph(model_id=model_id)
    frames: Dict[str, pd.DataFrame] = {}
    type_cfg: Dict[str, Dict[str, Any]] = {}

    # ---- materialize nodes
    for spec in node_specs:
        ntype = spec["type"]
        schema = spec["schema"]
        table = spec["table"]
        id_cols = list(spec["id"])
        label_cols = list(spec.get("label") or id_cols)
        model_id_col = spec.get("model_id_col") or "model_id"

        sel = f'SELECT * FROM {_q(schema, table)} WHERE "{model_id_col}" = :m'
        df = pd.read_sql(sql_text(sel), engine, params={"m": model_id})
        frames[f"node:{ntype}"] = df
        type_cfg[ntype] = {
            "id_cols": id_cols,
            "label_cols": label_cols,
            "model_id_col": model_id_col,
            "idx_by_cols": {},
        }
        if verbose:
            print(f"[build_graph] nodes: type='{ntype}' n={len(df)} from {schema}.{table}")

        _add_nodes_from_df(G, df, ntype=ntype, id_cols=id_cols, label_cols=label_cols)

    # index for existence checks
    node_index: Dict[str, Dict[Tuple[str, Tuple[Any, ...]], dict]] = {}
    for n, d in G.nodes(data=True):
        ntype, _ = n
        node_index.setdefault(ntype, {})[n] = d

    def ensure_index(ntype: str, cols: List[str]) -> Dict[Tuple[Any, ...], Tuple[str, Tuple[Any, ...]]]:
        cfg = type_cfg[ntype]
        key = tuple(cols)
        if key in cfg["idx_by_cols"]:
            return cfg["idx_by_cols"][key]
        idx = {}
        df = frames[f"node:{ntype}"]
        for rec in df.to_dict("records"):
            nk = _node_key(ntype, cfg["id_cols"], rec)
            idx[tuple(rec.get(c) for c in cols)] = nk
        cfg["idx_by_cols"][key] = idx
        return idx

    # ---- edges
    for es in edge_specs:
        mode = (es.get("mode") or "fk").lower()
        rel = es.get("rel") or "association"
        bidir = bool(es.get("bidirectional"))
        fk_label = es.get("fk_col") or "fk"

        if mode == "fk":
            src_type = es["src"]
            tgt_type = es["tgt"]
            on_map = es.get("on") or {}

            src_idx = ensure_index(src_type, list(on_map.keys()))
            tgt_idx = ensure_index(tgt_type, list(on_map.values()))

            src_df = frames[f"node:{src_type}"]
            for rec in src_df.to_dict("records"):
                src_key = src_idx.get(tuple(rec.get(c) for c in on_map.keys()))
                tgt_key = tgt_idx.get(tuple(rec.get(c) for c in on_map.values()))
                if not src_key or not tgt_key:
                    continue
                G.add_edge(src_key, tgt_key, **_edge_attrs(rel, fk_label))
                if bidir:
                    G.add_edge(tgt_key, src_key, **_edge_attrs(rel, fk_label))

        elif mode == "assoc":
            schema = es["schema"]
            table = es["table"]
            where = es.get("where") or "model_id = :m"
            params = es.get("params") or {"m": model_id}

            sel = f"SELECT * FROM {_q(schema, table)} WHERE {where}"
            df = pd.read_sql(sql_text(sel), engine, params=params)
            frames[f"edge:{schema}.{table}"] = df

            src = es["src"]
            tgt = es["tgt"]
            undirected = bool(es.get("undirected"))
            _add_assoc_edges(G, df=df, src=src, tgt=tgt, rel=rel, fk_label=fk_label, undirected=undirected)

        if verbose:
            print(f"[build_graph] edge mode='{mode}' rel='{rel}'")

    return G, frames


