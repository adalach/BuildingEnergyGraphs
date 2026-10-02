# beg/build/frames.py
from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Chunk-aware batch iteration and extraction
# -----------------------------------------------------------------------------
#
# The chunked payload these functions read:
#
# out = {
#   "configs_summary": ...,
#   "chunk_map": ...,
#   "data": {
#     "<config_name>": {
#       "chunks": [
#         {
#           "chunk_no": <optional metadata>,
#           "batches": [
#             {
#               "model_ids": [...],
#               "nodes": { "<node_table>": DataFrame, ... },
#               "edges": { "<edge_table>": DataFrame, ... },
#             },
#             ...
#           ]
#         },
#         ...
#       ]
#     },
#     ...
#   }
# }
#
# The chunk_map DataFrame (model_id -> chunk_no) routes a lookup to one chunk,
# so only that chunk's batches are scanned.
#
# Extraction returns:
#   nodes_dict: {table_name: DataFrame}
#   edges_dict: {table_name: DataFrame}
# -----------------------------------------------------------------------------


def build_model_to_chunk_map(
    chunk_map: pd.DataFrame,
    *,
    model_id_col: str = "model_id",
    chunk_col: str = "chunk_no",
) -> Dict[str, int]:
    """
    Build a fast lookup dict {model_id -> chunk_no} from chunk_map.

    Notes:
    - chunk_no values are 1-indexed and returned as-is; they become list indices
      (chunk_no - 1) only when selecting from out["data"][cfg]["chunks"].
    """
    if chunk_map is None or chunk_map.empty:
        return {}

    if model_id_col not in chunk_map.columns or chunk_col not in chunk_map.columns:
        raise KeyError(
            f"chunk_map missing required columns: '{model_id_col}', '{chunk_col}'. "
            f"Got: {list(chunk_map.columns)}"
        )

    mp = chunk_map.set_index(model_id_col)[chunk_col]
    return mp.astype(int).to_dict()


def _resolve_chunk_no(
    model_id: str,
    *,
    chunk_no: Optional[int] = None,
    model_to_chunk: Optional[Dict[str, int]] = None,
    chunk_map: Optional[pd.DataFrame] = None,
) -> Optional[int]:
    """
    Resolve the chunk number for a given model_id.

    Priority:
      1) explicit chunk_no argument
      2) model_to_chunk mapping
      3) chunk_map DataFrame (converted on demand)

    Returns:
      chunk_no (likely 1-indexed) or None if unavailable.
    """
    if chunk_no is not None:
        return int(chunk_no)

    if model_to_chunk is not None:
        got = model_to_chunk.get(str(model_id))
        return int(got) if got is not None else None

    if chunk_map is not None:
        tmp = build_model_to_chunk_map(chunk_map)
        got = tmp.get(str(model_id))
        return int(got) if got is not None else None

    return None


def _iter_batches_for_config(
    out: dict,
    config_name: str,
    *,
    chunk_no: Optional[int] = None,
) -> Iterable[dict]:
    """
    Yield batch dicts for a given config.

    If chunk_no is provided:
      - It is read as 1-indexed and selects out["data"][config_name]["chunks"][chunk_no - 1].
      - If that index is out of range, batches from all chunks are yielded.
    """
    cfg_payload = out.get("data", {}).get(config_name, {})
    chunks = cfg_payload.get("chunks", []) or []

    if chunk_no is None:
        for ch in chunks:
            for b in (ch.get("batches", []) or []):
                yield b
        return

    idx = int(chunk_no) - 1  # assume 1-indexed chunk numbers in metadata
    if 0 <= idx < len(chunks):
        ch = chunks[idx]
        for b in (ch.get("batches", []) or []):
            yield b
        return

    # Fallback: chunk_no doesn't match chunks layout; scan all chunks
    for ch in chunks:
        for b in (ch.get("batches", []) or []):
            yield b


def extract_model_dfs_from_out(
    out: dict,
    config_name: str,
    model_id: str,
    *,
    chunk_no: Optional[int] = None,
    model_to_chunk: Optional[Dict[str, int]] = None,
    chunk_map: Optional[pd.DataFrame] = None,
    model_id_col: str = "model_id",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Extract per-model nodes/edges DataFrames for one (config_name, model_id).

    Performance behavior:
      - If chunk routing metadata is available, only batches in that chunk are scanned.
      - Otherwise, all chunks/batches are scanned (fallback).

    Returns:
      nodes_dict: {node_table: DataFrame} filtered to this model_id
      edges_dict: {edge_table: DataFrame} filtered to this model_id
    """
    resolved_chunk_no = _resolve_chunk_no(
        str(model_id),
        chunk_no=chunk_no,
        model_to_chunk=model_to_chunk,
        chunk_map=chunk_map,
    )

    node_parts: Dict[str, List[pd.DataFrame]] = {}
    edge_parts: Dict[str, List[pd.DataFrame]] = {}

    for b in _iter_batches_for_config(out, config_name, chunk_no=resolved_chunk_no):
        # Even within a chunk, a batch contains multiple model_ids, so skip fast
        if model_id not in (b.get("model_ids") or []):
            continue

        # Nodes
        for node_table, df in (b.get("nodes") or {}).items():
            if df is None or not hasattr(df, "columns"):
                continue
            if model_id_col not in df.columns:
                continue
            sub = df[df[model_id_col] == model_id]
            if not sub.empty:
                node_parts.setdefault(node_table, []).append(sub)

        # Edges
        for edge_table, df in (b.get("edges") or {}).items():
            if df is None or not hasattr(df, "columns"):
                continue
            if model_id_col not in df.columns:
                continue
            sub = df[df[model_id_col] == model_id]
            if not sub.empty:
                edge_parts.setdefault(edge_table, []).append(sub)

    nodes = {k: pd.concat(v, ignore_index=True) for k, v in node_parts.items()}
    edges = {k: pd.concat(v, ignore_index=True) for k, v in edge_parts.items()}
    return nodes, edges


# -----------------------------------------------------------------------------
# Model id selection utilities
# -----------------------------------------------------------------------------


def choose_shared_model_ids(
    chunk_map: pd.DataFrame,
    configs_summary: pd.DataFrame,
    config_names: List[str],
    n: int,
) -> List[str]:
    """
    Choose N model_ids that are shared across all listed config_names.

    Steps:
    - Read configs_summary["model_ids"] for each config (list or stringified list)
    - Compute intersection across configs
    - Order the shared ids by first appearance in chunk_map["model_id"]
    """
    cfg_to_ids: Dict[str, set] = {}

    for cfg in config_names:
        row = configs_summary.loc[configs_summary["config_name"] == cfg]
        if row.empty:
            raise KeyError(f"Config not found in configs_summary: {cfg}")

        ids = row.iloc[0]["model_ids"]
        if isinstance(ids, str):
            ids = ast.literal_eval(ids)

        cfg_to_ids[cfg] = set(ids)

    shared = set.intersection(*cfg_to_ids.values()) if cfg_to_ids else set()
    if not shared:
        raise RuntimeError("No shared model_ids across the requested configs.")

    ordered = [mid for mid in chunk_map["model_id"].tolist() if mid in shared]
    return ordered[:n]


# -----------------------------------------------------------------------------
# DataFrame housekeeping helpers
# -----------------------------------------------------------------------------


def _stable_sort_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provide a stable, deterministic sort for node DataFrames if common keys exist.

    Sort preference order if present:
      1) idx
      2) id
      3) instance_name
    """
    keys: List[str] = []
    if "idx" in df.columns:
        keys.append("idx")
    if "id" in df.columns:
        keys.append("id")
    if "instance_name" in df.columns:
        keys.append("instance_name")

    if keys:
        return df.sort_values(keys, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Config-spec parsing helpers (nodes/edges columns in configs_summary)
# -----------------------------------------------------------------------------


def parse_nodes_spec(nodes_spec: Any) -> List[dict]:
    """
    Parse configs_summary["nodes"] into a Python list of dicts.

    Accepts:
      - None -> []
      - list -> returned as-is
      - stringified python literal -> ast.literal_eval
    """
    if nodes_spec is None:
        return []
    if isinstance(nodes_spec, list):
        return nodes_spec
    if isinstance(nodes_spec, str):
        return ast.literal_eval(nodes_spec)
    raise TypeError(f"Unsupported nodes_spec type: {type(nodes_spec)}")


def parse_edges_spec(edges_spec: Any) -> List[dict]:
    """
    Parse configs_summary["edges"] into a Python list of dicts.

    Accepts:
      - None -> []
      - list -> returned as-is
      - stringified python literal -> ast.literal_eval
    """
    if edges_spec is None:
        return []
    if isinstance(edges_spec, list):
        return edges_spec
    if isinstance(edges_spec, str):
        return ast.literal_eval(edges_spec)
    raise TypeError(f"Unsupported edges_spec type: {type(edges_spec)}")


def get_het_specs(configs_summary: pd.DataFrame, config_name: str) -> Dict[str, Any]:
    """
    Extract the heterogeneous graph specification for one config.

    Returns:
      {
        "config_name": <config_name>,
        "node_tables": [<table>, ...],
        "edges_list":  [ {edge_src, src_cat, tgt_cat, rel, bidirectional, ...}, ... ]
      }
    """
    row = configs_summary.loc[configs_summary["config_name"] == config_name]
    if row.empty:
        raise KeyError(f"Config not found: {config_name}")
    row = row.iloc[0]

    nodes_list = parse_nodes_spec(row["nodes"])
    edges_list = parse_edges_spec(row["edges"])

    node_tables = [d["table"] for d in nodes_list if "table" in d]
    return {"config_name": config_name, "node_tables": node_tables, "edges_list": edges_list}


__all__ = [
    # chunk-aware extraction
    "build_model_to_chunk_map",
    "extract_model_dfs_from_out",
    # selection
    "choose_shared_model_ids",
    # df utilities
    "_stable_sort_nodes",
    # config parsing
    "parse_nodes_spec",
    "parse_edges_spec",
    "get_het_specs",
]
