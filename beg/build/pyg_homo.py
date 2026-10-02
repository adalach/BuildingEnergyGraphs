# beg/build/pyg_homo.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch_geometric.data import Data

from beg.build.frames import (
    _stable_sort_nodes,
    build_model_to_chunk_map,
    choose_shared_model_ids,
    extract_model_dfs_from_out,
    parse_edges_spec,
    parse_nodes_spec,
)


def _is_identifier_like_col(col: str) -> bool:
    """
    Return True for technical/identifier columns that must never be used
    as ML features.
    """
    lc = str(col).lower()
    if not lc:
        return True
    if lc.startswith("__"):
        return True
    if lc in {
        "id",
        "idx",
        "index",
        "model_id",
        "node_id",
        "src_id",
        "tgt_id",
        "src_idx",
        "tgt_idx",
    }:
        return True
    if lc.endswith("_id") or lc.endswith("_idx") or lc.endswith("_key"):
        return True
    return False


# -----------------------------------------------------------------------------
# Homogeneous config helpers
# -----------------------------------------------------------------------------


def parse_homo_node_order(configs_summary: pd.DataFrame) -> Dict[str, List[str]]:
    """
    For each config, parse configs_summary["nodes"] and return:
      { config_name: [node_table_1, node_table_2, ...] }

    Node tables keep the order the configuration declares; the homogeneous builder
    merges them in that order.
    """
    out_order: Dict[str, List[str]] = {}
    for _, row in configs_summary.iterrows():
        cfg = row["config_name"]
        nodes_list = parse_nodes_spec(row["nodes"])
        out_order[cfg] = [d["table"] for d in nodes_list if "table" in d]
    return out_order


def parse_homo_edge_specs(configs_summary: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    For each config, parse configs_summary["edges"] and return:
      { config_name: [edge_spec_1, edge_spec_2, ...] }
    """
    out_specs: Dict[str, List[dict]] = {}
    for _, row in configs_summary.iterrows():
        cfg = row["config_name"]
        out_specs[cfg] = parse_edges_spec(row["edges"])
    return out_specs


def _clean_and_merge_node_tables(
    nodes_dict: Dict[str, pd.DataFrame],
    node_table_order: List[str],
    *,
    node_key: str,
) -> pd.DataFrame:
    """
    Merge multiple node tables into a single node_df (one row per node_key).

    Strategy:
      - Concatenate tables in config order
      - Add __node_table__ indicating provenance
      - For nodes appearing multiple times (same node_key across tables),
        keep the first row as base, then fill missing values from subsequent rows.
    """
    parts: List[pd.DataFrame] = []

    for nt in node_table_order:
        if nt not in nodes_dict:
            continue
        df = _stable_sort_nodes(nodes_dict[nt]).copy()
        df["__node_table__"] = nt
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[node_key])

    all_nodes = pd.concat(parts, ignore_index=True)

    if node_key not in all_nodes.columns:
        raise KeyError(f"Node tables missing required key column: {node_key}")

    # One row per node key: `groupby(...).first()` takes the first non-null value
    # of each column in table order.
    if all_nodes[node_key].is_unique:
        # The common case: nothing to merge, so skip the grouping entirely.
        return all_nodes.reset_index(drop=True)

    column_order = list(all_nodes.columns)
    merged = all_nodes.groupby(node_key, sort=False).first().reset_index()
    return merged[column_order]


def _numeric_tensor_from_df(df: pd.DataFrame, cols: List[str]) -> torch.Tensor:
    """
    Convert selected numeric columns into a float32 tensor.
    NaNs are replaced with 0.0.

    If cols is empty, returns shape [len(df), 0].
    """
    if df is None or df.empty or not cols:
        return torch.zeros((0 if df is None else len(df), 0), dtype=torch.float32)
    mat = df[cols].copy().fillna(0.0).to_numpy()
    return torch.tensor(mat, dtype=torch.float32)


def _infer_name_col(df: pd.DataFrame, *, exclude: Optional[Sequence[str]] = None) -> Optional[str]:
    exclude_set = set(exclude or [])
    candidates = [c for c in df.columns if c.endswith("_name") and c not in exclude_set]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_edge_endpoint_cols(
    *,
    config_name: str,
    edge_table: str,
    df: pd.DataFrame,
    edge_spec: Optional[dict],
) -> tuple[str, str]:
    """
    Resolve endpoint columns for a homogeneous edge table.

    Preference order:
      1) src_id/tgt_id
      2) config-defined src_col/tgt_col
      3) generic src/tgt
      4) heuristic unique *_name columns
    """
    if ("src_id" in df.columns) and ("tgt_id" in df.columns):
        return "src_id", "tgt_id"

    src_pref = edge_spec.get("src_col") if edge_spec else None
    tgt_pref = edge_spec.get("tgt_col") if edge_spec else None

    src_col_used = src_pref if src_pref in df.columns else None
    tgt_col_used = tgt_pref if tgt_pref in df.columns else None

    if src_col_used is None and "src" in df.columns:
        src_col_used = "src"
    if tgt_col_used is None and "tgt" in df.columns:
        tgt_col_used = "tgt"

    if src_col_used is None:
        src_col_used = _infer_name_col(df, exclude=[tgt_col_used, "src", "tgt"])
    if tgt_col_used is None:
        tgt_col_used = _infer_name_col(df, exclude=[src_col_used, "src", "tgt"])

    if src_col_used is None or tgt_col_used is None:
        raise KeyError(
            f"[{config_name}] edge table '{edge_table}' missing endpoint columns. "
            f"Expected src_id/tgt_id or config src_col/tgt_col. "
            f"Columns: {list(df.columns)}"
        )

    if (src_pref and src_col_used != src_pref) or (tgt_pref and tgt_col_used != tgt_pref):
        warnings.warn(
            f"[{config_name}] edge table '{edge_table}' missing configured endpoint columns "
            f"({src_pref}, {tgt_pref}); using ({src_col_used}, {tgt_col_used}) instead.",
            stacklevel=2,
        )

    return str(src_col_used), str(tgt_col_used)


# -----------------------------------------------------------------------------
# Homogeneous PyG builder (numeric-only)
# -----------------------------------------------------------------------------


def build_homo_data_for_model_numeric(
    *,
    config_name: str,
    model_id: str,
    nodes_dict: Dict[str, pd.DataFrame],
    edges_dict: Dict[str, pd.DataFrame],
    node_table_order: List[str],
    edges_list: Optional[List[dict]] = None,
    node_key: str = "instance_name",
    model_id_col: str = "model_id",
    keep_raw_dfs: bool = True,
    edge_feature_cols_expected: Optional[List[str]] = None,  # stable edge schema
) -> Data:
    """
    Build a numeric-only homogeneous PyG Data object.

    Pipeline assumptions:
      - All node/edge feature columns are numeric (categoricals already encoded).
      - Edge endpoints are resolved from src_id/tgt_id or config src_col/tgt_col.
      - Node identity is node_key (default: instance_name).
      - Some models may have zero edges: edge_index may be [2, 0].

    Important behavior:
      - Edges that reference nodes absent from the node tables add placeholder nodes,
        with 0-valued features and __node_table__ = "(missing_from_edges)" in node_df_raw.
      - A relation declared bidirectional also gets its reverse edges.

    Stable batching behavior (important for DataLoader collation):
      - If edge_feature_cols_expected is provided, edge_attr is forced to a stable
        feature dimension across all models in the config.
      - Edge-less models then get edge_attr of shape (0, D), not (0, 0).
    """
    # ---- nodes ----
    node_df = _clean_and_merge_node_tables(
        nodes_dict,
        node_table_order=node_table_order,
        node_key=node_key,
    ).copy()

    if node_key not in node_df.columns:
        raise KeyError(f"[{config_name}] node_df missing key col: {node_key}")

    node_df = _stable_sort_nodes(node_df).reset_index(drop=True)
    name_to_id = {name: i for i, name in enumerate(node_df[node_key].astype(str).tolist())}
    node_df["node_id"] = node_df[node_key].astype(str).map(name_to_id).astype(int)

    # ---- edges ----
    src_ids_all: List[int] = []
    tgt_ids_all: List[int] = []
    attr_blocks: List[pd.DataFrame] = []
    edge_specs_by_table = {
        str(e.get("edge_src")): e
        for e in (edges_list or [])
        if e.get("edge_src")
    }

    for edge_table, df in edges_dict.items():
        if df is None or df.empty:
            continue

        edge_spec = edge_specs_by_table.get(str(edge_table))
        src_col_used, tgt_col_used = _resolve_edge_endpoint_cols(
            config_name=config_name,
            edge_table=str(edge_table),
            df=df,
            edge_spec=edge_spec,
        )

        s_raw = df[src_col_used]
        t_raw = df[tgt_col_used]
        mask = (~s_raw.isna()) & (~t_raw.isna())

        s = s_raw.loc[mask].astype(str).reset_index(drop=True)
        t = t_raw.loc[mask].astype(str).reset_index(drop=True)
        df_sub = df.loc[mask].reset_index(drop=True).copy()
        df_sub["__edge_table__"] = edge_table

        # Add missing endpoint nodes (rare)
        for nm in pd.unique(pd.concat([s, t], ignore_index=True)):
            if nm not in name_to_id:
                name_to_id[nm] = len(name_to_id)

        src_ids = s.map(name_to_id).astype(int).to_list()
        tgt_ids = t.map(name_to_id).astype(int).to_list()
        src_ids_all.extend(src_ids)
        tgt_ids_all.extend(tgt_ids)

        # Keep non-feature cols like "__edge_table__" for debugging, but drop endpoints + model_id
        drop_cols = {model_id_col, "src_id", "tgt_id", "src", "tgt", src_col_used, tgt_col_used}
        attr_df = df_sub.drop(columns=[c for c in drop_cols if c in df_sub.columns], errors="ignore")
        attr_blocks.append(attr_df)

        # A relation the configuration declares bidirectional gets its reverse,
        # with the same attributes on both directions, as in the heterogeneous
        # builder: homogeneous and heterogeneous configurations must agree on
        # edge directionality.
        if bool((edge_spec or {}).get("bidirectional", False)):
            src_ids_all.extend(tgt_ids)
            tgt_ids_all.extend(src_ids)
            attr_blocks.append(attr_df.copy())

    edge_index = (
        torch.tensor([src_ids_all, tgt_ids_all], dtype=torch.long)
        if src_ids_all
        else torch.zeros((2, 0), dtype=torch.long)
    )

    edge_attr_df = (
        pd.concat(attr_blocks, ignore_index=True)
        if attr_blocks
        else pd.DataFrame(index=range(int(edge_index.size(1))))
    )

    # If edges introduced nodes, extend node_df to match mapping
    if len(name_to_id) != len(node_df):
        existing = set(node_df[node_key].astype(str).tolist())
        missing = [n for n in name_to_id.keys() if n not in existing]
        if missing:
            add_df = pd.DataFrame({node_key: missing})
            add_df["__node_table__"] = "(missing_from_edges)"
            add_df["node_id"] = add_df[node_key].astype(str).map(name_to_id).astype(int)
            node_df = pd.concat([node_df, add_df], ignore_index=True)

        node_df["node_id"] = node_df[node_key].astype(str).map(name_to_id).astype(int)
        node_df = node_df.sort_values("node_id", kind="mergesort").reset_index(drop=True)

    # ---- numeric features only ----
    node_ignore = {model_id_col, node_key, "node_id", "id", "idx"}
    node_feature_cols = [
        c
        for c in node_df.columns
        if (
            c not in node_ignore
            and not _is_identifier_like_col(c)
            and pd.api.types.is_numeric_dtype(node_df[c])
        )
    ]
    x = _numeric_tensor_from_df(node_df, node_feature_cols)

    # Edge numeric columns inferred from this model
    edge_ignore = {model_id_col, "src_id", "tgt_id", "src_idx", "tgt_idx", "id", "idx", "node_id"}
    edge_feature_cols = [
        c
        for c in edge_attr_df.columns
        if (
            c not in edge_ignore
            and not _is_identifier_like_col(c)
            and pd.api.types.is_numeric_dtype(edge_attr_df[c])
        )
    ]

    # -------------------------------------------------------------------------
    # Enforce a stable edge feature schema so DataLoader batches collate
    # -------------------------------------------------------------------------
    expected = (
        [c for c in edge_feature_cols_expected if not _is_identifier_like_col(c)]
        if edge_feature_cols_expected is not None
        else None
    )
    E = int(edge_index.size(1))

    if expected is not None:
        # Ensure all expected numeric columns exist (even if edge_attr_df is empty)
        for c in expected:
            if c not in edge_attr_df.columns:
                edge_attr_df[c] = 0.0

        # Fixed ordering
        edge_feature_cols = list(expected)

        if E == 0:
            # Important: keep feature dim consistent for edge-less models
            edge_attr = torch.zeros((0, len(expected)), dtype=torch.float32)
        else:
            # Reindex numeric features to expected schema and fill NaNs
            mat = edge_attr_df[edge_feature_cols].copy().fillna(0.0).to_numpy()
            edge_attr = torch.tensor(mat, dtype=torch.float32)
    else:
        # Default: infer per-model schema (may vary across models, not batch-safe)
        edge_attr = _numeric_tensor_from_df(edge_attr_df, edge_feature_cols)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.config_name = config_name
    data.model_id = model_id
    data.node_feature_cols = node_feature_cols
    data.edge_feature_cols = edge_feature_cols

    if keep_raw_dfs:
        data.node_df_raw = node_df
        data.edge_df_raw = edge_attr_df

    return data


# -----------------------------------------------------------------------------
# Batch builder over the model ids shared by several configurations
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HomoBuildResult:
    """
    Result bundle returned by build_homo_pyg_graphs(...).
    """
    model_ids: List[str]
    pyg_by_config: Dict[str, List[Data]]
    node_order_map: Dict[str, List[str]]


def build_homo_pyg_graphs(
    *,
    out: dict,
    configs_summary: pd.DataFrame,
    chunk_map: pd.DataFrame,
    homo_configs: Sequence[str],
    n_graphs: int,
    node_key: str = "instance_name",
    keep_raw_dfs: bool = True,
    use_chunk_map: bool = True,
) -> HomoBuildResult:
    """
    Build homogeneous PyG graphs for all homo_configs for a set of shared model_ids.

    Notes:
      - Uses choose_shared_model_ids(...) to pick model_ids shared across all configs.
      - Uses chunk_map to route each model_id to a chunk and avoid scanning all batches.

    Stable edge schema enforcement:
      - Within a single config, edge_attr feature dim must be constant for batching.
      - Some models may have 0 edges (E=0). If the config normally has D edge features,
        we must represent those as edge_attr shape (0, D) instead of (0, 0).
      - We discover the per-config edge feature schema from the first model that has
        at least one numeric edge feature, then enforce it for all subsequent models.
      - Any earlier edge-less graphs are retroactively patched to edge_attr (0, D).

    Returns:
      HomoBuildResult(model_ids, pyg_by_config, node_order_map)
    """
    homo_configs = list(homo_configs)

    node_order_map = parse_homo_node_order(configs_summary)
    edge_specs_map = parse_homo_edge_specs(configs_summary)
    model_ids = (
        choose_shared_model_ids(chunk_map, configs_summary, homo_configs, n_graphs)
        if homo_configs
        else []
    )

    model_to_chunk = build_model_to_chunk_map(chunk_map) if use_chunk_map else None

    pyg_by_config: Dict[str, List[Data]] = {}

    for cfg in homo_configs:
        order = node_order_map.get(cfg, [])
        edges_list = edge_specs_map.get(cfg, [])
        built: List[Data] = []

        # Per-config stable schema for edge features (discovered lazily)
        edge_cols_ref: Optional[List[str]] = None
        pending_no_edge: List[int] = []  # indices of graphs built before schema discovery

        for mid in model_ids:
            nodes_dict, edges_dict = extract_model_dfs_from_out(
                out,
                cfg,
                mid,
                model_to_chunk=model_to_chunk,
                chunk_map=None,
            )

            g = build_homo_data_for_model_numeric(
                config_name=cfg,
                model_id=mid,
                nodes_dict=nodes_dict,
                edges_dict=edges_dict,
                node_table_order=order,
                edges_list=edges_list,
                node_key=node_key,
                keep_raw_dfs=keep_raw_dfs,
                edge_feature_cols_expected=edge_cols_ref,  # enforce if known
            )
            built.append(g)

            # Discover schema from first graph that has numeric edge features
            if edge_cols_ref is None and len(getattr(g, "edge_feature_cols", [])) > 0:
                edge_cols_ref = list(g.edge_feature_cols)

                # Retroactively patch any earlier edge-less graphs
                for j in pending_no_edge:
                    gj = built[j]
                    if int(gj.edge_index.size(1)) != 0:
                        continue

                    gj.edge_attr = torch.zeros((0, len(edge_cols_ref)), dtype=torch.float32)
                    gj.edge_feature_cols = list(edge_cols_ref)

                    if keep_raw_dfs and hasattr(gj, "edge_df_raw") and isinstance(gj.edge_df_raw, pd.DataFrame):
                        for c in edge_cols_ref:
                            if c not in gj.edge_df_raw.columns:
                                # Create empty numeric column; length is 0 because there are 0 edges
                                gj.edge_df_raw[c] = pd.Series(dtype=float)

            # Track edge-less graphs built before schema discovery
            if edge_cols_ref is None and int(g.edge_index.size(1)) == 0:
                pending_no_edge.append(len(built) - 1)

            # Safety check: once schema is known, enforce consistent edge_attr dim
            if edge_cols_ref is not None:
                expected_dim = len(edge_cols_ref)
                got_dim = int(g.edge_attr.size(1)) if getattr(g, "edge_attr", None) is not None else None
                if got_dim != expected_dim:
                    raise RuntimeError(
                        f"[{cfg}] edge_attr dim mismatch for model_id={mid}: got {got_dim}, expected {expected_dim}"
                    )

        pyg_by_config[cfg] = built

    return HomoBuildResult(model_ids=model_ids, pyg_by_config=pyg_by_config, node_order_map=node_order_map)
