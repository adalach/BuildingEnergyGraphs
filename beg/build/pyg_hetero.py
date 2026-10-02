# beg/build/pyg_hetero.py - stable edge_attr dimensions per edge type
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from beg.build.frames import (
    _stable_sort_nodes,
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


def _numeric_tensor_from_df(df: pd.DataFrame, cols: List[str]) -> torch.Tensor:
    if df is None or df.empty or not cols:
        return torch.zeros((0 if df is None else len(df), 0), dtype=torch.float32)
    mat = df[cols].copy().fillna(0.0).to_numpy()
    return torch.tensor(mat, dtype=torch.float32)


def _node_feature_cols(df: pd.DataFrame, *, model_id_col: str, node_key: str) -> List[str]:
    ignore = {model_id_col, node_key, "node_id", "id", "idx"}
    return [
        c
        for c in df.columns
        if c not in ignore and not _is_identifier_like_col(c) and pd.api.types.is_numeric_dtype(df[c])
    ]


def _edge_feature_cols(
    df: pd.DataFrame,
    *,
    model_id_col: str,
    ignore_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    ignore = {model_id_col, "src_id", "tgt_id", "src_idx", "tgt_idx", "id", "idx", "node_id"}
    if ignore_cols:
        ignore.update({c for c in ignore_cols if c})
    return [
        c
        for c in df.columns
        if c not in ignore and not _is_identifier_like_col(c) and pd.api.types.is_numeric_dtype(df[c])
    ]


def _infer_name_col(df: pd.DataFrame, *, exclude: Optional[Sequence[str]] = None) -> Optional[str]:
    exclude_set = set(exclude or [])
    candidates = [c for c in df.columns if c.endswith("_name") and c not in exclude_set]
    if len(candidates) == 1:
        return candidates[0]
    return None


def build_hetero_data_for_model_numeric(
    *,
    config_name: str,
    model_id: str,
    nodes_dict: Dict[str, pd.DataFrame],
    edges_dict: Dict[str, pd.DataFrame],
    node_tables: List[str],
    edges_list: List[dict],
    node_key: str = "instance_name",
    model_id_col: str = "model_id",
    keep_raw_dfs: bool = True,
    edge_feature_cols_expected: Optional[Dict[Tuple[str, str, str], List[str]]] = None,
    filter_model_id: bool = True,  # allow shared dfs with multiple model_ids
) -> HeteroData:
    """
    Numeric-only HeteroData builder.

    Shared DataFrame support:
      - If nodes_dict / edges_dict contain multiple model_ids, set filter_model_id=True
        (default) to filter rows for the requested model_id.

    Stable batching behavior:
      - If edge_feature_cols_expected is provided, edge_attr feature dim is forced
        per edge type across all models in the config.
      - For edge-less stores, edge_attr becomes shape (0, D) instead of (0, 0).

    Bidirectional behavior:
      - Reverse edges keep the SAME relation name (`rel`) instead of creating `rev_*`.
      - If src/tgt node types are the same, both directions are merged into one edge store.
      - If src/tgt node types differ, reverse edges are stored under the swapped etype
        `(tgt_cat, rel, src_cat)`.
    """
    hd = HeteroData()
    hd.config_name = config_name
    hd.model_id = model_id

    # ---- node stores ----
    node_maps: Dict[str, Dict[str, int]] = {}
    node_dfs_raw: Dict[str, pd.DataFrame] = {}
    node_feature_cols_map: Dict[str, List[str]] = {}

    for nt in node_tables:
        df = nodes_dict.get(nt)
        if df is not None and filter_model_id and model_id_col in df.columns:
            df = df[df[model_id_col] == model_id].copy()

        if df is None or df.empty:
            hd[nt].x = torch.zeros((0, 0), dtype=torch.float32)
            node_maps[nt] = {}
            node_feature_cols_map[nt] = []
            if keep_raw_dfs:
                node_dfs_raw[nt] = df.copy() if df is not None else pd.DataFrame(columns=[node_key, "node_id"])
            continue

        if node_key not in df.columns:
            raise KeyError(f"[{config_name}] node table '{nt}' missing '{node_key}'. Columns: {list(df.columns)}")

        df = _stable_sort_nodes(df).reset_index(drop=True)

        name_to_id = {name: i for i, name in enumerate(df[node_key].astype(str).tolist())}
        node_maps[nt] = name_to_id
        df["node_id"] = df[node_key].astype(str).map(name_to_id).astype(int)

        feat_cols = _node_feature_cols(df, model_id_col=model_id_col, node_key=node_key)
        node_feature_cols_map[nt] = feat_cols
        hd[nt].x = _numeric_tensor_from_df(df, feat_cols)

        if keep_raw_dfs:
            node_dfs_raw[nt] = df

    # ---- edge stores ----
    edge_dfs_raw: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    edge_feature_cols_map: Dict[Tuple[str, str, str], List[str]] = {}
    dropped_edges: Dict[Tuple[str, str, str], int] = {}

    def _expected_cols_for(etype: Tuple[str, str, str]) -> Optional[List[str]]:
        if edge_feature_cols_expected is None:
            return None
        cols = edge_feature_cols_expected.get(etype)
        if cols is None:
            return None
        return [c for c in cols if not _is_identifier_like_col(c)]

    for e in edges_list:
        edge_src = e.get("edge_src")
        src_cat = e.get("src_cat")
        tgt_cat = e.get("tgt_cat")
        rel = e.get("rel", "to")
        bidir = bool(e.get("bidirectional", False))

        if edge_src is None or src_cat is None or tgt_cat is None:
            raise KeyError(f"[{config_name}] edge spec missing keys. Got: {e}")

        etype = (str(src_cat), str(rel), str(tgt_cat))
        # Reverse edges keep the same relation label; no "rev_*" relation names.
        rtype = (str(tgt_cat), str(rel), str(src_cat)) if bidir else None

        df = edges_dict.get(edge_src)
        if df is not None and filter_model_id and model_id_col in df.columns:
            df = df[df[model_id_col] == model_id].copy()

        expected_cols = _expected_cols_for(etype)
        expected_dim = len(expected_cols) if expected_cols is not None else None

        # Always create store (even if empty), respecting expected schema if given
        if df is None or df.empty:
            hd[etype].edge_index = torch.zeros((2, 0), dtype=torch.long)
            if expected_cols is None:
                hd[etype].edge_attr = torch.zeros((0, 0), dtype=torch.float32)
                edge_feature_cols_map[etype] = []
            else:
                hd[etype].edge_attr = torch.zeros((0, expected_dim), dtype=torch.float32)
                edge_feature_cols_map[etype] = list(expected_cols)

            if keep_raw_dfs:
                cols = ["src_id", "tgt_id", "__edge_table__"] + (list(expected_cols) if expected_cols else [])
                edge_dfs_raw[etype] = pd.DataFrame(columns=cols)

            if bidir and rtype is not None and rtype != etype:
                hd[rtype].edge_index = torch.zeros((2, 0), dtype=torch.long)
                if expected_cols is None:
                    hd[rtype].edge_attr = torch.zeros((0, 0), dtype=torch.float32)
                    edge_feature_cols_map[rtype] = []
                else:
                    hd[rtype].edge_attr = torch.zeros((0, expected_dim), dtype=torch.float32)
                    edge_feature_cols_map[rtype] = list(expected_cols)

                if keep_raw_dfs:
                    cols = ["src_id", "tgt_id", "__edge_table__"] + (list(expected_cols) if expected_cols else [])
                    edge_dfs_raw[rtype] = pd.DataFrame(columns=cols)

            continue

        # Resolve endpoint columns (prefer src_id/tgt_id if present)
        if ("src_id" in df.columns) and ("tgt_id" in df.columns):
            src_col_used, tgt_col_used = "src_id", "tgt_id"
        else:
            src_pref = e.get("src_col")
            tgt_pref = e.get("tgt_col")

            src_col_used = src_pref if src_pref in df.columns else None
            tgt_col_used = tgt_pref if tgt_pref in df.columns else None

            # Common generic cols
            if src_col_used is None and "src" in df.columns:
                src_col_used = "src"
            if tgt_col_used is None and "tgt" in df.columns:
                tgt_col_used = "tgt"

            # Heuristic fallback: unique *_name column not already used
            if src_col_used is None:
                src_col_used = _infer_name_col(df, exclude=[tgt_col_used, "src", "tgt"])
            if tgt_col_used is None:
                tgt_col_used = _infer_name_col(df, exclude=[src_col_used, "src", "tgt"])

            if src_col_used is None or tgt_col_used is None:
                raise KeyError(
                    f"[{config_name}] edge df '{edge_src}' missing endpoint columns. "
                    f"Expected src_id/tgt_id or config src_col/tgt_col. "
                    f"Got columns: {list(df.columns)}"
                )

            # Warn if we had to deviate from config
            if (src_pref and src_col_used != src_pref) or (tgt_pref and tgt_col_used != tgt_pref):
                warnings.warn(
                    f"[{config_name}] edge df '{edge_src}' missing configured endpoint columns "
                    f"({src_pref}, {tgt_pref}); using ({src_col_used}, {tgt_col_used}) instead.",
                    stacklevel=2,
                )

        s = df[src_col_used].astype(str)
        t = df[tgt_col_used].astype(str)
        base_mask = (~s.isna()) & (~t.isna())

        df_sub = df.loc[base_mask].reset_index(drop=True).copy()
        s = s[base_mask].reset_index(drop=True)
        t = t[base_mask].reset_index(drop=True)

        src_map = node_maps.get(str(src_cat), {})
        tgt_map = node_maps.get(str(tgt_cat), {})

        ok_mask = s.isin(src_map.keys()) & t.isin(tgt_map.keys())
        n_drop = int((~ok_mask).sum())
        if n_drop:
            dropped_edges[etype] = dropped_edges.get(etype, 0) + n_drop

        s_ok = s[ok_mask].reset_index(drop=True)
        t_ok = t[ok_mask].reset_index(drop=True)
        df_ok = df_sub.loc[ok_mask].reset_index(drop=True).copy()

        src_ids = s_ok.map(src_map).astype(int).to_list()
        tgt_ids = t_ok.map(tgt_map).astype(int).to_list()

        edge_index = (
            torch.tensor([src_ids, tgt_ids], dtype=torch.long)
            if src_ids
            else torch.zeros((2, 0), dtype=torch.long)
        )

        # numeric features: inferred OR forced to expected schema
        if expected_cols is None:
            feat_cols = _edge_feature_cols(
                df_ok,
                model_id_col=model_id_col,
                ignore_cols=[src_col_used, tgt_col_used],
            )
            edge_attr = _numeric_tensor_from_df(df_ok, feat_cols)
        else:
            for c in expected_cols:
                if c not in df_ok.columns:
                    df_ok[c] = 0.0
            feat_cols = list(expected_cols)

            if int(edge_index.size(1)) == 0:
                edge_attr = torch.zeros((0, len(expected_cols)), dtype=torch.float32)
            else:
                mat = df_ok[feat_cols].copy().fillna(0.0).to_numpy()
                edge_attr = torch.tensor(mat, dtype=torch.float32)

        hd[etype].edge_index = edge_index
        hd[etype].edge_attr = edge_attr
        edge_feature_cols_map[etype] = feat_cols

        if keep_raw_dfs:
            df_dbg = df_ok.copy()
            # Normalize endpoint columns for debugging
            if "src_id" not in df_dbg.columns:
                df_dbg["src_id"] = s_ok.astype(str).to_list()
            if "tgt_id" not in df_dbg.columns:
                df_dbg["tgt_id"] = t_ok.astype(str).to_list()
            df_dbg["__edge_table__"] = str(edge_src)
            keep_cols = [c for c in ["src_id", "tgt_id", "__edge_table__"] if c in df_dbg.columns] + feat_cols
            edge_dfs_raw[etype] = df_dbg[keep_cols].copy()

        if bidir and rtype is not None:
            rev_edge_index = torch.flip(edge_index, dims=[0])

            # Same node type on both ends: merge both directions into one etype store.
            if rtype == etype:
                hd[etype].edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
                hd[etype].edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
                if keep_raw_dfs and etype in edge_dfs_raw:
                    df_rev = edge_dfs_raw[etype].copy()
                    if "src_id" in df_rev.columns and "tgt_id" in df_rev.columns:
                        df_rev["src_id"], df_rev["tgt_id"] = df_rev["tgt_id"], df_rev["src_id"]
                    edge_dfs_raw[etype] = pd.concat([edge_dfs_raw[etype], df_rev], ignore_index=True)
            else:
                hd[rtype].edge_index = rev_edge_index
                hd[rtype].edge_attr = edge_attr
                edge_feature_cols_map[rtype] = feat_cols
                if keep_raw_dfs:
                    df_rev = edge_dfs_raw[etype].copy()
                    if "src_id" in df_rev.columns and "tgt_id" in df_rev.columns:
                        df_rev["src_id"], df_rev["tgt_id"] = df_rev["tgt_id"], df_rev["src_id"]
                    edge_dfs_raw[rtype] = df_rev

    hd.node_feature_cols = node_feature_cols_map
    hd.edge_feature_cols = edge_feature_cols_map
    hd.dropped_edges = dropped_edges
    if keep_raw_dfs:
        hd.node_df_raw = node_dfs_raw
        hd.edge_df_raw = edge_dfs_raw

    return hd
