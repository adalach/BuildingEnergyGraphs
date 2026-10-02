"""
Database-to-DataFrame step of the pipeline (tables or views).

This module coordinates:
1. Config loading
2. SQL transformations (tables in ep_transformed or views in ep)
3. Direct table fetch into pandas DataFrames

The graph builders in :mod:`beg.build` consume the returned DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from beg.pipeline.sql import execute_pipeline_sql


def build_graph_dataframes(
    configs_dir: Path,
    config_names: List[str],
    engine: Optional[Engine] = None,
    schema: Optional[str] = None,
    rebuild_transforms: bool = False,
    max_graphs: Optional[int] = None,
    verbose: bool = True,
    transform_mode: str = "tables",
    transform_schema: Optional[str] = None,
    raw_schema: str = "ep",
) -> Dict[str, Any]:
    """
    Run the transforms of ``config_names`` and fetch each configuration's node and edge tables.
    Tables mode fetches from `ep_transformed`; views mode fetches from `ep`.

    Returns ``{"configs_summary": configs_df, "data": {config_name: payload}}``, where each
    payload holds ``model_ids`` and ``nodes``/``edges`` dicts of table name to DataFrame.
    """
    configs_dir = Path(configs_dir)

    if engine is None:
        from beg.db import get_engine

        engine = get_engine()

    if verbose:
        print("="*80)
        print("Graph DataFrame builder (tables or views)")
        print("="*80)
        print(f"Configs:         {config_names}")
        print(f"Max graphs:      {max_graphs or 'all'}")
        print(f"Rebuild:         {rebuild_transforms}")
        print(f"Mode:            {transform_mode}")
        print(f"Schema:          {schema}")
        print("="*80)

    if verbose:
        print("\n[1/2] Loading configurations and executing SQL transforms...")

    # Resolve target schema for fetch
    if schema is None:
        if transform_mode == "tables":
            schema = transform_schema or "ep_transformed"
        else:
            schema = transform_schema or raw_schema

    configs_df = execute_pipeline_sql(
        configs_dir=configs_dir,
        config_names=config_names,
        rebuild_transforms=rebuild_transforms,
        engine=engine,
        schema=schema,
        verbose=verbose,
        transform_mode=transform_mode,
        transform_schema=transform_schema,
        raw_schema=raw_schema,
    )

    if verbose:
        print("\n[2/2] Fetching base tables directly (no schemas/chunks)...")

    data: Dict[str, Any] = {}

    for _, cfg in configs_df.iterrows():
        cfg_name = cfg.config_name
        cfg_nodes = cfg.nodes
        cfg_edges = cfg.edges

        main_node = cfg_nodes[0]["table"] if isinstance(cfg_nodes[0], dict) else cfg_nodes[0]
        with engine.connect() as con:
            rows = con.execute(
                sql_text(
                    f'SELECT DISTINCT model_id FROM "{schema}"."{main_node}" ORDER BY model_id'
                    + (f" LIMIT {int(max_graphs)}" if max_graphs else "")
                )
            ).fetchall()
        model_ids = [r[0] for r in rows]

        if verbose:
            print(f"  • {cfg_name}: {len(model_ids)} model_ids from {schema}.{main_node}")

        nodes_out: Dict[str, pd.DataFrame] = {}
        for node in cfg_nodes:
            table = node["table"] if isinstance(node, dict) else node
            if model_ids:
                sql = f'SELECT * FROM "{schema}"."{table}" WHERE model_id = ANY(:mids)'
                df = pd.read_sql(sql_text(sql), engine, params={"mids": model_ids})
            else:
                df = pd.DataFrame()
            nodes_out[table] = df

        edges_out: Dict[str, pd.DataFrame] = {}
        for edge in cfg_edges:
            edge_src = edge.get("edge_src", "")
            if edge_src.lower() == "fk":
                continue
            table = edge_src.replace(".sql", "")
            if model_ids:
                sql = f'SELECT * FROM "{schema}"."{table}" WHERE model_id = ANY(:mids)'
                df = pd.read_sql(sql_text(sql), engine, params={"mids": model_ids})
            else:
                df = pd.DataFrame()
            edges_out[table] = df

        data[cfg_name] = {
            "model_ids": model_ids,
            "nodes": nodes_out,
            "edges": edges_out,
        }

    return {
        "configs_summary": configs_df,
        "data": data,
    }
