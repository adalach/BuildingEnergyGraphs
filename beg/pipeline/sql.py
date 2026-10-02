"""
SQL pipeline: runs the configurations' transforms as tables or views.

This module:
1. Loads configs
2. Builds a unified SQL transform list
3. Executes transforms into ep_transformed (tables) or ep (views)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from beg.config.loader import load_configs_from_dir
from beg.db.utils import get_relkind
from beg.pipeline.transform_runner import (
    convert_relations_to_tables,
    drop_transform_outputs,
    ensure_schema,
    ensure_transform_output_indexes,
    extract_outputs,
    prepare_transform_inputs_for_sql,
    rewrite_sql,
)
from beg.sql import TRANSFORMS_DIR


def execute_pipeline_sql(
    configs_dir: Path,
    config_names: List[str],
    rebuild_transforms: bool,
    engine: Engine,
    schema: Optional[str] = None,
    verbose: bool = True,
    transform_mode: str = "tables",
    transform_schema: Optional[str] = None,
    raw_schema: str = "ep",
) -> pd.DataFrame:
    """
    Execute the SQL pipeline for selected configs in table or view mode.
    """
    if verbose:
        print("=" * 80)
        mode_label = "tables" if transform_mode == "tables" else "views"
        print(f"SQL PIPELINE ({mode_label})")
        print("=" * 80)
        print(f"Transform schema: {transform_schema or schema}")
        print(f"Rebuild mode:     {rebuild_transforms}")
        print(f"Configs:          {config_names}")
        print("=" * 80)

    if verbose:
        print("\n[1/3] Loading configurations...")

    configs_df = load_configs_from_dir(configs_dir, config_names)
    # Resolve target schema
    if transform_mode == "tables":
        resolved_schema = transform_schema or "ep_transformed"
    else:
        resolved_schema = transform_schema or raw_schema
    schema = schema or resolved_schema
    configs_df["base_schema"] = schema

    if verbose:
        print(f"  ✓ Loaded {len(configs_df)} configs")
        for _, cfg in configs_df.iterrows():
            print(
                f"    - {cfg.config_name} ({cfg.type}): "
                f"{len(cfg.nodes)} nodes, {len(cfg.edges)} edges, "
                f"{len(cfg.sql_transforms)} transforms"
            )

    if verbose:
        print("\n[2/3] Building unified SQL transform list...")

    unified_transforms = build_unified_transform_list(
        configs_df, verbose=verbose
    )
    if verbose:
        print(f"  ✓ Unified list: {len(unified_transforms)} transforms")

    shading_was_populated = ensure_required_source_tables(
        engine=engine,
        raw_schema=raw_schema,
        sql_transforms=unified_transforms,
        verbose=verbose,
    )
    effective_rebuild = rebuild_transforms or shading_was_populated
    if shading_was_populated and verbose:
        print(
            "  ! Shading metrics were populated during this run; rebuilding all SQL "
            "transforms.\n"
            "  ! Rebuild downstream cached DataFrames, NetworkX graphs, and PyG "
            "files from these refreshed transform outputs."
        )

    if verbose:
        print("\n[3/3] Executing SQL transforms...")

    transform_results = execute_sql_transforms(
        engine=engine,
        raw_schema=raw_schema,
        target_schema=schema,
        sql_transforms=unified_transforms,
        rebuild=effective_rebuild,
        verbose=verbose,
        transform_mode=transform_mode,
    )

    all_ok = all(transform_results.values())
    configs_df["sql_transforms_ok"] = all_ok
    if not all_ok:
        failed = [t for t, ok in transform_results.items() if not ok]
        error_msg = f"Failed transforms: {failed}"
        configs_df["error_message"] = error_msg
        if verbose:
            print(f"\n  ✗ Some transforms failed: {failed}")
        raise RuntimeError(error_msg)

    if verbose:
        print("\n" + "=" * 80)
        print("SQL PIPELINE COMPLETE")
        print("=" * 80)

    return configs_df


def _read_primary_order_file() -> List[str]:
    """
    Read PRIMARY_ORDER.txt (authoritative transform order).
    """
    transforms_dir = TRANSFORMS_DIR
    order_file = transforms_dir / "PRIMARY_ORDER.txt"
    if not order_file.exists():
        raise FileNotFoundError(f"PRIMARY_ORDER.txt not found at {order_file}")
    lines: List[str] = []
    with open(order_file, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#") or s.startswith("--"):
                continue
            lines.append(s)
    return lines


def build_unified_transform_list(
    configs_df: pd.DataFrame,
    verbose: bool = True,
) -> List[str]:
    """
    Build a unified transform list for selected configs using PRIMARY_ORDER.txt.
    """
    selected: Set[str] = set()
    for _, config in configs_df.iterrows():
        for t in config.sql_transforms:
            name = str(t)
            if not name.lower().endswith(".sql"):
                name = f"{name}.sql"
            name = name.replace("\\", "/")
            selected.add(name)

    primary_order = _read_primary_order_file()

    missing = [name for name in sorted(selected) if name not in primary_order]
    if missing:
        msg = (
            "Transforms referenced by configs but missing in PRIMARY_ORDER:\n"
            + "\n".join(f"  - {n}" for n in missing)
            + "\nAdd them to beg/sql/transforms/PRIMARY_ORDER.txt"
        )
        raise ValueError(msg)

    ordered = [name for name in primary_order if name in selected]

    if verbose:
        print(f"    PRIMARY_ORDER entries: {len(primary_order)} (source file)")
        print(f"    Selected transforms:   {len(selected)} (from configs)")
        print(f"    Ordered (to execute):  {len(ordered)}")

    return ordered


def _relation_exists(engine: Engine, schema: str, name: str) -> bool:
    return get_relkind(engine, schema, name) is not None


def _eligible_shading_model_count(engine: Engine, schema: str) -> int:
    with engine.connect() as con:
        return int(
            con.execute(
                text(
                    f'''SELECT COUNT(DISTINCT s.model_id)
                        FROM "{schema}"."ep_buildingsurface_detailed" s
                        WHERE s.outside_boundary_condition_outdoors = 1
                          AND s.surface_type_wall = 1
                          AND EXISTS (
                            SELECT 1
                            FROM "{schema}"."ep_shading_building_detailed" sh
                            WHERE sh.model_id = s.model_id
                          )'''
                )
            ).scalar_one()
        )


def _shading_table_stats(engine: Engine, schema: str, table_name: str):
    with engine.connect() as con:
        return con.execute(
            text(
                f'''WITH eligible AS (
                      SELECT DISTINCT s.model_id
                      FROM "{schema}"."ep_buildingsurface_detailed" s
                      WHERE s.outside_boundary_condition_outdoors = 1
                        AND s.surface_type_wall = 1
                        AND EXISTS (
                          SELECT 1
                          FROM "{schema}"."ep_shading_building_detailed" sh
                          WHERE sh.model_id = s.model_id
                        )
                    )
                    SELECT
                      (SELECT COUNT(*) FROM eligible) AS eligible_models,
                      COUNT(*) AS row_count,
                      COUNT(DISTINCT m.model_id) FILTER (
                        WHERE e.model_id IS NOT NULL
                      ) AS covered_models,
                      COUNT(*) FILTER (
                        WHERE e.model_id IS NOT NULL
                          AND (
                            COALESCE(m.shade_count, 0) > 0
                            OR COALESCE(m.ray_count, 0) > 0
                          )
                      ) AS nonzero_rows
                    FROM "{schema}"."{table_name}" m
                    LEFT JOIN eligible e ON e.model_id = m.model_id'''
            )
        ).one()


def ensure_required_source_tables(
    engine: Engine,
    raw_schema: str,
    sql_transforms: List[str],
    verbose: bool = True,
) -> bool:
    """Build mandatory derived source tables requested by selected transforms."""
    if "surface_shading_metrics.sql" not in sql_transforms:
        return False

    table_name = "surface_shading_metrics"
    relkind = get_relkind(engine, raw_schema, table_name)
    table_exists = relkind in {"r", "p"}
    row_count = 0
    covered_models = 0
    nonzero_rows = 0
    if table_exists:
        stats = _shading_table_stats(engine, raw_schema, table_name)
        eligible_models = int(stats.eligible_models)
        row_count = int(stats.row_count)
        covered_models = int(stats.covered_models)
        nonzero_rows = int(stats.nonzero_rows)
    else:
        eligible_models = _eligible_shading_model_count(engine, raw_schema)

    coverage_complete = table_exists and covered_models >= eligible_models
    if coverage_complete and (eligible_models == 0 or nonzero_rows > 0):
        if verbose:
            print(
                f"  ✓ Required shading source exists: {raw_schema}.{table_name} "
                f"({row_count} rows, {covered_models}/{eligible_models} eligible models)"
            )
        return False

    if coverage_complete and eligible_models > 0 and nonzero_rows == 0:
        raise RuntimeError(
            f"{raw_schema}.{table_name} covers {eligible_models} shade-bearing models "
            "but every shading metric is zero. Stopped before building graphs."
        )

    print(
        f"\n  ! Required shading source {raw_schema}.{table_name} is "
        f"{'incomplete' if table_exists else 'missing'}.\n"
        "    Running beg.pipeline.shading now."
    )
    try:
        from beg.pipeline.shading import populate_surface_shading_metrics

        row_count = populate_surface_shading_metrics(
            engine,
            schema=raw_schema,
            replace=False,
            resume=table_exists,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Automatic population of {raw_schema}.{table_name} failed. "
            "Stopped before building graphs."
        ) from exc

    if get_relkind(engine, raw_schema, table_name) not in {"r", "p"}:
        raise RuntimeError(
            f"The shading producer completed without creating "
            f"{raw_schema}.{table_name}. Stopped before building graphs."
        )

    stats = _shading_table_stats(engine, raw_schema, table_name)
    eligible_models = int(stats.eligible_models)
    row_count = int(stats.row_count)
    covered_models = int(stats.covered_models)
    nonzero_rows = int(stats.nonzero_rows)
    if covered_models < eligible_models:
        raise RuntimeError(
            f"The shading producer covered only {covered_models} of {eligible_models} "
            f"eligible models in {raw_schema}.{table_name}. Stopped before "
            "building graphs."
        )
    if eligible_models > 0 and (row_count <= 0 or nonzero_rows <= 0):
        raise RuntimeError(
            f"The shading producer covered {eligible_models} eligible models but "
            f"{raw_schema}.{table_name} contains no nonzero shading signal. Stopped "
            "before building graphs: an all-zero shading dataset is not accepted."
        )

    print(
        f"  ✓ Created {raw_schema}.{table_name}: {row_count} rows for "
        f"{covered_models}/{eligible_models} eligible models; "
        f"{nonzero_rows} rows contain shading signal."
    )
    return True


def execute_sql_transforms(
    engine: Engine,
    raw_schema: str,
    target_schema: str,
    sql_transforms: List[str],
    rebuild: bool,
    verbose: bool = True,
    transform_mode: str = "tables",
) -> Dict[str, bool]:
    """
    Execute transforms into target_schema, as tables or views.

    Without ``rebuild``, a transform whose outputs all exist is skipped even if its
    SQL has changed (see ``docs/limitations.md``). Returns ``{sql_file: succeeded}``.
    """
    results: Dict[str, bool] = {}
    transforms_dir = TRANSFORMS_DIR

    if transform_mode == "tables":
        ensure_schema(engine, target_schema, rebuild=False)

    if verbose:
        total = len(sql_transforms)
        print(f"    Total: {total} transforms")
        print("    Mode: REBUILD ALL" if rebuild else "    Mode: Skip existing, rebuild missing")

    for idx, sql_file in enumerate(sql_transforms, 1):
        sql_path = transforms_dir / sql_file
        if not sql_path.exists():
            if verbose:
                print(f"    [{idx}/{len(sql_transforms)}] ✗ Not found: {sql_file}")
            results[sql_file] = False
            continue

        sql_text = sql_path.read_text(encoding="utf-8")
        outputs = extract_outputs(sql_text, raw_schema=raw_schema)

        should_execute = rebuild
        if not rebuild and outputs:
            should_execute = not all(_relation_exists(engine, target_schema, o) for o in outputs)

        if not should_execute:
            if transform_mode == "tables":
                ensure_transform_output_indexes(
                    engine,
                    target_schema,
                    outputs,
                    verbose=verbose,
                )
            if verbose:
                print(f"    [{idx}/{len(sql_transforms)}] ↷ Skip (all outputs exist): {sql_file}")
            results[sql_file] = True
            continue

        if verbose:
            print(f"    [{idx}/{len(sql_transforms)}] ▶ Running: {sql_file}")

        try:
            if transform_mode == "tables":
                prepare_transform_inputs_for_sql(
                    engine=engine,
                    sql_text=sql_text,
                    raw_schema=raw_schema,
                    target_schema=target_schema,
                    verbose=verbose,
                )

            drop_transform_outputs(engine, target_schema, outputs)

            force_views = "pg_get_viewdef" in sql_text.lower()
            sql_exec = rewrite_sql(
                sql_text,
                raw_schema=raw_schema,
                target_schema=target_schema,
                transform_mode=transform_mode,
                force_views=force_views,
            )

            raw = engine.raw_connection()
            try:
                cur = raw.cursor()
                cur.execute(sql_exec)
                raw.commit()
                cur.close()
            finally:
                raw.close()

            # Ensure outputs are tables (table mode only)
            if transform_mode == "tables":
                # All of this transform's outputs together: converting them one
                # by one drops dependent views by cascade. See
                # transform_runner.convert_relations_to_tables.
                convert_relations_to_tables(engine, target_schema, outputs)
                ensure_transform_output_indexes(
                    engine,
                    target_schema,
                    outputs,
                    verbose=verbose,
                )

            if verbose:
                print(f"    [{idx}/{len(sql_transforms)}] ✓ Success: {sql_file}")
            results[sql_file] = True
        except Exception as e:
            if verbose:
                print(f"    [{idx}/{len(sql_transforms)}] ✗ Error: {sql_file}")
                print(f"        {e}")
            results[sql_file] = False

    return results
