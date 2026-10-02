"""
Database utility functions.

Centralized database operations for the beg pipeline.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine


def quote_identifier(schema: str, table: str) -> str:
    """
    Quote a schema.table identifier for SQL.

    Args:
        schema: Schema name
        table: Table/view name

    Returns:
        Quoted identifier string (e.g., '"ep"."zones_zone"')
    """
    return f'"{schema}"."{table}"'


def get_relkind(engine: Engine, schema: str, name: str) -> Optional[str]:
    """
    Return PostgreSQL relkind for the object if it exists:
      'v' = view, 'm' = materialized view, 'r' = table, 'p' = partitioned table.
    Returns None if object does not exist.
    """
    sql = """
        SELECT c.relkind
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname = :name
        LIMIT 1
    """
    try:
        with engine.connect() as con:
            rk = con.execute(sql_text(sql), {"schema": schema, "name": name}).scalar()
            return rk if rk is not None else None
    except Exception:
        return None


def relation_has_column(engine: Engine, schema: str, table: str, column: str) -> bool:
    """
    Return True when a relation exposes the requested column.
    """
    sql = """
        SELECT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema
              AND c.relname = :table
              AND c.relkind IN ('r','v','m','p')
              AND a.attnum > 0
              AND NOT a.attisdropped
              AND a.attname = :column
        )
    """
    with engine.connect() as con:
        return bool(
            con.execute(
                sql_text(sql),
                {"schema": schema, "table": table, "column": column},
            ).scalar()
        )


def index_exists(engine: Engine, schema: str, index_name: str) -> bool:
    """
    Return True when an index already exists in the schema.
    """
    sql = """
        SELECT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema
              AND c.relname = :index_name
              AND c.relkind = 'i'
        )
    """
    with engine.connect() as con:
        return bool(
            con.execute(
                sql_text(sql),
                {"schema": schema, "index_name": index_name},
            ).scalar()
        )


def _safe_index_name(relation: str, suffix: str, max_len: int = 63) -> str:
    """
    Build a deterministic PostgreSQL-safe index name.
    """
    raw = f"idx_{relation}_{suffix}"
    if len(raw) <= max_len:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    head_len = max_len - len(digest) - 1
    return f"{raw[:head_len]}_{digest}"


def ensure_btree_index(
    engine: Engine,
    schema: str,
    relation: str,
    columns: Sequence[str],
    *,
    index_name: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """
    Ensure a btree index exists on the requested table columns.

    Returns True when a new index was created in this call.
    Returns False when the table is missing, not a table, columns are missing,
    or the named index already exists.
    """
    if not columns:
        return False
    if get_relkind(engine, schema, relation) not in ("r", "p"):
        return False
    for column in columns:
        if not relation_has_column(engine, schema, relation, str(column)):
            return False

    suffix = "_".join(str(col) for col in columns)
    idx_name = index_name or _safe_index_name(relation, suffix)
    if index_exists(engine, schema, idx_name):
        return False

    quoted_cols = ", ".join(f'"{str(col)}"' for col in columns)
    sql = (
        f'CREATE INDEX IF NOT EXISTS "{idx_name}" '
        f'ON "{schema}"."{relation}" ({quoted_cols})'
    )
    with engine.begin() as con:
        con.execute(sql_text(sql))

    if verbose:
        print(f'      + index: "{schema}"."{idx_name}" on {relation}({", ".join(columns)})')
    return True


def ensure_standard_relation_indexes(
    engine: Engine,
    schema: str,
    relation: str,
    *,
    verbose: bool = False,
) -> list[str]:
    """
    Ensure the standard lookup indexes on a transform output table.

    Standard policy:
    - add `(model_id)` when `model_id` exists
    - add `(model_id, instance_name)` when both columns exist

    Runs ANALYZE when an index was created. Returns the names of the indexes created.
    """
    if get_relkind(engine, schema, relation) not in ("r", "p"):
        return []

    created: list[str] = []
    cols = set(get_all_columns(engine, schema, relation))
    if "model_id" in cols:
        idx_name = _safe_index_name(relation, "model_id")
        if ensure_btree_index(
            engine,
            schema,
            relation,
            ["model_id"],
            index_name=idx_name,
            verbose=verbose,
        ):
            created.append(idx_name)

    if {"model_id", "instance_name"}.issubset(cols):
        idx_name = _safe_index_name(relation, "model_id_instance_name")
        if ensure_btree_index(
            engine,
            schema,
            relation,
            ["model_id", "instance_name"],
            index_name=idx_name,
            verbose=verbose,
        ):
            created.append(idx_name)

    if created:
        with engine.begin() as con:
            con.execute(sql_text(f'ANALYZE "{schema}"."{relation}"'))
        if verbose:
            print(f'      + analyzed "{schema}"."{relation}" after index creation')

    return created


def get_all_columns(engine: Engine, schema: str, table: str) -> List[str]:
    """
    Get all column names for a relation (table, view, materialized view, partition).

    Uses pg_catalog to ensure materialized views are supported.

    Args:
        engine: SQLAlchemy engine
        schema: Schema name
        table: Table/view name

    Returns:
        List of column names in physical order
    """
    sql = """
        SELECT a.attname
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema
          AND c.relname = :table
          AND c.relkind IN ('r','v','m','p')
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
    """
    with engine.connect() as con:
        result = con.execute(sql_text(sql), {'schema': schema, 'table': table})
        return [row[0] for row in result]


def schema_exists(engine: Engine, schema: str) -> bool:
    """
    Check if a schema exists.

    Args:
        engine: SQLAlchemy engine
        schema: Schema name

    Returns:
        True if exists, False otherwise
    """
    sql = """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.schemata
            WHERE schema_name = :schema
        )
    """
    with engine.connect() as con:
        result = con.execute(sql_text(sql), {'schema': schema})
        return result.scalar()


def create_schema(engine: Engine, schema: str, verbose: bool = True) -> None:
    """
    Create a schema if it doesn't exist.

    Args:
        engine: SQLAlchemy engine
        schema: Schema name
        verbose: Print status
    """
    if not schema_exists(engine, schema):
        sql = f'CREATE SCHEMA "{schema}"'
        with engine.begin() as con:
            con.execute(sql_text(sql))
        if verbose:
            print(f"  Created schema: {schema}")
    else:
        if verbose:
            print(f"  Schema exists: {schema}")


def drop_schema(engine: Engine, schema: str, cascade: bool = True, verbose: bool = True) -> None:
    """
    Drop a schema.

    Args:
        engine: SQLAlchemy engine
        schema: Schema name
        cascade: Drop all objects in schema
        verbose: Print status
    """
    if schema_exists(engine, schema):
        cascade_clause = "CASCADE" if cascade else ""
        sql = f'DROP SCHEMA "{schema}" {cascade_clause}'
        with engine.begin() as con:
            con.execute(sql_text(sql))
        if verbose:
            print(f"  Dropped schema: {schema}")


