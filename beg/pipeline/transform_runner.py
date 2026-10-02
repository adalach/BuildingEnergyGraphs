"""
Helpers for executing one SQL transform, used by :mod:`beg.pipeline.sql`.

Output discovery, shim input views, schema rewriting and view-to-table conversion.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, Iterable, List, Set

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine

from beg.db.utils import (
    create_schema,
    drop_schema,
    ensure_standard_relation_indexes,
    get_relkind,
)


def ensure_schema(engine: Engine, schema: str, rebuild: bool = False) -> None:
    """
    Ensure target schema exists.
    If `rebuild=True`, drop the schema first.
    """
    if rebuild:
        drop_schema(engine, schema, cascade=True, verbose=False)
    create_schema(engine, schema, verbose=False)


def _unique_preserve(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_schema_refs(sql_source: str, schema: str) -> Set[str]:
    """
    Extract relation names referenced as `schema.relation` or `"schema"."relation"`.
    """
    refs: Set[str] = set()
    schema_re = re.escape(schema)
    pattern = re.compile(
        rf'(?is)(?:"{schema_re}"|{schema_re})\s*\.\s*(?:"([A-Za-z_][A-Za-z0-9_]*)"|([A-Za-z_][A-Za-z0-9_]*))'
    )
    for m in pattern.finditer(sql_source):
        name = (m.group(1) or m.group(2) or "").strip()
        if not name:
            continue
        # Skip placeholders from dynamic SQL fragments.
        if "%" in name:
            continue
        refs.add(name)
    return refs


def _extract_assigned_vars(sql_source: str) -> Dict[str, str]:
    """
    Extract simple PL/pgSQL assignments like:
      out_view := 'comp_element';
    """
    assigned: Dict[str, str] = {}
    for m in re.finditer(r"(?im)\b([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*'([A-Za-z_][A-Za-z0-9_]*)'\s*;", sql_source):
        assigned[m.group(1)] = m.group(2)
    return assigned


def extract_outputs(sql_source: str, raw_schema: str = "ep") -> List[str]:
    """
    Infer transform output relation names.

    Strategy:
    - Parse explicit CREATE statements in the raw schema.
    - Parse explicit DROP ... IF EXISTS in the raw schema.
    - Fallback to variables like `out_view := '...'` used in dynamic SQL.
    """
    outputs: List[str] = []
    schema_re = re.escape(raw_schema)

    create_pat = re.compile(
        rf'(?is)\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+VIEW|VIEW|TABLE)\s+'
        rf'(?:"{schema_re}"|{schema_re})\s*\.\s*(?:"([A-Za-z_][A-Za-z0-9_]*)"|([A-Za-z_][A-Za-z0-9_]*))'
    )
    drop_pat = re.compile(
        rf'(?is)\bDROP\s+(?:MATERIALIZED\s+VIEW|VIEW|TABLE)\s+IF\s+EXISTS\s+'
        rf'(?:"{schema_re}"|{schema_re})\s*\.\s*(?:"([A-Za-z_][A-Za-z0-9_]*)"|([A-Za-z_][A-Za-z0-9_]*))'
    )

    for m in create_pat.finditer(sql_source):
        outputs.append((m.group(1) or m.group(2)).strip())
    for m in drop_pat.finditer(sql_source):
        outputs.append((m.group(1) or m.group(2)).strip())

    # Dynamic SQL fallback (used by e.g. comp_element / buildingsurface_features).
    assigned = _extract_assigned_vars(sql_source)
    for var_name, relation in assigned.items():
        if re.search(r"(?i)out_(view|table)$", var_name) or var_name.lower() == "out":
            outputs.append(relation)

    return _unique_preserve(outputs)


def prepare_transform_inputs_for_sql(
    engine: Engine,
    sql_text: str | None = None,
    raw_schema: str = "ep",
    target_schema: str = "ep_transformed",
    verbose: bool = True,
    sql_source: str | None = None,
) -> None:
    """
    Prepare shim inputs in the target schema.

    For each relation referenced as `raw_schema.relation`:
    - If it already exists in target schema, use it.
    - Else if it exists in raw schema, create shim view:
        target_schema.relation -> SELECT * FROM raw_schema.relation
    """
    # beg.pipeline.sql passes `sql_text`; `sql_source` is accepted as an alias.
    source = sql_text if sql_text is not None else sql_source
    if source is None:
        raise TypeError("prepare_transform_inputs_for_sql requires `sql_text` (or `sql_source`).")

    refs = _extract_schema_refs(source, schema=raw_schema)
    outputs = set(extract_outputs(source, raw_schema=raw_schema))

    created: List[str] = []
    for rel in sorted(refs):
        if rel in outputs:
            continue

        if get_relkind(engine, target_schema, rel) is not None:
            continue

        if get_relkind(engine, raw_schema, rel) is None:
            continue

        with engine.begin() as con:
            con.execute(
                sa_text(
                    f'CREATE VIEW "{target_schema}"."{rel}" AS '
                    f'SELECT * FROM "{raw_schema}"."{rel}"'
                )
            )
        created.append(rel)

    if verbose and created:
        preview = ", ".join(created[:8])
        suffix = " ..." if len(created) > 8 else ""
        print(
            f"      + prepared {len(created)} shim input(s) in "
            f'"{target_schema}": {preview}{suffix}'
        )


def drop_transform_outputs(engine: Engine, schema: str, outputs: Iterable[str]) -> None:
    """
    Drop output relations from target schema regardless of relation kind.
    """
    for out in _unique_preserve(outputs):
        relkind = get_relkind(engine, schema, out)
        if relkind is None:
            continue

        if relkind in ("r", "p"):
            stmt = f'DROP TABLE IF EXISTS "{schema}"."{out}" CASCADE'
        elif relkind == "m":
            stmt = f'DROP MATERIALIZED VIEW IF EXISTS "{schema}"."{out}" CASCADE'
        else:
            stmt = f'DROP VIEW IF EXISTS "{schema}"."{out}" CASCADE'

        with engine.begin() as con:
            con.execute(sa_text(stmt))


def rewrite_sql(
    sql_source: str,
    raw_schema: str = "ep",
    target_schema: str = "ep_transformed",
    transform_mode: str = "tables",
    force_views: bool = False,
) -> str:
    """
    Rewrite SQL schema references from `raw_schema` to `target_schema`.

    Notes:
    - In view mode with same schema, SQL is returned unchanged.
    - `force_views` is ignored: CREATE statements are kept as authored and
      relation-kind conversion is left to `convert_relations_to_tables`.
    """
    _ = force_views  # accepted and ignored

    if transform_mode == "views" and raw_schema == target_schema:
        return sql_source

    old = re.escape(raw_schema)
    out = sql_source

    # Schema-qualified object references: ep.foo / "ep"."foo"
    out = re.sub(rf'(?i)"{old}"\s*\.\s*"', f'"{target_schema}"."', out)
    out = re.sub(rf'(?i)\b{old}\s*\.\s*', f"{target_schema}.", out)

    # Information schema checks in PL/pgSQL often use explicit schema strings.
    for key in ("table_schema", "schema_name", "nspname"):
        out = re.sub(
            rf"(?i)\b{key}\b\s*=\s*'{old}'",
            f"{key}='{target_schema}'",
            out,
        )

    return out


def convert_relation_to_table(engine: Engine, schema: str, relation: str) -> None:
    """Convert one view or materialized view into a table of the same name.

    Safe only when nothing else in the same conversion batch is built on this
    relation — the drop cascades. Use :func:`convert_relations_to_tables` for a
    transform's full output list.
    """
    convert_relations_to_tables(engine, schema, [relation])


def convert_relations_to_tables(engine: Engine, schema: str, relations: Iterable[str]) -> None:
    """Convert every named view or materialized view into a table, order-independently.

    All of them are copied into temporary tables first, then the originals are
    dropped, then the copies renamed. A transform may build ``X_cache`` as a
    materialized view and ``X`` as a view selecting from it; dropping ``X_cache``
    cascades to ``X``, so converting them one at a time can lose ``X``.
    """
    targets: list[tuple[str, str, str]] = []  # (relation, relkind, tmp_name)
    for relation in _unique_preserve(list(relations)):
        relkind = get_relkind(engine, schema, relation)
        if relkind in ("v", "m"):
            targets.append((relation, relkind, f"__tmp_{relation}_{uuid.uuid4().hex[:8]}"))

    if not targets:
        return

    with engine.begin() as con:
        for relation, _relkind, tmp_name in targets:
            con.execute(
                sa_text(
                    f'CREATE TABLE "{schema}"."{tmp_name}" AS '
                    f'SELECT * FROM "{schema}"."{relation}"'
                )
            )

        for relation, relkind, _tmp_name in targets:
            keyword = "MATERIALIZED VIEW" if relkind == "m" else "VIEW"
            con.execute(sa_text(f'DROP {keyword} IF EXISTS "{schema}"."{relation}" CASCADE'))

        for relation, _relkind, tmp_name in targets:
            con.execute(sa_text(f'ALTER TABLE "{schema}"."{tmp_name}" RENAME TO "{relation}"'))


def ensure_transform_output_indexes(
    engine: Engine,
    schema: str,
    outputs: Iterable[str],
    *,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Ensure standard lookup indexes on transform outputs that are materialized as tables.
    """
    created: Dict[str, List[str]] = {}
    for out in _unique_preserve(outputs):
        created_indexes = ensure_standard_relation_indexes(
            engine,
            schema,
            out,
            verbose=verbose,
        )
        if created_indexes:
            created[out] = created_indexes
    return created


__all__ = [
    "ensure_schema",
    "prepare_transform_inputs_for_sql",
    "drop_transform_outputs",
    "rewrite_sql",
    "extract_outputs",
    "convert_relation_to_table",
    "convert_relations_to_tables",
    "ensure_transform_output_indexes",
]
