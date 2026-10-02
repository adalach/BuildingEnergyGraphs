"""Resolving which object a label-based reference points at.

An EnergyPlus field declared ``object_list`` names one or more *labels*, and each
label is advertised by one or more object types. A reference is unambiguous only
when the labels resolve to exactly one object type for which a table was actually
built; everything else is left to the registry pass in
:mod:`beg.ingest.registries`.

Every function takes ``label_to_objects`` and ``object_to_main_tbl`` explicitly.
"""

from __future__ import annotations

from sqlalchemy import text as sa_text

from beg.ingest.identifiers import qident


def list_existing_fk_names(engine, schema: str) -> set[str]:
    """Return the set of existing FK constraint names in the given schema."""
    sql = """
        SELECT tc.constraint_name
        FROM information_schema.table_constraints tc
        WHERE tc.table_schema = :schema
          AND tc.constraint_type = 'FOREIGN KEY'
    """
    with engine.connect() as con:
        return {row[0] for row in con.execute(sa_text(sql), {"schema": schema}).fetchall()}


def table_row_estimates(engine, schema: str) -> dict[str, int]:
    """Estimated row counts from pg_class (cheap; useful for auto-index decisions)."""
    sql = """
        SELECT c.relname, c.reltuples::bigint AS est_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relkind = 'r'
    """
    with engine.connect() as con:
        rows = con.execute(sa_text(sql), {"schema": schema}).fetchall()
    return {name: int(est) for name, est in rows}


def candidate_targets_set(labels, label_to_objects: dict, object_to_main_tbl: dict) -> set[str]:
    """Objects advertising any of ``labels``, restricted to those with a built table."""
    objs: set[str] = set()
    for lbl in labels or []:
        objs |= label_to_objects.get(lbl, set())
    return {o for o in objs if o in object_to_main_tbl}


def candidate_targets_for_labels(
    labels, label_to_objects: dict, object_to_main_tbl: dict
) -> list[str]:
    """Sorted list of candidate target object names for the given labels."""
    return sorted(candidate_targets_set(labels, label_to_objects, object_to_main_tbl))


def fully_qualified(schema_name: str, table_name: str) -> str:
    """Return a schema-qualified, quoted table name (e.g. ``"ep"."ep_construction"``)."""
    return f"{qident(schema_name)}.{qident(table_name)}"


def should_queue_index(
    table_name: str,
    create_src_indexes,
    est_row_counts: dict[str, int],
    min_rows_for_index: int,
) -> bool:
    """Decide whether to queue an index on a table's ``(model_id, source column)``.

    Decided by the table's size, not by which column: the same threshold applies
    to every foreign-key source column in a table.
    """
    if create_src_indexes is True:
        return True
    if create_src_indexes == "auto":
        return est_row_counts.get(table_name, 0) >= min_rows_for_index
    return False


