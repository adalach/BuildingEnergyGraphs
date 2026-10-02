"""Label-based foreign keys for unambiguous object references.

A foreign key is added only where the field's labels resolve to exactly one
target object type. Ambiguous references — a label advertised by several object
types — are left to :mod:`beg.ingest.registries`, which resolves them against
the data rather than the schema.

Constraints are created ``DEFERRABLE INITIALLY DEFERRED NOT VALID``: the model
set is ingested object by object, so a reference is routinely written before its
target row exists, and ``NOT VALID`` skips scanning the rows already loaded.
Constraints that already exist are skipped, so a re-run is idempotent.
"""

from __future__ import annotations

from sqlalchemy import text as sa_text

from beg.ingest.foreign_keys_advanced import (
    candidate_targets_set,
    fully_qualified,
    list_existing_fk_names,
    should_queue_index,
    table_row_estimates,
)
from beg.ingest.identifiers import qident, safe_constraint_name


def add_label_based_fks(
    engine,
    fk_candidates,
    schema: str,
    object_to_main_tbl: dict,
    label_to_objects: dict,
    create_src_indexes=False,
    min_rows_for_index: int = 50_000,
):
    """Add foreign keys for references whose labels identify exactly one target.

    Each candidate is a dict with ``src_tbl`` (a SQLAlchemy Table), ``src_col``,
    ``labels`` and optionally ``src_obj`` for reporting.

    ``create_src_indexes`` is ``False`` (skip), ``True`` (always) or ``"auto"``,
    which indexes only tables estimated at ``min_rows_for_index`` rows or more.

    Returns ``(added, skipped)``: a list of dicts describing each constraint
    created, and a list of ``(candidate, reason)`` pairs.
    """
    existing_fk_names = list_existing_fk_names(engine, schema)
    est = table_row_estimates(engine, schema) if create_src_indexes == "auto" else {}

    added: list[dict] = []
    skipped: list[tuple] = []
    index_requests: set[tuple[str, str]] = set()

    with engine.begin() as con:
        for cand in fk_candidates:
            src_tbl = cand.get("src_tbl")
            src_col = cand.get("src_col")
            labels = cand.get("labels") or []
            src_obj = cand.get("src_obj")

            if src_tbl is None or src_col is None or not labels:
                skipped.append((cand, "incomplete_candidate"))
                continue

            obj_candidates = candidate_targets_set(labels, label_to_objects, object_to_main_tbl)

            if not obj_candidates:
                skipped.append((cand, "no_target_object"))
                continue
            if len(obj_candidates) > 1:
                skipped.append((cand, f"ambiguous:{sorted(obj_candidates)}"))
                continue

            # Exactly one target -> create FK
            tgt_obj = next(iter(obj_candidates))
            tgt_tbl = object_to_main_tbl[tgt_obj]

            fk_name = safe_constraint_name("fk", src_tbl.name, src_col, tgt_tbl.name)
            if fk_name in existing_fk_names:
                skipped.append((cand, "exists"))
                continue

            fq_src = fully_qualified(schema, src_tbl.name)
            fq_tgt = fully_qualified(schema, tgt_tbl.name)
            con.execute(
                sa_text(
                    f"""
                ALTER TABLE {fq_src}
                ADD CONSTRAINT {qident(fk_name)}
                FOREIGN KEY (model_id, {qident(src_col)})
                REFERENCES {fq_tgt} (model_id, instance_name)
                DEFERRABLE INITIALLY DEFERRED NOT VALID
            """
                )
            )
            existing_fk_names.add(fk_name)

            added.append(
                {
                    "src_obj": src_obj,
                    "src_tbl": src_tbl.name,
                    "src_col": src_col,
                    "tgt_obj": tgt_obj,
                    "tgt_tbl": tgt_tbl.name,
                }
            )

            if should_queue_index(
                src_tbl.name, create_src_indexes, est, min_rows_for_index
            ):
                index_requests.add((src_tbl.name, src_col))

    # Optional helper indexes
    if index_requests:
        with engine.begin() as con:
            for tbl_name, col_name in sorted(index_requests):
                ix_name = safe_constraint_name("ix", tbl_name, f"model_id_{col_name}")
                fq_tbl = fully_qualified(schema, tbl_name)
                con.execute(
                    sa_text(
                        f"""
                    CREATE INDEX IF NOT EXISTS {qident(ix_name)}
                    ON {fq_tbl} (model_id, {qident(col_name)});
                """
                    )
                )

    return added, skipped
