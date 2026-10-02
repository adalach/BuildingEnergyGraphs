"""Registry tables for references whose label matches more than one object type.

When a field's labels resolve to several possible target objects, no single
foreign key can express the reference. Instead one *registry head* table is
created per ambiguous label, holding every ``(model_id, instance_name)`` any
candidate object declares, together with the object and table finally preferred
for that name. Ambiguous fields then key against the registry head.

The preference rule is *first hit in alphabetical order of object name*. It is
arbitrary but deterministic: the same dataset always resolves the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import Column, ForeignKey, Integer, String, Table, UniqueConstraint
from sqlalchemy import text as sa_text

from beg.ingest.foreign_keys_advanced import (
    candidate_targets_for_labels,
    fully_qualified,
    list_existing_fk_names,
)
from beg.ingest.identifiers import qident, safe_constraint_name, safe_table_name, sanitize
from beg.ingest.tables import SchemaBuild


@dataclass
class RegistryResult:
    """Which labels were ambiguous, how big each registry is, and the FKs attached."""

    labels: dict = field(default_factory=dict)
    rows: list = field(repr=False, default_factory=list)
    added_fks: list = field(repr=False, default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(r["rows"] for r in self.rows)

    @property
    def total_filled(self) -> int:
        return sum(r["preferred_filled"] for r in self.rows)

    def summary(self) -> pd.DataFrame:
        """One row per ambiguous label: candidates, registry size, fill completeness."""
        if not self.rows:
            return pd.DataFrame(
                columns=["label", "candidate_objects", "rows", "preferred_filled"]
            )
        return pd.DataFrame(self.rows).sort_values("label").reset_index(drop=True)

    def fks_frame(self) -> pd.DataFrame:
        """One row per foreign key attached to a registry head."""
        if not self.added_fks:
            return pd.DataFrame(columns=["src_obj", "src_table", "src_col", "label", "constraint"])
        return pd.DataFrame(self.added_fks)


def label_to_head_name(label: str) -> str:
    """``'MaterialName'`` -> ``'ep_edge_materialname'`` (sanitized and length-safe)."""
    return safe_table_name(f"ep_edge_{sanitize(label)}")


def _meta_table_lookup(build: SchemaBuild, table_name: str):
    tbl = build.meta.tables.get(f"{build.schema}.{table_name}")
    if tbl is not None:
        return tbl
    return build.meta.tables.get(table_name)


def ensure_registry_head_model_index(engine, build: SchemaBuild, head: str) -> None:
    """Ensure the head table has a plain ``model_id`` index, even after an interrupted run."""
    ix_name = safe_constraint_name("ix", head, "model_id")
    fq_head = fully_qualified(build.schema, head)
    with engine.begin() as con:
        con.execute(
            sa_text(
                f"CREATE INDEX IF NOT EXISTS {qident(ix_name)} ON {fq_head} (model_id)"
            )
        )


def ensure_registry_head(engine, build: SchemaBuild, label: str) -> Table:
    """Create, or fetch, the head table for one label's registry."""
    head = label_to_head_name(label)

    # An interrupted run can leave a stale in-memory Table definition behind,
    # sometimes with an auto-created Index attached. Rebuild it cleanly.
    existing = _meta_table_lookup(build, head)
    if existing is not None:
        build.meta.remove(existing)

    ep_head = Table(
        head,
        build.meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("model_id", String, ForeignKey("ep_model.model_id"), nullable=False),
        Column("instance_name", String, nullable=False),
        Column("preferred_object", String, nullable=False),
        Column("preferred_table", String, nullable=False),
        UniqueConstraint(
            "model_id", "instance_name", name=safe_constraint_name("uq", head, "name")
        ),
        extend_existing=True,
        schema=build.schema,
    )

    build.meta.create_all(engine, tables=[ep_head])
    ensure_registry_head_model_index(engine, build, head)
    return ep_head


def ambiguous_labels(skipped_fks, build: SchemaBuild, label_to_objects: dict) -> dict:
    """Labels that appeared in an ``ambiguous`` skip, with their candidate objects."""
    out: dict[str, set[str]] = {}
    for cand, reason in skipped_fks:
        if not (isinstance(reason, str) and reason.startswith("ambiguous")):
            continue
        for lbl in cand.get("labels") or []:
            objs = candidate_targets_for_labels(
                [lbl], label_to_objects, build.object_to_main_tbl
            )
            if len(objs) > 1:
                out.setdefault(lbl, set()).update(objs)
    return {lbl: sorted(v) for lbl, v in out.items()}


def populate_registry_head(engine, build: SchemaBuild, label: str, candidate_objs: list) -> None:
    """Insert every distinct ``(model_id, instance_name)`` from each candidate table.

    ``preferred_*`` are filled with empty strings to satisfy NOT NULL and then
    resolved by :func:`fill_registry_preferred_first_hit`.
    """
    head = ensure_registry_head(engine, build, label).name
    fq_head = fully_qualified(build.schema, head)
    with engine.begin() as con:
        for obj in candidate_objs:
            tgt_tbl = build.object_to_main_tbl.get(obj)
            if tgt_tbl is None:
                continue
            con.execute(
                sa_text(
                    f"""
                INSERT INTO {fq_head} (model_id, instance_name, preferred_object, preferred_table)
                SELECT DISTINCT t.model_id, t.instance_name, ''::text, ''::text
                FROM {fully_qualified(build.schema, tgt_tbl.name)} AS t
                ON CONFLICT (model_id, instance_name) DO NOTHING
            """
                )
            )


def fill_registry_preferred_first_hit(
    engine, build: SchemaBuild, label: str, candidate_objs: list
) -> None:
    """Resolve ``preferred_*`` to the first candidate object holding that name.

    "First" is alphabetical by object name. One pass, so no row is left unfilled
    as long as it came from some candidate table.
    """
    head = label_to_head_name(label)

    unions = []
    params: dict = {}
    for i, obj in enumerate(sorted(candidate_objs)):
        tgt_tbl = build.object_to_main_tbl.get(obj)
        if tgt_tbl is None:
            continue
        unions.append(
            f"SELECT model_id, instance_name, :obj{i} AS object_type, "
            f":tbl{i} AS table_name, {i} AS ord "
            f"FROM {fully_qualified(build.schema, tgt_tbl.name)}"
        )
        params[f"obj{i}"] = obj
        params[f"tbl{i}"] = tgt_tbl.name

    if not unions:
        return

    fq_head = fully_qualified(build.schema, head)
    union_sql = " \nUNION ALL\n".join(unions)
    sql = f"""
        WITH candidates AS (
            {union_sql}
        ),
        hits AS (
            SELECT h.id, c.object_type, c.table_name, c.ord
            FROM {fq_head} AS h
            JOIN candidates c
              ON c.model_id = h.model_id AND c.instance_name = h.instance_name
        ),
        first_hit AS (
            SELECT DISTINCT ON (id) id, object_type, table_name
            FROM hits
            ORDER BY id, ord
        )
        UPDATE {fq_head} AS h
        SET preferred_object = f.object_type,
            preferred_table  = f.table_name
        FROM first_hit f
        WHERE f.id = h.id
    """
    with engine.begin() as con:
        con.execute(sa_text(sql), params)


def add_fk_to_registry(engine, build: SchemaBuild, src_tbl: Table, src_col: str, label: str) -> str:
    """Attach a foreign key from ``(model_id, src_col)`` to the label's registry head."""
    head = label_to_head_name(label)
    fk_name = safe_constraint_name("fk", src_tbl.name, src_col, head)
    if fk_name in list_existing_fk_names(engine, build.schema):
        return fk_name

    sql = f"""
        ALTER TABLE {fully_qualified(build.schema, src_tbl.name)}
        ADD CONSTRAINT {qident(fk_name)}
        FOREIGN KEY (model_id, {qident(src_col)})
        REFERENCES {fully_qualified(build.schema, head)} (model_id, instance_name)
        DEFERRABLE INITIALLY DEFERRED NOT VALID
    """
    with engine.begin() as con:
        con.execute(sa_text(sql))
    return fk_name


def registry_preview(engine, build: SchemaBuild, labels: dict, per_label: int = 3) -> pd.DataFrame:
    """A few resolved rows per registry, for eyeballing the preference rule."""
    rows = []
    for lbl in sorted(labels):
        fq = fully_qualified(build.schema, label_to_head_name(lbl))
        with engine.connect() as con:
            sample = con.execute(
                sa_text(
                    f"""
                SELECT model_id, instance_name, preferred_object, preferred_table
                FROM {fq} ORDER BY model_id, instance_name LIMIT {int(per_label)}
            """
                )
            ).fetchall()
        for r in sample:
            rows.append({"label": lbl, **dict(r._mapping)})
    return pd.DataFrame(
        rows,
        columns=["label", "model_id", "instance_name", "preferred_object", "preferred_table"],
    )


def build_registries(engine, build: SchemaBuild, skipped_fks, label_to_objects: dict) -> RegistryResult:
    """Create, populate and wire up a registry for every ambiguous label.

    Runs the whole pass: detect ambiguous labels, build and fill each head table,
    measure it, then attach foreign keys from the fields that were skipped.
    """
    labels = ambiguous_labels(skipped_fks, build, label_to_objects)
    result = RegistryResult(labels=labels)

    for lbl, objs in labels.items():
        populate_registry_head(engine, build, lbl, objs)
    for lbl, objs in labels.items():
        fill_registry_preferred_first_hit(engine, build, lbl, objs)

    for lbl, objs in sorted(labels.items()):
        fq = fully_qualified(build.schema, label_to_head_name(lbl))
        with engine.connect() as con:
            total = con.execute(sa_text(f"SELECT COUNT(*) FROM {fq}")).scalar() or 0
            filled = (
                con.execute(
                    sa_text(
                        f"SELECT COUNT(*) FROM {fq} "
                        "WHERE preferred_object <> '' AND preferred_table <> ''"
                    )
                ).scalar()
                or 0
            )
        result.rows.append(
            {
                "label": lbl,
                "candidate_objects": len(objs),
                "rows": int(total),
                "preferred_filled": int(filled),
            }
        )

    for cand, reason in skipped_fks:
        if not (isinstance(reason, str) and reason.startswith("ambiguous")):
            continue
        src_tbl = cand.get("src_tbl")
        src_col = cand.get("src_col")
        cand_labels = cand.get("labels") or []
        if src_tbl is None or src_col is None or not cand_labels:
            continue

        lbl = next((label for label in cand_labels if label in labels), None)
        if lbl is None:
            continue

        fk_name = add_fk_to_registry(engine, build, src_tbl, src_col, lbl)
        result.added_fks.append(
            {
                "src_obj": cand.get("src_obj"),
                "src_table": src_tbl.name,
                "src_col": src_col,
                "label": lbl,
                "constraint": fk_name,
            }
        )

    return result
