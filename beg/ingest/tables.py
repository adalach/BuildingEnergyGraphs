"""Turning parsed schema metadata into the ``ep`` table set.

One *main* table per EnergyPlus object type, plus one *child* table per
array-of-objects field that the model set actually populates. Scalar fields
become columns on the main table, carrying the schema's type, default, enum and
range constraints.

Everything the later stages need is collected in :class:`SchemaBuild`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    inspect,
)
from sqlalchemy import text as sa_text

from beg.ingest.ep_schema import EnergyPlusSchema
from beg.ingest.identifiers import (
    safe_column_name,
    safe_constraint_name,
    safe_table_name,
    shorten_identifier,
)
from beg.ingest.models import ep_model_table
from beg.ingest.schema_map import (
    build_check_enum,
    build_check_minmax,
    classify_fields,
    get,
    sa_type_from_schema,
    server_default_to_str,
    set_server_default,
)


@dataclass
class SchemaBuild:
    """The in-memory table set, and the maps the ingestion and FK stages need.

    ``table_map`` maps an EnergyPlus object name to ``(main_table, {field: child_table})``.
    ``scalar_col_map`` maps ``(object, original_field)`` to its sanitized column.
    ``object_child_colmap`` maps ``(object, array_field)`` to ``{child_prop: column}``.
    ``fk_candidates`` accumulates every ``object_list`` field for the FK stage.
    """

    meta: MetaData
    schema: str
    ep_model: Table
    table_map: dict = field(repr=False, default_factory=dict)
    scalar_col_map: dict = field(repr=False, default_factory=dict)
    object_child_colmap: dict = field(repr=False, default_factory=dict)
    fk_candidates: list = field(repr=False, default_factory=list)
    skipped_missing_schema: list = field(repr=False, default_factory=list)

    @property
    def object_to_main_tbl(self) -> dict:
        return {obj: pair[0] for obj, pair in self.table_map.items()}

    @property
    def table_to_object(self) -> dict:
        return {pair[0].name: obj for obj, pair in self.table_map.items()}

    @property
    def reference_columns(self) -> set[tuple[str, str]]:
        """``(table, column)`` pairs the EnergyPlus schema declares as references.

        These name another object — ``Construction.layer_5`` names a material,
        ``BuildingSurface:Detailed.zone_name`` names a zone. They look like
        low-cardinality text to a post-processing pass that only counts distinct
        values, but they are identifiers, and encoding or dropping one destroys a
        relationship rather than compressing a feature.
        """
        return {
            (c["src_tbl"].name, c["src_col"])
            for c in self.fk_candidates
            if c.get("src_tbl") is not None and c.get("src_col")
        }

    @property
    def main_tables(self) -> list[Table]:
        return [pair[0] for pair in self.table_map.values()]

    @property
    def child_tables(self) -> list[Table]:
        return [t for _, children in self.table_map.values() for t in children.values()]

    def summary(self) -> pd.DataFrame:
        """One-row frame describing the table set that was defined."""
        return pd.DataFrame(
            [
                {
                    "main_tables": len(self.main_tables),
                    "child_tables": len(self.child_tables),
                    "total_tables": len(self.main_tables) + len(self.child_tables) + 1,
                    "scalar_columns": len(self.scalar_col_map),
                    "fk_candidates": len(self.fk_candidates),
                    "objects_skipped_not_in_schema": len(self.skipped_missing_schema),
                }
            ]
        )

    def tables_frame(self) -> pd.DataFrame:
        """One row per object type: its main table, column count and child tables."""
        rows = []
        for obj, (main, children) in sorted(self.table_map.items()):
            rows.append(
                {
                    "object": obj,
                    "main_table": main.name,
                    "columns": len(main.columns),
                    "child_tables": len(children),
                    "children": ", ".join(sorted(t.name for t in children.values())) or "-",
                }
            )
        return pd.DataFrame(rows)

    def columns_frame(self, obj_name: str) -> pd.DataFrame:
        """Column definitions of one object's main table, as SQLAlchemy holds them."""
        main, _ = self.table_map[obj_name]
        return pd.DataFrame(
            [
                {
                    "column": c.name,
                    "type": str(c.type),
                    "nullable": c.nullable,
                    "primary_key": c.primary_key,
                    "server_default": server_default_to_str(getattr(c, "server_default", None)),
                }
                for c in main.columns
            ]
        )


def _unique_name_factory(taken: set[str]):
    """Return a function handing out column names unique within ``taken``."""

    def unique(raw: str) -> str:
        base = safe_column_name(raw)
        if base not in taken:
            taken.add(base)
            return base
        for i in range(1, 1000):
            cand = shorten_identifier(f"{base}_{i}")
            if cand not in taken:
                taken.add(cand)
                return cand
        raise RuntimeError(f"Could not find unique column name for {raw}")

    return unique


def create_object_tables(
    build: SchemaBuild, obj_name: str, df_obj: pd.DataFrame, model_samples: list
):
    """Define the main table and any child tables for one EnergyPlus object.

    Child tables are created only for array-of-objects fields that are non-empty
    somewhere in ``model_samples``, so the schema stays as narrow as the data.
    Scalar ``object_list`` fields are recorded on ``build.fk_candidates``.
    """
    tname_main = safe_table_name(f"ep_{obj_name}")

    # Split fields; arrays of scalars are ignored (the schema uses arrays of objects)
    _scalars, arrays = classify_fields(df_obj)

    # Detect which array-of-objects fields are actually used (non-empty) in the dataset
    used_object_arrays = set()
    array_field_names = {fo for fo, _ in arrays}
    for inst in model_samples:
        if not isinstance(inst, dict):
            continue
        for fname, val in inst.items():
            if isinstance(val, list) and val and fname in array_field_names:
                used_object_arrays.add(fname)

    # --- MAIN TABLE -----------------------------------------------------------
    unique_col = _unique_name_factory({"id", "model_id", "instance_name"})

    cols = [
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("model_id", String, ForeignKey("ep_model.model_id"), nullable=False, index=True),
        Column("instance_name", String, nullable=False),
    ]
    uq_name = safe_constraint_name("uq", tname_main, "inst")
    table_level_constraints = [UniqueConstraint("model_id", "instance_name", name=uq_name)]

    for _, r in df_obj.iterrows():
        ftype = get(r, "type")
        if ftype in ("array", "object"):
            continue
        orig_fname = r["field"]
        colname = unique_col(orig_fname)
        build.scalar_col_map[(obj_name, orig_fname)] = colname

        coltype = sa_type_from_schema(get(r, "type"))
        required = bool(get(r, "required"))
        default = get(r, "default")

        col = Column(colname, coltype, nullable=not required)
        set_server_default(col, default)

        enum_vals = get(r, "enum")
        if isinstance(enum_vals, list):
            c = build_check_enum(tname_main, colname, enum_vals)
            if c is not None:
                table_level_constraints.append(c)

        c = build_check_minmax(
            tname_main,
            colname,
            minimum=get(r, "minimum"),
            maximum=get(r, "maximum"),
            exclusive_min=bool(get(r, "exclusiveMinimum")),
            exclusive_max=bool(get(r, "exclusiveMaximum")),
        )
        if c is not None:
            table_level_constraints.append(c)

        # object_list fields become foreign key candidates, resolved after the
        # whole table set exists
        if get(r, "data_type") == "object_list":
            labels = get(r, "object_list") or []
            if labels:
                build.fk_candidates.append(
                    {
                        "src_obj": obj_name,
                        "src_tbl": None,  # filled once main_tbl exists
                        "src_col": colname,
                        "labels": labels,
                    }
                )

        cols.append(col)

    main_tbl = Table(
        tname_main, build.meta, *cols, *table_level_constraints, extend_existing=True
    )

    for item in build.fk_candidates:
        if item["src_obj"] == obj_name and item["src_tbl"] is None:
            item["src_tbl"] = main_tbl

    # --- CHILD TABLES: arrays of objects only ---------------------------------
    child_object_tbls: dict[str, Table] = {}

    for fname, items in arrays:
        if fname not in used_object_arrays:
            continue

        if isinstance(items, dict):
            props = items.get("properties", {}) or {}
            required_props = set(items.get("required", []) or [])
        else:
            props = get(items, "properties") or {}
            required_props = set(get(items, "required") or [])

        tname_child = safe_table_name(f"{tname_main}__{fname}")
        unique_child_col = _unique_name_factory({"id", "parent_id", "position"})

        cols_child = [
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column(
                "parent_id",
                Integer,
                ForeignKey(f"{tname_main}.id"),
                nullable=False,
                index=True,
            ),
            Column("position", Integer, nullable=False),
            UniqueConstraint(
                "parent_id", "position", name=safe_constraint_name("uq", tname_child, "pos")
            ),
            CheckConstraint(
                "position >= 0", name=safe_constraint_name("ck", tname_child, "position_nonneg")
            ),
        ]
        tbl_constraints = []
        prop_map = {}

        for p_name, p_def in props.items():
            scol = unique_child_col(p_name)
            prop_map[p_name] = scol

            col = Column(
                scol, sa_type_from_schema(p_def.get("type")), nullable=p_name not in required_props
            )
            set_server_default(col, p_def.get("default"))

            if isinstance(p_def.get("enum"), list):
                c = build_check_enum(tname_child, scol, p_def["enum"])
                if c is not None:
                    tbl_constraints.append(c)

            c = build_check_minmax(
                tname_child,
                scol,
                minimum=p_def.get("minimum"),
                maximum=p_def.get("maximum"),
                exclusive_min=bool(p_def.get("exclusiveMinimum")),
                exclusive_max=bool(p_def.get("exclusiveMaximum")),
            )
            if c is not None:
                tbl_constraints.append(c)

            cols_child.append(col)

        ct = Table(tname_child, build.meta, *cols_child, *tbl_constraints, extend_existing=True)
        child_object_tbls[fname] = ct
        build.object_child_colmap[(obj_name, fname)] = prop_map

    return main_tbl, child_object_tbls


def sample_instances(epjson_files, used_objects: set[str]) -> dict:
    """Collect example instances per object type, to decide which arrays are populated."""
    samples = defaultdict(list)
    for fp in epjson_files:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        for obj_name in used_objects:
            insts = data.get(obj_name)
            if isinstance(insts, dict):
                samples[obj_name].extend(list(insts.values()))
            elif isinstance(insts, list):
                samples[obj_name].extend(insts)
    return samples


def build_table_set(
    schema: EnergyPlusSchema,
    used_objects: set[str],
    epjson_files,
    schema_name: str,
) -> SchemaBuild:
    """Define ``ep_model`` and one table set per used object type.

    Nothing is written to the database here; the result is a
    :class:`SchemaBuild` holding SQLAlchemy definitions only.
    """
    meta = MetaData(schema=schema_name)
    build = SchemaBuild(meta=meta, schema=schema_name, ep_model=ep_model_table(meta))

    samples = sample_instances(epjson_files, used_objects)

    for obj_name in sorted(used_objects):
        df = schema.tables.get(obj_name)
        if df is None:
            build.skipped_missing_schema.append(obj_name)
            continue
        main, children = create_object_tables(build, obj_name, df, samples[obj_name])
        build.table_map[obj_name] = (main, children)

    return build


def render_restart_safe_add_column_sql(engine, col: Column) -> str:
    """Render an ADD COLUMN fragment for a column that post-processing removed.

    Restored as nullable: rows already in the database have no value for a
    physically dropped column, while newly ingested rows populate it.
    """
    preparer = engine.dialect.identifier_preparer
    return f"{preparer.quote(col.name)} {col.type.compile(dialect=engine.dialect)}"


def add_missing_columns_for_table(engine, table: Table) -> list[str]:
    """Add columns present in the SQLAlchemy definition but missing in the live table."""
    inspector = inspect(engine)
    existing_cols = {c["name"] for c in inspector.get_columns(table.name, schema=table.schema)}
    missing_cols = [c for c in table.columns if c.name not in existing_cols]
    if not missing_cols:
        return []

    preparer = engine.dialect.identifier_preparer
    if table.schema:
        fq_table = f"{preparer.quote_schema(table.schema)}.{preparer.quote(table.name)}"
    else:
        fq_table = preparer.quote(table.name)

    restored = []
    with engine.begin() as con:
        for col in missing_cols:
            col_sql = render_restart_safe_add_column_sql(engine, col)
            con.execute(sa_text(f"ALTER TABLE {fq_table} ADD COLUMN IF NOT EXISTS {col_sql}"))
            restored.append(col.name)
    return restored


def list_schema_tables(engine, schema: str) -> list[str]:
    """Every table name currently present in a schema."""
    sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
        ORDER BY table_name
    """
    with engine.connect() as con:
        return [r[0] for r in con.execute(sa_text(sql), {"schema": schema}).fetchall()]


def materialise(engine, build: SchemaBuild, reconcile_columns: bool = False) -> pd.DataFrame:
    """Create every defined table that does not exist yet.

    ``meta.create_all`` is idempotent. When the schema was *not* dropped first,
    pass ``reconcile_columns=True`` to restore columns a previous run's
    post-processing pruned; otherwise ingestion would silently skip those fields.

    Returns a frame of restored columns, empty when there was nothing to do.
    """
    build.meta.create_all(engine)

    restored_rows: list[dict] = []
    if reconcile_columns:
        for tbl in build.meta.sorted_tables:
            restored = add_missing_columns_for_table(engine, tbl)
            if restored:
                restored_rows.append({"table": tbl.name, "restored_columns": ", ".join(restored)})

    return pd.DataFrame(restored_rows, columns=["table", "restored_columns"])


def materialisation_report(engine, build: SchemaBuild) -> pd.DataFrame:
    """Compare what was defined against what the database holds."""
    present = set(list_schema_tables(engine, build.schema))
    groups = {
        "main": [t.name for t in build.main_tables],
        "child": [t.name for t in build.child_tables],
        "registry": [build.ep_model.name],
    }
    return pd.DataFrame(
        [
            {
                "kind": kind,
                "defined": len(names),
                "present": sum(1 for n in names if n in present),
                "missing": sum(1 for n in names if n not in present),
            }
            for kind, names in groups.items()
        ]
    )
