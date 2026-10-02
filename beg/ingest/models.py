"""The model registry and the epJSON -> relational row mapping.

``ep.ep_model`` is the registry: one row per ingested epJSON file, keyed by a
``model_id`` derived from the file's ``Building`` object, and deduplicated on the
file's SHA-256. Everything else in the ``ep`` schema references it.

The helpers below take the tables they operate on as arguments, because the
table set is built at run time from whichever EnergyPlus objects the model set
actually uses.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    and_,
    bindparam,
    delete,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from beg.ingest.schema_map import get
from beg.settings import data_dir


def ep_model_table(meta: MetaData) -> Table:
    """Define ``ep_model`` on ``meta`` and return it.

    ``extend_existing`` keeps this callable more than once against the same
    MetaData, which is what a re-run notebook cell does.
    """
    return Table(
        "ep_model",
        meta,
        Column("model_id", String, primary_key=True),
        Column("file_name", String, nullable=False),
        Column("file_path", Text, nullable=False),
        Column("sha256", String(64), nullable=False, unique=True),
        Column("imported_at", DateTime(timezone=True), nullable=False),
        extend_existing=True,
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 8 KiB binary chunks."""
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def portable_file_path(path: Path) -> str:
    """Render a model's path relative to ``data_dir()``, falling back to its name.

    The registry is part of a published database dump, so it must not carry an
    absolute path belonging to whoever ran the ingestion.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(data_dir()))
    except ValueError:
        return resolved.name


def insert_model(con, file_path: Path, ep_model: Table):
    """Insert a file row into ``ep_model`` if not present. Returns ``(model_id, is_new)``.

    Deduplication is by SHA-256. ``model_id`` comes from the file's ``Building``
    instance name, falling back to the file stem, and is suffixed ``_2``, ``_3``
    … if a different file already claimed it.
    """
    sh = sha256_file(file_path)

    # Derive model_id from Building.instance_name or file name
    try:
        with file_path.open("r", encoding="utf-8") as f:
            j = json.load(f)
        b = j.get("Building") or j.get("building") or {}
        if isinstance(b, dict) and b:
            base_model_id = next(iter(b.keys()))
        else:
            base_model_id = file_path.stem
    except Exception:
        base_model_id = file_path.stem

    # Ensure uniqueness of model_id (avoid PK conflicts) by suffixing if needed
    candidate = base_model_id
    i = 2
    while True:
        existing = con.execute(
            select(ep_model.c.model_id).where(ep_model.c.model_id == bindparam("mid")),
            {"mid": candidate},
        ).fetchone()
        if existing is None:
            break
        # If an existing row has the same SHA -> reuse its model_id
        row = con.execute(
            select(ep_model.c.model_id).where(ep_model.c.sha256 == bindparam("sha")),
            {"sha": sh},
        ).fetchone()
        if row is not None:
            return row[0], False
        candidate = f"{base_model_id}_{i}"
        i += 1

    # Insert, deduping primarily by SHA-256; also ignore PK conflicts
    stmt = (
        pg_insert(ep_model)
        .values(
            model_id=candidate,
            file_name=file_path.name,
            file_path=portable_file_path(file_path),
            sha256=sh,
            imported_at=datetime.now(timezone.utc),
        )
        .on_conflict_do_nothing()
        .returning(ep_model.c.model_id)
    )

    model_id = con.execute(stmt).scalar()
    if model_id is not None:
        return model_id, True  # newly inserted

    # Existing row -> try fetch by sha256 first, then by candidate model_id
    row = con.execute(
        select(ep_model.c.model_id).where(ep_model.c.sha256 == bindparam("sha")), {"sha": sh}
    ).fetchone()
    if row is not None:
        return row[0], False
    row = con.execute(
        select(ep_model.c.model_id).where(ep_model.c.model_id == bindparam("mid")),
        {"mid": candidate},
    ).fetchone()
    if row is not None:
        return row[0], False
    return candidate, False


def upsert_main_return_id(con, main_tbl, row_data):
    """Upsert a main row on ``(model_id, instance_name)`` and return its ``id``.

    Used when the duplicate policy is ``update``. Identity columns are excluded
    from the UPDATE SET so only the scalar payload is refreshed.
    """
    stmt = pg_insert(main_tbl).values(**row_data)

    # Exclude immutable / identity columns from update set
    update_set = {
        k: stmt.excluded[k]
        for k in row_data.keys()
        if k not in ("id", "model_id", "instance_name")
    }

    # ON CONFLICT ... DO UPDATE to keep the latest scalar values
    res = con.execute(
        stmt.on_conflict_do_update(
            index_elements=["model_id", "instance_name"], set_=update_set
        ).returning(main_tbl.c.id)
    )
    return res.scalar_one()


def insert_main_return_id(con, main_tbl, row_data):
    """Insert a main row and return its ``id``, or fetch the id if it already exists.

    Used when the duplicate policy is ``skip``. The ``except`` is broad because
    dialects raise different concrete classes for a unique violation.
    """
    try:
        return con.execute(
            main_tbl.insert().values(**row_data).returning(main_tbl.c.id)
        ).scalar_one()
    except Exception:
        # Likely unique-violation -> lookup the existing row id
        where_clause = and_(
            main_tbl.c.model_id == row_data["model_id"],
            main_tbl.c.instance_name == row_data["instance_name"],
        )
        res = con.execute(select(main_tbl.c.id).where(where_clause))
        return res.scalar_one()


def load_epjson(fp: Path) -> dict:
    """Load an epJSON file from disk (UTF-8)."""
    return json.loads(Path(fp).read_text(encoding="utf-8"))


def normalize_instances_container(instances):
    """Yield ``(instance_name, fields_dict)`` for either a dict or a list container."""
    if isinstance(instances, dict):
        # Dict form: keys are instance names, values are the field dicts
        return instances.items()
    # List form: use the numeric index as a synthetic instance name
    return enumerate(instances)


def build_scalar_field_map(obj_name: str, schema_tables: dict, scalar_col_map: dict) -> dict:
    """Map schema field -> (sanitized column name, schema row) for scalar fields only."""
    df_schema = schema_tables[obj_name].set_index("field", drop=False)

    field_map: dict = {}
    for fname, sr in df_schema.iterrows():
        ftype = get(sr, "type")
        if ftype in ("array", "object"):
            # Arrays/objects are handled via child tables
            continue
        colname = scalar_col_map.get((obj_name, fname))
        if colname:
            field_map[fname] = (colname, sr)
    return field_map


def row_data_from_fields(model_id, inst_name, field_map: dict, fields: dict) -> dict:
    """Build the main-row payload, applying schema defaults when a field is missing."""
    row = {"model_id": model_id, "instance_name": str(inst_name)}
    for fname, (colname, sr) in field_map.items():
        default_val = get(sr, "default")
        row[colname] = fields.get(fname, default_val)
    return row


def insert_or_upsert_main(
    con, main_tbl, row_data: dict, is_new_file: bool, duplicate_policy: str = "skip"
) -> int:
    """Insert or upsert a main row according to ``duplicate_policy``. Return parent id."""
    if is_new_file or duplicate_policy == "skip":
        return insert_main_return_id(con, main_tbl, row_data)
    return upsert_main_return_id(con, main_tbl, row_data)


def replace_children(
    con,
    obj_name: str,
    fields: dict,
    parent_id: int,
    child_obj_tbls: dict,
    object_child_colmap: dict,
    parent_is_new: bool = False,
):
    """Delete and reinsert array-of-object children.

    Returns ``(rows_inserted, per_field Counter)``. Deleting first makes the step
    idempotent, so re-ingesting a model does not duplicate its child rows.

    ``parent_is_new`` skips that delete: a newly inserted parent row has no
    children yet.
    """
    inserted_total = 0
    per_field = Counter()

    for fname, tbl in child_obj_tbls.items():
        # --- Delete existing children for this parent (idempotent step) ---
        if not parent_is_new:
            con.execute(delete(tbl).where(tbl.c.parent_id == parent_id))

        # --- Insert new children (preserve order with `position`) ---
        arr = fields.get(fname) or []
        if not arr:
            continue

        prop_map = object_child_colmap[(obj_name, fname)]  # original prop -> sanitized column
        rows = []
        for idx, obj in enumerate(arr):
            row = {"parent_id": parent_id, "position": idx}
            for orig_prop, scol in prop_map.items():
                row[scol] = obj.get(orig_prop)
            rows.append(row)

        con.execute(tbl.insert(), rows)  # bulk insert

        n = len(rows)
        inserted_total += n
        per_field[f"{obj_name}.{fname}"] += n

    return inserted_total, per_field


def insert_main_rows_batched(con, main_tbl, rows: list[dict]) -> list[int]:
    """Insert every main row of one object type in a single statement.

    Returns the new ``id`` values, in the order the rows were given, which is
    what lets the caller attach each instance's child rows afterwards.
    PostgreSQL returns ``RETURNING`` rows in insertion order for a single
    multi-row ``INSERT``, and SQLAlchemy preserves that correspondence for its
    batched form.

    Only safe on the insert path — a conflict would abort the whole statement
    and poison the transaction. The upsert path uses
    :func:`upsert_main_return_id` row by row.
    """
    if not rows:
        return []
    result = con.execute(main_tbl.insert().returning(main_tbl.c.id), rows)
    return [r[0] for r in result]


def child_rows_for(
    obj_name: str,
    field_name: str,
    parent_ids: list[int],
    instance_fields: list[dict],
    object_child_colmap: dict,
) -> list[dict]:
    """Build every child row of one array field, across all parents of an object type."""
    prop_map = object_child_colmap[(obj_name, field_name)]
    rows: list[dict] = []
    # strict=True: if the ids and the instances ever fall out of step, children
    # would be silently attached to the wrong parents or dropped. Fail instead.
    for parent_id, fields in zip(parent_ids, instance_fields, strict=True):
        for idx, obj in enumerate(fields.get(field_name) or []):
            row = {"parent_id": parent_id, "position": idx}
            for orig_prop, scol in prop_map.items():
                row[scol] = obj.get(orig_prop)
            rows.append(row)
    return rows


def count_instances_in_container(instances) -> int:
    """Count how many instances are present in a dict/list container (for stats)."""
    if isinstance(instances, dict):
        return len(instances)
    if isinstance(instances, list):
        return len(instances)
    return 1  # scalars / nulls (rare)
