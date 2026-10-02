"""Post-processing the raw ``ep`` schema into something a model can consume.

Three passes, in this order:

1. drop columns with no variation across the dataset,
2. convert text booleans (``Yes``/``No``) to ``1``/``0``,
3. one-hot encode low-cardinality categorical text columns.

These run at *ingestion*, which means **the feature width of the ``ep`` schema is
a property of the model set, not only of the configuration**. Two ingestions of
different model sets are not directly comparable; ``docs/database.md`` explains
the consequences and ``docs/limitations.md`` records what it costs.

Every DDL step runs in its own transaction. The tables carry deferred foreign
keys, and batching these statements raises "pending trigger events".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import text as sa_text

from beg.ingest.identifiers import ID_MAX, qident, shorten_identifier

#: Structural columns that must survive all three passes.
PROTECTED_COLUMNS = frozenset(
    {
        "id",
        "model_id",
        "instance_name",
        "parent_id",
        "position",
        "imported_at",
        "file_name",
        "file_path",
        "sha256",
        "preferred_object",
        "preferred_table",
    }
)

BOOLEAN_TRUE_VALUES = frozenset({"yes", "true", "on"})
BOOLEAN_FALSE_VALUES = frozenset({"no", "false", "off"})
BOOLEAN_ALL_VALUES = BOOLEAN_TRUE_VALUES | BOOLEAN_FALSE_VALUES

#: Column types the boolean and one-hot passes consider.
TEXT_TYPES = ("text", "character varying", "varchar", "character", "char")


@dataclass
class PostProcessResult:
    """What each pass changed, as frames ready to display."""

    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)
    converted: pd.DataFrame = field(default_factory=pd.DataFrame)
    encoded: pd.DataFrame = field(default_factory=pd.DataFrame)
    columns_analyzed: int = 0

    def summary(self) -> pd.DataFrame:
        """One row per pass: how many columns it touched."""
        encoded_ok = (
            int(self.encoded["ok"].sum()) if "ok" in self.encoded else len(self.encoded)
        )
        converted_ok = (
            int(self.converted["ok"].sum()) if "ok" in self.converted else len(self.converted)
        )
        new_cols = (
            int(self.encoded.loc[self.encoded["ok"], "new_columns"].sum())
            if "ok" in self.encoded and not self.encoded.empty
            else 0
        )
        return pd.DataFrame(
            [
                {"pass": "invariant columns dropped", "columns": len(self.dropped), "detail": ""},
                {
                    "pass": "text booleans -> 0/1",
                    "columns": converted_ok,
                    "detail": f"{len(self.converted) - converted_ok} failed",
                },
                {
                    "pass": "categoricals one-hot encoded",
                    "columns": encoded_ok,
                    "detail": f"{new_cols} new binary columns",
                },
            ]
        )

    def as_dict(self) -> dict:
        """Counts for the run summary JSON."""
        return {
            "columns_analyzed": self.columns_analyzed,
            "no_variation_columns_removed": len(self.dropped),
            "boolean_text_columns_converted": int(self.converted["ok"].sum())
            if "ok" in self.converted
            else 0,
            "categorical_columns_encoded": int(self.encoded["ok"].sum())
            if "ok" in self.encoded
            else 0,
        }


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #


def get_all_user_tables(engine, schema: str) -> list[str]:
    """Every base table in the schema."""
    sql = """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = :schema AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    with engine.connect() as con:
        return [row[0] for row in con.execute(sa_text(sql), {"schema": schema}).fetchall()]


def get_table_columns(engine, schema: str, table: str) -> list[str]:
    """Column names of one table, in ordinal order."""
    sql = """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
    """
    with engine.connect() as con:
        return [
            row[0]
            for row in con.execute(sa_text(sql), {"schema": schema, "table": table}).fetchall()
        ]


def get_column_type(engine, schema: str, table: str, column: str) -> str | None:
    """The declared data type of one column."""
    sql = """
        SELECT data_type FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table AND column_name = :column
    """
    with engine.connect() as con:
        return con.execute(
            sa_text(sql), {"schema": schema, "table": table, "column": column}
        ).scalar()


def count_distinct_values(engine, schema: str, table: str, column: str) -> int:
    """Distinct non-NULL values in a column.

    ``0`` means every row is NULL, ``1`` means a single value regardless of how
    many NULLs accompany it, ``2+`` means real variation. NULL does not count as
    a value: ``["Yes", "Yes", NULL]`` has no variation.
    """
    fq_table = f"{qident(schema)}.{qident(table)}"
    qcol = qident(column)

    with engine.connect() as con:
        non_null = (
            con.execute(sa_text(f"SELECT COUNT(*) FROM {fq_table} WHERE {qcol} IS NOT NULL")).scalar()
            or 0
        )
        if non_null == 0:
            return 0
        return (
            con.execute(
                sa_text(f"SELECT COUNT(DISTINCT {qcol}) FROM {fq_table} WHERE {qcol} IS NOT NULL")
            ).scalar()
            or 0
        )


def get_sample_value(engine, schema: str, table: str, column: str, distinct_count: int) -> str:
    """One representative value, truncated, for the report."""
    if distinct_count == 0:
        return "<all NULL>"
    fq_table = f"{qident(schema)}.{qident(table)}"
    qcol = qident(column)
    try:
        with engine.connect() as con:
            result = con.execute(
                sa_text(f"SELECT {qcol} FROM {fq_table} WHERE {qcol} IS NOT NULL LIMIT 1")
            ).scalar()
        return "<all NULL>" if result is None else str(result)[:50]
    except Exception:
        return "<error>"


def get_distinct_values(engine, schema: str, table: str, column: str, limit: int = 100) -> set:
    """Distinct non-NULL values as stripped strings, up to ``limit``."""
    fq_table = f"{qident(schema)}.{qident(table)}"
    qcol = qident(column)
    try:
        with engine.connect() as con:
            rows = con.execute(
                sa_text(
                    f"SELECT DISTINCT {qcol} FROM {fq_table} WHERE {qcol} IS NOT NULL LIMIT {int(limit)}"
                )
            ).fetchall()
        return {str(r[0]).strip() for r in rows if r[0] is not None}
    except Exception:
        return set()


def get_distinct_values_for_onehot(engine, schema: str, table: str, column: str) -> list:
    """Distinct non-NULL values, ordered, for deterministic column naming."""
    fq_table = f"{qident(schema)}.{qident(table)}"
    qcol = qident(column)
    try:
        with engine.connect() as con:
            rows = con.execute(
                sa_text(
                    f"SELECT DISTINCT {qcol} FROM {fq_table} WHERE {qcol} IS NOT NULL ORDER BY {qcol}"
                )
            ).fetchall()
        return [str(r[0]).strip() for r in rows if r[0] is not None]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Pass 1 - drop columns with no variation
# --------------------------------------------------------------------------- #


def drop_column(engine, schema: str, table: str, column: str) -> None:
    """Drop a column, cascading to dependent views and constraints."""
    with engine.begin() as con:
        con.execute(
            sa_text(
                f"ALTER TABLE {qident(schema)}.{qident(table)} DROP COLUMN {qident(column)} CASCADE"
            )
        )


def prune_invariant_columns(
    engine,
    schema: str,
    protected: frozenset = PROTECTED_COLUMNS,
    dry_run: bool = False,
    reference_columns: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Drop every column that holds at most one distinct non-NULL value.

    ``reference_columns`` — ``(table, column)`` pairs naming another object — are
    dropped only when *entirely* empty: a reference holding a single name still
    records which rows have it (``Construction.layer_5`` is NULL for constructions
    with four layers). See :attr:`beg.ingest.tables.SchemaBuild.reference_columns`.

    Returns ``(dropped_frame, columns_analyzed)``. With ``dry_run=True`` nothing
    is dropped and the frame is the list of candidates.
    """
    rows = []
    analyzed = 0
    references = reference_columns or set()

    for table in get_all_user_tables(engine, schema):
        for column in get_table_columns(engine, schema, table):
            analyzed += 1
            if column in protected:
                continue

            distinct = count_distinct_values(engine, schema, table, column)
            is_reference = (table, column) in references
            # A reference with a value in even one row still says which rows have it.
            drop_at = 0 if is_reference else 1
            if distinct <= drop_at:
                rows.append(
                    {
                        "table": table,
                        "column": column,
                        "reason": "all NULL" if distinct == 0 else "single value",
                        "sample_value": get_sample_value(engine, schema, table, column, distinct),
                    }
                )

    if not dry_run:
        for row in rows:
            drop_column(engine, schema, row["table"], row["column"])

    return pd.DataFrame(rows, columns=["table", "column", "reason", "sample_value"]), analyzed


# --------------------------------------------------------------------------- #
# Pass 2 - text booleans to 0/1
# --------------------------------------------------------------------------- #


def is_boolean_column(distinct_values: set) -> bool:
    """True when every distinct value is boolean-like, ignoring case."""
    if not distinct_values:
        return False
    return {v.lower() for v in distinct_values}.issubset(BOOLEAN_ALL_VALUES)


def convert_column_to_numeric_boolean(engine, schema: str, table: str, column: str) -> bool:
    """Replace a text boolean column with an INTEGER 0/1 column of the same name.

    Done as add-temp / populate / drop / rename, each in its own transaction, so
    deferred foreign keys do not trip on pending trigger events.
    """
    fq_table = f"{qident(schema)}.{qident(table)}"
    qcol = qident(column)
    temp_col = qident(f"{column}_numeric_temp")

    try:
        with engine.begin() as con:
            con.execute(sa_text(f"ALTER TABLE {fq_table} ADD COLUMN {temp_col} INTEGER"))

        for value, numeric in (
            *((v, 1) for v in BOOLEAN_TRUE_VALUES),
            *((v, 0) for v in BOOLEAN_FALSE_VALUES),
        ):
            with engine.begin() as con:
                con.execute(
                    sa_text(
                        f"UPDATE {fq_table} SET {temp_col} = {numeric} WHERE LOWER({qcol}) = :val"
                    ),
                    {"val": value},
                )

        with engine.begin() as con:
            con.execute(sa_text(f"ALTER TABLE {fq_table} DROP COLUMN {qcol}"))
        with engine.begin() as con:
            con.execute(sa_text(f"ALTER TABLE {fq_table} RENAME COLUMN {temp_col} TO {qcol}"))
        return True
    except Exception:
        try:
            with engine.begin() as con:
                con.execute(sa_text(f"ALTER TABLE {fq_table} DROP COLUMN IF EXISTS {temp_col}"))
        except Exception:
            pass  # cleanup is best-effort; the failure is already reported
        return False


def coerce_boolean_columns(
    engine,
    schema: str,
    protected: frozenset = PROTECTED_COLUMNS,
    reference_columns: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Convert every text column whose values are all ``Yes``/``No``-like to 0/1.

    ``reference_columns`` are skipped: an object named ``On`` is still a name.
    """
    rows = []
    references = reference_columns or set()
    for table in get_all_user_tables(engine, schema):
        for column in get_table_columns(engine, schema, table):
            if column in protected or (table, column) in references:
                continue
            if get_column_type(engine, schema, table, column) not in TEXT_TYPES:
                continue

            values = get_distinct_values(engine, schema, table, column)
            if not is_boolean_column(values):
                continue

            rows.append(
                {
                    "table": table,
                    "column": column,
                    "values": ", ".join(sorted(values)),
                    "ok": convert_column_to_numeric_boolean(engine, schema, table, column),
                }
            )
    return pd.DataFrame(rows, columns=["table", "column", "values", "ok"])


# --------------------------------------------------------------------------- #
# Pass 3 - one-hot encode low-cardinality categoricals
# --------------------------------------------------------------------------- #


def sanitize_column_suffix(value: str) -> str:
    """Safe column suffix for a categorical value (``'Wall-Exterior'`` -> ``'wall_exterior'``)."""
    safe = re.sub(r"[^0-9a-z_]", "_", value.lower())
    safe = re.sub(r"__+", "_", safe).strip("_")
    if safe and safe[0].isdigit():
        safe = "v_" + safe
    return safe


def one_hot_encode_column(
    engine, schema: str, table: str, column: str, distinct_values: list
) -> tuple[bool, int]:
    """Replace a categorical column with one binary column per value.

    Returns ``(ok, new_column_count)``. Rows where the original was NULL are left
    NULL in every new column, so "unknown" stays distinguishable from "false".
    """
    fq_table = f"{qident(schema)}.{qident(table)}"
    qcol = qident(column)

    # Distinct categories can sanitize to the same identifier -- "Wall-Exterior"
    # and "Wall Exterior" both become `wall_exterior`, as do values differing
    # only in case. Colliding names get a short digest of the original value
    # appended, which keeps them distinct and deterministic.
    new_columns: dict[str, str] = {}
    for value in distinct_values:
        name = shorten_identifier(f"{column}_{sanitize_column_suffix(value)}", maxlen=ID_MAX)
        if name in new_columns:
            digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]
            name = shorten_identifier(f"{name}_{digest}", maxlen=ID_MAX)
        new_columns[name] = value

    try:
        for name in new_columns:
            with engine.begin() as con:
                con.execute(sa_text(f"ALTER TABLE {fq_table} ADD COLUMN {qident(name)} INTEGER"))

        for name, original_value in new_columns.items():
            qnew = qident(name)
            with engine.begin() as con:
                con.execute(
                    sa_text(f"UPDATE {fq_table} SET {qnew} = 1 WHERE {qcol} = :val"),
                    {"val": original_value},
                )
            with engine.begin() as con:
                con.execute(
                    sa_text(
                        f"UPDATE {fq_table} SET {qnew} = 0 "
                        f"WHERE {qcol} IS NOT NULL AND {qcol} != :val"
                    ),
                    {"val": original_value},
                )

        with engine.begin() as con:
            con.execute(sa_text(f"ALTER TABLE {fq_table} DROP COLUMN {qcol}"))
        return True, len(new_columns)
    except Exception:
        for name in new_columns:
            try:
                with engine.begin() as con:
                    con.execute(
                        sa_text(f"ALTER TABLE {fq_table} DROP COLUMN IF EXISTS {qident(name)}")
                    )
            except Exception:
                pass  # cleanup is best-effort
        return False, 0


def one_hot_encode_columns(
    engine,
    schema: str,
    min_categories: int = 2,
    max_categories: int = 5,
    protected: frozenset = PROTECTED_COLUMNS,
    reference_columns: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """One-hot encode text columns holding between ``min`` and ``max`` distinct values.

    The upper bound keeps the schema from exploding on free-text fields; the
    lower bound skips columns that carry no choice at all.

    ``reference_columns`` are never encoded: how many distinct names a reference
    holds depends on the model set, and encoding it would replace a join to the
    named object (and the foreign key built on it) with indicator columns.
    """
    rows = []
    references = reference_columns or set()
    for table in get_all_user_tables(engine, schema):
        for column in get_table_columns(engine, schema, table):
            if column in protected or (table, column) in references:
                continue
            if get_column_type(engine, schema, table, column) not in TEXT_TYPES:
                continue

            values = get_distinct_values_for_onehot(engine, schema, table, column)
            if not (min_categories <= len(values) <= max_categories):
                continue

            ok, made = one_hot_encode_column(engine, schema, table, column, values)
            rows.append(
                {
                    "table": table,
                    "column": column,
                    "categories": ", ".join(values[:10]) + ("..." if len(values) > 10 else ""),
                    "new_columns": made,
                    "ok": ok,
                }
            )
    return pd.DataFrame(rows, columns=["table", "column", "categories", "new_columns", "ok"])


