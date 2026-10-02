"""EnergyPlus schema field metadata -> SQLAlchemy column types and constraints.

``Energy+.schema.epJSON`` describes every EnergyPlus object as a JSON Schema
fragment. These helpers translate one field's metadata into the column type,
server default and CHECK constraints the ingestion stage puts on the table.

The schema is the sole source of the column set, so a field the schema does not
declare gets no column and is dropped silently at ingestion. ``docs/data.md``
explains why the shipped schema version has to match the model set.
"""

from __future__ import annotations

import math

import pandas as pd
from sqlalchemy import Boolean, CheckConstraint, Float, Integer, Text
from sqlalchemy import text as sa_text

from beg.ingest.identifiers import qident, safe_constraint_name


def build_check_enum(table_name: str, col_name: str, values):
    """Create an IN(...) enum check constraint for a column, or None if no values."""
    if not values:
        return None

    def lit(v):
        return "'" + str(v).replace("'", "''") + "'" if isinstance(v, str) else str(v)

    clause = ", ".join(lit(v) for v in values)
    cname = safe_constraint_name("ck", table_name, col_name, "enum")
    return CheckConstraint(f"{qident(col_name)} IN ({clause})", name=cname)


def build_check_minmax(table_name: str, col_name: str,
                       minimum=None, maximum=None,
                       exclusive_min: bool = False, exclusive_max: bool = False):
    """Create a numeric min/max check constraint for a column, or None if not applicable."""
    parts = []
    colq = qident(col_name)
    if minimum is not None:
        op = ">" if exclusive_min else ">="
        parts.append(f"{colq} {op} {minimum}")
    if maximum is not None:
        op = "<" if exclusive_max else "<="
        parts.append(f"{colq} {op} {maximum}")
    if not parts:
        return None
    cname = safe_constraint_name("ck", table_name, col_name, "bounds")
    return CheckConstraint(" AND ".join(parts), name=cname)


def return_as_list(v):
    """Wrap a value in a list.

    The schema's ``reference`` field is either a string or a list of strings;
    ``None`` becomes an empty list so callers can always iterate.
    """
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def get(series, key):
    """Safe getter from pandas Series that treats NaN as None."""
    v = series.get(key, None)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def sa_type_from_schema(ftype: str):
    """Map JSON schema 'type' to SQLAlchemy column type (broad mapping)."""
    if ftype == "integer":
        return Integer
    if ftype == "number":
        return Float
    if ftype == "boolean":
        return Boolean
    # Treat strings and everything else as Text by default
    return Text


def classify_fields(df_obj: pd.DataFrame):
    """Split a field-metadata DataFrame into scalar and array-like fields.

    Returns ``(scalars, arrays)`` where ``scalars`` are rows with non-array,
    non-object types and ``arrays`` is a list of ``(field_name, items_schema)``.
    An ``object``-typed field is treated as a 1:1 payload and joins ``arrays``,
    because it also becomes a child table.
    """
    scalars, arrays = [], []
    for _, r in df_obj.iterrows():
        ftype = get(r, "type")
        fname = r["field"]
        if ftype == "array":
            # Arrays of objects and arrays of scalars both become child tables;
            # the table builder reads `items` to decide which columns it gets.
            arrays.append((fname, get(r, "items")))
        elif ftype == "object":
            # Treat object as an object-like payload (1:1) stored in a child table
            arrays.append((fname, r))
        else:
            scalars.append(r)
    return scalars, arrays


def server_default_to_str(default_clause):
    """Render a SQLAlchemy DefaultClause as a string.

    Avoids boolean evaluation of a ClauseElement, which raises rather than
    returning a truth value.
    """
    if default_clause is None:
        return None
    try:
        arg = getattr(default_clause, "arg", None)
        if arg is None:
            return str(default_clause)
        # TextClause has .text; literals are just values
        txt = getattr(arg, "text", None)
        return txt if txt is not None else str(arg)
    except Exception:
        return str(default_clause)


def set_server_default(col, default):
    """Attach a SQL-level server default, quoting strings so the SQL stays valid."""
    if default is None:
        return
    if isinstance(default, str):
        col.server_default = sa_text(repr(default))  # adds quotes
    else:
        col.server_default = sa_text(str(default))
