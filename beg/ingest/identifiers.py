"""SQL-safe table, column and constraint names derived from EnergyPlus names.

EnergyPlus object and field names contain spaces, punctuation and mixed case, and
are frequently longer than PostgreSQL's 63-character identifier limit. Every
identifier the ingestion stage creates passes through :func:`sanitize` and then
through the length, digit-prefix and reserved-word fixes below, so that the same
EnergyPlus name always maps to the same SQL identifier.

Two quoting helpers exist in the wider codebase and they are not equivalent:
:func:`qident` escapes embedded double quotes, ``beg.db.utils.quote_identifier``
does not. Use this one when building constraint SQL by hand.
"""

from __future__ import annotations

import hashlib
import re

#: PostgreSQL's maximum identifier length. Longer names are hashed down to fit.
ID_MAX = 63

#: PostgreSQL system columns; a field of the same name gets an ``r_`` prefix.
PG_SYSTEM_COLS = {"tableoid", "xmin", "xmax", "cmin", "cmax", "ctid", "oid"}

#: Reserved words that would need quoting everywhere; prefixed with ``r_`` instead.
PG_KEYWORDS = {"user", "order", "select", "where", "table", "group", "limit"}


def shorten_identifier(name: str, maxlen: int = ID_MAX) -> str:
    """Shorten identifier with _<8hex> suffix if longer than maxlen."""
    if len(name) <= maxlen:
        return name
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    keep = maxlen - 1 - len(h)
    return f"{name[:keep]}_{h}"


def prefix_if_digit_start(name: str) -> str:
    """If a column name starts with a digit it adds a prefix c_"""
    return ("c_" + name) if name[:1].isdigit() else name


def avoid_reserved(name: str) -> str:
    """Prefix ``r_`` when the name collides with a reserved word or system column."""
    return ("r_" + name) if (name in PG_SYSTEM_COLS or name in PG_KEYWORDS) else name


def sanitize(name: str) -> str:
    """Turn EnergyPlus names into safe SQL identifiers (before reserved/length fixes)."""
    name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
    name = re.sub(r'__+', '_', name)
    return name.strip('_').lower()


def safe_base_name(raw: str) -> str:
    return avoid_reserved(prefix_if_digit_start(sanitize(raw)))


def safe_table_name(raw: str) -> str:
    return shorten_identifier(safe_base_name(raw))


def safe_column_name(raw: str) -> str:
    return shorten_identifier(safe_base_name(raw))


def safe_constraint_name(prefix: str, table: str, col: str | None = None, extra: str = "") -> str:
    base = "_".join(x for x in [prefix, table, col, extra] if x)
    return shorten_identifier(base)


def qident(name: str) -> str:
    """Double-quote identifier safely for constraint SQL strings."""
    return '"' + name.replace('"', '""') + '"'
