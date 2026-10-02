"""Engine construction. The only place in the project that builds a DSN."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from beg.settings import dsn as configured_dsn

#: Schema holding the raw ingested epJSON objects and the weather tables.
RAW_SCHEMA = "ep"

#: Schema holding the node and edge tables produced by the SQL transforms.
TRANSFORMED_SCHEMA = "ep_transformed"


@lru_cache(maxsize=4)
def _cached_engine(url: str, echo: bool) -> Engine:
    return create_engine(url, echo=echo, pool_pre_ping=True, future=True)


def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Return a SQLAlchemy engine for the configured database.

    Parameters
    ----------
    url:
        Optional explicit DSN. Omit it in notebooks and scripts so that the
        single ``BEG_DSN`` setting stays authoritative; it exists for tests that
        need to point at a throwaway database.
    echo:
        Log every statement. Useful when debugging a transform.
    """
    return _cached_engine(url or configured_dsn(), echo)


def maintenance_engine(url: str | None = None) -> Engine:
    """Engine bound to the server's ``postgres`` database.

    Needed to create or drop the project database itself, which cannot be done
    from a connection to that same database.
    """
    target = make_url(url or configured_dsn()).set(database="postgres")
    return create_engine(target, isolation_level="AUTOCOMMIT", future=True)


def database_name(url: str | None = None) -> str:
    return make_url(url or configured_dsn()).database or ""


def check_connection(url: str | None = None) -> tuple[bool, str]:
    """Try to connect. Returns ``(ok, message)`` and never raises."""
    try:
        with get_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "connected"
    except Exception as exc:  # noqa: BLE001 - the message is the product here
        return False, f"{type(exc).__name__}: {exc}".split("\n")[0]


def server_version(url: str | None = None) -> str | None:
    try:
        with get_engine(url).connect() as conn:
            return conn.execute(text("SHOW server_version")).scalar_one()
    except Exception:  # noqa: BLE001
        return None
