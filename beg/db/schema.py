"""Creating the project database and its two schemas, idempotently."""

from __future__ import annotations

from sqlalchemy import text

from beg.db.engine import (
    RAW_SCHEMA,
    TRANSFORMED_SCHEMA,
    database_name,
    get_engine,
    maintenance_engine,
)


def database_exists(url: str | None = None) -> bool:
    name = database_name(url)
    with maintenance_engine(url).connect() as conn:
        found = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
        ).scalar()
    return bool(found)


def ensure_database(url: str | None = None) -> bool:
    """Create the project database if it does not exist. Returns True if created."""
    if database_exists(url):
        return False
    name = database_name(url)
    with maintenance_engine(url).connect() as conn:
        # Identifier cannot be bound as a parameter; the name comes from the
        # operator's own DSN, and is quoted here. UTF8 whatever the server's
        # default: the SQL transforms carry non-ASCII characters.
        conn.execute(text(f'CREATE DATABASE "{name}" ENCODING \'UTF8\' TEMPLATE template0'))
    return True


def create_schemas(url: str | None = None) -> list[str]:
    """Create ``ep`` and ``ep_transformed`` if missing. Returns the schemas created."""
    created: list[str] = []
    engine = get_engine(url)
    with engine.begin() as conn:
        for schema in (RAW_SCHEMA, TRANSFORMED_SCHEMA):
            exists = conn.execute(
                text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :s"),
                {"s": schema},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE SCHEMA "{schema}"'))
                created.append(schema)
    return created


def init_database(url: str | None = None) -> dict[str, object]:
    """Bring an empty server to the state the pipeline expects.

    Creates the database and both schemas. Safe to run repeatedly; it never
    drops anything.
    """
    result: dict[str, object] = {"database": database_name(url)}
    result["database_created"] = ensure_database(url)
    result["schemas_created"] = create_schemas(url)
    return result


def drop_schemas(url: str | None = None, confirm: bool = False) -> None:
    """Drop both schemas and everything in them. Requires ``confirm=True``.

    This deletes every ingested model and every transform output. The raw
    epJSON files are untouched, so the state is rebuildable by re-running
    notebooks 01 and 02.
    """
    if not confirm:
        raise ValueError("refusing to drop schemas without confirm=True")
    with get_engine(url).begin() as conn:
        for schema in (TRANSFORMED_SCHEMA, RAW_SCHEMA):
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
