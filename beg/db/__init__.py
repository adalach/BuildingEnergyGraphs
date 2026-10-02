"""Database access for beg.

One connection setting (``BEG_DSN``), one engine factory, two schemas:

``ep``
    Raw ingested epJSON objects plus the weather tables, one table per
    EnergyPlus object type used by the model set.
``ep_transformed``
    Node and edge tables produced by the SQL transforms in ``beg/sql``.
"""

from beg.db.engine import (
    RAW_SCHEMA,
    TRANSFORMED_SCHEMA,
    check_connection,
    get_engine,
    server_version,
)
from beg.db.schema import create_schemas, database_exists, ensure_database, init_database
from beg.db.status import format_status, status_report

__all__ = [
    "RAW_SCHEMA",
    "TRANSFORMED_SCHEMA",
    "get_engine",
    "check_connection",
    "server_version",
    "ensure_database",
    "database_exists",
    "create_schemas",
    "init_database",
    "status_report",
    "format_status",
]
