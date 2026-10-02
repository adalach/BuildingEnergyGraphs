"""A readable answer to "where am I in the pipeline?".

Reports the database side (server reachable, schemas present, how many tables
and rows) and the artefact side (which generated directories exist), so a new
user can tell which notebook to run next without opening one.
"""

from __future__ import annotations

from sqlalchemy import text

from beg.db.engine import (
    RAW_SCHEMA,
    TRANSFORMED_SCHEMA,
    check_connection,
    database_name,
    get_engine,
    server_version,
)
from beg.settings import output_dir, redacted_dsn

#: Tables whose row counts say most about ingestion progress.
KEY_TABLES = [
    (RAW_SCHEMA, "ep_model"),
    (RAW_SCHEMA, "weather_city"),
]

#: Generated artefacts, in pipeline order, with the notebook that produces them.
ARTEFACT_STAGES = [
    ("pyg_graphs", "BEG_02_Graphs"),
    ("pyg_datasets", "BEG_03_Datasets"),
    ("GNNruns", "BEG_04_Train"),
]


def _schema_table_count(conn, schema: str) -> int | None:
    return conn.execute(
        text(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = :s AND table_type = 'BASE TABLE'"
        ),
        {"s": schema},
    ).scalar()


def status_report(url: str | None = None) -> dict[str, object]:
    """Collect status as plain data. Never raises on a missing database."""
    report: dict[str, object] = {
        "dsn": redacted_dsn(url),
        "database": database_name(url),
    }
    ok, message = check_connection(url)
    report["connected"] = ok
    report["message"] = message

    if ok:
        report["server_version"] = server_version(url)
        with get_engine(url).connect() as conn:
            report["database_size"] = conn.execute(
                text("SELECT pg_size_pretty(pg_database_size(current_database()))")
            ).scalar()

            schemas: dict[str, int | None] = {}
            for schema in (RAW_SCHEMA, TRANSFORMED_SCHEMA):
                schemas[schema] = _schema_table_count(conn, schema)
            report["schemas"] = schemas

            rows: dict[str, int | None] = {}
            for schema, table in KEY_TABLES:
                try:
                    rows[f"{schema}.{table}"] = conn.execute(
                        text(f'SELECT count(*) FROM "{schema}"."{table}"')
                    ).scalar()
                except Exception:  # noqa: BLE001 - an absent table is a normal state here
                    rows[f"{schema}.{table}"] = None
            report["row_counts"] = rows

    out = output_dir()
    report["output_dir"] = str(out)
    artefacts: dict[str, dict[str, object]] = {}
    for name, producer in ARTEFACT_STAGES:
        path = out / name
        artefacts[name] = {
            "producer": producer,
            "exists": path.exists(),
            "entries": len(list(path.iterdir())) if path.is_dir() else 0,
        }
    report["artefacts"] = artefacts
    return report


def _mark(ok: object) -> str:
    return "[ok]" if ok else "[  ]"


def format_status(report: dict[str, object] | None = None, url: str | None = None) -> str:
    """Render :func:`status_report` for a terminal or a notebook cell."""
    r = report if report is not None else status_report(url)
    lines = ["Building Energy Graphs - status", ""]
    lines.append(f"  DSN              {r['dsn']}")
    lines.append(f"  {_mark(r['connected'])} connection    {r['message']}")

    if r.get("connected"):
        lines.append(f"       server       PostgreSQL {r.get('server_version')}")
        lines.append(f"       size         {r.get('database_size')}")
        lines.append("")
        lines.append("  Schemas")
        for schema, count in (r.get("schemas") or {}).items():
            present = count is not None
            suffix = f"{count} tables" if present else "missing (run: beg db init)"
            lines.append(f"  {_mark(present)} {schema:<16} {suffix}")
        lines.append("")
        lines.append("  Ingested")
        for name, count in (r.get("row_counts") or {}).items():
            suffix = f"{count:,} rows" if count is not None else "table not present"
            lines.append(f"  {_mark(count not in (None, 0))} {name:<16} {suffix}")
    else:
        lines.append("")
        lines.append("  Start a server first - see docs/installation.md")
        lines.append("  Quickest route:  docker compose up -d  &&  beg db init")

    lines.append("")
    lines.append(f"  Artefacts in {r['output_dir']}")
    for name, info in (r.get("artefacts") or {}).items():
        suffix = (
            f"{info['entries']} entries" if info["exists"] else f"not built ({info['producer']})"
        )
        lines.append(f"  {_mark(info['exists'])} {name:<16} {suffix}")
    return "\n".join(lines)
