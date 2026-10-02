"""``beg`` command line entry point.

    beg db init        create the database and its schemas
    beg db status      what exists so far, in the database and on disk
    beg db url         the configured DSN, with the password redacted
    beg doctor         check the environment before running a notebook
"""

from __future__ import annotations

import argparse
import importlib
import sys


def _cmd_db_init(args: argparse.Namespace) -> int:
    from beg.db import format_status, init_database

    result = init_database()
    created_db = "created" if result["database_created"] else "already present"
    print(f"database {result['database']}: {created_db}")
    schemas = result["schemas_created"]
    print(f"schemas created: {', '.join(schemas) if schemas else 'none needed'}")
    print()
    print(format_status())
    return 0


def _cmd_db_status(args: argparse.Namespace) -> int:
    from beg.db import format_status

    print(format_status())
    return 0


def _cmd_db_url(args: argparse.Namespace) -> int:
    from beg.settings import redacted_dsn

    if not getattr(args, "parts", False):
        print(redacted_dsn())
        return 0

    from sqlalchemy.engine import make_url

    from beg.db import RAW_SCHEMA, TRANSFORMED_SCHEMA
    from beg.settings import DSN_ENV_VAR, dsn

    url = make_url(dsn())
    rows = [
        ("Host", url.host or "127.0.0.1"),
        ("Port", str(url.port or 5432)),
        ("Database", url.database or ""),
        ("Username", url.username or ""),
        ("Password", "the POSTGRES_PASSWORD line in your .env file"),
        ("Schemas", f"{RAW_SCHEMA}, {TRANSFORMED_SCHEMA}"),
        ("Driver", "PostgreSQL"),
    ]
    width = max(len(label) for label, _ in rows)
    print("Connection settings for a GUI client such as DBeaver, pgAdmin or DataGrip:\n")
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")
    print(
        f"\nThese come from {DSN_ENV_VAR}. The password is never printed; it is the one you"
        "\nset in .env, which is untracked. See docs/database.md."
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from beg.db import check_connection
    from beg.settings import data_dir, dsn, output_dir, repo_root

    ok = True
    print(f"python           {sys.version.split()[0]}")
    print(f"repository       {repo_root()}")
    print()

    from beg.requirements import NOTEBOOK_NEEDS, REQUIREMENTS, install_command, missing

    absent = {r.module for r in missing()}
    for extra in ("core", "train", "viz", "notebooks"):
        print(f"  {extra}")
        for r in (r for r in REQUIREMENTS if r.extra == extra):
            if r.module in absent:
                print(f"    [  ] {r.distribution:<18} missing - needed for {r.purpose}")
            else:
                version = getattr(importlib.import_module(r.module), "__version__", "")
                print(f"    [ok] {r.distribution:<18} {version}")
    ok = not missing([r.module for r in REQUIREMENTS if r.extra == "core"])

    print()
    for notebook, needs in NOTEBOOK_NEEDS.items():
        gaps = missing(needs)
        fix = f"   {install_command(notebook, [r.module for r in gaps])}" if gaps else ""
        print(f"  [{'ok' if not gaps else '  '}] {notebook:<16}{fix}")
    print()

    configured = dsn(required=False)
    if not configured:
        print("  [  ] BEG_DSN          unset (cp .env.example .env)")
        ok = False
    else:
        connected, message = check_connection()
        print(f"  [{'ok' if connected else '  '}] database         {message}")
        ok = ok and connected

    for label, path in [("data dir", data_dir()), ("output dir", output_dir())]:
        print(f"  [{'ok' if path.exists() else '  '}] {label:<16} {path}")

    print()
    print("ready" if ok else "not ready - see docs/installation.md")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database setup and inspection")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("init", help="create the database and schemas").set_defaults(func=_cmd_db_init)
    db_sub.add_parser("status", help="report database and artefact state").set_defaults(
        func=_cmd_db_status
    )
    p_url = db_sub.add_parser("url", help="print the configured DSN, redacted")
    p_url.add_argument(
        "--parts",
        action="store_true",
        help="print host, port, database and user separately, for a GUI client such as DBeaver",
    )
    p_url.set_defaults(func=_cmd_db_url)

    sub.add_parser("doctor", help="check the environment").set_defaults(func=_cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
