"""A browsable picture of the ``ep`` schema as it was actually ingested.

The schema is derived from the model set at run time, so the only authoritative
description of it is the live database. This module reads it back and renders
it, with :mod:`beg.report.graph_uml` supplying the column introspection and the
DBML text.

Two levels of detail:

* :func:`schema_overview` — a DataFrame of tables, column counts and row counts.
  Always available, no browser needed.
* :func:`write_schema_preview` — a single self-contained HTML page with an entity
  diagram, rendered client-side by Mermaid.

The full ``ep`` schema runs to roughly a hundred tables, which is unreadable as
one diagram, so the preview takes a ``tables`` argument and defaults to the most
connected tables.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text as sa_text

#: Mermaid is loaded from a CDN, so the generated page needs network on first
#: open. Rendering happens in the reader's browser.
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs"


def schema_overview(engine, schema: str) -> pd.DataFrame:
    """One row per table: column count and live row count, largest first."""
    sql = """
        SELECT c.relname AS table_name,
               CASE c.relkind WHEN 'r' THEN 'table' WHEN 'v' THEN 'view'
                              WHEN 'm' THEN 'matview' ELSE c.relkind::text END AS kind,
               (SELECT COUNT(*) FROM information_schema.columns ic
                 WHERE ic.table_schema = n.nspname AND ic.table_name = c.relname) AS columns,
               COALESCE(c.reltuples, 0)::bigint AS approx_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relkind IN ('r', 'v', 'm')
        ORDER BY approx_rows DESC, c.relname
    """
    with engine.connect() as con:
        rows = con.execute(sa_text(sql), {"schema": schema}).fetchall()
    return pd.DataFrame(rows, columns=["table_name", "kind", "columns", "approx_rows"])


def foreign_key_edges(engine, schema: str) -> list[dict]:
    """Every foreign key in the schema, as ``{src, tgt, src_col}`` dicts.

    The label-based keys are composite — ``(model_id, <reference column>)`` — so
    ``src_col`` names the reference column and drops ``model_id``, which leads every
    key; a key on ``model_id`` alone keeps it. Duplicate ``(src, tgt, src_col)``
    keys appear once.
    """
    sql = """
        SELECT src.relname AS src_table,
               tgt.relname AS tgt_table,
               array_agg(att.attname ORDER BY k.ord) AS src_columns
        FROM pg_constraint con
        JOIN pg_class src ON src.oid = con.conrelid
        JOIN pg_class tgt ON tgt.oid = con.confrelid
        JOIN pg_namespace n ON n.oid = src.relnamespace
        JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
        JOIN pg_attribute att ON att.attrelid = src.oid AND att.attnum = k.attnum
        WHERE con.contype = 'f' AND n.nspname = :schema
        GROUP BY con.oid, src.relname, tgt.relname
        ORDER BY src.relname, tgt.relname
    """
    with engine.connect() as con:
        rows = con.execute(sa_text(sql), {"schema": schema}).fetchall()

    edges, seen = [], set()
    for src_table, tgt_table, columns in rows:
        informative = [c for c in columns if c != "model_id"] or list(columns)
        label = ", ".join(informative)
        key = (src_table, tgt_table, label)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"src": src_table, "tgt": tgt_table, "src_col": label})
    return edges


def most_connected_tables(engine, schema: str, top_n: int = 20) -> list[str]:
    """The ``top_n`` tables with the most foreign keys attached, busiest first."""
    edges = foreign_key_edges(engine, schema)
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["src"]] = degree.get(edge["src"], 0) + 1
        degree[edge["tgt"]] = degree.get(edge["tgt"], 0) + 1

    overview = schema_overview(engine, schema).set_index("table_name")
    rows = overview["approx_rows"].to_dict()

    ranked = sorted(degree, key=lambda t: (-degree[t], -rows.get(t, 0), t))
    if len(ranked) >= top_n:
        return ranked[:top_n]

    # Pad with the largest unconnected tables so the page is not near-empty
    extra = [t for t in overview.index if t not in set(ranked)]
    return ranked + extra[: top_n - len(ranked)]


def schema_cfg(engine, schema: str, tables: list[str] | None = None) -> dict:
    """Build a ``graph_uml`` configuration describing the live schema.

    ``graph_uml`` expects the node and edge sets a graph configuration would
    supply; here they come from the database itself — tables as nodes, real
    foreign keys as edges.
    """
    overview = schema_overview(engine, schema)
    names = tables if tables is not None else overview["table_name"].tolist()
    keep = set(names)

    edges = [e for e in foreign_key_edges(engine, schema) if e["src"] in keep and e["tgt"] in keep]
    return {
        "prefix": f"{schema} schema",
        "description": f"{len(names)} tables, {len(edges)} foreign keys",
        "nodes": list(names),
        "edges_fks": edges,
    }


def _mermaid_er(engine, schema: str, cfg: dict, max_columns: int) -> str:
    """Render a Mermaid ``erDiagram`` for the chosen tables.

    Column and key introspection comes from ``GraphUML``; the ER text is assembled here.
    """
    from beg.report.graph_uml import GraphUML

    builder = GraphUML(engine=engine, schema=schema, enable_bool_detection=False)

    lines = ["erDiagram"]
    for table in cfg["nodes"]:
        cols = builder._table_columns(table)
        if not cols:
            continue
        pks = set(builder._table_primary_keys(table))
        shown = cols[:max_columns]

        lines.append(f"    {builder._sanitize(table)} {{")
        for col_name, col_type in shown:
            key = " PK" if col_name in pks else ""
            lines.append(
                f"        {builder._dbml_type(col_type)} {builder._sanitize(col_name)}{key}"
            )
        if len(cols) > len(shown):
            lines.append(f"        more plus_{len(cols) - len(shown)}_columns")
        lines.append("    }")

    for edge in cfg["edges_fks"]:
        src = builder._sanitize(edge["src"])
        tgt = builder._sanitize(edge["tgt"])
        label = builder._sanitize(edge.get("src_col") or "references")
        lines.append(f'    {tgt} ||--o{{ {src} : "{label}"')

    return "\n".join(lines)


def write_schema_preview(
    engine,
    schema: str,
    path: Path,
    tables: list[str] | None = None,
    top_n: int = 20,
    max_columns: int = 10,
) -> Path:
    """Write a self-contained HTML page showing the schema as linked blocks.

    With no explicit ``tables``, the ``top_n`` most connected tables are shown. The
    page also carries the DBML text, which can be pasted into dbdiagram.io for an
    interactive view.

    Returns the path written.
    """
    overview = schema_overview(engine, schema)
    if tables is None:
        tables = most_connected_tables(engine, schema, top_n)

    cfg = schema_cfg(engine, schema, tables)
    mermaid = _mermaid_er(engine, schema, cfg, max_columns)

    from beg.report.graph_uml import GraphUML

    builder = GraphUML(engine=engine, schema=schema, enable_bool_detection=False)
    dbml = builder.render_dbml(cfg, max_columns=max_columns, append_ellipsis=True)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(schema)} schema</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         margin: 0 auto; max-width: 1200px; padding: 2rem 1rem; line-height: 1.5; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
  p.sub {{ color: #666; margin-top: 0; }}
  .diagram {{ overflow-x: auto; border: 1px solid #ddd; border-radius: 6px; padding: 1rem;
              background: #fff; }}
  details {{ margin-top: 1.5rem; }}
  summary {{ cursor: pointer; font-weight: 600; }}
  pre {{ background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow-x: auto;
         font-size: 0.8rem; }}
  table {{ border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.3rem 0.6rem; text-align: left; }}
  th {{ background: #f6f8fa; }}
</style>
</head>
<body>
<h1>{html.escape(schema)} schema</h1>
<p class="sub">{html.escape(cfg["description"])} &middot;
showing {len(tables)} of {len(overview)} relations, up to {max_columns} columns each.</p>

<div class="diagram"><pre class="mermaid">{html.escape(mermaid)}</pre></div>

<details>
<summary>All relations in this schema</summary>
{overview.to_html(index=False, border=0)}
</details>

<details>
<summary>DBML — paste into dbdiagram.io for an interactive view</summary>
<pre>{html.escape(dbml)}</pre>
</details>

<script type="module">
  import mermaid from "{MERMAID_CDN}";
  mermaid.initialize({json.dumps({"startOnLoad": True, "theme": "neutral",
                                  "er": {"layoutDirection": "LR"}})});
</script>
</body>
</html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
