"""
graph_uml.py

Utilities to generate:
1. Mermaid UML (classDiagram) text
2. DBML text
3. dbdiagram.io iframe HTML

from:
- a graph configuration dict `cfg` (see `GraphUML.render_mermaid_text`)
- a live database schema accessible via SQLAlchemy.

Designed for use inside Jupyter notebooks or scripts, e.g.:

    from beg.report.graph_uml import GraphUML, GraphUMLConfig

    builder = GraphUML(
        engine=ENGINE,
        schema=SCHEMA,
        enable_bool_detection=False,  # optional performance tweak
    )

    mermaid_text, dbml_text, iframe_html = builder.build_all(cfg)

To draw the ingested ``ep`` schema itself, use :mod:`beg.report.db_schema_preview`,
which builds the configuration from the live database and calls into this module.
"""

from __future__ import annotations

import base64
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine

# ----------------------------------------------------------------------
# Configuration dataclass
# ----------------------------------------------------------------------


@dataclass
class GraphUMLConfig:
    """
    Optional tuning parameters for GraphUML.

    Attributes
    ----------
    enable_bool_detection : bool
        If True, integer columns are sampled to see if they are 0/1-only and
        then treated as bool. Turn off for maximum speed.
    bool_sample_rows : int
        Number of distinct non-null values to sample per int column when
        detecting boolean-ish columns. Small (e.g. 10–50) is usually enough.
    """

    enable_bool_detection: bool = True
    bool_sample_rows: int = 20


# ----------------------------------------------------------------------
# Main builder class
# ----------------------------------------------------------------------


class GraphUML:
    """
    Main builder class.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to the target database.
    schema : str
        Schema name where the tables live (e.g. "public").
    cfg_options : GraphUMLConfig, optional
        Performance/behavior tuning options. If omitted, default options are used.
    **config_kwargs :
        Convenience kwargs forwarded to GraphUMLConfig, e.g.:

            GraphUML(engine, schema, enable_bool_detection=False)

        You must not pass both cfg_options and config_kwargs.

    Usage
    -----
    builder = GraphUML(engine=ENGINE, schema=SCHEMA)
    mermaid, dbml, iframe = builder.build_all(cfg)
    """

    def __init__(
        self,
        engine: Engine,
        schema: str,
        cfg_options: Optional[GraphUMLConfig] = None,
        **config_kwargs: Any,
    ) -> None:
        self.engine = engine
        self.schema = schema

        # Either a full GraphUMLConfig OR keyword args to construct one
        if cfg_options is not None and config_kwargs:
            raise ValueError("Pass either cfg_options or keyword config options, not both.")

        if cfg_options is not None:
            self.options = cfg_options
        else:
            self.options = GraphUMLConfig(**config_kwargs)

        # Introspection helper (reused / cached inside SQLAlchemy)
        self._insp = inspect(self.engine)

        # Caches for speed (avoid repeated DB introspection)
        self._column_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._pk_cache: Dict[str, List[str]] = {}
        self._boolish_cache: Dict[Tuple[str, str], bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_all(
        self,
        cfg: Dict[str, Any],
        iframe_height: int = 600,
        *,
        max_web_cols: int = 5,
    ) -> Tuple[str, str, str]:
        """
        Generate:
        1. Mermaid UML text
        2. DBML text
        3. dbdiagram.io iframe HTML

        Parameters
        ----------
        cfg : dict
            Graph configuration (see `render_mermaid_text`).
        iframe_height : int
            Height of the generated <iframe> in pixels.
        max_web_cols : int, keyword-only
            Limit columns per class/table in the embedded web preview
            (the returned Mermaid/DBML strings remain full).

        Returns
        -------
        (mermaid_text, dbml_text, iframe_html)
        """
        mermaid = self.render_mermaid_text(cfg)
        # Full DBML for download/copy use
        dbml = self.render_dbml(cfg)
        # Truncated DBML for web preview to avoid excessively wide tables
        iframe_dbml = self.render_dbml(
            cfg,
            max_columns=max_web_cols,
            append_ellipsis=True,
        )
        iframe_html = self.dbdiagram_iframe_html_from_dbml(
            iframe_dbml,
            height=iframe_height,
        )
        return mermaid, dbml, iframe_html

    # ------------------------------------------------------------------
    # Mermaid generation
    # ------------------------------------------------------------------

    def render_mermaid_text(self, cfg: Dict[str, Any]) -> str:
        """
        Render a Mermaid classDiagram snippet from the configuration.

        Keys read from `cfg`:
        - cfg["prefix"]: optional string for top-level heading
        - cfg["description"]: optional description below heading
        - cfg["nodes_categories"], cfg["nodes"]: define nodes/categories
        - cfg["edges"], cfg["edges_fks"], cfg["edges_views"]: define edges

        Returns
        -------
        full_markdown_block : str
            Markdown with a heading + ```mermaid block.
        """
        prefix = cfg.get("prefix", "diagram")
        desc = (cfg.get("description") or "").strip()
        nodes = self._collect_nodes(cfg)
        edges = self._collect_edges(cfg)

        # Ensure edge endpoints appear as nodes, preserving insertion order
        seen = set(nodes)
        for src, tgt, _ in edges:
            for n in (src, tgt):
                if n and n not in seen:
                    seen.add(n)
                    nodes.append(n)

        lines: List[str] = ["```mermaid", "classDiagram"]

        # Classes with attributes
        for n in nodes:
            cls_name = self._sanitize(n)
            cols = self._table_columns(n)
            if cols:
                lines.append(f"    class {cls_name} {{")
                for col_name, col_type in cols:
                    safe_name = self._sanitize(col_name)
                    safe_type = (col_type or "").replace("|", "/")
                    lines.append(f"        {safe_name} : {safe_type}")
                lines.append("    }")
            else:
                # Node with no known columns -> just a bare class
                lines.append(f"    class {cls_name}")

        # Edges: preserve directionality, treat bidir as <-->
        for src, tgt, bidir in edges:
            arrow = "<-->" if bidir else "-->"
            lines.append(
                f"    {self._sanitize(src)} {arrow} {self._sanitize(tgt)}"
            )

        lines.append("```")

        header = [f"# {prefix}"]
        if desc:
            header.append(desc)

        return "\n".join(header + ["\n".join(lines)])

    # ------------------------------------------------------------------
    # DBML generation
    # ------------------------------------------------------------------

    def render_dbml(
        self,
        cfg: Dict[str, Any],
        max_columns: Optional[int] = None,
        append_ellipsis: bool = False,
    ) -> str:
        """
        Render DBML text from the same configuration.

        Includes:
        - Table definitions with types inferred from the DB.
        - Primary keys (using SQLAlchemy's get_pk_constraint).
        - Ref: relationships inferred from cfg["edges"] etc.

        Parameters
        -------
        cfg : dict
            Graph configuration.
        max_columns : int, optional
            If provided, limit each table to this many columns (used for
            web previews to keep the iframe manageable).
        append_ellipsis : bool
            When max_columns truncates columns, append a placeholder column
            carrying a short note so the web preview shows an explicit “…”.

        Returns
        -------
        dbml : str
        """
        nodes = self._collect_nodes(cfg)
        edges = self._collect_edges(cfg)

        # Ensure edge endpoints appear as nodes, as in Mermaid
        seen = set(nodes)
        for src, tgt, _ in edges:
            for n in (src, tgt):
                if n and n not in seen:
                    seen.add(n)
                    nodes.append(n)

        lines: List[str] = []

        # ---- Tables ----
        for n in nodes:
            cols = self._table_columns(n)
            if not cols:
                # No columns detected: skip in DBML
                continue

            # Apply optional truncation for web preview use-cases
            truncated = False
            extra_count = 0
            if max_columns is not None and len(cols) > max_columns:
                extra_count = len(cols) - max_columns
                cols = cols[:max_columns]
                truncated = True

            table_name = self._sanitize(n)
            lines.append(f"Table {table_name} {{")

            pks = set(self._table_primary_keys(n))

            for col_name, col_type in cols:
                col_name_safe = self._sanitize(col_name)
                dbml_type = self._dbml_type(col_type)
                attrs = " [pk]" if col_name in pks else ""
                lines.append(f"  {col_name_safe} {dbml_type}{attrs}")

            # If we truncated, add an ellipsis placeholder row for the iframe view.
            if truncated and append_ellipsis:
                ellipsis_note = "note: '... (+{} more)'".format(extra_count)
                lines.append(f"  __ellipsis__ varchar [{ellipsis_note}]")

            lines.append("}")
            lines.append("")  # blank line between tables

        # ---- Relationships (Refs) ----

        def pick_ref_col(table_name: str) -> Optional[str]:
            """
            Pick a representative column name (unsanitized) for a table.

            Heuristic:
            - Prefer primary key if present
            - Else a column literally named 'id'
            - Else the first column
            """
            cols_local = self._table_columns(table_name)
            if not cols_local:
                return None
            names = [c[0] for c in cols_local]

            # Prefer primary key if we have one
            pks_local = self._table_primary_keys(table_name)
            if pks_local:
                return pks_local[0]

            # Otherwise prefer a column literally named 'id'
            for cand in ("id", "ID"):
                if cand in names:
                    return cand

            # Fallback: first column
            return names[0]

        ref_lines: List[str] = []

        for src, tgt, _bidir in edges:
            if not src or not tgt:
                continue
            if src == tgt:
                # Self-edges are more conceptual; skip in DBML
                continue

            src_col = pick_ref_col(src)
            tgt_col = pick_ref_col(tgt)
            if not src_col or not tgt_col:
                continue

            src_tbl = self._sanitize(src)
            tgt_tbl = self._sanitize(tgt)
            src_col_safe = self._sanitize(src_col)
            tgt_col_safe = self._sanitize(tgt_col)

            # '-' as a neutral, symmetric connection; cardinality is not modelled.
            ref_lines.append(
                f"Ref: {src_tbl}.({src_col_safe}) - {tgt_tbl}.({tgt_col_safe})"
            )

        if ref_lines:
            lines.append("")
            lines.extend(ref_lines)

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # dbdiagram.io iframe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def dbdiagram_embed_url_from_dbml(dbml: str) -> str:
        """
        Convert DBML text into a dbdiagram.io embed URL using the `?c=` parameter.

        Returns
        -------
        url : str
        """
        b64 = base64.b64encode(dbml.encode("utf-8")).decode("ascii")
        param = urllib.parse.quote(b64, safe="")
        return f"https://dbdiagram.io/embed?c={param}"

    @classmethod
    def dbdiagram_iframe_html_from_dbml(
        cls,
        dbml: str,
        height: int = 600,
        width: str = "100%",
    ) -> str:
        """
        Create an <iframe> HTML snippet that embeds a dbdiagram.io diagram
        based on the given DBML.

        Parameters
        ----------
        dbml : str
            DBML diagram definition.
        height : int
            iframe height in pixels.
        width : str
            iframe width (e.g. "100%" or "800").

        Returns
        -------
        iframe_html : str
        """
        src = cls.dbdiagram_embed_url_from_dbml(dbml)
        return (
            f'<iframe src="{src}" width="{width}" height="{height}" '
            f'style="border:0" loading="lazy"></iframe>'
        )

    # ------------------------------------------------------------------
    # Internal helpers: nodes, edges, introspection
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(name: Any) -> str:
        """
        Replace non-alphanumeric/underscore chars with '_' to keep Mermaid/DBML happy.
        """
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(name or ""))

    @staticmethod
    def _collect_nodes(cfg: Dict[str, Any]) -> List[str]:
        """
        Collect node identifiers from cfg["nodes_categories"] and cfg["nodes"].
        Uses an order-preserving set-like behavior.
        """
        nodes: List[str] = []
        seen = set()

        def add(n: Optional[str]) -> None:
            if not n or n in seen:
                return
            seen.add(n)
            nodes.append(n)

        # Categories
        for n in (cfg.get("nodes_categories") or {}).keys():
            add(n)

        # Explicit nodes
        for node in cfg.get("nodes") or []:
            if isinstance(node, str):
                add(node)
            elif isinstance(node, dict):
                if "table" in node:
                    add(node["table"])
                elif "name" in node:
                    add(node["name"])
                elif len(node) == 1:
                    # Single-key dict -> use the key
                    add(next(iter(node.keys())))

        return nodes

    @staticmethod
    def _collect_edges(cfg: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
        """
        Collect edges as (src, tgt, bidirectional) triples from cfg.
        Supports:
        - cfg["edges"]
        - cfg["edges_fks"]
        - cfg["edges_views"]
        """
        edges: List[Tuple[str, str, bool]] = []
        seen = set()

        def add(src: Optional[str], tgt: Optional[str], bidir: bool) -> None:
            if not src or not tgt:
                return
            key = (src, tgt, bool(bidir))
            if key in seen:
                return
            seen.add(key)
            edges.append(key)

        # Explicit edges list
        for e in cfg.get("edges") or []:
            add(
                e.get("src_cat") or e.get("src") or e.get("source_category"),
                e.get("tgt_cat") or e.get("tgt") or e.get("target_category"),
                e.get("bidirectional", False),
            )

        # FK-based edges
        for e in cfg.get("edges_fks") or []:
            add(e.get("src"), e.get("tgt"), e.get("bidirectional", False))

        # View-based edges
        for e in (cfg.get("edges_views") or {}).values():
            add(
                e.get("src_cat") or e.get("src"),
                e.get("tgt_cat") or e.get("tgt"),
                e.get("bidirectional", False),
            )

        return edges

    # ------------------------------------------------------------------
    # Introspection: columns, PKs, bool detection, type mapping
    # ------------------------------------------------------------------

    def _table_columns(self, table_name: str) -> List[Tuple[str, str]]:
        """
        Return a list of (column_name, friendly_type) for a given table.

        Results are cached per table for speed.
        """
        if table_name in self._column_cache:
            return self._column_cache[table_name]

        cols: List[Tuple[str, str]] = []
        try:
            for col in self._insp.get_columns(table_name, schema=self.schema):
                raw_type = str(col.get("type") or "")
                name = col.get("name")
                is_bool = False

                # Optional: detect "integer but only 0/1" as bool
                if (
                    self.options.enable_bool_detection
                    and "int" in raw_type.lower()
                    and name is not None
                ):
                    is_bool = self._is_booleanish(table_name, name)

                friendly = self._friendly_type(raw_type, is_bool)
                cols.append((name, friendly))
        except Exception:
            # If anything goes wrong, we just treat as no columns
            cols = []

        self._column_cache[table_name] = cols
        return cols

    def _table_primary_keys(self, table_name: str) -> List[str]:
        """
        Return a list of primary key column names for a given table.

        Cached per table for speed.
        """
        if table_name in self._pk_cache:
            return self._pk_cache[table_name]
        try:
            pk_info = self._insp.get_pk_constraint(table_name, schema=self.schema) or {}
            cols = pk_info.get("constrained_columns") or []
        except Exception:
            cols = []
        self._pk_cache[table_name] = cols
        return cols

    def _is_booleanish(self, table_name: str, col_name: str) -> bool:
        """
        Heuristic: check if an integer column only contains values {0,1}.

        Uses a limited sample of distinct values for speed, controlled by
        GraphUMLConfig.bool_sample_rows. Results are cached per (table, column).
        """
        key = (table_name, col_name)
        if key in self._boolish_cache:
            return self._boolish_cache[key]

        if self.options.bool_sample_rows <= 0:
            self._boolish_cache[key] = False
            return False

        try:
            sql = sa_text(
                f'SELECT DISTINCT "{col_name}" '
                f'FROM "{self.schema}"."{table_name}" '
                f'WHERE "{col_name}" IS NOT NULL '
                f'LIMIT :n'
            )
            with self.engine.connect() as con:
                values = {row[0] for row in con.execute(
                    sql, {"n": self.options.bool_sample_rows}
                )}
            result = bool(values) and values.issubset({0, 1})
        except Exception:
            result = False

        self._boolish_cache[key] = result
        return result

    @staticmethod
    def _friendly_type(raw_type: str, is_bool: bool = False) -> str:
        """
        Map DB-specific column types to simple, language-agnostic types
        for Mermaid (and as an intermediate for DBML).
        """
        if is_bool:
            return "bool"
        t = (raw_type or "").lower()
        if "bool" in t:
            return "bool"
        if any(k in t for k in ["char", "text", "json"]):
            return "str"
        if any(k in t for k in ["int", "serial"]):
            return "int"
        if any(k in t for k in ["double", "numeric", "real", "float", "decimal"]):
            return "float"
        if "timestamp" in t or "date" in t:
            return "timestamp"
        return raw_type or ""

    @staticmethod
    def _dbml_type(typ: str) -> str:
        """
        Map the simplified types used in Mermaid/inspection to DBML types.
        """
        t = (typ or "").lower()
        if t == "int":
            return "int"
        if t == "float":
            return "float"
        if t == "bool":
            return "boolean"
        if t == "timestamp":
            return "timestamp"
        if t == "str":
            return "varchar"
        # Fallback
        return "varchar"
