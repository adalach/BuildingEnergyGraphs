"""
Shared HTML renderer for interactive NetworkX graph previews.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.parse
from collections import Counter
from typing import Any, Dict, Mapping, Optional

import networkx as nx


def _viridis_hex(n: int):
    try:
        import numpy as np
        from matplotlib import cm
        from matplotlib import colors as mcolors

        if n <= 0:
            return []

        vir = cm.get_cmap("viridis")(np.linspace(0, 1, 256))
        pla = cm.get_cmap("plasma")(np.linspace(0, 1, 256))
        pla_rev = pla[::-1]

        vir_y = int(np.argmax(vir[:, 1]))
        pla_rev_y = int(np.argmax(pla_rev[:, 1]))

        joined = np.vstack([vir[: vir_y + 1], pla_rev[pla_rev_y:]])
        cmap = cm.colors.LinearSegmentedColormap.from_list(
            "viridis_to_revplasma_at_yellow", joined
        )

        return [mcolors.to_hex(c) for c in cmap(np.linspace(0, 1, n))]
    except Exception:
        def hsl(i, count):
            h = (i / max(count, 1)) * 360.0
            return f"hsl({h:.0f},65%,50%)"

        return [hsl(i, n) for i in range(n)]


def _sanitize(val: Any) -> Any:
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    return str(val)


def _json_safe(val: Any) -> Any:
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(v) for v in val]
    return str(val)


UML_EXCLUDED_NODE_KEYS = {
    "id",
    "label",
    "name",
    "instance_name",
    "node_type",
    "type",
    "category",
    "X",
    "Y",
    "x_dim",
    "y_dim",
}
MAX_DBML_EMBED_URL_LENGTH = 6000


def _node_type(d: dict) -> str:
    return str(d.get("node_type") or d.get("type") or "unknown")


def _node_label(n: Any, d: dict) -> str:
    return str(d.get("instance_name") or d.get("name") or n)


def _edge_type(d: dict) -> str:
    return str(d.get("edge_type") or d.get("group") or d.get("rel") or "relation")


def _is_graph(x: Any) -> bool:
    return isinstance(x, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))


def _extract_mermaid_code(mermaid_text: str) -> str:
    if not mermaid_text:
        return ""

    lines = mermaid_text.splitlines()
    block_lines = []
    in_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```mermaid"):
            in_block = True
            continue
        if in_block and stripped.startswith("```"):
            break
        if in_block:
            block_lines.append(line.rstrip())

    if block_lines:
        return "\n".join(block_lines).strip()

    idx = mermaid_text.find("classDiagram")
    if idx >= 0:
        return mermaid_text[idx:].strip()

    return mermaid_text.strip()


def _safe_diagram_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(name or "")).strip("_")
    if not cleaned:
        cleaned = "node_type"
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


def _unique_diagram_names(names: list[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    used: set[str] = set()

    for name in names:
        base = _safe_diagram_name(name)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        mapping[name] = candidate

    return mapping


def _dbdiagram_embed_url_from_dbml(dbml: str) -> str:
    b64 = base64.b64encode(dbml.encode("utf-8")).decode("ascii")
    param = urllib.parse.quote(b64, safe="")
    return f"https://dbdiagram.io/embed?c={param}"


def _dbdiagram_iframe_html_from_dbml(
    dbml: str,
    *,
    height: int = 600,
    width: str = "100%",
) -> str:
    src = _dbdiagram_embed_url_from_dbml(dbml)
    if len(src) > MAX_DBML_EMBED_URL_LENGTH:
        return ""
    return (
        f'<iframe src="{src}" width="{width}" height="{height}" '
        f'style="border:0" loading="lazy"></iframe>'
    )


def _normalize_uml_cfgs(uml_cfgs: Any) -> Dict[str, Dict[str, Any]]:
    if not uml_cfgs:
        return {}

    if isinstance(uml_cfgs, Mapping):
        out = {}
        for key, value in uml_cfgs.items():
            if isinstance(value, Mapping):
                out[str(key)] = dict(value)
        return out

    out = {}
    for cfg in uml_cfgs:
        if isinstance(cfg, Mapping):
            prefix = cfg.get("prefix")
            if prefix:
                out[str(prefix)] = dict(cfg)
    return out


def _merge_graphs_for_uml(models: Mapping[str, nx.Graph]) -> nx.MultiDiGraph:
    merged = nx.MultiDiGraph()

    for model_name, G in models.items():
        node_map: Dict[Any, str] = {}

        for node_id, attrs in G.nodes(data=True):
            merged_id = f"{model_name}::{node_id}"
            node_map[node_id] = merged_id
            merged.add_node(
                merged_id,
                node_type=_node_type(attrs),
                **{
                    str(key): _sanitize(value)
                    for key, value in attrs.items()
                    if str(key) not in {"node_type", "type"}
                },
            )

        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            for idx, (u, v, _key, attrs) in enumerate(G.edges(keys=True, data=True)):
                if u not in node_map or v not in node_map:
                    continue
                merged.add_edge(
                    node_map[u],
                    node_map[v],
                    key=f"{model_name}::{idx}",
                    edge_type=_edge_type(attrs),
                )
        else:
            for idx, (u, v, attrs) in enumerate(G.edges(data=True)):
                if u not in node_map or v not in node_map:
                    continue
                merged.add_edge(
                    node_map[u],
                    node_map[v],
                    key=f"{model_name}::{idx}",
                    edge_type=_edge_type(attrs),
                )

    return merged


def _graph_uml_stats(
    G: nx.MultiDiGraph,
) -> tuple[Counter[str], Counter[tuple[str, str, str]]]:
    node_counts: Counter[str] = Counter(
        str(attrs.get("node_type", "unknown"))
        for _, attrs in G.nodes(data=True)
    )
    edge_counts: Counter[tuple[str, str, str]] = Counter()

    for u, v, attrs in G.edges(data=True):
        src_type = str(G.nodes[u].get("node_type", "unknown"))
        tgt_type = str(G.nodes[v].get("node_type", "unknown"))
        e_type = str(attrs.get("edge_type", "edge"))
        edge_counts[(src_type, tgt_type, e_type)] += 1

    return node_counts, edge_counts


def _graph_uml_text_summary(G: nx.MultiDiGraph, cfg_name: str, cfg_desc: str) -> str:
    node_counts, edge_counts = _graph_uml_stats(G)
    node_schema = _graph_uml_node_schema(G)

    lines = [cfg_name]
    if cfg_desc:
        lines.extend(["", cfg_desc])

    lines.extend(["", "Node types"])
    for node_type in sorted(node_counts):
        lines.append(f"- {node_type} ({node_counts[node_type]})")
        schema_lines = node_schema.get(node_type, [])
        if schema_lines:
            for prop_name, prop_type in schema_lines:
                lines.append(f"  - {prop_name}: {prop_type}")
        else:
            lines.append("  - no exported properties")

    lines.extend(["", "Edge links"])
    if edge_counts:
        for src_type, tgt_type, e_type in sorted(edge_counts):
            lines.append(
                f"- {src_type} -> {tgt_type} ({e_type}): {edge_counts[(src_type, tgt_type, e_type)]}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines)


def _graph_uml_dbml_text(G: nx.MultiDiGraph, cfg_name: str) -> str:
    node_counts, edge_counts = _graph_uml_stats(G)
    if not node_counts:
        return ""

    node_schema = _graph_uml_node_schema(G)
    sorted_types = sorted(node_counts)
    table_names = _unique_diagram_names([f"{node_type}_{node_counts[node_type]}" for node_type in sorted_types])
    lines: list[str] = [f"// {cfg_name} property-first DBML"]

    for node_type, table_key in zip(sorted_types, [f"{node_type}_{node_counts[node_type]}" for node_type in sorted_types]):
        table_name = table_names[table_key]
        lines.extend(
            [
                "",
                f"// {node_type} ({node_counts[node_type]})",
                f"Table {table_name} {{",
                "  id int [pk]",
            ]
        )
        for prop_name, prop_type in node_schema.get(node_type, []):
            lines.append(f"  {prop_name} {_dbml_type_name(prop_type)}")
        lines.append("}")

    ref_pairs = {
        tuple(sorted((src_type, tgt_type), key=str))
        for src_type, tgt_type, _edge_type in edge_counts.keys()
    }
    if ref_pairs:
        lines.append("")
        for src_type, tgt_type in sorted(ref_pairs):
            lines.append(
                f"Ref: {table_names[f'{src_type}_{node_counts[src_type]}']}.(id) - {table_names[f'{tgt_type}_{node_counts[tgt_type]}']}.(id)"
            )

    return "\n".join(lines).strip()


def _graph_uml_mermaid_text(G: nx.MultiDiGraph) -> str:
    node_counts, edge_counts = _graph_uml_stats(G)
    if not node_counts:
        return ""

    node_schema = _graph_uml_node_schema(G)
    display_names = {
        node_type: f"{node_type} ({node_counts[node_type]})"
        for node_type in sorted(node_counts)
    }
    class_names = {
        node_type: _safe_diagram_name(display_names[node_type])
        for node_type in sorted(node_counts)
    }

    lines = ["classDiagram"]
    for node_type in sorted(node_counts):
        class_name = class_names[node_type]
        lines.append(f'    class {class_name}["{display_names[node_type]}"] {{')
        schema_lines = node_schema.get(node_type, [])
        if schema_lines:
            for prop_name, prop_type in schema_lines:
                lines.append(f"        {prop_name} : {prop_type}")
        else:
            lines.append("        ")
        lines.append("    }")

    for src_type, tgt_type, e_type in sorted(edge_counts):
        lines.append(
            f"    {class_names[src_type]} --> {class_names[tgt_type]} : {_safe_diagram_name(e_type)}"
        )
    return "\n".join(lines)


def _graph_uml_node_schema(G: nx.MultiDiGraph) -> Dict[str, list[tuple[str, str]]]:
    schema_map: Dict[str, Dict[str, set[str]]] = {}
    for _, attrs in G.nodes(data=True):
        node_type = str(attrs.get("node_type", "unknown"))
        prop_types = schema_map.setdefault(node_type, {})
        for key, value in attrs.items():
            key_str = str(key)
            if key_str in UML_EXCLUDED_NODE_KEYS:
                continue
            prop_types.setdefault(key_str, set()).add(_infer_schema_type(value))

    out: Dict[str, list[tuple[str, str]]] = {}
    for node_type, props in schema_map.items():
        out[node_type] = [
            (prop_name, _merge_schema_types(type_names))
            for prop_name, type_names in sorted(props.items())
        ]
    return out


def _infer_schema_type(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _merge_schema_types(type_names: set[str]) -> str:
    if not type_names:
        return "str"
    if type_names == {"unknown"}:
        return "str"
    if "str" in type_names:
        return "str"
    numeric = type_names - {"unknown"}
    if numeric <= {"int"}:
        return "int"
    if numeric <= {"int", "float"}:
        return "float"
    if numeric <= {"bool"}:
        return "bool"
    if numeric <= {"bool", "int"}:
        return "int"
    if numeric <= {"bool", "int", "float"}:
        return "float"
    return "str"


def _dbml_type_name(schema_type: str) -> str:
    if schema_type == "bool":
        return "boolean"
    if schema_type == "str":
        return "text"
    if schema_type == "unknown":
        return "text"
    return schema_type


def _build_default_uml_payload(
    cfg_name: str,
    cfg_desc: str,
    models: Mapping[str, nx.Graph],
) -> Dict[str, Any]:
    merged = _merge_graphs_for_uml(models)
    if merged.number_of_nodes() == 0:
        return {}

    mermaid_code = _graph_uml_mermaid_text(merged)
    if not mermaid_code:
        return {}

    dbml_text = _graph_uml_dbml_text(merged, cfg_name)
    iframe_html = (
        _dbdiagram_iframe_html_from_dbml(dbml_text)
        if dbml_text
        else ""
    )

    return {
        "mermaid": mermaid_code,
        "text": _graph_uml_text_summary(merged, cfg_name, cfg_desc),
        "dbml": dbml_text,
        "iframe_html": iframe_html,
    }


def _build_uml_payloads(
    cfg_map: Mapping[str, Mapping[str, nx.Graph]],
    cfg_subtitles: Mapping[str, str],
    uml_builder: Any,
    uml_cfgs: Any,
    uml_iframe_height: int,
) -> Dict[str, Dict[str, Any]]:
    uml_cfg_map = _normalize_uml_cfgs(uml_cfgs) if uml_builder is not None else {}
    payloads: Dict[str, Dict[str, Any]] = {}

    for cfg_name, models in cfg_map.items():
        cfg_desc = cfg_subtitles.get(cfg_name, "")
        payload: Dict[str, Any] = {}
        merged = _merge_graphs_for_uml(models)
        summary_text = (
            _graph_uml_text_summary(merged, cfg_name, cfg_desc)
            if merged.number_of_nodes() > 0
            else ""
        )

        explicit_cfg = uml_cfg_map.get(cfg_name)
        if uml_builder is not None and explicit_cfg is not None:
            try:
                mermaid_text, dbml_text, iframe_html = uml_builder.build_all(
                    explicit_cfg,
                    iframe_height=uml_iframe_height,
                )
                mermaid_code = _extract_mermaid_code(mermaid_text)
                dbml_clean = (dbml_text or "").strip()
                iframe_clean = (iframe_html or "").strip()
                if mermaid_code or dbml_clean or iframe_clean:
                    payload = {
                        "mermaid": mermaid_code,
                        "text": summary_text or dbml_clean,
                        "dbml": dbml_clean,
                        "iframe_html": (
                            iframe_clean
                            or (
                                _dbdiagram_iframe_html_from_dbml(
                                    dbml_clean,
                                    height=uml_iframe_height,
                                )
                                if dbml_clean
                                else ""
                            )
                        ),
                    }
            except Exception:
                payload = {}

        if not payload:
            payload = _build_default_uml_payload(cfg_name, cfg_desc, models)

        payloads[cfg_name] = payload

    return payloads


def build_pyvis_html(G: nx.Graph, out_html: str = "graph.html", **kwargs) -> str:
    """
    Single-graph renderer.
    Delegates to build_pyvis_html_multi with a {model -> Graph} mapping.
    """
    if not _is_graph(G):
        raise TypeError("build_pyvis_html expects a NetworkX Graph")

    return build_pyvis_html_multi({"graph": G}, out_html=out_html, **kwargs)


def build_pyvis_html_multi(
    data: Mapping[str, Any],
    out_html: str = "web-graphs.html",
    *,
    height: str = "550px",
    width: str = "100%",
    directed: Optional[bool] = None,
    config_meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
    **kwargs: Any,
) -> str:
    """
    Render multiple graphs with nested tabs (config -> model) and a per-config legend.

    Input shapes accepted:
      - {model_name: Graph}
      - {config_name: {model_name: Graph}}

    Recognized extra kwargs:
      - uml_builder: object with build_all(cfg, ...) -> (mermaid_text, text, html)
      - uml_cfgs: config mapping/list keyed by config prefix
      - uml_iframe_height: forwarded to uml_builder.build_all()
      - model_preview_urls: mapping {model_key -> iframe src} for Honeybee previews
      - model_preview_hint: text shown under "no 3D preview", e.g. how to enable them

    Other extra kwargs are ignored.
    """
    uml_builder = kwargs.pop("uml_builder", None)
    uml_cfgs = kwargs.pop("uml_cfgs", None)
    uml_iframe_height = int(kwargs.pop("uml_iframe_height", 600))
    model_preview_urls = {
        str(key): str(value)
        for key, value in (kwargs.pop("model_preview_urls", {}) or {}).items()
    }
    model_preview_hint = str(kwargs.pop("model_preview_hint", "") or "")
    _ = kwargs

    if not data:
        raise ValueError("build_pyvis_html_multi: 'data' is empty")

    if all(_is_graph(g) for g in data.values()):
        cfg_map: Dict[str, Dict[str, nx.Graph]] = {"conf": dict(data)}
    else:
        cfg_map = {}
        for cfg, models in data.items():
            if isinstance(models, Mapping) and all(_is_graph(g) for g in models.values()):
                cfg_map[str(cfg)] = dict(models)
            else:
                raise TypeError(
                    "build_pyvis_html_multi expects {model→Graph} or {config→(model→Graph)}"
                )

    cfg_names = list(cfg_map.keys())
    model_key_order = {cfg: list(cfg_map[cfg].keys()) for cfg in cfg_names}

    if directed is None:
        directed = any(nx.is_directed(G) for models in cfg_map.values() for G in models.values())

    global_node_types: set[str] = set()
    global_edge_types: set[str] = set()
    for _cfg, _models in cfg_map.items():
        for _m, _G in _models.items():
            global_node_types.update(_node_type(d) for _, d in _G.nodes(data=True))
            global_edge_types.update(_edge_type(d) for _, _, d in _G.edges(data=True))

    global_node_list = sorted(global_node_types)
    global_edge_list = sorted(global_edge_types)
    global_node_palette = dict(zip(global_node_list, _viridis_hex(len(global_node_list))))
    global_edge_palette = dict(zip(global_edge_list, _viridis_hex(len(global_edge_list))))

    nodes_payload: Dict[str, Dict[str, Dict[str, Any]]] = {}
    edges_payload: Dict[str, Dict[str, Dict[str, Any]]] = {}
    node_meta_all: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    edge_meta_all: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    graph_meta_all: Dict[str, Dict[str, Dict[str, Any]]] = {}
    cfg_node_order: Dict[str, list[str]] = {}
    cfg_edge_order: Dict[str, list[str]] = {}

    for cfg, models in cfg_map.items():
        present_nodes = set()
        present_edges = set()
        for _, G in models.items():
            present_nodes.update(_node_type(d) for _, d in G.nodes(data=True))
            present_edges.update(_edge_type(d) for _, _, d in G.edges(data=True))

        cfg_node_order[cfg] = [t for t in global_node_list if t in present_nodes]
        cfg_edge_order[cfg] = [t for t in global_edge_list if t in present_edges]

        nodes_payload[cfg] = {}
        edges_payload[cfg] = {}
        node_meta_all[cfg] = {}
        edge_meta_all[cfg] = {}
        graph_meta_all[cfg] = {}

        for model_name, G in models.items():
            key_name = str(model_name)
            is_dir = nx.is_directed(G)
            in_deg = dict(G.in_degree()) if is_dir else dict(G.degree())

            vis_nodes = []
            node_meta: Dict[str, Dict[str, Any]] = {}

            # Detect if this graph has spatial coordinates on nodes
            _has_positions = any(
                "x" in dd and "y" in dd
                for _, dd in G.nodes(data=True)
            )

            for n, d in G.nodes(data=True):
                nid = str(n)
                ntype = _node_type(d)
                bg = global_node_palette.get(ntype, "#999999")
                label = _node_label(n, d)
                value = max(1, int(in_deg.get(n, 0)))

                node_obj: Dict[str, Any] = {
                    "id": nid,
                    "label": label,
                    "_origLabel": label,
                    "title": label,
                    "color": {
                        "background": bg,
                        "border": "#1a1a1a",
                        "highlight": {"background": bg, "border": "#000"},
                        "hover": {"background": bg, "border": "#333"},
                    },
                    "shape": "dot",
                    "value": value,
                    "borderWidth": 0,
                    "physics": True,
                    "fixed": False,
                    "hidden": False,
                    "_nodeType": ntype,
                }

                # If the graph carries spatial coordinates, pin nodes at those positions
                if _has_positions and "x" in d and "y" in d:
                    try:
                        node_obj["x"] = float(d["x"])
                        node_obj["y"] = float(d["y"])
                        node_obj["fixed"] = True
                        node_obj["physics"] = False
                    except (TypeError, ValueError):
                        pass

                vis_nodes.append(node_obj)

                meta = {k: _sanitize(v) for k, v in d.items()}
                meta.setdefault("type", ntype)
                meta.setdefault("label", label)
                node_meta[nid] = meta

            vis_edges = []
            edge_meta: Dict[str, Dict[str, Any]] = {}

            if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
                for u, v, k, d in G.edges(keys=True, data=True):
                    uid, vid = str(u), str(v)
                    etype = _edge_type(d)
                    ecolor = global_edge_palette.get(etype, "#888888")
                    eid = f"{uid}→{vid}#{k}"
                    vis_edges.append(
                        {
                            "id": eid,
                            "from": uid,
                            "to": vid,
                            "title": etype,
                            "color": ecolor,
                            "hidden": False,
                            "edgeType": etype,
                        }
                    )
                    emeta = {k2: _sanitize(v2) for k2, v2 in d.items()}
                    emeta.setdefault("from", uid)
                    emeta.setdefault("to", vid)
                    emeta.setdefault("type", etype)
                    edge_meta[eid] = emeta
            else:
                for idx, (u, v, d) in enumerate(G.edges(data=True)):
                    uid, vid = str(u), str(v)
                    etype = _edge_type(d)
                    ecolor = global_edge_palette.get(etype, "#888888")
                    eid = f"{uid}→{vid}#{idx}"
                    vis_edges.append(
                        {
                            "id": eid,
                            "from": uid,
                            "to": vid,
                            "title": etype,
                            "color": ecolor,
                            "hidden": False,
                            "edgeType": etype,
                        }
                    )
                    emeta = {k2: _sanitize(v2) for k2, v2 in d.items()}
                    emeta.setdefault("from", uid)
                    emeta.setdefault("to", vid)
                    emeta.setdefault("type", etype)
                    edge_meta[eid] = emeta

            nodes_payload[cfg][key_name] = {"items": vis_nodes}
            edges_payload[cfg][key_name] = {"items": vis_edges}
            node_meta_all[cfg][key_name] = node_meta
            edge_meta_all[cfg][key_name] = edge_meta
            graph_meta_all[cfg][key_name] = {
                str(k): _json_safe(v)
                for k, v in G.graph.items()
            }

    style_css = r"""
    <style>
      * { box-sizing: border-box; }
      body {
        margin: 0; padding: 0; padding-top: 60px;
        font: 13px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        background: #ffffff;
      }
      .cfg-tabs-wrapper{
        position: fixed; top:0; left:0; right:0; z-index:1100;
        background:#fff; border-bottom:2px solid #e0e0e0;
        box-shadow:0 2px 8px rgba(0,0,0,0.08);
        padding:10px 16px;
      }
      .section-tab-rail{
        position:fixed; top:var(--section-tabs-top,84px); left:12px;
        height:calc(100vh - var(--section-tabs-top,84px) - 12px); width:30px;
        z-index:1050; display:flex; flex-direction:column; gap:6px;
      }
      .section-tab{
        flex:1 1 0; min-height:0; width:30px; padding:6px 2px;
        border:1px solid #c9ccd3; border-radius:10px; background:#e4e4e4;
        color:#333; cursor:pointer; display:flex; align-items:center; justify-content:center;
        transition:background 0.18s, border-color 0.18s, box-shadow 0.18s, color 0.18s;
        box-shadow:inset 0 1px 0 rgba(255,255,255,0.55);
      }
      .section-tab:hover{
        background:#dddddd; border-color:#bcc0c8;
      }
      .section-tab.active{
        background:#ffffff; border-color:#b9bec8; color:#111;
        box-shadow:0 1px 4px rgba(15,23,42,0.10);
      }
      .section-tab span{
        display:block; writing-mode:vertical-rl; transform:rotate(180deg);
        white-space:normal; text-align:center; font-size:10px; font-weight:700;
        letter-spacing:0.25px; line-height:1.05;
      }
      .cfg-header{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:nowrap;}
      .cfg-header .cfg-tabs{flex:1;min-width:0;}
      .cfg-tabs{
        display:flex; gap:8px; flex-wrap:nowrap; overflow-x:auto; overflow-y:hidden;
        white-space:nowrap; scrollbar-width:thin; -webkit-overflow-scrolling:touch;
      }
      .cfg-tabs::-webkit-scrollbar{height:8px;}
      .cfg-tabs::-webkit-scrollbar-thumb{background:#d0d7e2; border-radius:999px;}
      .cfg-tab{
        flex:0 0 auto;
        padding:8px 16px; border:2px solid transparent; border-radius:8px;
        cursor:pointer; background:#f5f5f5; font-weight:500; transition:all 0.2s;
        user-select:none;
      }
      .cfg-tab:hover{background:#e8e8e8;}
      .cfg-tab.active{background:#e8f0ff; border-color:#7aa7ff; color:#0066cc;}
      .cfg-subtitle{
        margin-top:6px; font-size:12px; color:#666;
        max-width: 100%; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
      }

      .cfg-block{margin:0; padding:0; border:0; background:transparent; box-shadow:none;}
      .section-panel{
        position:relative; margin:20px 16px 0 56px; padding:16px;
        background:#fff; border:1px solid #d6d9df; border-radius:12px;
        box-shadow:0 1px 3px rgba(0,0,0,0.06); overflow:visible;
      }
      .preview-section{margin-top:16px;}
      .cfg-row{display:flex;gap:16px;align-items:flex-start;}
      .right-col{flex:1;min-width:0;display:flex;flex-direction:column;}
      .preview-main-row{display:flex; gap:16px; align-items:stretch;}
      .graph-preview-col{flex:2 1 0; min-width:0;}
      .model-preview-col{flex:1 1 0; min-width:280px; display:flex;}
      .model-preview-card{
        width:100%; display:flex; flex-direction:column;
        border:1px solid #d7dce4; border-radius:10px; overflow:hidden; background:#ffffff;
      }
      .model-preview-header{
        padding:10px 12px; border-bottom:1px solid #e3e7ee; background:#f5f7fb;
      }
      .model-preview-title{font-size:13px; font-weight:700; color:#223;}
      .model-preview-subtitle{
        margin-top:2px; font-size:11px; color:#667085;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
      }
      .model-preview-stage{
        position:relative; flex:1 1 auto; min-height:650px; background:#fff;
      }
      .model-preview-stage iframe{
        display:block; width:100%; height:100%; min-height:650px; border:0;
      }
      .model-preview-empty{
        position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
        padding:24px; text-align:center; color:#667085; font-size:12px; line-height:1.45;
      }
      .section-panel[data-section-tab-label]{
        scroll-margin-top:calc(var(--section-tabs-top,84px) + 10px);
      }

      .legend{
        position:sticky; top:75px; max-width:280px; flex:0 0 280px;
        background:#fff; border:1px solid #e0e0e0; border-radius:10px;
        box-shadow:0 2px 8px rgba(0,0,0,.08);
        max-height: calc(100vh - 90px);
        display:flex; flex-direction:column; overflow:hidden;
      }
      .legend .leg-content{overflow-y:auto; overflow-x:hidden; padding:8px; flex:1;}
      .leg-section{margin-bottom:4px;}
      .leg-section-header{
        display:flex; align-items:center; justify-content:space-between;
        font-weight:600; padding:6px 8px; margin:0;
        color:#333; font-size:11px; text-transform:uppercase;
        letter-spacing:0.5px; background:#f8f9fa; border-radius:4px;
      }
      .leg-section-header .header-left{display:flex; align-items:center; gap:6px; flex:1; cursor:pointer;}
      .toggle-all-checkbox{width:14px; height:14px; cursor:pointer; margin:0;}
      .toggle-arrow{font-size:10px; transition:transform 0.2s; color:#666; cursor:pointer;}
      .leg-section.collapsed .toggle-arrow{transform:rotate(-90deg);}
      .leg-section.collapsed .leg-col{display:none;}
      .leg-col{display:flex; flex-direction:column; gap:2px; padding:4px 0;}

      .leg-item{
        display:flex; align-items:center; gap:6px;
        padding:0 8px; height:24px; color:#333; font-size:12px;
        cursor:pointer; border-radius:4px; transition:all 0.15s; position:relative;
      }
      .leg-item:hover{background:#f0f2f5;}
      .leg-item.selected{background:#e3f2fd; border-left:3px solid #2196f3; padding-left:5px;}
      .leg-item input[type="checkbox"]{width:14px; height:14px; cursor:pointer; margin:0; flex-shrink:0;}
      .swatch{display:inline-block; width:10px; height:10px; border:1px solid rgba(0,0,0,0.2); border-radius:2px; flex-shrink:0;}
      .edge-swatch{display:inline-block; width:16px; height:3px; border-radius:2px; flex-shrink:0;}
      .leg-item-label{flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0;}

      .model-tabs{display:flex; gap:6px; margin:0 0 12px 0; flex-wrap:wrap;}
      .model-tab{
        padding:6px 12px; border:1px solid #ddd; border-radius:6px;
        cursor:pointer; background:#fafafa; font-size:12px; font-weight:500;
        transition:all 0.2s;
      }
      .model-tab:hover{background:#f0f0f0;}
      .model-tab.active{background:#eef6ff; border-color:#7aa7ff; color:#0066cc;}

      .net-wrap{position:relative; border-radius:8px; overflow:hidden;}
      .hidden{display:none;}

      .export-buttons{position:absolute;top:10px;right:10px;z-index:11;display:flex;gap:6px;}
      .export-buttons button{
        padding:4px 10px;font-size:12px;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer;
        box-shadow:0 1px 3px rgba(0,0,0,0.12);
      }
      .export-buttons button:hover{background:#f5f5f5;}
      .graph-props-btn{
        position:absolute; left:10px; bottom:10px; z-index:11;
        width:36px; height:36px; border:1px solid #c7cfdb; border-radius:10px;
        background:rgba(255,255,255,0.97); cursor:pointer; display:flex;
        align-items:center; justify-content:center; box-shadow:0 2px 8px rgba(0,0,0,0.12);
      }
      .graph-props-btn:hover{background:#f5f8ff; border-color:#9bb6e9;}
      .graph-props-btn.active{background:#e8f0ff; border-color:#7aa7ff;}
      .graph-props-btn svg{display:block}

      .selection-info{
        position:absolute;bottom:12px;right:12px;z-index:10;background:rgba(255,255,255,0.97);
        border:1px solid #d0d0d0;border-radius:8px;padding:8px 16px;font-size:13px;font-weight:500;
        box-shadow:0 4px 12px rgba(0,0,0,0.15);white-space:nowrap;color:#333;
      }
      .selection-info .count{
        display:inline-block;margin:0 4px;padding:3px 8px;background:#e8f0ff;border-radius:5px;
        font-weight:600;color:#0066cc;font-size:12px;
      }
      .selection-info button{
        padding:6px 14px;border:1px solid #0066cc;border-radius:6px;background:#0066cc;color:#fff;cursor:pointer;
        font-weight:600;font-size:13px;
      }
      .selection-info button:hover{background:#0052a3;}

      .details{
        margin-top:16px; padding:12px;
        border:1px solid #e8e8e8; border-radius:8px;
        background:#ffffff;
      }
      .details .hint{color:#666; font-style:italic; padding:8px 0;}
      .details .section{margin-bottom:16px; border:1px solid #ddd; border-radius:8px; overflow:hidden; background:#fff;}
      .details .section-header{
        display:flex; justify-content:space-between; align-items:center;
        padding:12px 16px; background:#f8f9fa; cursor:pointer; user-select:none; font-weight:600;
        border-bottom:1px solid #e8e8e8;
      }
      .details .section-header-main{display:flex; align-items:center; gap:8px; min-width:0;}
      .details .section-header-actions{display:flex; align-items:center; gap:6px; flex-shrink:0;}
      .details .json-bulk-btn{
        width:24px; height:24px; padding:0; border:1px solid #c9ced6; border-radius:6px;
        background:#fff; color:#334155; cursor:pointer; font-size:14px; font-weight:700;
        line-height:1; display:inline-flex; align-items:center; justify-content:center;
      }
      .details .json-bulk-btn:hover{background:#f3f4f6;}
      .details .json-bulk-btn:disabled{opacity:0.45; cursor:not-allowed; background:#f8f8f8;}
      .details .section.collapsed .section-content{display:none;}
      .details .section.collapsed .toggle-icon{transform:rotate(-90deg);}
      .details .toggle-icon{font-size:14px; transition:transform 0.2s;}
      .details .section-content{padding:12px 16px;}
      .details .table-controls{
        display:flex; gap:8px; margin-bottom:12px; padding:10px; background:#f8f9fa; border-radius:6px;
        align-items:center; border:1px solid #e8e8e8;
      }
      .details .table-controls button{
        padding:6px 12px; border:1px solid #ccc; border-radius:5px; background:#fff; cursor:pointer;
        font-size:12px; font-weight:500;
      }
      .details .table-wrapper{margin-bottom:14px; overflow-x:auto;}
      .details table{border-collapse:collapse; width:100%; min-width:520px;}
      .details th, .details td{border-bottom:1px solid #e8e8e8; padding:8px 10px; text-align:left; vertical-align:top;}
      .details th{background:#f8f9fa; font-weight:600; position:sticky; top:0; font-size:11px; text-transform:uppercase; letter-spacing:0.3px; color:#555;}
      .details tr:hover{background:#fafbfc;}
      .details .tbl-header{display:flex; justify-content:space-between; align-items:center; margin:12px 0 8px; padding-bottom:8px; border-bottom:1px solid #e8e8e8;}
      .details .tbl-title{font-weight:600; color:#333; font-size:13px;}
      .details .tbl-buttons{display:flex; gap:4px;}
      .details .tbl-buttons button{padding:4px 10px; border:1px solid #ccc; border-radius:4px; background:#fff; cursor:pointer; font-size:11px;}
      .details .table-wrapper.collapsed table{display:none;}
      .details .nowrap{white-space:nowrap;}
      .details .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size:11px;}
      .details .wrap-anywhere{
        white-space:normal; overflow-wrap:anywhere; word-break:break-word;
        line-height:1.25; font-size:11px; max-width:220px;
      }
      .details .json-cell{min-width:180px; max-width:320px;}
      .details .json-cell-wrap{display:flex; align-items:flex-start; gap:8px; max-width:100%; min-width:0;}
      .details .json-cell-wrap.expanded{display:block;}
      .details .json-cell-toggle{
        width:22px; height:22px; padding:0; flex:0 0 22px; border:1px solid #c9ced6; border-radius:6px;
        background:#fff; color:#334155; cursor:pointer; font-size:14px; font-weight:700; line-height:1;
      }
      .details .json-cell-toggle:hover{background:#f3f4f6;}
      .details .json-cell-preview{
        display:block; max-width:240px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        color:#334155; font-size:10px; line-height:1.15;
      }
      .details .json-cell-pretty{
        display:none; margin:0; white-space:pre-wrap; word-break:break-word;
        background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 12px;
        color:#1f2937; font-size:10px; line-height:1.3; max-width:320px;
      }
      .details .json-cell-wrap.expanded .json-cell-toggle{margin:0 0 8px 0;}
      .details .json-cell-wrap.expanded .json-cell-preview{display:none;}
      .details .json-cell-wrap.expanded .json-cell-pretty{display:block;}

      .uml-section{
        background:#fafafa;
      }
      .uml-header{
        display:flex; justify-content:space-between; align-items:flex-start;
        gap:12px; flex-wrap:wrap; margin-bottom:12px;
      }
      .uml-title{font-size:14px; font-weight:700; color:#222;}
      .uml-subtitle{font-size:12px; color:#666; margin-top:4px;}
      .uml-actions{display:flex; gap:8px; flex-wrap:wrap;}
      .uml-actions button{
        padding:6px 12px; border:1px solid #ccc; border-radius:6px;
        background:#fff; cursor:pointer; font-size:12px; font-weight:600;
      }
      .uml-actions button:hover{background:#f5f5f5;}
      .uml-actions button:disabled{
        opacity:0.5; cursor:not-allowed; background:#f3f3f3;
      }
      .uml-stage{
        background:#fff; border:1px solid #ddd; border-radius:8px;
        padding:16px; overflow:auto; min-height:180px;
      }
      .uml-stage.uml-stage-iframe{
        padding:0; overflow:hidden;
      }
      .uml-stage .hint{color:#666; font-style:italic;}
      .uml-stage svg{
        display:block; margin:0 auto; max-width:82%; height:auto;
      }
      .uml-stage iframe{
        display:block; width:100%; min-height:600px;
      }
      .uml-render-cache{display:none;}
      @media (max-width: 1100px){
        .section-panel{margin-left:16px;}
        .section-tab-rail{display:none;}
        .preview-main-row{flex-direction:column;}
        .model-preview-col{min-width:0;}
        .model-preview-stage,
        .model-preview-stage iframe{min-height:420px;}
      }
    </style>
    """

    cfg_subtitles = {}
    if config_meta:
        for c in cfg_names:
            meta = config_meta.get(c, {})
            desc = (meta.get("description") or "").strip()
            cfg_subtitles[c] = desc

    uml_payloads = _build_uml_payloads(
        cfg_map,
        cfg_subtitles,
        uml_builder=uml_builder,
        uml_cfgs=uml_cfgs,
        uml_iframe_height=uml_iframe_height,
    )

    cfg_tabs_html = []
    cfg_blocks_html = []

    for i, cfg in enumerate(cfg_names):
        subtitle = cfg_subtitles.get(cfg, "")
        subtitle_html = f"<div class='cfg-subtitle' title='{subtitle}'>{subtitle}</div>" if subtitle else ""
        _ = subtitle_html
        cfg_tabs_html.append(f"<div class='cfg-tab' data-cfg='{i}'>{cfg}</div>")

        node_types = cfg_node_order[cfg]
        edge_types = cfg_edge_order[cfg]

        legend_nodes = "".join(
            f"<div class='leg-item' data-type='{t}' data-kind='node' title='{t}'>"
            f"<input type='checkbox' checked onclick='event.stopPropagation()'>"
            f"<span class='swatch' style='background:{global_node_palette[t]}'></span>"
            f"<span class='leg-item-label'>{t}</span>"
            f"</div>"
            for t in node_types
        )
        legend_edges = "".join(
            f"<div class='leg-item' data-etype='{e}' data-kind='edge' title='{e}'>"
            f"<input type='checkbox' checked onclick='event.stopPropagation()'>"
            f"<span class='edge-swatch' style='background:{global_edge_palette[e]}'></span>"
            f"<span class='leg-item-label'>{e}</span>"
            f"</div>"
            for e in edge_types
        )

        model_keys = list(cfg_map[cfg].keys())
        model_tabs = "".join(
            f"<div class='model-tab' data-cfg='{i}' data-model='{j}'>{model_keys[j]}</div>"
            for j in range(len(model_keys))
        )

        model_containers = "".join(
            f"<div id='net_{i}_{j}' class='net-wrap hidden' style='height:{height}; width:{width};'>"
            f"<div class='export-buttons'>"
            f"<button id='btn-export-png_{i}_{j}' data-cfg='{i}' data-model='{j}'>PNG</button>"
            f"</div>"
            f"<button id='btn-graph-props_{i}_{j}' class='graph-props-btn' data-cfg='{i}' data-model='{j}' title='Select graph properties' aria-pressed='false'>"
            f"<svg width='18' height='18' viewBox='0 0 18 18' aria-hidden='true'>"
            f"<line x1='5' y1='4.5' x2='13' y2='4.5' stroke='#4b5d7a' stroke-width='1.4'/>"
            f"<line x1='5' y1='4.5' x2='9' y2='13.5' stroke='#4b5d7a' stroke-width='1.4'/>"
            f"<line x1='13' y1='4.5' x2='9' y2='13.5' stroke='#4b5d7a' stroke-width='1.4'/>"
            f"<circle cx='5' cy='4.5' r='2.2' fill='#ffffff' stroke='#4b5d7a' stroke-width='1.4'/>"
            f"<circle cx='13' cy='4.5' r='2.2' fill='#ffffff' stroke='#4b5d7a' stroke-width='1.4'/>"
            f"<circle cx='9' cy='13.5' r='2.2' fill='#ffffff' stroke='#4b5d7a' stroke-width='1.4'/>"
            f"</svg>"
            f"</button>"
            f"<div id='selection-info_{i}_{j}' class='selection-info'>No nodes selected</div>"
            f"<div id='mynetwork_{i}_{j}' style='height:100%; width:100%; border:1px solid #ddd; border-radius:6px;'></div>"
            f"</div>"
            for j in range(len(model_keys))
        )

        uml_payload = uml_payloads.get(cfg, {})
        has_mermaid = bool(uml_payload.get("mermaid"))
        has_text = bool(uml_payload.get("text"))
        has_dbml = bool(uml_payload.get("dbml"))
        has_iframe = bool(uml_payload.get("iframe_html"))
        uml_hint = (
            "Rendering structure diagram..."
            if has_mermaid
            else (
                "Loading DBML viewer..."
                if has_iframe
                else "Diagram unavailable for this configuration."
            )
        )
        uml_panel = (
            f"<section class='section-panel uml-section' id='uml_panel_{i}' data-section-tab-label='Structure Diagram'>"
            f"  <div class='uml-header'>"
            f"    <div>"
            f"      <div class='uml-title'>Structure Diagram</div>"
            f"      <div class='uml-subtitle'>DBML or Mermaid view for the selected configuration.</div>"
            f"    </div>"
            f"    <div class='uml-actions'>"
            f"      <button id='btn-uml-png_{i}' data-cfg='{i}'{' disabled' if not has_mermaid else ''}>PNG</button>"
            f"      <button id='btn-uml-mermaid_{i}' data-cfg='{i}'{' disabled' if not has_mermaid else ''}>Mermaid</button>"
            f"      <button id='btn-uml-txt_{i}' data-cfg='{i}'{' disabled' if not has_text else ''} title='Download structural text export'>TXT</button>"
            f"      <button id='btn-uml-dbml_{i}' data-cfg='{i}'{' disabled' if not has_dbml else ''}>DBML</button>"
            f"    </div>"
            f"  </div>"
            f"  <div class='uml-stage' id='uml_stage_{i}'><div class='hint'>{uml_hint}</div></div>"
            f"  <div class='uml-render-cache' id='uml_export_stage_{i}' aria-hidden='true'></div>"
            f"</section>"
        )

        preview_panel = (
            f"<section class='section-panel preview-section' id='cfg_section_{i}' data-section-tab-label='Graph Preview'>"
            f"  <div class='cfg-row' id='cfg_row_{i}'>"
            f"    <div class='legend' id='legend_{i}'>"
            f"      <div class='leg-content'>"
            f"        <div class='leg-section'>"
            f"          <div class='leg-section-header'>"
            f"            <div class='header-left' onclick='this.parentElement.parentElement.classList.toggle(\"collapsed\")'>"
            f"              <input type='checkbox' class='toggle-all-checkbox toggle-all-nodes' checked onclick='event.stopPropagation()' title='Show/hide all node types'>"
            f"              <span>Node types</span>"
            f"            </div>"
            f"            <span class='toggle-arrow' onclick='this.parentElement.parentElement.classList.toggle(\"collapsed\")'>▼</span>"
            f"          </div>"
            f"          <div class='leg-col'>{legend_nodes}</div>"
            f"        </div>"
            + (
                f"<div class='leg-section'>"
                f"  <div class='leg-section-header'>"
                f"    <div class='header-left' onclick='this.parentElement.parentElement.classList.toggle(\"collapsed\")'>"
                f"      <input type='checkbox' class='toggle-all-checkbox toggle-all-edges' checked onclick='event.stopPropagation()' title='Show/hide all edge types'>"
                f"      <span>Edge types</span>"
                f"    </div>"
                f"    <span class='toggle-arrow' onclick='this.parentElement.parentElement.classList.toggle(\"collapsed\")'>▼</span>"
                f"  </div>"
                f"  <div class='leg-col'>{legend_edges}</div>"
                f"</div>"
                if edge_types else ""
            )
            + (
                "        <div class='leg-section'>"
                "          <div class='leg-section-header'>"
                "            <div class='header-left' onclick='this.parentElement.parentElement.classList.toggle(\"collapsed\")'>"
                "              <span>Options</span>"
                "            </div>"
                "            <span class='toggle-arrow' onclick='this.parentElement.parentElement.classList.toggle(\"collapsed\")'>▼</span>"
                "          </div>"
                "          <div class='leg-col'>"
                "            <label class='leg-item' style='cursor:default'><input type='checkbox' class='chk-show-isolates' checked onclick='event.stopPropagation()'> <span class='leg-item-label'>Show isolates</span></label>"
                "            <label class='leg-item' style='cursor:default'><input type='checkbox' class='chk-show-labels' onclick='event.stopPropagation()'> <span class='leg-item-label'>Show labels</span></label>"
                "          </div>"
                "        </div>"
            )
            + (
                f"      </div>"
                f"    </div>"
                f"    <div class='right-col'>"
                f"      <div class='model-tabs' id='tabs_{i}'>{model_tabs}</div>"
                f"      <div class='preview-main-row'>"
                f"        <div class='graph-preview-col'><div class='models' id='models_{i}'>{model_containers}</div></div>"
                f"        <aside class='model-preview-col'>"
                f"          <div class='model-preview-card'>"
                f"            <div class='model-preview-header'>"
                f"              <div class='model-preview-title'>Honeybee Model</div>"
                f"              <div class='model-preview-subtitle' id='model_preview_subtitle_{i}'>No model selected</div>"
                f"            </div>"
                f"            <div class='model-preview-stage' id='model_preview_stage_{i}'></div>"
                f"          </div>"
                f"        </aside>"
                f"      </div>"
                f"      <div class='details' id='details_{i}'><div class='hint'>Select nodes. ⌘/Ctrl-click toggles; Shift-drag for box select. Drag a node to pin; double-click to toggle pin.</div></div>"
                f"    </div>"
                f"  </div>"
                f"</section>"
            )
        )

        cfg_block = (
            f"<div class='cfg-block' id='cfg_{i}'>"
            f"  {preview_panel}"
            f"  {uml_panel}"
            f"</div>"
        )
        cfg_blocks_html.append(cfg_block)

    inject_js = (
        "<script>\n"
        "(function(){\n"
        "  const DATA_NODES = " + json.dumps(nodes_payload, ensure_ascii=False) + ";\n"
        "  const DATA_EDGES = " + json.dumps(edges_payload, ensure_ascii=False) + ";\n"
        "  const NODE_META  = " + json.dumps(node_meta_all, ensure_ascii=False) + ";\n"
        "  const EDGE_META  = " + json.dumps(edge_meta_all, ensure_ascii=False) + ";\n"
        "  const GRAPH_META = " + json.dumps(graph_meta_all, ensure_ascii=False) + ";\n"
        "  const CFG_NAMES  = " + json.dumps(cfg_names, ensure_ascii=False) + ";\n"
        "  const MODEL_KEYS = " + json.dumps(model_key_order, ensure_ascii=False) + ";\n"
        "  const UML_DATA   = " + json.dumps(uml_payloads, ensure_ascii=False) + ";\n"
        "  const MODEL_PREVIEW_URLS = " + json.dumps(model_preview_urls, ensure_ascii=False) + ";\n"
        "  const MODEL_PREVIEW_HINT = " + json.dumps(model_preview_hint, ensure_ascii=False) + ";\n"
        "  function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}\n"
        "  const PIN_FIRST = ['id','from','to','src','tgt','name','type','node_type','edge_type','instance_name','surface_type','zone_name','space_name','construction_name','outside_boundary_condition'];\n"
        "  const GRAPH_PIN_FIRST = ['config_name','derived_from','model_id'];\n"
        "  const GRAPH_FEATURE_VALUE_KEYS = ['graph_feature_values','weather_city'];\n"
        "  const GRAPH_FEATURE_COL_KEYS = ['graph_feature_cols','weather_feature_cols'];\n"
        "  function orderColumns(keys){const set=new Set(keys);const pinned=PIN_FIRST.filter(k=>set.has(k));const rest=keys.filter(k=>!PIN_FIRST.includes(k)).sort((a,b)=>a.localeCompare(b));return pinned.concat(rest);}\n"
        "  function orderGraphKeys(keys){const set=new Set(keys);const pinned=GRAPH_PIN_FIRST.filter(k=>set.has(k));const rest=keys.filter(k=>!GRAPH_PIN_FIRST.includes(k)).sort((a,b)=>a.localeCompare(b));return pinned.concat(rest);}\n"
        "  function extractGraphFeatureRows(graphMeta){\n"
        "    let featureValues={};\n"
        "    for(const key of GRAPH_FEATURE_VALUE_KEYS){\n"
        "      if(graphMeta && graphMeta[key] && typeof graphMeta[key]==='object' && !Array.isArray(graphMeta[key])){featureValues=graphMeta[key]; break;}\n"
        "    }\n"
        "    let cols=[];\n"
        "    for(const key of GRAPH_FEATURE_COL_KEYS){\n"
        "      if(graphMeta && Array.isArray(graphMeta[key])){cols=graphMeta[key].map(v=>String(v)); break;}\n"
        "    }\n"
        "    if(!cols.length) cols=Object.keys(featureValues).sort((a,b)=>a.localeCompare(b));\n"
        "    return cols.filter(k=>Object.prototype.hasOwnProperty.call(featureValues, k)).map(k=>({property:k, value:featureValues[k]}));\n"
        "  }\n"
        "  function extractGraphMetaRows(graphMeta, cfgName, modelKey){\n"
        "    const meta=Object.assign({}, graphMeta||{});\n"
        "    meta.config_name=cfgName;\n"
        "    if(modelKey!==null && modelKey!==undefined) meta.model_id=modelKey;\n"
        "    delete meta.config_full_name;\n"
        "    delete meta.graph_feature_values;\n"
        "    delete meta.graph_feature_cols;\n"
        "    delete meta.weather_city;\n"
        "    delete meta.weather_feature_cols;\n"
        "    return orderGraphKeys(Object.keys(meta)).map(key=>({property:key, value:meta[key]}));\n"
        "  }\n"
        "  const cfgTabs = Array.from(document.querySelectorAll('.cfg-tab'));\n"
        "  const cfgBlocks = Array.from(document.querySelectorAll('.cfg-block'));\n"
        "  const sectionTabRail = document.getElementById('section_tab_rail');\n"
        "  const networks = {};\n"
        "  const hiddenStates = {};\n"
        "  let currentModelKey = null;\n"
        "  let sectionTabTargets = [];\n"
        "  let sectionTabTicking = false;\n"
        "  function sectionHeaderOffset(){\n"
        "    const header=document.querySelector('.cfg-tabs-wrapper');\n"
        "    const height=header?Math.ceil(header.getBoundingClientRect().height):60;\n"
        "    return height+12;\n"
        "  }\n"
        "  function visibleSectionTargets(){\n"
        "    return Array.from(document.querySelectorAll('[data-section-tab-label]')).filter(el=>{\n"
        "      if(!(el instanceof HTMLElement)) return false;\n"
        "      if(el.offsetParent===null) return false;\n"
        "      const style=window.getComputedStyle(el);\n"
        "      return style.display!=='none' && style.visibility!=='hidden';\n"
        "    });\n"
        "  }\n"
        "  function ensureSectionTargetId(el, idx){\n"
        "    if(!el.id) el.id='section_tab_target_'+idx;\n"
        "    return el.id;\n"
        "  }\n"
        "  function updateSectionTabActiveState(){\n"
        "    if(!sectionTabRail||!sectionTabTargets.length) return;\n"
        "    const offset=sectionHeaderOffset()+6;\n"
        "    let active=sectionTabTargets[0];\n"
        "    const passed=sectionTabTargets.filter(el=>el.getBoundingClientRect().top<=offset);\n"
        "    if(passed.length) active=passed[passed.length-1];\n"
        "    sectionTabRail.querySelectorAll('.section-tab').forEach(btn=>{\n"
        "      btn.classList.toggle('active', btn.getAttribute('data-target')===active.id);\n"
        "    });\n"
        "  }\n"
        "  function scheduleSectionTabUpdate(){\n"
        "    if(sectionTabTicking) return;\n"
        "    sectionTabTicking=true;\n"
        "    window.requestAnimationFrame(()=>{\n"
        "      sectionTabTicking=false;\n"
        "      updateSectionTabActiveState();\n"
        "    });\n"
        "  }\n"
        "  function refreshSectionTabs(){\n"
        "    if(!sectionTabRail) return;\n"
        "    document.documentElement.style.setProperty('--section-tabs-top', sectionHeaderOffset()+'px');\n"
        "    sectionTabTargets=visibleSectionTargets();\n"
        "    if(!sectionTabTargets.length){\n"
        "      sectionTabRail.innerHTML='';\n"
        "      sectionTabRail.classList.add('hidden');\n"
        "      return;\n"
        "    }\n"
        "    sectionTabRail.classList.remove('hidden');\n"
        "    sectionTabRail.innerHTML=sectionTabTargets.map((el,idx)=>{\n"
        "      const id=ensureSectionTargetId(el, idx);\n"
        "      const label=el.getAttribute('data-section-tab-label')||('Section '+(idx+1));\n"
        "      return `<button class=\"section-tab\" data-target=\"${id}\" title=\"${esc(label)}\"><span>${esc(label)}</span></button>`;\n"
        "    }).join('');\n"
        "    sectionTabRail.querySelectorAll('.section-tab').forEach(btn=>{\n"
        "      btn.addEventListener('click',()=>{\n"
        "        const target=document.getElementById(btn.getAttribute('data-target')||'');\n"
        "        if(!target) return;\n"
        "        const top=target.getBoundingClientRect().top+window.scrollY-sectionHeaderOffset();\n"
        "        window.scrollTo({top: Math.max(0, top), behavior:'smooth'});\n"
        "      });\n"
        "    });\n"
        "    updateSectionTabActiveState();\n"
        "  }\n"
        "  window.refreshSectionTabs=refreshSectionTabs;\n"
        "  function downloadText(content, filename){\n"
        "    const blob=new Blob([content||''], {type:'text/plain;charset=utf-8'});\n"
        "    const url=URL.createObjectURL(blob);\n"
        "    const a=document.createElement('a');\n"
        "    a.href=url; a.download=filename; a.click();\n"
        "    setTimeout(()=>URL.revokeObjectURL(url), 0);\n"
        "  }\n"
        "  function downloadSvgAsPng(svgEl, filename){\n"
        "    if(!svgEl){alert('UML SVG unavailable'); return;}\n"
        "    const rect=svgEl.getBoundingClientRect();\n"
        "    const vb=(svgEl.viewBox&&svgEl.viewBox.baseVal)?svgEl.viewBox.baseVal:null;\n"
        "    const width=Math.max(1, Math.ceil((vb&&vb.width)||rect.width||1200));\n"
        "    const height=Math.max(1, Math.ceil((vb&&vb.height)||rect.height||800));\n"
        "    let source=new XMLSerializer().serializeToString(svgEl);\n"
        "    if(!source.match(/^<svg[^>]+xmlns=\\\"http:\\/\\/www\\.w3\\.org\\/2000\\/svg\\\"/)) source=source.replace(/^<svg/, '<svg xmlns=\"http://www.w3.org/2000/svg\"');\n"
        "    if(!source.match(/^<svg[^>]+xmlns:xlink=\\\"http:\\/\\/www\\.w3\\.org\\/1999\\/xlink\\\"/)) source=source.replace(/^<svg/, '<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\"');\n"
        "    const img=new Image();\n"
        "    img.onload=()=>{\n"
        "      const canvas=document.createElement('canvas');\n"
        "      canvas.width=width; canvas.height=height;\n"
        "      const ctx=canvas.getContext('2d');\n"
        "      if(!ctx){alert('Canvas unavailable'); return;}\n"
        "      ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,width,height);\n"
        "      ctx.drawImage(img,0,0,width,height);\n"
        "      const a=document.createElement('a');\n"
        "      a.href=canvas.toDataURL('image/png');\n"
        "      a.download=filename;\n"
        "      a.click();\n"
        "    };\n"
        "    img.onerror=()=>alert('Failed to export UML PNG');\n"
        "    img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(source);\n"
        "  }\n"
        "  async function renderMermaidSvg(code, renderId){\n"
        "    if(!(window.mermaid&&window.mermaid.render)) return '';\n"
        "    const out=await window.mermaid.render(renderId, code);\n"
        "    return out.svg;\n"
        "  }\n"
        "  async function renderUmlPanels(){\n"
        "    const canRenderMermaid=!!(window.mermaid&&window.mermaid.initialize&&window.mermaid.render);\n"
        "    if(canRenderMermaid) window.mermaid.initialize({startOnLoad:false, securityLevel:'loose', theme:'base', themeVariables:{fontSize:'10px'}});\n"
        "    for(let cfgIdx=0; cfgIdx<CFG_NAMES.length; cfgIdx++){\n"
        "      const cfgName=CFG_NAMES[cfgIdx];\n"
        "      const data=UML_DATA[cfgName]||{};\n"
        "      const stage=document.getElementById('uml_stage_'+cfgIdx);\n"
        "      const exportStage=document.getElementById('uml_export_stage_'+cfgIdx);\n"
        "      if(!stage) continue;\n"
        "      const hasIframe=!!data.iframe_html;\n"
        "      const hasMermaid=!!data.mermaid;\n"
        "      const renderIframe = hasIframe && !hasMermaid;\n"
        "      stage.classList.toggle('uml-stage-iframe', renderIframe);\n"
        "      if(exportStage) exportStage.innerHTML='';\n"
        "      if(!renderIframe && !hasMermaid){\n"
        "        stage.innerHTML='<div class=\"hint\">Diagram unavailable for this configuration.</div>';\n"
        "        continue;\n"
        "      }\n"
        "      let renderedSvg='';\n"
        "      if(hasMermaid && canRenderMermaid){\n"
        "        try{\n"
        "          renderedSvg=await renderMermaidSvg(data.mermaid, 'uml_render_'+cfgIdx+'_'+Date.now());\n"
        "        }catch(err){\n"
        "          console.error('Failed to render Mermaid for', cfgName, err);\n"
        "        }\n"
        "      }\n"
        "      if(renderIframe){\n"
        "        stage.innerHTML=data.iframe_html;\n"
        "        if(exportStage && renderedSvg) exportStage.innerHTML=renderedSvg;\n"
        "        continue;\n"
        "      }\n"
        "      if(renderedSvg){\n"
        "        stage.innerHTML=renderedSvg;\n"
        "        if(exportStage) exportStage.innerHTML=renderedSvg;\n"
        "        continue;\n"
        "      }\n"
        "      if(hasMermaid && !canRenderMermaid){\n"
        "        stage.innerHTML='<div class=\"hint\">Mermaid renderer unavailable in this page.</div>';\n"
        "        continue;\n"
        "      }\n"
        "      stage.innerHTML='<div class=\"hint\">Failed to render structure diagram.</div>';\n"
        "    }\n"
        "  }\n"
        "  function bindUmlExports(){\n"
        "    CFG_NAMES.forEach((cfgName,cfgIdx)=>{\n"
        "      const data=UML_DATA[cfgName]||{};\n"
        "      const btnMermaid=document.getElementById('btn-uml-mermaid_'+cfgIdx);\n"
        "      if(btnMermaid) btnMermaid.addEventListener('click',()=>downloadText(data.mermaid||'', cfgName+'_uml.mmd'));\n"
        "      const btnTxt=document.getElementById('btn-uml-txt_'+cfgIdx);\n"
        "      if(btnTxt) btnTxt.addEventListener('click',()=>downloadText(data.text||data.dbml||data.mermaid||'', cfgName+'_uml.txt'));\n"
        "      const btnDbml=document.getElementById('btn-uml-dbml_'+cfgIdx);\n"
        "      if(btnDbml) btnDbml.addEventListener('click',()=>downloadText(data.dbml||'', cfgName+'_uml.dbml'));\n"
        "      const btnPng=document.getElementById('btn-uml-png_'+cfgIdx);\n"
        "      if(btnPng) btnPng.addEventListener('click',()=>{\n"
        "        const stage=document.getElementById('uml_stage_'+cfgIdx);\n"
        "        const exportStage=document.getElementById('uml_export_stage_'+cfgIdx);\n"
        "        const svg=(exportStage&&exportStage.querySelector('svg'))||(stage&&stage.querySelector('svg'));\n"
        "        if(!svg){alert('UML diagram has not finished rendering yet'); return;}\n"
        "        downloadSvgAsPng(svg, cfgName+'_uml.png');\n"
        "      });\n"
        "    });\n"
        "  }\n"
        "  function applySelectionStyles(netObj){\n"
        "    const nodes=netObj.nodes; const sel=new Set(netObj.network.getSelectedNodes());\n"
        "    const updates=[]; nodes.getIds().forEach(id=>updates.push({id, borderWidth: sel.has(id)?4:0}));\n"
        "    nodes.update(updates);\n"
        "  }\n"
        "  function applyGraphPropsButtonState(cfgIdx, modelIdx){\n"
        "    const btn=document.getElementById('btn-graph-props_'+cfgIdx+'_'+modelIdx);\n"
        "    const obj=networks[cfgIdx]&&networks[cfgIdx][modelIdx];\n"
        "    const selected=!!(obj&&obj.graphPropsSelected);\n"
        "    if(!btn) return;\n"
        "    btn.classList.toggle('active', selected);\n"
        "    btn.setAttribute('aria-pressed', selected ? 'true' : 'false');\n"
        "    btn.title=selected ? 'Deselect graph properties' : 'Select graph properties';\n"
        "  }\n"
        "  function recomputeVisibility(cfgIdx){\n"
        "    const st = hiddenStates[cfgIdx]; if(!st) return;\n"
        "    const mMap = networks[cfgIdx] || {};\n"
        "    Object.values(mMap).forEach(obj=>{\n"
        "      const nodes=obj.nodes, edges=obj.edges;\n"
        "      const labelUpdates=[];\n"
        "      nodes.getIds().forEach(id=>{\n"
        "        const n=nodes.get(id); if(!n) return;\n"
        "        const orig=(n._origLabel!==undefined)?n._origLabel:(n.label||'');\n"
        "        if(n._origLabel===undefined) n._origLabel=orig;\n"
        "        const desired=st.showLabels?orig:'';\n"
        "        if((n.label||'')!==desired) labelUpdates.push({id, label: desired});\n"
        "      });\n"
        "      if(labelUpdates.length) nodes.update(labelUpdates);\n"
        "      const nUpdates=[];\n"
        "      nodes.getIds().forEach(id=>{\n"
        "        const n=nodes.get(id); const t=n&&(n._nodeType||'unknown');\n"
        "        const hide=st.nodeTypes.has(t);\n"
        "        if(!!n.hidden!==hide) nUpdates.push({id, hidden: hide});\n"
        "      });\n"
        "      if(nUpdates.length) nodes.update(nUpdates);\n"
        "      const eUpdates=[];\n"
        "      edges.getIds().forEach(id=>{\n"
        "        const e=edges.get(id);\n"
        "        const et=(e&&e.edgeType)?e.edgeType:((e&&e.title)||'relation');\n"
        "        const fromHidden=!!(nodes.get(e.from)&&nodes.get(e.from).hidden);\n"
        "        const toHidden=!!(nodes.get(e.to)&&nodes.get(e.to).hidden);\n"
        "        const hide=st.edgeTypes.has(et)||fromHidden||toHidden;\n"
        "        if(!!e.hidden!==hide) eUpdates.push({id, hidden: hide});\n"
        "      });\n"
        "      if(eUpdates.length) edges.update(eUpdates);\n"
        "      if(!st.showIsolates){\n"
        "        const hasEdge=new Set();\n"
        "        edges.getIds().forEach(id=>{const e=edges.get(id); if(!e.hidden){hasEdge.add(e.from); hasEdge.add(e.to);}});\n"
        "        const isoUpd=[];\n"
        "        nodes.getIds().forEach(id=>{\n"
        "          const n=nodes.get(id); if(n.hidden) return;\n"
        "          const isIso=!hasEdge.has(id);\n"
        "          if(!!n.hidden!==!!isIso) isoUpd.push({id, hidden: isIso});\n"
        "        });\n"
        "        if(isoUpd.length) nodes.update(isoUpd);\n"
        "      }\n"
        "    });\n"
        "  }\n"
        "  function updateSelectionInfo(cfgIdx, modelIdx){\n"
        "    const infoDiv=document.getElementById('selection-info_'+cfgIdx+'_'+modelIdx);\n"
        "    if(!infoDiv) return;\n"
        "    const obj=networks[cfgIdx]&&networks[cfgIdx][modelIdx];\n"
        "    if(!obj){infoDiv.textContent='No nodes selected'; return;}\n"
        "    const net=obj.network, nodes=obj.nodes, edges=obj.edges;\n"
        "    const selIds=net.getSelectedNodes().filter(id=>{const n=nodes.get(id); return n && !n.hidden;});\n"
        "    const selEdgeIds=net.getSelectedEdges().filter(id=>{const e=edges.get(id); return e && !e.hidden;});\n"
        "    if(!selIds.length && !selEdgeIds.length && !obj.graphPropsSelected){\n"
        "      infoDiv.innerHTML='<button id=\"select-all-btn-'+cfgIdx+'_'+modelIdx+'\">Select All</button>';\n"
        "      const btn=document.getElementById('select-all-btn-'+cfgIdx+'_'+modelIdx);\n"
        "      if(btn){btn.onclick=()=>{\n"
        "        const visibleNodeIds=nodes.getIds().filter(id=>{const n=nodes.get(id); return n && !n.hidden;});\n"
        "        const visibleEdgeIds=edges.getIds().filter(id=>{const e=edges.get(id); return e && !e.hidden;});\n"
        "        obj.graphPropsSelected=true; applyGraphPropsButtonState(cfgIdx, modelIdx);\n"
        "        net.setSelection({nodes:visibleNodeIds, edges:visibleEdgeIds});\n"
        "        applySelectionStyles(obj); updateSelectionInfo(cfgIdx, modelIdx); renderDetails(cfgIdx, modelIdx);\n"
        "      };}\n"
        "      return;\n"
        "    }\n"
        "    let html='';\n"
        "    if(obj.graphPropsSelected){\n"
        "      html+='<span class=\"count\">1</span> graph';\n"
        "    }\n"
        "    if(selIds.length){\n"
        "      if(html) html+=' | ';\n"
        "      const typeCounts={};\n"
        "      selIds.forEach(id=>{const n=nodes.get(id); const t=(n&&n._nodeType)||'unknown'; typeCounts[t]=(typeCounts[t]||0)+1;});\n"
        "      const numTypes=Object.keys(typeCounts).length;\n"
        "      html+=`<span class=\"count\">${selIds.length}</span> node${selIds.length!==1?'s':''}`;\n"
        "      if(numTypes<=3){\n"
        "        html+=': ' + Object.entries(typeCounts).sort((a,b)=>a[0].localeCompare(b[0]))\n"
        "          .map(([t,c])=>`<span class=\"count\">${c}</span> ${esc(t)}`).join(', ');\n"
        "      }\n"
        "    }\n"
        "    if(selEdgeIds.length){\n"
        "      if(html) html+=' | ';\n"
        "      const edgeCounts={};\n"
        "      selEdgeIds.forEach(id=>{const e=edges.get(id); const et=(e&&e.edgeType)?e.edgeType:'relation'; edgeCounts[et]=(edgeCounts[et]||0)+1;});\n"
        "      const num=Object.keys(edgeCounts).length;\n"
        "      html+=`<span class=\"count\">${selEdgeIds.length}</span> edge${selEdgeIds.length!==1?'s':''}`;\n"
        "      if(num<=3){\n"
        "        html+=': ' + Object.entries(edgeCounts).sort((a,b)=>a[0].localeCompare(b[0]))\n"
        "          .map(([t,c])=>`<span class=\"count\">${c}</span> ${esc(t)}`).join(', ');\n"
        "      }\n"
        "    }\n"
        "    infoDiv.innerHTML=html;\n"
        "  }\n"
        "  function truncateText(text, maxLen){\n"
        "    const raw=String(text||'');\n"
        "    if(raw.length<=maxLen) return raw;\n"
        "    return raw.slice(0, Math.max(0, maxLen-1))+'…';\n"
        "  }\n"
        "  function parseStructuredString(value){\n"
        "    if(typeof value!=='string') return null;\n"
        "    const trimmed=value.trim();\n"
        "    if(trimmed.length<2) return null;\n"
        "    const looksStructured=(trimmed.startsWith('{')&&trimmed.endsWith('}'))||(trimmed.startsWith('[')&&trimmed.endsWith(']'));\n"
        "    if(!looksStructured) return null;\n"
        "    try{\n"
        "      const parsed=JSON.parse(trimmed);\n"
        "      if(!parsed || typeof parsed!=='object') return null;\n"
        "      return {raw: trimmed, pretty: JSON.stringify(parsed, null, 2)};\n"
        "    }catch(err){\n"
        "      return null;\n"
        "    }\n"
        "  }\n"
        "  function setJsonCellExpanded(wrap, expanded){\n"
        "    if(!(wrap instanceof HTMLElement)) return;\n"
        "    wrap.classList.toggle('expanded', !!expanded);\n"
        "    const btn=wrap.querySelector('.json-cell-toggle');\n"
        "    if(btn){\n"
        "      btn.textContent=expanded?'-':'+';\n"
        "      btn.setAttribute('aria-expanded', expanded?'true':'false');\n"
        "      btn.setAttribute('title', expanded?'Collapse JSON':'Expand JSON');\n"
        "    }\n"
        "  }\n"
        "  window.toggleJsonCell=function(btn){\n"
        "    const wrap=btn&&btn.closest?btn.closest('.json-cell-wrap'):null;\n"
        "    if(!wrap) return false;\n"
        "    setJsonCellExpanded(wrap, !wrap.classList.contains('expanded'));\n"
        "    return false;\n"
        "  };\n"
        "  window.toggleJsonSection=function(sectionId, expanded){\n"
        "    const section=document.getElementById(sectionId);\n"
        "    if(!section) return false;\n"
        "    section.querySelectorAll('.json-cell-wrap').forEach(wrap=>setJsonCellExpanded(wrap, expanded));\n"
        "    return false;\n"
        "  };\n"
        "  function renderTableCell(value, extraClass){\n"
        "    let v=value;\n"
        "    if(v===null||v===undefined) v='';\n"
        "    const classes=[];\n"
        "    if(extraClass) classes.push(extraClass);\n"
        "    if(typeof v==='number') classes.push('nowrap','mono');\n"
        "    else if(typeof v==='string') classes.push('wrap-anywhere');\n"
        "    const sv=String(v);\n"
        "    const structured=parseStructuredString(sv);\n"
        "    if(structured){\n"
        "      classes.push('json-cell');\n"
        "      return `<td class=\"${classes.join(' ')}\" title=\"${esc(structured.raw)}\"><div class=\"json-cell-wrap\"><button type=\"button\" class=\"json-cell-toggle\" aria-expanded=\"false\" title=\"Expand JSON\" onclick=\"event.stopPropagation();return window.toggleJsonCell(this);\">+</button><span class=\"json-cell-preview mono\">${esc(truncateText(structured.raw, 50))}</span><pre class=\"json-cell-pretty mono\">${esc(structured.pretty)}</pre></div></td>`;\n"
        "    }\n"
        "    let shown=sv;\n"
        "    const MAX=120;\n"
        "    if(sv.length>MAX) shown=sv.slice(0,MAX-1)+'…';\n"
        "    return `<td class=\"${classes.join(' ')}\" title=\"${esc(sv)}\">${esc(shown)}</td>`;\n"
        "  }\n"
        "  function renderDetails(cfgIdx, modelIdx){\n"
        "    const block=document.getElementById('details_'+cfgIdx);\n"
        "    const obj=networks[cfgIdx]&&networks[cfgIdx][modelIdx];\n"
        "    if(!obj){block.innerHTML='<div class=\"hint\">No nodes or edges selected.</div>'; return;}\n"
        "    const net=obj.network, nodes=obj.nodes, edges=obj.edges;\n"
        "    const cfgName=CFG_NAMES[cfgIdx];\n"
        "    const modelKey = (MODEL_KEYS[cfgName] && MODEL_KEYS[cfgName][modelIdx]) ? MODEL_KEYS[cfgName][modelIdx] : null;\n"
        "    const nodeMeta = (modelKey && (NODE_META[cfgName]||{})[modelKey]) ? (NODE_META[cfgName]||{})[modelKey] : {};\n"
        "    const graphMeta = (modelKey && (GRAPH_META[cfgName]||{})[modelKey]) ? (GRAPH_META[cfgName]||{})[modelKey] : {};\n"
        "    const selIds=net.getSelectedNodes().filter(id=>{const n=nodes.get(id); return n && !n.hidden;});\n"
        "    const selEdgeIds=net.getSelectedEdges().filter(id=>{const e=edges.get(id); return e && !e.hidden;});\n"
        "    if(!selIds.length && !selEdgeIds.length && !obj.graphPropsSelected){block.innerHTML='<div class=\"hint\">No nodes, edges, or graph properties selected.</div>'; return;}\n"
        "    window.toggleSection=function(id){const sec=document.getElementById(id); if(sec) sec.classList.toggle('collapsed');};\n"
        "    let html='';\n"
        "    if(obj.graphPropsSelected){\n"
        "      html+='<div class=\"section\" id=\"graph-section-'+cfgIdx+'\">'\n"
        "           +'<div class=\"section-header\" onclick=\"toggleSection(\\'graph-section-'+cfgIdx+'\\')\">'\n"
        "           +'<div class=\"section-header-main\"><span>Graph Properties</span></div>'\n"
        "           +'<div class=\"section-header-actions\">'\n"
        "           +'<button type=\"button\" class=\"json-bulk-btn\" title=\"Expand all JSON cells\" onclick=\"event.stopPropagation();return window.toggleJsonSection(\\'graph-section-'+cfgIdx+'\\', true);\">+</button>'\n"
        "           +'<button type=\"button\" class=\"json-bulk-btn\" title=\"Collapse all JSON cells\" onclick=\"event.stopPropagation();return window.toggleJsonSection(\\'graph-section-'+cfgIdx+'\\', false);\">-</button>'\n"
        "           +'<span class=\"toggle-icon\">▼</span></div></div>'\n"
        "           +'<div class=\"section-content\">';\n"
        "      const featureRows=extractGraphFeatureRows(graphMeta);\n"
        "      const metaRows=extractGraphMetaRows(graphMeta, cfgName, modelKey);\n"
        "      html+='<div class=\"table-controls\">'\n"
        "           +'<button id=\"head-all-graph-btn-'+cfgIdx+'\">Head (3 rows)</button>'\n"
        "           +'<button id=\"all-all-graph-btn-'+cfgIdx+'\">All</button>'\n"
        "           +'</div>';\n"
        "      if(featureRows.length){\n"
        "        html+='<div class=\"table-wrapper\" data-section=\"graph\"><div class=\"tbl-header\"><div class=\"tbl-title\">Graph-level ML features ('+featureRows.length+')</div></div>';\n"
        "        html+='<table><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody class=\"table-body\">';\n"
        "        html+=featureRows.map(r=>'<tr>'+renderTableCell(r.property, 'mono')+renderTableCell(r.value, '')+'</tr>').join('');\n"
        "        html+='</tbody></table></div>';\n"
        "      }\n"
        "      if(metaRows.length){\n"
        "        html+='<div class=\"table-wrapper\" data-section=\"graph\"><div class=\"tbl-header\"><div class=\"tbl-title\">Graph metadata ('+metaRows.length+')</div></div>';\n"
        "        html+='<table><thead><tr><th>Property</th><th>Value</th></tr></thead><tbody class=\"table-body\">';\n"
        "        html+=metaRows.map(r=>'<tr>'+renderTableCell(r.property, 'mono')+renderTableCell(r.value, '')+'</tr>').join('');\n"
        "        html+='</tbody></table></div>';\n"
        "      }\n"
        "      if(!featureRows.length && !metaRows.length){\n"
        "        html+='<div class=\"hint\">No graph-level properties stored for this graph.</div>';\n"
        "      }\n"
        "      html+='</div></div>';\n"
        "    }\n"
        "    if(selIds.length){\n"
        "      html+='<div class=\"section\" id=\"nodes-section-'+cfgIdx+'\">'\n"
        "           +'<div class=\"section-header\" onclick=\"toggleSection(\\'nodes-section-'+cfgIdx+'\\')\">'\n"
        "           +'<div class=\"section-header-main\"><span>Nodes ('+selIds.length+')</span></div>'\n"
        "           +'<div class=\"section-header-actions\">'\n"
        "           +'<button type=\"button\" class=\"json-bulk-btn\" title=\"Expand all JSON cells\" onclick=\"event.stopPropagation();return window.toggleJsonSection(\\'nodes-section-'+cfgIdx+'\\', true);\">+</button>'\n"
        "           +'<button type=\"button\" class=\"json-bulk-btn\" title=\"Collapse all JSON cells\" onclick=\"event.stopPropagation();return window.toggleJsonSection(\\'nodes-section-'+cfgIdx+'\\', false);\">-</button>'\n"
        "           +'<span class=\"toggle-icon\">▼</span></div></div>'\n"
        "           +'<div class=\"section-content\">';\n"
        "      const groups={};\n"
        "      selIds.forEach(id=>{const meta=Object.assign({id}, nodeMeta[id]||{}); const typ=meta.type||meta.node_type||'unknown'; (groups[typ] ||= []).push(meta);});\n"
        "      html+='<div class=\"table-controls\">'\n"
        "           +'<button id=\"head-all-nodes-btn-'+cfgIdx+'\">Head (3 rows)</button>'\n"
        "           +'<button id=\"all-all-nodes-btn-'+cfgIdx+'\">All</button>'\n"
        "           +'</div>';\n"
        "      let tableIdx=0;\n"
        "      Object.keys(groups).sort().forEach(typ=>{\n"
        "        const rows=groups[typ];\n"
        "        const keySet=new Set(); rows.forEach(r=>Object.keys(r).forEach(k=>keySet.add(k)));\n"
        "        const cols=orderColumns([...keySet]);\n"
        "        html+=`<div class=\"table-wrapper\" data-section=\"nodes\" data-table-idx=\"${tableIdx}\">`;\n"
        "        html+='<div class=\"tbl-header\"><div class=\"tbl-title\">Type: '+esc(typ)+' ('+rows.length+')</div></div>';\n"
        "        html+='<table><thead><tr>'+cols.map(c=>'<'+'th>'+esc(c)+'</'+'th>').join('')+'</tr></thead><tbody class=\"table-body\">';\n"
        "        html+=rows.map(r=>'<tr>'+cols.map(c=>renderTableCell(r[c], c==='id'?'mono':'')).join('')+'</tr>').join('');\n"
        "        html+='</tbody></table></div>';\n"
        "        tableIdx++;\n"
        "      });\n"
        "      html+='</div></div>';\n"
        "    }\n"
        "    if(selEdgeIds.length){\n"
        "      html+='<div class=\"section\" id=\"edges-section-'+cfgIdx+'\">'\n"
        "           +'<div class=\"section-header\" onclick=\"toggleSection(\\'edges-section-'+cfgIdx+'\\')\">'\n"
        "           +'<div class=\"section-header-main\"><span>Edges ('+selEdgeIds.length+')</span></div>'\n"
        "           +'<div class=\"section-header-actions\">'\n"
        "           +'<button type=\"button\" class=\"json-bulk-btn\" title=\"Expand all JSON cells\" onclick=\"event.stopPropagation();return window.toggleJsonSection(\\'edges-section-'+cfgIdx+'\\', true);\">+</button>'\n"
        "           +'<button type=\"button\" class=\"json-bulk-btn\" title=\"Collapse all JSON cells\" onclick=\"event.stopPropagation();return window.toggleJsonSection(\\'edges-section-'+cfgIdx+'\\', false);\">-</button>'\n"
        "           +'<span class=\"toggle-icon\">▼</span></div></div>'\n"
        "           +'<div class=\"section-content\">';\n"
        "      const eGroups={};\n"
        "      const edgeMeta = (modelKey && (EDGE_META[cfgName]||{})[modelKey]) ? (EDGE_META[cfgName]||{})[modelKey] : {};\n"
        "      selEdgeIds.forEach(id=>{\n"
        "        const e=edges.get(id); const et=(e&&e.edgeType)?e.edgeType:'relation';\n"
        "        const meta = Object.assign({id}, edgeMeta[id]||{});\n"
        "        const fromNode=nodes.get(e.from); const toNode=nodes.get(e.to);\n"
        "        meta.from = (fromNode?(fromNode._origLabel||fromNode.label):e.from);\n"
        "        meta.to = (toNode?(toNode._origLabel||toNode.label):e.to);\n"
        "        meta.type = et;\n"
        "        (eGroups[et] ||= []).push(meta);\n"
        "      });\n"
        "      html+='<div class=\"table-controls\">'\n"
        "           +'<button id=\"head-all-edges-btn-'+cfgIdx+'\">Head (3 rows)</button>'\n"
        "           +'<button id=\"all-all-edges-btn-'+cfgIdx+'\">All</button>'\n"
        "           +'</div>';\n"
        "      Object.keys(eGroups).sort().forEach(typ=>{\n"
        "        const rows=eGroups[typ];\n"
        "        const keySet=new Set(); rows.forEach(r=>Object.keys(r).forEach(k=>keySet.add(k)));\n"
        "        const cols=orderColumns([...keySet]);\n"
        "        html+='<div class=\"table-wrapper\"><div class=\"tbl-header\"><div class=\"tbl-title\">Type: '+esc(typ)+' ('+rows.length+')</div></div>';\n"
        "        html+='<table><thead><tr>'+cols.map(c=>'<'+'th>'+esc(c)+'</'+'th>').join('')+'</tr></thead><tbody class=\"table-body\">';\n"
        "        html+=rows.map(r=>'<tr>'+cols.map(c=>renderTableCell(r[c], c==='id'?'mono':'')).join('')+'</tr>').join('');\n"
        "        html+='</tbody></table></div>';\n"
        "      });\n"
        "      html+='</div></div>';\n"
        "    }\n"
        "    block.innerHTML=html;\n"
        "    block.querySelectorAll('.section').forEach(sec=>{\n"
        "      const hasJson=!!sec.querySelector('.json-cell-wrap');\n"
        "      sec.querySelectorAll('.json-bulk-btn').forEach(btn=>{btn.disabled=!hasJson;});\n"
        "    });\n"
        "    const hN=document.getElementById('head-all-nodes-btn-'+cfgIdx); const aN=document.getElementById('all-all-nodes-btn-'+cfgIdx);\n"
        "    if(hN && aN){\n"
        "      hN.onclick=()=>{block.querySelectorAll('.table-wrapper[data-section=\"nodes\"] .table-body').forEach(tb=>{Array.from(tb.querySelectorAll('tr')).forEach((tr,i)=>tr.style.display=i<3?'':'none');});};\n"
        "      aN.onclick=()=>{block.querySelectorAll('.table-wrapper[data-section=\"nodes\"] .table-body').forEach(tb=>{Array.from(tb.querySelectorAll('tr')).forEach(tr=>tr.style.display='');});};\n"
        "    }\n"
        "    const hG=document.getElementById('head-all-graph-btn-'+cfgIdx); const aG=document.getElementById('all-all-graph-btn-'+cfgIdx);\n"
        "    if(hG && aG){\n"
        "      hG.onclick=()=>{block.querySelectorAll('#graph-section-'+cfgIdx+' .table-wrapper[data-section=\"graph\"] .table-body').forEach(tb=>{Array.from(tb.querySelectorAll('tr')).forEach((tr,i)=>tr.style.display=i<3?'':'none');});};\n"
        "      aG.onclick=()=>{block.querySelectorAll('#graph-section-'+cfgIdx+' .table-wrapper[data-section=\"graph\"] .table-body').forEach(tb=>{Array.from(tb.querySelectorAll('tr')).forEach(tr=>tr.style.display='');});};\n"
        "    }\n"
        "    const hE=document.getElementById('head-all-edges-btn-'+cfgIdx); const aE=document.getElementById('all-all-edges-btn-'+cfgIdx);\n"
        "    if(hE && aE){\n"
        "      hE.onclick=()=>{block.querySelectorAll('#edges-section-'+cfgIdx+' .table-body').forEach(tb=>{Array.from(tb.querySelectorAll('tr')).forEach((tr,i)=>tr.style.display=i<3?'':'none');});};\n"
        "      aE.onclick=()=>{block.querySelectorAll('#edges-section-'+cfgIdx+' .table-body').forEach(tb=>{Array.from(tb.querySelectorAll('tr')).forEach(tr=>tr.style.display='');});};\n"
        "    }\n"
        "  }\n"
        "  function renderModelPreview(cfgIdx, modelIdx){\n"
        "    const stage=document.getElementById('model_preview_stage_'+cfgIdx);\n"
        "    const subtitle=document.getElementById('model_preview_subtitle_'+cfgIdx);\n"
        "    const cfgName=CFG_NAMES[cfgIdx];\n"
        "    const modelKey=(MODEL_KEYS[cfgName] && MODEL_KEYS[cfgName][modelIdx]) ? MODEL_KEYS[cfgName][modelIdx] : null;\n"
        "    if(subtitle) subtitle.textContent=modelKey || 'No model selected';\n"
        "    if(!stage) return;\n"
        "    if(!modelKey){\n"
        "      stage.innerHTML='<div class=\"model-preview-empty\">No model selected.</div>';\n"
        "      return;\n"
        "    }\n"
        "    const src=MODEL_PREVIEW_URLS[modelKey];\n"
        "    if(!src){\n"
        "      stage.innerHTML='<div class=\"model-preview-empty\">No 3D preview for <strong>'+esc(modelKey)+'</strong>.'+(MODEL_PREVIEW_HINT?'<br><br>'+esc(MODEL_PREVIEW_HINT):'')+'</div>';\n"
        "      return;\n"
        "    }\n"
        "    stage.innerHTML='<iframe loading=\"lazy\" src=\"'+esc(src)+'\" title=\"Honeybee preview '+esc(modelKey)+'\"></iframe>';\n"
        "  }\n"
        "  function init(){\n"
        "    CFG_NAMES.forEach((cfgName,cfgIdx)=>{\n"
        "      hiddenStates[cfgIdx]={nodeTypes:new Set(), edgeTypes:new Set(), showIsolates:true, showLabels:false};\n"
        "      networks[cfgIdx]={};\n"
        "      const modelMapN = DATA_NODES[cfgName];\n"
        "      const modelMapE = DATA_EDGES[cfgName];\n"
        "      const keys = (MODEL_KEYS[cfgName] || Object.keys(modelMapN));\n"
        "      let j=0;\n"
        "      keys.forEach(modelName =>{\n"
        "        const container=document.getElementById('mynetwork_'+cfgIdx+'_'+j);\n"
        "        const nodes=new vis.DataSet(modelMapN[modelName].items);\n"
        "        const edges=new vis.DataSet(modelMapE[modelName].items);\n"
        "        const nodeCount=nodes.length; const edgeCount=edges.length;\n"
        "        const isLarge=(nodeCount>500||edgeCount>1000);\n"
        "        const hasSpatial=modelMapN[modelName].items.some(function(n){return typeof n.x==='number'&&typeof n.y==='number';});\n"
        "        const options={\n"
        "          nodes:{shape:'dot', scaling:{min:10,max:25}},\n"
        "          edges:{arrows:{to:{enabled:" + ("true" if directed else "false") + "}}, smooth:false, selectionWidth:2},\n"
        "          interaction:{hover:!isLarge, dragNodes:true, tooltipDelay:isLarge?300:100, multiselect:true, selectConnectedEdges:false},\n"
        "          physics:hasSpatial?{enabled:false}:{enabled:true, solver:isLarge?'barnesHut':'forceAtlas2Based', stabilization:{iterations:isLarge?100:200}}\n"
        "        };\n"
        "        const network=new vis.Network(container, {nodes, edges}, options);\n"
        "        const graphMeta=(GRAPH_META[cfgName]&&GRAPH_META[cfgName][modelName])?GRAPH_META[cfgName][modelName]:{};\n"
        "        networks[cfgIdx][j]={network,nodes,edges,graphMeta,graphPropsSelected:false};\n"
        "        const idx=j;\n"
        "        const btnPng=document.getElementById('btn-export-png_'+cfgIdx+'_'+idx);\n"
        "        if(btnPng){btnPng.addEventListener('click',()=>{const canvas=network.canvas && network.canvas.frame && network.canvas.frame.canvas; if(!canvas){alert('Canvas unavailable');return;} const dataUrl=canvas.toDataURL('image/png'); const a=document.createElement('a'); a.href=dataUrl; a.download=(cfgName||'graph')+'_'+(modelName||'model')+'.png'; a.click();});}\n"
        "        const btnGraph=document.getElementById('btn-graph-props_'+cfgIdx+'_'+idx);\n"
        "        if(btnGraph){btnGraph.addEventListener('click',()=>{const obj=networks[cfgIdx][idx]; if(!obj) return; obj.graphPropsSelected=!obj.graphPropsSelected; applyGraphPropsButtonState(cfgIdx, idx); updateSelectionInfo(cfgIdx, idx); renderDetails(cfgIdx, idx);});}\n"
        "        network.on('click',(params)=>{\n"
        "          const ev=params.event&&params.event.srcEvent;\n"
        "          const hasMod=!!(ev&&(ev.metaKey||ev.ctrlKey||ev.shiftKey));\n"
        "          if((params.nodes||[]).length===0 && (params.edges||[]).length===0 && !hasMod){const obj=networks[cfgIdx][idx]; network.unselectAll(); if(obj){obj.graphPropsSelected=false; applyGraphPropsButtonState(cfgIdx, idx);} applySelectionStyles(networks[cfgIdx][idx]); updateSelectionInfo(cfgIdx, idx); renderDetails(cfgIdx, idx);}\n"
        "        });\n"
        "        network.on('select',()=>{applySelectionStyles(networks[cfgIdx][idx]); updateSelectionInfo(cfgIdx, idx); renderDetails(cfgIdx, idx);});\n"
        "        network.on('deselectNode',()=>{applySelectionStyles(networks[cfgIdx][idx]); updateSelectionInfo(cfgIdx, idx); renderDetails(cfgIdx, idx);});\n"
        "        network.on('selectNode',()=>{applySelectionStyles(networks[cfgIdx][idx]); updateSelectionInfo(cfgIdx, idx); renderDetails(cfgIdx, idx);});\n"
        "        network.on('selectEdge',()=>{applySelectionStyles(networks[cfgIdx][idx]); updateSelectionInfo(cfgIdx, idx); renderDetails(cfgIdx, idx);});\n"
        "        network.on('deselectEdge',()=>{applySelectionStyles(networks[cfgIdx][idx]); updateSelectionInfo(cfgIdx, idx); renderDetails(cfgIdx, idx);});\n"
        "        network.on('dragEnd',(params)=>{if(params&&params.nodes&&params.nodes.length){nodes.update(params.nodes.map(nid=>({id:nid, fixed:{x:true,y:true}})));}});\n"
        "        network.on('doubleClick',(params)=>{if(params&&params.nodes&&params.nodes.length){const nid=params.nodes[0]; const node=nodes.get(nid); const isFixed=node&&node.fixed&&(node.fixed.x||node.fixed.y); nodes.update({id:nid, fixed:{x:!isFixed,y:!isFixed}});}});\n"
        "        j++;\n"
        "      });\n"
        "    });\n"
        "    function activateCfg(idx){\n"
        "      cfgTabs.forEach((t,i)=>t.classList.toggle('active', i===idx));\n"
        "      cfgBlocks.forEach((b,i)=>b.style.display=(i===idx)?'block':'none');\n"
        "      const modelTabs=Array.from(document.querySelectorAll('.model-tab[data-cfg=\"'+idx+'\"]'));\n"
        "      const cfgName=CFG_NAMES[idx];\n"
        "      const modelKeys=MODEL_KEYS[cfgName] || [];\n"
        "      let targetIdx=0;\n"
        "      if(currentModelKey!==null){\n"
        "        const found=modelKeys.indexOf(currentModelKey);\n"
        "        if(found>=0) targetIdx=found;\n"
        "      }\n"
        "      if(modelTabs[targetIdx]) activateModel(idx,targetIdx);\n"
        "      setTimeout(refreshSectionTabs,0);\n"
        "    }\n"
        "    function activateModel(cfgIdx, modelIdx){\n"
        "      const cfgName=CFG_NAMES[cfgIdx];\n"
        "      const modelKeys=MODEL_KEYS[cfgName] || [];\n"
        "      if(modelIdx>=0 && modelIdx<modelKeys.length) currentModelKey=modelKeys[modelIdx];\n"
        "      const modelTabs=Array.from(document.querySelectorAll('.model-tab[data-cfg=\"'+cfgIdx+'\"]'));\n"
        "      modelTabs.forEach((t,i)=>t.classList.toggle('active', i===modelIdx));\n"
        "      const cont=document.getElementById('models_'+cfgIdx);\n"
        "      Array.from(cont.children).forEach((c,i)=>c.classList.toggle('hidden', i!==modelIdx));\n"
        "      recomputeVisibility(cfgIdx);\n"
        "      const obj=networks[cfgIdx]&&networks[cfgIdx][modelIdx];\n"
        "      if(obj){setTimeout(()=>{try{obj.network.redraw(); obj.network.fit({animation:false});}catch(e){}},0);}\n"
        "      applyGraphPropsButtonState(cfgIdx, modelIdx);\n"
        "      renderModelPreview(cfgIdx, modelIdx);\n"
        "      updateSelectionInfo(cfgIdx, modelIdx);\n"
        "      renderDetails(cfgIdx, modelIdx);\n"
        "    }\n"
        "    cfgTabs.forEach((tab,i)=>tab.addEventListener('click',()=>activateCfg(i)));\n"
        "    document.querySelectorAll('.model-tabs').forEach(mt=>{\n"
        "      mt.addEventListener('click',(e)=>{\n"
        "        const t=e.target; if(!(t instanceof HTMLElement)) return;\n"
        "        if(!t.classList.contains('model-tab')) return;\n"
        "        const cfgIdx=parseInt(t.getAttribute('data-cfg'));\n"
        "        const sibs=Array.from(mt.querySelectorAll('.model-tab'));\n"
        "        const idx=sibs.indexOf(t);\n"
        "        if(idx>=0) activateModel(cfgIdx, idx);\n"
        "      });\n"
        "    });\n"
        "    cfgBlocks.forEach((block, cfgIdx)=>{\n"
        "      const legend=document.getElementById('legend_'+cfgIdx);\n"
        "      const st=hiddenStates[cfgIdx];\n"
        "      function activeModelIdx(){\n"
        "        const tabs=Array.from(document.querySelectorAll('.model-tab[data-cfg=\"'+cfgIdx+'\"]'));\n"
        "        return tabs.findIndex(t=>t.classList.contains('active'));\n"
        "      }\n"
        "      legend.addEventListener('change',(e)=>{\n"
        "        const t=e.target; if(!(t instanceof HTMLInputElement)) return;\n"
        "        if(t.classList.contains('chk-show-isolates')){st.showIsolates=!!t.checked; recomputeVisibility(cfgIdx); return;}\n"
        "        if(t.classList.contains('chk-show-labels')){st.showLabels=!!t.checked; recomputeVisibility(cfgIdx); return;}\n"
        "        if(t.classList.contains('toggle-all-nodes')){\n"
        "          const isChecked=t.checked;\n"
        "          legend.querySelectorAll('.leg-item[data-kind=\"node\"] input[type=\"checkbox\"]').forEach(cb=>{\n"
        "            cb.checked=isChecked;\n"
        "            const item=cb.closest('.leg-item'); const typ=item.getAttribute('data-type');\n"
        "            if(isChecked) st.nodeTypes.delete(typ); else st.nodeTypes.add(typ);\n"
        "          });\n"
        "          recomputeVisibility(cfgIdx); const mi=activeModelIdx(); if(mi>=0){updateSelectionInfo(cfgIdx,mi); renderDetails(cfgIdx,mi);} return;\n"
        "        }\n"
        "        if(t.classList.contains('toggle-all-edges')){\n"
        "          const isChecked=t.checked;\n"
        "          legend.querySelectorAll('.leg-item[data-kind=\"edge\"] input[type=\"checkbox\"]').forEach(cb=>{\n"
        "            cb.checked=isChecked;\n"
        "            const item=cb.closest('.leg-item'); const et=item.getAttribute('data-etype');\n"
        "            if(isChecked) st.edgeTypes.delete(et); else st.edgeTypes.add(et);\n"
        "          });\n"
        "          recomputeVisibility(cfgIdx); const mi=activeModelIdx(); if(mi>=0){updateSelectionInfo(cfgIdx,mi); renderDetails(cfgIdx,mi);} return;\n"
        "        }\n"
        "        const item=t.closest('.leg-item[data-kind]'); if(!item) return;\n"
        "        const kind=item.getAttribute('data-kind'); const isChecked=t.checked;\n"
        "        if(kind==='node'){const typ=item.getAttribute('data-type'); if(isChecked) st.nodeTypes.delete(typ); else st.nodeTypes.add(typ);}\n"
        "        if(kind==='edge'){const et=item.getAttribute('data-etype'); if(isChecked) st.edgeTypes.delete(et); else st.edgeTypes.add(et);}\n"
        "        recomputeVisibility(cfgIdx);\n"
        "        const mi=activeModelIdx(); if(mi>=0){updateSelectionInfo(cfgIdx,mi); renderDetails(cfgIdx,mi);} \n"
        "      });\n"
        "    });\n"
        "    window.addEventListener('scroll', scheduleSectionTabUpdate, {passive:true});\n"
        "    window.addEventListener('resize', refreshSectionTabs);\n"
        "    bindUmlExports();\n"
        "    renderUmlPanels();\n"
        "    const firstCfg=CFG_NAMES[0];\n"
        "    if(firstCfg && MODEL_KEYS[firstCfg] && MODEL_KEYS[firstCfg].length){currentModelKey=MODEL_KEYS[firstCfg][0];}\n"
        "    activateCfg(0);\n"
        "  }\n"
        "  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', init); else init();\n"
        "})();\n"
        "</script>\n"
    )

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Graphs</title>"
        "<link rel='stylesheet' href='https://unpkg.com/vis-network@9.1.2/styles/vis-network.min.css'>"
        + style_css
        + "</head><body>"
        "<div class='section-tab-rail' id='section_tab_rail'></div>"
        f"<div class='cfg-tabs-wrapper'><div class='cfg-header'><div class='cfg-tabs'>{''.join(cfg_tabs_html)}</div></div></div>"
        f"{''.join(cfg_blocks_html)}"
        "<script src='https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js'></script>"
        "<script src='https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js'></script>"
        + inject_js
        + "</body></html>"
    )

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


__all__ = ["build_pyvis_html", "build_pyvis_html_multi"]
