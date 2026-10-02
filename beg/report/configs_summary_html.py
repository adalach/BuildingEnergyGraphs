from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd


def _convert_to_json_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def _extract_features_from_graph(G: nx.Graph, model_id: str) -> Dict[str, Any]:
    features_info: Dict[str, Any] = {
        "model_id": model_id,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "node_types": {},
        "edge_types": {},
    }

    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get("node_type")
        if not node_type:
            if isinstance(node, tuple) and len(node) > 0:
                node_type = node[0]
            else:
                node_type = "unknown"

        if node_type not in features_info["node_types"]:
            features_info["node_types"][node_type] = {"count": 0, "features": {}}

        features_info["node_types"][node_type]["count"] += 1

        if not features_info["node_types"][node_type]["features"]:
            for key, value in node_data.items():
                json_value = _convert_to_json_serializable(value)

                if json_value is None:
                    val_type = "NULL"
                elif isinstance(value, bool):
                    val_type = "BOOLEAN"
                elif isinstance(value, (int, np.integer)):
                    val_type = "INTEGER"
                elif isinstance(value, (float, np.floating)):
                    val_type = "DOUBLE"
                elif isinstance(value, (pd.Timestamp, datetime)):
                    val_type = "TIMESTAMP"
                else:
                    val_type = "VARCHAR"

                features_info["node_types"][node_type]["features"][key] = {
                    "type": val_type,
                    "sample": json_value,
                }

    is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))
    if is_multigraph:
        for u, v, key in G.edges(keys=True):
            edge_data = G[u][v][key]
            edge_type = edge_data.get("rel", edge_data.get("group", "unknown"))

            if edge_type not in features_info["edge_types"]:
                features_info["edge_types"][edge_type] = {"count": 0, "features": {}}

            features_info["edge_types"][edge_type]["count"] += 1

            if not features_info["edge_types"][edge_type]["features"]:
                for attr_key, value in edge_data.items():
                    if attr_key in ["rel", "fk_col", "group"]:
                        continue
                    json_value = _convert_to_json_serializable(value)

                    if json_value is None:
                        val_type = "NULL"
                    elif isinstance(value, bool):
                        val_type = "BOOLEAN"
                    elif isinstance(value, (int, np.integer)):
                        val_type = "INTEGER"
                    elif isinstance(value, (float, np.floating)):
                        val_type = "DOUBLE"
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        val_type = "TIMESTAMP"
                    else:
                        val_type = "VARCHAR"

                    features_info["edge_types"][edge_type]["features"][attr_key] = {
                        "type": val_type,
                        "sample": json_value,
                    }
    else:
        for u, v in G.edges():
            edge_data = G[u][v]
            edge_type = edge_data.get("rel", edge_data.get("group", "unknown"))

            if edge_type not in features_info["edge_types"]:
                features_info["edge_types"][edge_type] = {"count": 0, "features": {}}

            features_info["edge_types"][edge_type]["count"] += 1

            if not features_info["edge_types"][edge_type]["features"]:
                for attr_key, value in edge_data.items():
                    if attr_key in ["rel", "fk_col", "group"]:
                        continue
                    json_value = _convert_to_json_serializable(value)

                    if json_value is None:
                        val_type = "NULL"
                    elif isinstance(value, bool):
                        val_type = "BOOLEAN"
                    elif isinstance(value, (int, np.integer)):
                        val_type = "INTEGER"
                    elif isinstance(value, (float, np.floating)):
                        val_type = "DOUBLE"
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        val_type = "TIMESTAMP"
                    else:
                        val_type = "VARCHAR"

                    features_info["edge_types"][edge_type]["features"][attr_key] = {
                        "type": val_type,
                        "sample": json_value,
                    }

    return features_info


_GRAPH_TYPE_NAMES = {
    "homo": "homogeneous",
    "homogeneous": "homogeneous",
    "hetero": "heterogeneous",
    "heterogeneous": "heterogeneous",
}


def _graph_type(conf_name: str, metadata: Dict[str, Any]) -> str:
    """``homogeneous`` or ``heterogeneous``, whatever spelling the caller used.

    The page's badge compares against the long form. Callers pass either the long
    form from a config file or the short ``homo``/``hetero`` of ``ConfigId.graph_type``.
    Unknown spellings fall back to the configuration registry, and to ``N/A`` when
    the registry does not know ``conf_name`` either.
    """
    declared = str(metadata.get("type") or "").strip().lower()
    if declared in _GRAPH_TYPE_NAMES:
        return _GRAPH_TYPE_NAMES[declared]
    try:
        from beg.config.naming import resolve

        return _GRAPH_TYPE_NAMES[resolve(conf_name).graph_type]
    except KeyError:
        return "N/A"


def _config_file(configs_dir: Path, conf_name: str) -> Path | None:
    """The JSON declaring ``conf_name``, e.g. ``C1.json``; ``None`` for C2 and C7, which have none."""
    from beg.config.naming import resolve

    config = resolve(conf_name)
    return None if config.is_derived else configs_dir / config.config_filename


def build_configs_summary_html(
    all_graphs_data: Dict[str, Dict[str, nx.Graph]],
    configs_dir: Path | str,
    configs_to_analyze: Iterable[str],
    output_html: Path | str,
    model_ids: Optional[Iterable[str]] = None,
    config_metadata_override: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Build an interactive HTML summary table for multiple configs and models.

    Parameters:
      - all_graphs_data: {config_name: {model_id: Graph}}
      - configs_dir: directory holding the configuration JSON files
      - configs_to_analyze: list of config prefixes to include
      - output_html: path to write the HTML file
      - model_ids: optional list of model IDs to include (default: union of graphs)
      - config_metadata_override: optional per-config metadata for synthetic or
        derived configs that do not have a backing JSON file
    """
    configs_dir = Path(configs_dir)
    output_html = Path(output_html)
    configs_to_analyze = list(configs_to_analyze)

    if model_ids is None:
        model_ids = sorted(
            {
                mid
                for conf_name in configs_to_analyze
                for mid in all_graphs_data.get(conf_name, {}).keys()
            }
        )
    else:
        model_ids = list(model_ids)

    print(f"{'='*70}")
    print("GENERATING CONFIGURATION SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\nModel IDs to include: {len(model_ids)}")
    print(f"Configurations to analyze: {len(configs_to_analyze)}")
    print(f"Output: {output_html}")

    all_features: Dict[str, Dict[str, Any]] = {}
    print(f"\n{'='*70}")
    print("Extracting features from graphs...")
    print(f"{'='*70}")

    for conf_name in configs_to_analyze:
        print(f"\n[{conf_name}] Extracting features...")
        if conf_name not in all_graphs_data:
            print(f"  No graphs found for {conf_name}, skipping")
            continue

        all_features[conf_name] = {}
        for model_id in model_ids:
            if model_id in all_graphs_data[conf_name]:
                G = all_graphs_data[conf_name][model_id]
                features = _extract_features_from_graph(G, model_id)
                all_features[conf_name][model_id] = features
                print(
                    f"  [{model_id}] {features['node_count']} nodes, {features['edge_count']} edges"
                )
            else:
                print(f"  [{model_id}] Graph not found")

    print("\nFeature extraction complete")
    print(f"  Configurations with features: {len(all_features)}")
    print(f"  Total graphs analyzed: {sum(len(models) for models in all_features.values())}")

    print(f"\n{'='*70}")
    print("Loading configuration metadata...")
    print(f"{'='*70}")

    configs_metadata: Dict[str, Dict[str, Any]] = dict(config_metadata_override or {})
    for conf_name in configs_to_analyze:
        if conf_name in configs_metadata:
            print(f"  [{conf_name}] Using provided metadata override")
            continue
        config_path = _config_file(configs_dir, conf_name)
        if config_path is not None and config_path.exists():
            try:
                raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
                configs_metadata[conf_name] = {
                    "id": raw_cfg.get("id", "N/A"),
                    "type": raw_cfg.get("type", "N/A"),
                    "weather": raw_cfg.get("weather", "N/A"),
                    "description": raw_cfg.get("description", ""),
                    "edges": raw_cfg.get("edges", []),
                }
                print(
                    f"  [{conf_name}] Loaded metadata (ID: {configs_metadata[conf_name]['id']})"
                )
            except Exception as e:
                print(f"  [{conf_name}] Error loading: {e}")
        else:
            print(f"  [{conf_name}] Config file not found: {config_path}")

    print(f"\n{'='*70}")
    print("Generating HTML table...")
    print(f"{'='*70}")

    configs_data_json = []
    for conf_name in configs_to_analyze:
        if conf_name not in all_features or not all_features[conf_name]:
            continue

        metadata = configs_metadata.get(conf_name, {})
        first_model_id = list(all_features[conf_name].keys())[0]
        first_features = all_features[conf_name][first_model_id]

        node_type_names = list(first_features["node_types"].keys())
        edge_type_names = list(first_features["edge_types"].keys())
        display_name = conf_name.replace("_", "_<wbr>")

        edges_config = metadata.get("edges", [])
        edge_bidirectional = {}
        edge_src_to_tgt = {}
        for edge in edges_config:
            if isinstance(edge, dict):
                edge_rel = edge.get("rel", "unknown")
                edge_bidirectional[edge_rel] = edge.get("bidirectional", False)
                edge_src_to_tgt[edge_rel] = {
                    "src": edge.get("src_cat", ""),
                    "tgt": edge.get("tgt_cat", ""),
                }

        config_entry = {
            "name": conf_name,
            "display_name": display_name,
            "id": metadata.get("id", "N/A"),
            "type": _graph_type(conf_name, metadata),
            "weather": metadata.get("weather", "N/A"),
            "node_type_names": node_type_names,
            "edge_type_names": edge_type_names,
            "edge_bidirectional": edge_bidirectional,
            "edge_src_to_tgt": edge_src_to_tgt,
            "models": all_features[conf_name],
        }
        configs_data_json.append(config_entry)

    def parse_id_for_sort(config_entry: Dict[str, Any]):
        id_str = config_entry["id"]
        if id_str == "N/A":
            return (999, "")
        import re

        match = re.match(r"C(\d+)(_.*)?", id_str)
        if match:
            num = int(match.group(1))
            suffix = match.group(2) or ""
            return (num, suffix)
        return (999, id_str)

    configs_data_json.sort(key=parse_id_for_sort)
    configs_json_str = json.dumps(configs_data_json, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Configurations Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #ffffff;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}

        h1 {{
            color: #333;
            margin: 0;
        }}

        .model-selector {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .model-selector label {{
            font-weight: 600;
            color: #555;
        }}

        .model-selector select {{
            padding: 8px 12px;
            border: 2px solid #667eea;
            border-radius: 6px;
            font-size: 14px;
            background: white;
            color: #333;
            cursor: pointer;
            min-width: 200px;
        }}

        .model-selector select:hover {{
            border-color: #764ba2;
        }}

        .model-selector select:focus {{
            outline: none;
            border-color: #764ba2;
            box-shadow: 0 0 0 3px rgba(118, 75, 162, 0.1);
        }}

        .table-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
            padding: 20px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            table-layout: fixed;
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
        }}

        th:nth-child(1), td:nth-child(1) {{ width: 150px; }}
        th:nth-child(2), td:nth-child(2) {{ width: 80px; }}
        th:nth-child(3), td:nth-child(3) {{ width: 60px; }}
        th:nth-child(4), td:nth-child(4) {{ width: 120px; }}
        th:nth-child(5), td:nth-child(5) {{ width: 100px; }}
        th:nth-child(6), td:nth-child(6) {{ width: 300px; }}
        th:nth-child(7), td:nth-child(7) {{ width: 300px; }}

        td:nth-child(1) {{
            word-break: break-word;
            hyphens: auto;
        }}

        tr:hover {{ background-color: #f9f9f9; }}

        .expandable {{
            cursor: pointer;
            color: #667eea;
            font-weight: 500;
            user-select: none;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}

        .expandable:hover {{
            color: #764ba2;
            text-decoration: underline;
        }}

        .expandable-icon {{
            display: inline-block;
            transition: transform 0.2s;
        }}

        .expandable-icon.expanded {{ transform: rotate(90deg); }}

        .expandable-content {{
            display: none;
            margin-top: 8px;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            overflow-x: hidden;
            word-wrap: break-word;
        }}

        .expandable-content.show {{ display: block; }}

        .nested-content {{
            background-color: #fff;
            border-left: 3px solid #667eea;
            margin-left: 15px;
            padding-left: 10px;
        }}

        .node-type-container {{ margin: 8px 0; }}

        .feature-list {{ margin: 0; padding-left: 20px; }}

        .feature-item {{
            margin: 4px 0;
            font-family: 'Courier New', monospace;
        }}

        .feature-name {{ color: #2c3e50; font-weight: 500; }}
        .feature-type {{ color: #7f8c8d; font-size: 11px; }}
        .feature-sample {{ color: #27ae60; font-size: 11px; }}

        .badge {{
            display: inline-block;
            padding: 3px 6px;
            border-radius: 8px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .badge-hom {{ background-color: #e3f2fd; color: #1976d2; }}
        .badge-het {{ background-color: #fce4ec; color: #c2185b; }}

        sup {{ font-size: 0.7em; font-weight: normal; }}

        .count-badge {{
            background-color: #e8f5e9;
            color: #2e7d32;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 10px;
            margin-left: 5px;
        }}

        .small-text {{ font-size: 11px; color: #666; font-weight: normal; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Graph Configurations Summary</h1>
        <div class="model-selector">
            <label for="modelSelect">Reference Model:</label>
            <select id="modelSelect" onchange="updateTable()"></select>
        </div>
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Config Name</th>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Weather</th>
                    <th>Graph Stats</th>
                    <th>Node Types & Features</th>
                    <th>Edge Types & Features</th>
                </tr>
            </thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>

    <script>
        const configsData = {configs_json_str};
        let currentModelId = null;

        function formatIdWithSuperscript(id) {{
            return id.replace(/_(.+)/, '<sup>$1</sup>');
        }}

        function init() {{
            const modelIds = new Set();
            configsData.forEach(config => {{
                Object.keys(config.models).forEach(mid => modelIds.add(mid));
            }});

            const select = document.getElementById('modelSelect');
            Array.from(modelIds).sort().forEach((mid, idx) => {{
                const option = document.createElement('option');
                option.value = mid;
                option.textContent = mid;
                if (idx === 0) {{
                    option.selected = true;
                    currentModelId = mid;
                }}
                select.appendChild(option);
            }});

            updateTable();
        }}

        function updateTable() {{
            const select = document.getElementById('modelSelect');
            currentModelId = select.value;
            renderTable();
        }}

        function renderTable() {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';

            configsData.forEach((config, configIdx) => {{
                const modelData = config.models[currentModelId];
                if (!modelData) return;

                const row = document.createElement('tr');
                row.innerHTML += `<td><strong>${{config.display_name}}</strong></td>`;

                const formattedId = formatIdWithSuperscript(config.id);
                row.innerHTML += `<td>${{formattedId}}</td>`;

                const typeBadge = config.type === 'homogeneous' ?
                    '<span class="badge badge-hom">HOM</span>' :
                    '<span class="badge badge-het">HET</span>';
                row.innerHTML += `<td>${{typeBadge}}</td>`;

                row.innerHTML += `<td>${{config.weather}}</td>`;
                row.innerHTML += `<td>${{modelData.node_count}} nodes<br>${{modelData.edge_count}} edges</td>`;
                row.innerHTML += `<td>${{renderNodeTypes(config, modelData, configIdx)}}</td>`;
                row.innerHTML += `<td>${{renderEdgeTypes(config, modelData, configIdx)}}</td>`;
                tbody.appendChild(row);
            }});
        }}

        function renderNodeTypes(config, modelData, configIdx) {{
            let html = '';
            Object.entries(modelData.node_types).forEach(([nodeType, nodeInfo], nIdx) => {{
                const featureCount = Object.keys(nodeInfo.features).length;
                const expandId = `node-${{configIdx}}-${{nIdx}}`;
                html += `
                    <div class="node-type-container">
                        <div class="expandable" onclick="toggleExpand('${{expandId}}')">
                            <span class="expandable-icon" id="icon-${{expandId}}">▶</span>
                            <span class="feature-name">${{nodeType}}</span>
                            <span class="count-badge">${{nodeInfo.count}} nodes, ${{featureCount}} features</span>
                        </div>
                        <div class="expandable-content nested-content" id="${{expandId}}">
                            <ul class="feature-list">
                `;
                Object.entries(nodeInfo.features).forEach(([fname, finfo]) => {{
                    const sampleStr = finfo.sample !== null && finfo.sample !== undefined ?
                        String(finfo.sample).substring(0, 50) : 'NULL';
                    html += `
                        <li class="feature-item">
                            <span class="feature-name">${{fname}}</span>
                            <span class="feature-type">(${{finfo.type}})</span>
                            <span class="feature-sample">= ${{sampleStr}}</span>
                        </li>
                    `;
                }});
                html += `
                            </ul>
                        </div>
                    </div>
                `;
            }});
            return html;
        }}

        function renderEdgeTypes(config, modelData, configIdx) {{
            let html = '';
            Object.entries(modelData.edge_types).forEach(([edgeType, edgeInfo], eIdx) => {{
                const featureCount = Object.keys(edgeInfo.features).length;
                const expandId = `edge-${{configIdx}}-${{eIdx}}`;

                const isBidirectional = config.edge_bidirectional[edgeType] || false;
                const arrow = isBidirectional ? '<->' : '->';

                const edgeSrcTgt = config.edge_src_to_tgt[edgeType] || {{}};
                const srcNodeType = edgeSrcTgt.src || '';
                const tgtNodeType = edgeSrcTgt.tgt || '';

                let connectionStr = '';
                if (srcNodeType && tgtNodeType) {{
                    connectionStr = `${{srcNodeType}} ${{arrow}} ${{tgtNodeType}}`;
                }} else if (config.node_type_names.length === 1) {{
                    connectionStr = `${{config.node_type_names[0]}} ${{arrow}} ${{config.node_type_names[0]}}`;
                }} else if (config.node_type_names.length > 1) {{
                    connectionStr = `${{config.node_type_names[0]}} ${{arrow}} ${{config.node_type_names[1]}}`;
                }} else {{
                    connectionStr = `node ${{arrow}} node`;
                }}

                html += `
                    <div class="node-type-container">
                        <div class="expandable" onclick="toggleExpand('${{expandId}}')">
                            <span class="expandable-icon" id="icon-${{expandId}}">▶</span>
                            <span class="feature-name">${{edgeType}}</span>
                            <span class="small-text">(${{connectionStr}})</span>
                            <span class="count-badge">${{edgeInfo.count}} edges, ${{featureCount}} features</span>
                        </div>
                        <div class="expandable-content nested-content" id="${{expandId}}">
                            <ul class="feature-list">
                `;
                Object.entries(edgeInfo.features).forEach(([fname, finfo]) => {{
                    const sampleStr = finfo.sample !== null && finfo.sample !== undefined ?
                        String(finfo.sample).substring(0, 50) : 'NULL';
                    html += `
                        <li class="feature-item">
                            <span class="feature-name">${{fname}}</span>
                            <span class="feature-type">(${{finfo.type}})</span>
                            <span class="feature-sample">= ${{sampleStr}}</span>
                        </li>
                    `;
                }});
                html += `
                            </ul>
                        </div>
                    </div>
                `;
            }});
            return html;
        }}

        function toggleExpand(id) {{
            const content = document.getElementById(id);
            const icon = document.getElementById('icon-' + id);
            if (content.classList.contains('show')) {{
                content.classList.remove('show');
                icon.classList.remove('expanded');
            }} else {{
                content.classList.add('show');
                icon.classList.add('expanded');
            }}
        }}

        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>
"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")

    print("\nHTML table generated successfully!")
    print(f"  Output: {output_html}")
    print(f"  Configurations included: {len(configs_data_json)}")
    print(f"  Model IDs available: {len(model_ids)}")
    print(f"\n{'='*70}")
    print("SUMMARY TABLE GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOpen the table: file://{output_html.absolute()}")

    return output_html
