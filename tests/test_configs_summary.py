"""The configuration summary page labels homogeneous graphs as homogeneous."""

import json
import re

import networkx as nx

from beg.config.naming import CONFIGS
from beg.report.configs_summary_html import build_configs_summary_html


def _entries(html: str) -> list[dict]:
    match = re.search(r"const configsData = (\[.*?\]);\n", html, re.S)
    assert match, "embedded configuration data not found"
    return json.loads(match.group(1))


def test_graph_type_badges_follow_the_registry(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node("z0", node_type="zones_zone")
    graphs = {c.full_name: {"m": graph} for c in CONFIGS}
    # The notebook passes ConfigId.graph_type, the short spelling.
    meta = {c.full_name: {"id": c.id, "type": c.graph_type} for c in CONFIGS}

    out = build_configs_summary_html(graphs, tmp_path, list(graphs), tmp_path / "s.html",
                                     config_metadata_override=meta)
    types = {e["id"]: e["type"] for e in _entries(out.read_text())}

    assert types == {c.id: {"homo": "homogeneous", "hetero": "heterogeneous"}[c.graph_type]
                     for c in CONFIGS}
    assert {i for i, t in types.items() if t == "homogeneous"} == {"C1", "C2", "C6", "C7"}
