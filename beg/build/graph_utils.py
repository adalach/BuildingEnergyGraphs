"""
Graph utility functions.

Common utilities for graph construction (NetworkX and PyTorch Geometric).
"""

from __future__ import annotations

from typing import Any


def pyg_to_nx(
    graph: Any,
    *,
    node_type: str = "node",
    edge_type: str = "unified",
):
    """A NetworkX view of an exported PyG graph: topology plus node and edge types.

    Features are not copied; for attribute-level reporting use
    :func:`beg.build.build_nx_graph_for_model`.

    ``node_type`` and ``edge_type`` name the single type of a homogeneous graph.
    """
    import networkx as nx

    G = nx.MultiDiGraph()

    if hasattr(graph, "node_types"):
        for ntype in graph.node_types:
            for i in range(int(graph[ntype].num_nodes)):
                G.add_node(f"{ntype}:{i}", node_type=ntype, index=i)
        for etype in graph.edge_types:
            src, rel, tgt = etype
            edge_index = getattr(graph[etype], "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                continue
            for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True):
                G.add_edge(
                    f"{src}:{s}", f"{tgt}:{t}",
                    rel=rel, edge_type=rel, src_type=src, tgt_type=tgt,
                )
        return G

    for i in range(int(graph.num_nodes)):
        G.add_node(f"{node_type}:{i}", node_type=node_type, index=i)
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None and edge_index.numel():
        for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True):
            G.add_edge(
                f"{node_type}:{s}", f"{node_type}:{t}",
                rel=edge_type, edge_type=edge_type,
                src_type=node_type, tgt_type=node_type,
            )
    return G
