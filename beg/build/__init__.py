"""Graph construction from the pipeline's node and edge DataFrames.

NetworkX builders are exported here directly. The PyTorch Geometric builders
live in :mod:`beg.build.pyg_homo` and :mod:`beg.build.pyg_hetero` and are
imported explicitly, so that the data half of the pipeline stays usable on a
machine without torch installed.
"""

from beg.build.networkx_graphs import (
    BuiltGraph,
    build_nx_graph_for_model,
)

__all__ = [
    "BuiltGraph",
    "build_nx_graph_for_model",
]
