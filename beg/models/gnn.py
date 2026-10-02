# GNNmodels.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GINConv,
    GINEConv,
    HeteroConv,
    Set2Set,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import scatter


# Graph-level feature conditioning (optional, homo models)
DEFAULT_GRAPH_FEAT_KEY = "weather_city"
DEFAULT_GRAPH_FEAT_DIM = 223  # current weather_city length
# If True, replace NaN/Inf in graph-level features with zeros (in-memory only).
SANITIZE_GRAPH_FEATS = True
SANITIZE_FILL_VALUE = 0.0


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

GraphT = Union[Data, HeteroData]
EdgeType = Tuple[str, str, str]
Metadata = Tuple[List[str], List[EdgeType]]  # (node_types, edge_types)


# -----------------------------------------------------------------------------
# Small utilities (dims, pooling, safe x handling)
# -----------------------------------------------------------------------------

def infer_homo_in_dim(g: Data) -> int:
    x = getattr(g, "x", None)
    if isinstance(x, torch.Tensor) and x.ndim == 2:
        return int(x.size(-1))
    return 0


def infer_hetero_in_dims(g: HeteroData) -> Dict[str, int]:
    in_dims: Dict[str, int] = {}
    for nt in list(g.node_types):
        x = getattr(g[nt], "x", None)
        in_dims[nt] = int(x.size(-1)) if isinstance(x, torch.Tensor) and x.ndim == 2 else 0
    return in_dims


def infer_hetero_edge_dims(g: HeteroData) -> Dict[EdgeType, int]:
    edge_dims: Dict[EdgeType, int] = {}
    for et in list(g.edge_types):
        ea = getattr(g[et], "edge_attr", None)
        if isinstance(ea, torch.Tensor) and ea.ndim == 2:
            edge_dims[et] = int(ea.size(-1))
        else:
            # GINEConv requires an edge_dim; if edge_attr is missing, we treat it as a 1D zero feature.
            edge_dims[et] = 1
    return edge_dims


def hetero_default_pool_node_types(edge_types: Sequence[EdgeType]) -> List[str]:
    """
    Default pooling set for hetero graphs:
      - pool destination node types only (dst types in edge_types)
      - this naturally excludes node types that never appear as a destination
        (e.g., source-only context nodes like 'weather_city').
    """
    return sorted({tgt for (_, _, tgt) in edge_types})


def safe_node_x(
    data: HeteroData,
    node_type: str,
    *,
    device: torch.device,
    in_dim: int,
) -> torch.Tensor:
    """
    Return a feature matrix for a node store, even if x is missing/None.
    """
    store = data[node_type]
    num_nodes = int(getattr(store, "num_nodes", 0) or 0)

    x = getattr(store, "x", None)
    if not isinstance(x, torch.Tensor):
        return torch.zeros((num_nodes, int(in_dim)), device=device, dtype=torch.float32)
    x = x.to(device)
    if x.ndim != 2:
        raise ValueError(f"{node_type}.x must be rank-2, got shape={tuple(x.shape)}")
    return x.float()


def device_from_graph(g: GraphT) -> torch.device:
    if isinstance(g, Data):
        x = getattr(g, "x", None)
        if isinstance(x, torch.Tensor):
            return x.device
        ei = getattr(g, "edge_index", None)
        if isinstance(ei, torch.Tensor):
            return ei.device
        return torch.device("cpu")

    # HeteroData
    for nt in g.node_types:
        x = getattr(g[nt], "x", None)
        if isinstance(x, torch.Tensor):
            return x.device
    for et in g.edge_types:
        ei = getattr(g[et], "edge_index", None)
        if isinstance(ei, torch.Tensor):
            return ei.device
    return torch.device("cpu")


def get_or_make_batch(store, x: torch.Tensor) -> torch.Tensor:
    batch = getattr(store, "batch", None)
    if isinstance(batch, torch.Tensor):
        return batch
    # Single-graph inference: no batch vector -> assume one graph (all zeros)
    return torch.zeros((x.size(0),), device=x.device, dtype=torch.long)


# -----------------------------------------------------------------------------
# Helper: infer graph-level feature dimension (optional)
# -----------------------------------------------------------------------------

def infer_graph_feat_dim(g: Data, key: str) -> int:
    """
    Infer the feature dimension F of a graph-level tensor stored on a PyG Data object.

    Supports:
      - data.weather_city = tensor([F])      -> shape [F]
      - data.weather_city = tensor([[...]]) -> shape [1, F] (recommended per-graph shape)

    Returns:
      int: Feature dimension F (0 if missing).
    """
    v = getattr(g, key, None)
    if isinstance(v, torch.Tensor):
        if v.ndim == 2:
            # Recommended per-graph storage: [1, F]
            # (Batched -> [B, F])
            return int(v.size(-1))
        if v.ndim == 1:
            # Common manual storage: [F]
            # (We will unsqueeze in forward so it behaves like [1, F])
            return int(v.size(0))
    return 0


def get_graph_feature(
    data: Data,
    key: str,
    *,
    device: torch.device,
    fallback_dim: int,
) -> torch.Tensor:
    """
    Return a [B, F] graph-level feature tensor from `data.<key>`, robust to PyG batching.

    Supports:
      - per-graph storage:
          [1, F]  (recommended)  -> batched: [B, F]
          [F]     (common)       -> batched: [B*F] (PyG concatenation) OR [F] (single graph)

      - batched storage:
          [B, F]  (already aligned)

    Missing -> zeros([B, F]).

    Important:
      - This function is ONLY about *extracting* features.
      - Whether the model actually uses them is controlled in the model forward.
    """
    # Determine B (num graphs) robustly
    batch = getattr(data, "batch", None)
    if isinstance(batch, torch.Tensor) and batch.numel() > 0:
        num_graphs = int(batch.max().item()) + 1
    else:
        num_graphs = int(getattr(data, "num_graphs", 1) or 1)

    Fdim = int(fallback_dim)
    if Fdim <= 0:
        return torch.zeros((num_graphs, 0), device=device, dtype=torch.float32)

    v = getattr(data, key, None)
    if not isinstance(v, torch.Tensor):
        return torch.zeros((num_graphs, Fdim), device=device, dtype=torch.float32)

    v = v.to(device).float()

    # v: [F] (single graph) OR [B*F] (batched concatenation)
    if v.ndim == 1:
        if v.numel() == Fdim:
            out = v.view(1, Fdim)
            if num_graphs > 1:
                out = out.expand(num_graphs, Fdim)
        elif v.numel() == num_graphs * Fdim:
            out = v.view(num_graphs, Fdim)
        else:
            raise ValueError(
                f"Graph-level feature '{key}' unexpected 1D shape {tuple(v.shape)} "
                f"for num_graphs={num_graphs}, F={Fdim}"
            )

    # v: [1, F] (single graph) OR [B, F] (batched)
    elif v.ndim == 2:
        if v.size(-1) != Fdim:
            raise ValueError(f"Graph-level feature '{key}' has shape {tuple(v.shape)} but expected last dim {Fdim}")
        if v.size(0) == 1 and num_graphs > 1:
            out = v.expand(num_graphs, Fdim)
        elif v.size(0) == num_graphs:
            out = v
        else:
            raise ValueError(
                f"Graph-level feature '{key}' has shape {tuple(v.shape)} but expected first dim 1 or {num_graphs}"
            )
    else:
        raise ValueError(f"Graph-level feature '{key}' unsupported ndim={v.ndim}, shape={tuple(v.shape)}")

    if SANITIZE_GRAPH_FEATS:
        out = torch.nan_to_num(out, nan=SANITIZE_FILL_VALUE, posinf=SANITIZE_FILL_VALUE, neginf=SANITIZE_FILL_VALUE)

    return out


# -----------------------------------------------------------------------------
# HOMOGENEOUS MODELS (with optional graph-level feature conditioning)
# -----------------------------------------------------------------------------

class GCNRegressor(nn.Module):
    """
    Homogeneous baseline: L-layer GCN + mean pooling + MLP head.

    Optional graph-level conditioning (additive, NOT concatenation):
      - If enabled AND data has `graph_feat_key`, we do:
            g <- g + W(graph_feats)
        where W is a Linear(F -> hidden), bias=False so W(0)=0.
      - If disabled OR attribute missing, behavior is exactly baseline.
    """
    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        use_graph_feats: bool = False,
        graph_feat_dim: int = 0,
        graph_feat_key: str = DEFAULT_GRAPH_FEAT_KEY,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.dropout = float(dropout)

        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.use_graph_feats = bool(use_graph_feats)
        self.graph_feat_dim = int(graph_feat_dim)
        self.graph_feat_key = str(graph_feat_key)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        self.graph_feat_proj = None
        if self.use_graph_feats:
            if self.graph_feat_dim <= 0:
                raise ValueError("use_graph_feats=True requires graph_feat_dim > 0")
            self.graph_feat_proj = nn.Linear(self.graph_feat_dim, hidden, bias=False)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < (self.num_layers - 1):
                x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_mean_pool(x, batch)  # [B, hidden]

        if self.use_graph_feats and hasattr(data, self.graph_feat_key):
            w = get_graph_feature(data, self.graph_feat_key, device=g.device, fallback_dim=self.graph_feat_dim)  # [B, F]
            g = g + self.graph_feat_proj(w)

        return self.head(g).view(-1)


class GraphSAGEAttentionRegressor(nn.Module):
    """
    Homogeneous: L-layer GraphSAGE + AttentionalAggregation pooling + MLP head.

    Optional graph-level conditioning (additive):
      - enabled: g <- g + W(graph_feats) if attribute exists
      - disabled/missing: baseline behavior
    """
    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        use_graph_feats: bool = False,
        graph_feat_dim: int = 0,
        graph_feat_key: str = DEFAULT_GRAPH_FEAT_KEY,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.dropout = float(dropout)

        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.use_graph_feats = bool(use_graph_feats)
        self.graph_feat_dim = int(graph_feat_dim)
        self.graph_feat_key = str(graph_feat_key)

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(self.num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        )

        self.graph_feat_proj = None
        if self.use_graph_feats:
            if self.graph_feat_dim <= 0:
                raise ValueError("use_graph_feats=True requires graph_feat_dim > 0")
            self.graph_feat_proj = nn.Linear(self.graph_feat_dim, hidden, bias=False)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < (self.num_layers - 1):
                x = F.dropout(x, p=self.dropout, training=self.training)

        g = self.pool(x, batch)  # [B, hidden]

        if self.use_graph_feats and hasattr(data, self.graph_feat_key):
            w = get_graph_feature(data, self.graph_feat_key, device=g.device, fallback_dim=self.graph_feat_dim)
            g = g + self.graph_feat_proj(w)

        return self.head(g).view(-1)


class GINRegressor(nn.Module):
    """
    Homogeneous: L-layer GIN + pooling (mean or Set2Set) + MLP head.

    Optional graph-level conditioning (additive):
      - enabled: g <- g + W(graph_feats) if attribute exists
      - disabled/missing: baseline behavior
    """
    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        pooling: str = "mean",
        set2set_steps: int = 3,
        use_graph_feats: bool = False,
        graph_feat_dim: int = 0,
        graph_feat_key: str = DEFAULT_GRAPH_FEAT_KEY,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.dropout = float(dropout)

        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.pooling = str(pooling).lower()
        self.set2set_steps = int(set2set_steps)

        self.use_graph_feats = bool(use_graph_feats)
        self.graph_feat_dim = int(graph_feat_dim)
        self.graph_feat_key = str(graph_feat_key)

        def _gin_mlp(din: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(din, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

        self.convs = nn.ModuleList()
        self.convs.append(GINConv(_gin_mlp(in_channels)))
        for _ in range(self.num_layers - 1):
            self.convs.append(GINConv(_gin_mlp(hidden)))

        if self.pooling == "set2set":
            self.pool = Set2Set(hidden, processing_steps=self.set2set_steps)
            pooled_dim = 2 * hidden
        elif self.pooling == "mean":
            self.pool = None
            pooled_dim = hidden
        else:
            raise ValueError(f"Unsupported pooling='{pooling}'. Use 'mean' or 'set2set'.")

        self.graph_feat_proj = None
        if self.use_graph_feats:
            if self.graph_feat_dim <= 0:
                raise ValueError("use_graph_feats=True requires graph_feat_dim > 0")
            self.graph_feat_proj = nn.Linear(self.graph_feat_dim, pooled_dim, bias=False)

        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < (self.num_layers - 1):
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pool is None:
            g = global_mean_pool(x, batch)  # [B, hidden]
        else:
            g = self.pool(x, batch)         # [B, 2*hidden]

        if self.use_graph_feats and hasattr(data, self.graph_feat_key):
            w = get_graph_feature(data, self.graph_feat_key, device=g.device, fallback_dim=self.graph_feat_dim)  # [B, F]
            g = g + self.graph_feat_proj(w)

        return self.head(g).view(-1)


# -----------------------------------------------------------------------------
# HETEROGENEOUS POOLING
# -----------------------------------------------------------------------------

class PerTypeAttentionPooling(nn.Module):
    """
    Attention pooling per node type, then combine pooled vectors (sum).
    """
    def __init__(self, hidden: int, node_types: Sequence[str]):
        super().__init__()
        self.hidden = int(hidden)
        self.node_types = list(node_types)

        self.pools = nn.ModuleDict()
        for nt in self.node_types:
            gate = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.pools[nt] = AttentionalAggregation(gate_nn=gate)

    def forward(self, x_dict: Dict[str, torch.Tensor], data: HeteroData, pool_node_types: Sequence[str]) -> torch.Tensor:
        g_vec: Optional[torch.Tensor] = None

        for nt in pool_node_types:
            if nt not in data.node_types:
                continue
            x = x_dict.get(nt, None)
            if x is None:
                continue

            batch = get_or_make_batch(data[nt], x)

            if nt in self.pools:
                pooled = self.pools[nt](x, batch)
            else:
                pooled = global_mean_pool(x, batch)

            g_vec = pooled if g_vec is None else (g_vec + pooled)

        if g_vec is None:
            dev = device_from_graph(data)
            num_graphs = int(getattr(data, "num_graphs", 1) or 1)
            g_vec = torch.zeros((num_graphs, self.hidden), device=dev, dtype=torch.float32)

        return g_vec


# -----------------------------------------------------------------------------
# HETEROGENEOUS MODELS
# -----------------------------------------------------------------------------

class HeteroGINRegressor(nn.Module):
    """
    Heterogeneous GIN using per-relation GINEConv inside HeteroConv.

    Pooling modes (graph-level):
      - "per_type_mean": mean-pool nodes per type, then average type embeddings
      - "per_type_sum":  sum-pool nodes per type, then average type embeddings
      - "per_type_attention": mean-pool nodes per type, then softmax-weight type embeddings

    Notes:
      - Supports missing edge_attr by substituting zeros (edge_dim=1).
      - Preserves node representations for types that receive no messages in a layer.
      - Optional LayerNorm after each message-passing layer (`use_layernorm=True`).
    """
    def __init__(
        self,
        metadata: Metadata,
        in_dims: Mapping[str, int],
        edge_dims: Mapping[EdgeType, int],
        hidden: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        pool_node_types: Optional[Sequence[str]] = None,
        pooling: str = "per_type_attention",
        rel_aggr: str = "sum",
        use_layernorm: bool = False,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

        self.hidden = int(hidden)
        self.dropout = float(dropout)

        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.in_dims = {str(k): int(v) for k, v in in_dims.items()}
        self.edge_dims = {et: int(edge_dims.get(et, 1)) for et in self.edge_types}

        if pool_node_types is None:
            self.pool_node_types = hetero_default_pool_node_types(self.edge_types)
        else:
            self.pool_node_types = list(pool_node_types)

        self.pooling = str(pooling).lower()
        if self.pooling not in {"per_type_mean", "per_type_sum", "per_type_attention"}:
            raise ValueError("pooling must be 'per_type_mean', 'per_type_sum', or 'per_type_attention'")

        self.rel_aggr = str(rel_aggr)
        self.use_layernorm = bool(use_layernorm)

        # -------------------------
        # Node encoders: type-specific x -> hidden
        # -------------------------
        self.node_enc = nn.ModuleDict()
        for nt in self.node_types:
            din = int(self.in_dims.get(nt, 0))
            if din <= 0:
                self.node_enc[nt] = nn.Identity()
            else:
                self.node_enc[nt] = nn.Sequential(
                    nn.Linear(din, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden),
                )

        # -------------------------
        # Message passing layers
        # -------------------------
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            conv_dict: Dict[EdgeType, nn.Module] = {}
            for rel in self.edge_types:
                mlp = nn.Sequential(
                    nn.Linear(self.hidden, self.hidden),
                    nn.ReLU(),
                    nn.Linear(self.hidden, self.hidden),
                )
                conv_dict[rel] = GINEConv(nn=mlp, edge_dim=int(self.edge_dims.get(rel, 1)))
            self.convs.append(HeteroConv(conv_dict, aggr=self.rel_aggr))

        # -------------------------
        # Optional post-layer normalization
        # -------------------------
        self.layer_norms = (
            nn.ModuleList([nn.LayerNorm(self.hidden) for _ in range(self.num_layers)])
            if self.use_layernorm
            else None
        )

        # -------------------------
        # Attention over pooled node-type embeddings (only if pooling == per_type_attention)
        # -------------------------
        self.type_att = nn.ModuleDict()
        if self.pooling == "per_type_attention":
            for nt in self.pool_node_types:
                self.type_att[str(nt)] = nn.Linear(self.hidden, 1)

        # -------------------------
        # Head
        # -------------------------
        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )

    def _pool_per_type(self, x: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        if self.pooling == "per_type_sum":
            return global_add_pool(x, batch_vec)
        return global_mean_pool(x, batch_vec)

    def _pool(self, x_dict: Dict[str, torch.Tensor], data: HeteroData) -> torch.Tensor:
        pooled_by_type: List[torch.Tensor] = []
        type_names: List[str] = []

        for nt in self.pool_node_types:
            if nt not in data.node_types:
                continue
            x = x_dict.get(nt, None)
            if x is None:
                continue
            b = get_or_make_batch(data[nt], x)
            pooled_by_type.append(self._pool_per_type(x, b))
            type_names.append(nt)

        if not pooled_by_type:
            dev = device_from_graph(data)
            num_graphs = int(getattr(data, "num_graphs", 1) or 1)
            return torch.zeros((num_graphs, self.hidden), device=dev, dtype=torch.float32)

        if self.pooling in {"per_type_mean", "per_type_sum"}:
            return torch.stack(pooled_by_type, dim=0).mean(dim=0)

        # per_type_attention:
        scores: List[torch.Tensor] = []
        for pooled, nt in zip(pooled_by_type, type_names):
            key = str(nt)
            att = self.type_att[key] if key in self.type_att else None
            if att is None:
                scores.append(torch.zeros((pooled.size(0), 1), device=pooled.device, dtype=pooled.dtype))
            else:
                scores.append(att(pooled))  # [B, 1]
        score_mat = torch.cat(scores, dim=1)                # [B, T]
        w = torch.softmax(score_mat, dim=1)                 # [B, T]
        pooled_stack = torch.stack(pooled_by_type, dim=1)   # [B, T, H]
        return (w.unsqueeze(-1) * pooled_stack).sum(dim=1)  # [B, H]

    def forward(self, data: HeteroData) -> torch.Tensor:
        dev = device_from_graph(data)

        # Encode nodes -> hidden
        x_dict: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            din = int(self.in_dims.get(nt, 0))
            x = safe_node_x(data, nt, device=dev, in_dim=din)
            enc = self.node_enc[nt]
            if isinstance(enc, nn.Identity):
                x_dict[nt] = torch.zeros((x.size(0), self.hidden), device=dev, dtype=torch.float32)
            else:
                x_dict[nt] = enc(x)

        # Edge attrs dict (zeros if missing)
        edge_attr_dict: Dict[EdgeType, torch.Tensor] = {}
        for rel in self.edge_types:
            ei = data.edge_index_dict.get(rel, None)
            if ei is None:
                continue
            ea = getattr(data[rel], "edge_attr", None)

            exp_dim = int(self.edge_dims.get(rel, 1))
            if not isinstance(ea, torch.Tensor):
                edge_attr_dict[rel] = torch.zeros((int(ei.size(1)), exp_dim), device=dev, dtype=torch.float32)
                continue

            ea = ea.to(dev).float()
            if ea.ndim == 1:
                ea = ea.view(-1, 1)
            if ea.ndim != 2:
                raise ValueError(f"{rel}.edge_attr must be rank-2, got shape={tuple(ea.shape)}")
            if int(ea.size(-1)) != exp_dim:
                raise ValueError(f"{rel}.edge_attr dim mismatch: expected {exp_dim}, got {int(ea.size(-1))}")
            edge_attr_dict[rel] = ea

        # Message passing
        for layer_idx, conv in enumerate(self.convs):
            x_prev = x_dict
            out_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)

            # Preserve representations for node types that didn't get updated in this layer
            x_dict = {}
            for nt, prev in x_prev.items():
                h = out_dict.get(nt, None)
                if h is None:
                    h = prev
                if self.layer_norms is not None:
                    h = self.layer_norms[layer_idx](h)
                x_dict[nt] = F.dropout(F.relu(h), p=self.dropout, training=self.training)

        g = self._pool(x_dict, data)
        return self.head(g).view(-1)


class RGCNLikeHeteroRegressor(nn.Module):
    """
    Heterogeneous "R-GCN-like" regressor WITHOUT converting HeteroData to a homogeneous graph.

    Implementation:
      - per-node-type encoder to hidden
      - per-relation linear transform applied to source node embeddings
      - scatter-sum messages into destination nodes
      - destination node update: ReLU(m_agg) (+ optional residual)
      - repeat for L layers
      - pool only destination node types by default (excludes source-only types like 'weather_city')
      - per-type attention pooling (or per-type mean) over pooled node types
    """
    def __init__(
        self,
        metadata: Metadata,
        in_dims: Mapping[str, int],
        hidden: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        pool_node_types: Optional[Sequence[str]] = None,
        pooling: str = "per_type_attention",  # "per_type_attention" or "per_type_mean"
        use_residual: bool = True,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

        self.hidden = int(hidden)
        self.dropout = float(dropout)

        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.use_residual = bool(use_residual)
        self.in_dims = {str(k): int(v) for k, v in in_dims.items()}

        if pool_node_types is None:
            self.pool_node_types = hetero_default_pool_node_types(self.edge_types)
        else:
            self.pool_node_types = list(pool_node_types)

        # encoders to hidden
        self.encoders = nn.ModuleDict()
        for nt in self.node_types:
            din = int(self.in_dims.get(nt, 0))
            self.encoders[nt] = nn.Linear(din, hidden) if din > 0 else nn.Identity()

        # relation-specific transforms per layer: ModuleList[ModuleDict]
        self.rel_lins = nn.ModuleList()
        for _ in range(self.num_layers):
            md = nn.ModuleDict()
            for et in self.edge_types:
                key = self._etype_key(et)
                md[key] = nn.Linear(hidden, hidden, bias=False)
            self.rel_lins.append(md)

        self.pooling = str(pooling).lower()
        if self.pooling == "per_type_attention":
            self.pool = PerTypeAttentionPooling(hidden=hidden, node_types=self.node_types)
        elif self.pooling == "per_type_mean":
            self.pool = None
        else:
            raise ValueError("pooling must be 'per_type_attention' or 'per_type_mean'")

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _etype_key(et: EdgeType) -> str:
        return f"{et[0]}__{et[1]}__{et[2]}"

    def _encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        dev = device_from_graph(data)
        x_dict: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            din = int(self.in_dims.get(nt, 0))
            x = safe_node_x(data, nt, device=dev, in_dim=din)
            enc = self.encoders[nt]
            if isinstance(enc, nn.Identity):
                xh = torch.zeros((x.size(0), self.hidden), device=dev, dtype=torch.float32)
            else:
                xh = enc(x)
            x_dict[nt] = F.relu(xh)
        return x_dict

    def _rgcn_layer(
        self,
        data: HeteroData,
        x_dict: Dict[str, torch.Tensor],
        rel_lins: nn.ModuleDict,
    ) -> Dict[str, torch.Tensor]:
        dev = device_from_graph(data)

        # initialize accumulators per node type
        out: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            n = int(getattr(data[nt], "num_nodes", 0) or 0)
            out[nt] = torch.zeros((n, self.hidden), device=dev, dtype=torch.float32)

        # relation-wise message passing: msg = W_r x_src; aggregate on dst with scatter-sum
        for et in self.edge_types:
            if et not in data.edge_types:
                continue
            ei = getattr(data[et], "edge_index", None)
            if not isinstance(ei, torch.Tensor) or ei.numel() == 0:
                continue

            src, _, dst = et
            key = self._etype_key(et)
            W = rel_lins[key]

            src_x = x_dict.get(src, None)
            dst_acc = out.get(dst, None)
            if src_x is None or dst_acc is None:
                continue

            src_idx = ei[0].to(dev)
            dst_idx = ei[1].to(dev)

            msg = W(src_x[src_idx])  # [E, hidden]
            agg = scatter(msg, dst_idx, dim=0, dim_size=dst_acc.size(0), reduce="sum")  # [N_dst, hidden]
            out[dst] = out[dst] + agg

        # update (optionally residual)
        new_x: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            h = F.relu(out[nt])
            if self.use_residual and nt in x_dict and x_dict[nt].shape == h.shape:
                h = h + x_dict[nt]
            new_x[nt] = h
        return new_x

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = self._encode(data)

        for i in range(self.num_layers):
            x_dict = self._rgcn_layer(data, x_dict, self.rel_lins[i])
            if i < (self.num_layers - 1):
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        if self.pool is None:
            # per-type mean pooling + sum
            g_vec: Optional[torch.Tensor] = None
            for nt in self.pool_node_types:
                if nt not in data.node_types:
                    continue
                x = x_dict.get(nt, None)
                if x is None:
                    continue
                batch = get_or_make_batch(data[nt], x)
                pooled = global_mean_pool(x, batch)
                g_vec = pooled if g_vec is None else (g_vec + pooled)

            if g_vec is None:
                dev = device_from_graph(data)
                num_graphs = int(getattr(data, "num_graphs", 1) or 1)
                g_vec = torch.zeros((num_graphs, self.hidden), device=dev, dtype=torch.float32)
        else:
            g_vec = self.pool(x_dict, data, self.pool_node_types)

        return self.head(g_vec).view(-1)


class HeteroSAGERegressor(nn.Module):
    """
    Heterogeneous GraphSAGE via per-relation HeteroConv(SAGEConv) + per-type pooling.

    Key behavior:
      - pool only destination node types by default (excludes source-only nodes like 'weather_city')
      - supports per-type attention pooling (recommended) or per-type mean pooling
      - handles x=None and x with 0 dims robustly
      - ignores edge_attr
    """
    def __init__(
        self,
        metadata: Metadata,
        in_dims: Mapping[str, int],
        hidden: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        pool_node_types: Optional[Sequence[str]] = None,
        pooling: str = "per_type_attention",  # "per_type_attention" or "per_type_mean"
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

        self.hidden = int(hidden)
        self.dropout = float(dropout)

        self.num_layers = int(num_layers)
        if self.num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.in_dims = {str(k): int(v) for k, v in in_dims.items()}

        if pool_node_types is None:
            self.pool_node_types = hetero_default_pool_node_types(self.edge_types)
        else:
            self.pool_node_types = list(pool_node_types)

        # encoders to hidden
        self.encoders = nn.ModuleDict()
        for nt in self.node_types:
            din = int(self.in_dims.get(nt, 0))
            self.encoders[nt] = nn.Linear(din, hidden) if din > 0 else nn.Identity()

        # L-layer per-relation SAGE
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in self.edge_types}, aggr="sum")
            )

        self.pooling = str(pooling).lower()
        if self.pooling == "per_type_attention":
            self.pool = PerTypeAttentionPooling(hidden=hidden, node_types=self.node_types)
        elif self.pooling == "per_type_mean":
            self.pool = None
        else:
            raise ValueError("pooling must be 'per_type_attention' or 'per_type_mean'")

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, 1),
        )

    def _encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        dev = device_from_graph(data)
        x_dict: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            din = int(self.in_dims.get(nt, 0))
            x = safe_node_x(data, nt, device=dev, in_dim=din)
            enc = self.encoders[nt]
            if isinstance(enc, nn.Identity):
                xh = torch.zeros((x.size(0), self.hidden), device=dev, dtype=torch.float32)
            else:
                xh = enc(x)
            x_dict[nt] = F.relu(xh)
        return x_dict

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = self._encode(data)

        for i, conv in enumerate(self.convs):
            # HeteroConv returns only destination node types; keep old reps for non-updated types.
            out = conv(x_dict, data.edge_index_dict)
            x_dict = {**x_dict, **out}  # keep old reps for non-updated types (e.g., weather_city)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            if i < (self.num_layers - 1):
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        if self.pool is None:
            # per-type mean pooling + sum
            g_vec: Optional[torch.Tensor] = None
            for nt in self.pool_node_types:
                if nt not in data.node_types:
                    continue
                x = x_dict.get(nt, None)
                if x is None:
                    continue
                batch = get_or_make_batch(data[nt], x)
                pooled = global_mean_pool(x, batch)
                g_vec = pooled if g_vec is None else (g_vec + pooled)

            if g_vec is None:
                dev = device_from_graph(data)
                num_graphs = int(getattr(data, "num_graphs", 1) or 1)
                g_vec = torch.zeros((num_graphs, self.hidden), device=dev, dtype=torch.float32)
        else:
            g_vec = self.pool(x_dict, data, self.pool_node_types)

        return self.head(g_vec).view(-1)


# -----------------------------------------------------------------------------
# ModelSpec + registry (for summaries, reproducible instantiation)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    graph_kind: str  # "homo" or "hetero"
    family: str
    description: str
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    def create(self, sample_graph: GraphT, **overrides) -> nn.Module:
        """
        Instantiate model from a representative graph (to infer dims/metadata).
        """
        kwargs = dict(self.default_kwargs)
        kwargs.update(overrides)

        if self.graph_kind == "homo":
            if not isinstance(sample_graph, Data):
                raise TypeError(f"{self.name}: expected Data sample_graph, got {type(sample_graph)}")
            in_dim = infer_homo_in_dim(sample_graph)

            # --- graph-level features (optional; additive conditioning; backward compatible) ---
            graph_feat_key = str(kwargs.pop("graph_feat_key", DEFAULT_GRAPH_FEAT_KEY))

            # allow explicit override from experiment config:
            use_graph_feats = kwargs.pop("use_graph_feats", None)
            if use_graph_feats is None:
                # default: enable ONLY if the attribute exists on the sample graph
                use_graph_feats = hasattr(sample_graph, graph_feat_key)
            use_graph_feats = bool(use_graph_feats)

            # dimension: prefer explicit override, then inferred, then default fallback
            if "graph_feat_dim" in kwargs:
                graph_feat_dim = int(kwargs.pop("graph_feat_dim"))
            else:
                inferred = infer_graph_feat_dim(sample_graph, graph_feat_key)
                graph_feat_dim = int(inferred if inferred > 0 else DEFAULT_GRAPH_FEAT_DIM)

            if not use_graph_feats:
                graph_feat_dim = 0  # keep clean invariant

            if self.name == "GCNRegressor":
                return GCNRegressor(
                    in_channels=in_dim,
                    use_graph_feats=use_graph_feats,
                    graph_feat_dim=graph_feat_dim,
                    graph_feat_key=graph_feat_key,
                    **kwargs,
                )
            if self.name == "GraphSAGEAttentionRegressor":
                return GraphSAGEAttentionRegressor(
                    in_channels=in_dim,
                    use_graph_feats=use_graph_feats,
                    graph_feat_dim=graph_feat_dim,
                    graph_feat_key=graph_feat_key,
                    **kwargs,
                )
            if self.name == "GINRegressor":
                return GINRegressor(
                    in_channels=in_dim,
                    use_graph_feats=use_graph_feats,
                    graph_feat_dim=graph_feat_dim,
                    graph_feat_key=graph_feat_key,
                    **kwargs,
                )

            raise KeyError(f"Unknown homo model name: {self.name}")

        if self.graph_kind == "hetero":
            if not isinstance(sample_graph, HeteroData):
                raise TypeError(f"{self.name}: expected HeteroData sample_graph, got {type(sample_graph)}")
            node_types = list(sample_graph.node_types)
            edge_types = list(sample_graph.edge_types)
            in_dims = infer_hetero_in_dims(sample_graph)
            edge_dims = infer_hetero_edge_dims(sample_graph)
            metadata: Metadata = (node_types, edge_types)

            # default pooling set (dst node types only)
            pool_types = kwargs.pop("pool_node_types", None)
            if pool_types is None:
                pool_types = hetero_default_pool_node_types(edge_types)

            if self.name == "RGCNLikeHeteroRegressor":
                return RGCNLikeHeteroRegressor(
                    metadata=metadata,
                    in_dims=in_dims,
                    pool_node_types=pool_types,
                    **kwargs,
                )
            if self.name == "HeteroGINRegressor":
                return HeteroGINRegressor(
                    metadata=metadata,
                    in_dims=in_dims,
                    edge_dims=edge_dims,
                    pool_node_types=pool_types,
                    **kwargs,
                )
            if self.name == "HeteroSAGERegressor":
                return HeteroSAGERegressor(
                    metadata=metadata,
                    in_dims=in_dims,
                    pool_node_types=pool_types,
                    **kwargs,
                )
            raise KeyError(f"Unknown hetero model name: {self.name}")

        raise ValueError(f"Invalid graph_kind={self.graph_kind}")


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # HOMO
    "gcn": ModelSpec(
        name="GCNRegressor",
        graph_kind="homo",
        family="GCN",
        description="GCN regressor (default: 2 layers) with mean pooling; optional additive graph-level feature conditioning.",
        default_kwargs={"hidden": 128, "dropout": 0.1, "num_layers": 2},
    ),
    "sage_attn": ModelSpec(
        name="GraphSAGEAttentionRegressor",
        graph_kind="homo",
        family="GraphSAGE",
        description="GraphSAGE regressor (default: 2 layers) with attention pooling; optional additive graph-level feature conditioning.",
        default_kwargs={"hidden": 128, "dropout": 0.1, "num_layers": 2},
    ),
    "gin_mean": ModelSpec(
        name="GINRegressor",
        graph_kind="homo",
        family="GIN",
        description="GIN regressor (default: 2 layers) with mean pooling; optional additive graph-level feature conditioning.",
        default_kwargs={"hidden": 128, "dropout": 0.1, "num_layers": 2, "pooling": "mean"},
    ),
    "gin_set2set": ModelSpec(
        name="GINRegressor",
        graph_kind="homo",
        family="GIN",
        description="GIN regressor (default: 2 layers) with Set2Set pooling; optional additive graph-level feature conditioning.",
        default_kwargs={"hidden": 128, "dropout": 0.1, "num_layers": 2, "pooling": "set2set", "set2set_steps": 3},
    ),

    # HETERO
    "rgcn": ModelSpec(
        name="RGCNLikeHeteroRegressor",
        graph_kind="hetero",
        family="R-GCN-like",
        description=(
            "Typed, relation-specific linear message passing (R-GCN core idea) without converting HeteroData to homo; "
            "pools destination node types only by default (excludes source-only nodes like 'weather_city'). "
            "Default: 2 layers."
        ),
        default_kwargs={"hidden": 128, "dropout": 0.1, "num_layers": 2, "pooling": "per_type_attention", "use_residual": True},
    ),
    "heterogin": ModelSpec(
        name="HeteroGINRegressor",
        graph_kind="hetero",
        family="HeteroGIN",
        description=(
            "Per-relation HeteroConv(GINEConv) with per-type pooling; pools destination node types only by default "
            "(excludes source-only nodes like 'weather_city'). Default: 2 layers."
        ),
        default_kwargs={"hidden": 128, "dropout": 0.1, "num_layers": 2, "pooling": "per_type_attention"},
    ),
    # Alias (older notebooks/scripts)
    "hetero_gin": ModelSpec(
        name="HeteroGINRegressor",
        graph_kind="hetero",
        family="HeteroGIN",
        description="Alias for 'heterogin'.",
        default_kwargs={"hidden": 128, "dropout": 0.1, "num_layers": 2, "pooling": "per_type_attention"},
    ),
    "heterosage": ModelSpec(
        name="HeteroSAGERegressor",
        graph_kind="hetero",
        family="HeteroSAGE",
        description=(
            "Per-relation HeteroConv(SAGEConv) with per-type pooling; pools destination node types only by default "
            "(excludes source-only nodes like 'weather_city'). Default: 2 layers."
        ),
        default_kwargs={"hidden": 128, "dropout": 0.1, "num_layers": 2, "pooling": "per_type_attention"},
    ),
}


def list_models(graph_kind: Optional[str] = None) -> List[str]:
    keys = sorted(MODEL_REGISTRY.keys())
    if graph_kind is None:
        return keys
    return [k for k in keys if MODEL_REGISTRY[k].graph_kind == graph_kind]


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key='{key}'. Available: {list_models()}")
    return MODEL_REGISTRY[key]


def create_model(key: str, sample_graph: GraphT, **overrides) -> nn.Module:
    """
    Convenience wrapper:
      model = create_model("heterosage", g0, hidden=256, dropout=0.2, num_layers=3)
    """
    spec = get_model_spec(key)
    return spec.create(sample_graph, **overrides)


def summarize_registry() -> List[Dict[str, Any]]:
    """
    Handy for experiment tables/logging.
    """
    rows = []
    for k in list_models():
        s = MODEL_REGISTRY[k]
        rows.append(
            {
                "key": k,
                "name": s.name,
                "graph_kind": s.graph_kind,
                "family": s.family,
                "description": s.description,
                "default_kwargs": dict(s.default_kwargs),
            }
        )
    return rows
