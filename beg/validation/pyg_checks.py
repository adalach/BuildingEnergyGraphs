"""
Checks on PyTorch-Geometric datasets.

Two artefact kinds are covered:

* per-model graph files ``outputs/pyg_graphs/{homo|hetero}/<Cn_config>/<model_id>.pt``
  holding a ``Data`` or ``HeteroData`` (plus the parquet manifests)
* the training packages ``outputs/pyg_datasets/<Cn_config>/training_ready.pt``
  exported by ``beg.datasets.tools.export_training_ready_package``

``torch`` / ``torch_geometric`` are imported lazily; when they are missing the
checks return SKIP results instead of raising, so the rest of the suite still
runs in light-weight environments.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from beg.validation.core import CheckResult, examples
from beg.validation.spec import ConfigSpec

SCOPE = "pyg"
Triple = Tuple[str, str, str]


def _torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:  # pragma: no cover
        return None


def _pyg_types():
    try:
        from torch_geometric.data import Data, HeteroData  # type: ignore

        return Data, HeteroData
    except Exception:  # pragma: no cover
        return None, None


def torch_available() -> bool:
    return _torch() is not None and _pyg_types()[0] is not None


def load_graph_file(path: Path) -> Any:
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch not available")
    try:
        return torch.load(path, weights_only=False, map_location="cpu")
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


# --------------------------------------------------------------------------
# tensor helpers
# --------------------------------------------------------------------------

def _finite_report(x, name: str, kw: dict) -> List[CheckResult]:
    torch = _torch()
    r: List[CheckResult] = []
    if x is None or x.numel() == 0:
        return r
    if not torch.is_floating_point(x):
        return r
    nan = int(torch.isnan(x).sum())
    inf = int(torch.isinf(x).sum())
    if nan or inf:
        r.append(CheckResult.fail("pyg.tensor_finite", f"{name}: {nan} NaN and {inf} inf entries", **kw))
    return r


def _edge_index_report(
    ei,
    n_src: int,
    n_tgt: int,
    name: str,
    kw: dict,
    *,
    allow_self_loops: bool = False,
    same_node_type: bool = True,
) -> List[CheckResult]:
    """Structural checks on one edge_index.

    `same_node_type` says whether source and target index the *same* node set.
    Only then does `edge_index[0] == edge_index[1]` mean a self loop: in a
    heterogeneous store like ("comp_element", "part_of", "comp_aperture"), index 3
    on each side is a different node.
    """
    torch = _torch()
    r: List[CheckResult] = []
    if ei is None:
        r.append(CheckResult.fail("pyg.edge_index_present", f"{name}: edge_index missing", **kw))
        return r
    if ei.dim() != 2 or ei.size(0) != 2:
        r.append(CheckResult.fail("pyg.edge_index_shape", f"{name}: edge_index shape {tuple(ei.shape)} (expected [2, E])", **kw))
        return r
    if ei.dtype != torch.long:
        r.append(CheckResult.fail("pyg.edge_index_dtype", f"{name}: edge_index dtype {ei.dtype} (expected int64)", **kw))
    if ei.numel() == 0:
        return r
    if int(ei[0].min()) < 0 or int(ei[0].max()) >= n_src or int(ei[1].min()) < 0 or int(ei[1].max()) >= n_tgt:
        r.append(CheckResult.fail("pyg.edge_index_bounds", f"{name}: edge_index out of bounds (src∈[{int(ei[0].min())},{int(ei[0].max())}] of {n_src}, tgt∈[{int(ei[1].min())},{int(ei[1].max())}] of {n_tgt})", **kw))
    if not allow_self_loops and same_node_type:
        loops = int((ei[0] == ei[1]).sum())
        if loops:
            r.append(CheckResult.fail("pyg.no_self_loops", f"{name}: {loops} self loops", **kw))
    # duplicates
    key = ei[0] * max(n_tgt, 1) + ei[1]
    dups = int(key.numel() - torch.unique(key).numel())
    if dups:
        r.append(CheckResult.fail("pyg.no_duplicate_edges", f"{name}: {dups} duplicated (src,tgt) pairs", **kw))
    return r


def _is_symmetric(ei, n: int) -> Tuple[bool, int]:
    if ei.numel() == 0:
        return True, 0
    fwd = set((ei[0] * n + ei[1]).tolist())
    rev = set((ei[1] * n + ei[0]).tolist())
    return fwd == rev, len(fwd - rev)


# --------------------------------------------------------------------------
# single graph
# --------------------------------------------------------------------------

def check_homo_data(data: Any, spec: Optional[ConfigSpec], *, model_id: Optional[str] = None, expect_symmetric: Optional[bool] = None, graph_feat_key: Optional[str] = None, graph_feat_dim: Optional[int] = None) -> List[CheckResult]:
    torch = _torch()
    kw = dict(config=spec.name if spec else None, model_id=str(model_id or getattr(data, "model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    x = getattr(data, "x", None)
    if x is None:
        return [CheckResult.fail("pyg.x_present", "Data.x missing", **kw)]
    n = int(x.size(0))
    if n == 0:
        r.append(CheckResult.fail("pyg.nonempty", "graph has 0 nodes", **kw))
    if x.dim() != 2 or x.size(1) == 0:
        r.append(CheckResult.fail("pyg.x_shape", f"x shape {tuple(x.shape)} (expected [N, F>0])", **kw))
    if not torch.is_floating_point(x):
        r.append(CheckResult.fail("pyg.x_dtype", f"x dtype {x.dtype} (expected float)", **kw))
    r += _finite_report(x, "x", kw)
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None and int(num_nodes) != n:
        r.append(CheckResult.fail("pyg.num_nodes_matches_x", f"num_nodes={num_nodes} but x has {n} rows", **kw))
    ei = getattr(data, "edge_index", None)
    r += _edge_index_report(ei, n, n, "edge_index", kw)
    if ei is not None and ei.numel() and expect_symmetric:
        sym, k = _is_symmetric(ei, n)
        if not sym:
            r.append(CheckResult.fail("pyg.edges_symmetric", f"{k} edges without their reverse although all relations are bidirectional", **kw))
    ea = getattr(data, "edge_attr", None)
    if ea is not None and ei is not None:
        if ea.size(0) != ei.size(1):
            r.append(CheckResult.fail("pyg.edge_attr_rows", f"edge_attr has {ea.size(0)} rows but edge_index has {ei.size(1)} edges", **kw))
        r += _finite_report(ea, "edge_attr", kw)
    if ei is not None and ei.numel() and n > 1:
        deg = torch.bincount(ei.reshape(-1), minlength=n)
        iso = int((deg == 0).sum())
        if iso:
            r.append(CheckResult.warn("pyg.isolated_nodes", f"{iso}/{n} isolated nodes", **kw))
    elif n > 1 and (ei is None or ei.numel() == 0):
        r.append(CheckResult.warn("pyg.has_edges", f"{n} nodes but no edges", **kw))
    y = getattr(data, "y", None)
    if y is not None:
        r += _finite_report(y.float() if not torch.is_floating_point(y) else y, "y", kw)
    if graph_feat_key:
        gf = getattr(data, graph_feat_key, None)
        if gf is None:
            r.append(CheckResult.fail("pyg.graph_feature_present", f"graph-level feature '{graph_feat_key}' missing", **kw))
        else:
            r += _finite_report(gf, graph_feat_key, kw)
            if graph_feat_dim is not None and gf.numel() != graph_feat_dim and gf.size(-1) != graph_feat_dim:
                r.append(CheckResult.fail("pyg.graph_feature_dim", f"'{graph_feat_key}' has {gf.numel()} values, expected {graph_feat_dim}", **kw))
            if gf.numel() and float(gf.abs().sum()) == 0.0:
                r.append(CheckResult.fail("pyg.graph_feature_nonzero", f"'{graph_feat_key}' is all zeros", **kw))
    mid_attr = getattr(data, "model_id", None)
    if model_id is not None and mid_attr is not None and str(mid_attr) != str(model_id):
        r.append(CheckResult.fail("pyg.model_id_matches", f"Data.model_id={mid_attr!r} but expected {model_id!r}", **kw))
    if not any(x.status.value == "FAIL" for x in r):
        r.append(CheckResult.ok("pyg.graph_valid", f"N={n} F={int(x.size(1))} E={int(ei.size(1)) if ei is not None else 0}", **kw))
    return r


def check_hetero_data(hd: Any, spec: Optional[ConfigSpec], *, model_id: Optional[str] = None) -> List[CheckResult]:
    torch = _torch()
    kw = dict(config=spec.name if spec else None, model_id=str(model_id or getattr(hd, "model_id", "")), scope=SCOPE)
    r: List[CheckResult] = []
    node_types = list(hd.node_types)
    edge_types = [tuple(et) for et in hd.edge_types]
    sizes: Dict[str, int] = {}
    for nt in node_types:
        x = getattr(hd[nt], "x", None)
        if x is None:
            r.append(CheckResult.fail("pyg.x_present", f"node store '{nt}' has no x", **kw))
            sizes[nt] = int(getattr(hd[nt], "num_nodes", 0) or 0)
            continue
        sizes[nt] = int(x.size(0))
        if x.dim() != 2:
            r.append(CheckResult.fail("pyg.x_shape", f"'{nt}'.x shape {tuple(x.shape)}", **kw))
        elif x.size(0) > 0 and x.size(1) == 0:
            r.append(CheckResult.fail("pyg.x_features", f"'{nt}' has {x.size(0)} nodes but 0 features", **kw))
        r += _finite_report(x, f"{nt}.x", kw)
    if spec is not None:
        expected_nodes = set(spec.node_tables)
        missing = sorted(expected_nodes - set(node_types))
        extra = sorted(set(node_types) - expected_nodes)
        if missing:
            r.append(CheckResult.fail("pyg.node_types_match_config", f"node types missing: {missing}", **kw))
        if extra:
            r.append(CheckResult.fail("pyg.node_types_match_config", f"node types not in config: {extra}", **kw))
        empty_mandatory = [nt for nt in expected_nodes if nt in sizes and sizes[nt] == 0 and "aperture" not in nt]
        if empty_mandatory:
            r.append(CheckResult.fail("pyg.node_store_nonempty", f"mandatory node stores empty: {empty_mandatory}", **kw))
        for t in spec.singleton_node_tables:
            if sizes.get(t, 0) > 1:
                r.append(CheckResult.fail("pyg.singleton_node_type", f"'{t}' has {sizes[t]} nodes (expected 1)", **kw))
        exp_triples = spec.expected_triples
        unexpected = [et for et in edge_types if et not in exp_triples]
        if unexpected:
            r.append(CheckResult.fail("pyg.edge_types_match_config", f"edge types not derivable from config: {unexpected}", **kw))
        for e in spec.edges:
            if e.forward not in edge_types and not (e.is_self_type and e.reverse in edge_types):
                if e.rel in {"adjacent_surface", "adjacent_zone"} and sizes.get(e.src_cat, 0) < 2:
                    continue
                if sizes.get(e.src_cat, 0) == 0 or sizes.get(e.tgt_cat, 0) == 0:
                    continue
                store_missing = CheckResult.warn if e.rel in {"adjacent_surface", "adjacent_zone"} else CheckResult.fail
                r.append(store_missing("pyg.edge_types_present", f"relation {e.forward} has no edge store", **kw))
        for fwd in spec.one_directional_triples:
            rev = (fwd[2], fwd[1], fwd[0])
            if rev in edge_types and getattr(hd[rev], "edge_index", None) is not None and int(hd[rev].edge_index.size(1)) > 0:
                r.append(CheckResult.fail("pyg.one_directional_respected", f"{fwd} is one-directional but reverse store {rev} has edges", **kw))
    for et in edge_types:
        st = hd[et]
        ei = getattr(st, "edge_index", None)
        r += _edge_index_report(
            ei, sizes.get(et[0], 0), sizes.get(et[2], 0), str(et), kw,
            same_node_type=(et[0] == et[2]),
        )
        ea = getattr(st, "edge_attr", None)
        if ea is not None and ei is not None:
            if ea.dim() != 2:
                r.append(CheckResult.fail("pyg.edge_attr_shape", f"{et}: edge_attr shape {tuple(ea.shape)}", **kw))
            elif ea.size(0) != ei.size(1):
                r.append(CheckResult.fail("pyg.edge_attr_rows", f"{et}: edge_attr rows {ea.size(0)} ≠ edges {ei.size(1)}", **kw))
            r += _finite_report(ea, f"{et}.edge_attr", kw)
        # reverse store must mirror forward store for bidirectional cross-type relations
        if spec is not None and et[0] != et[2]:
            rev = (et[2], et[1], et[0])
            espec = next((e for e in spec.edges if e.forward == et), None)
            if espec is not None and espec.bidirectional:
                if rev not in edge_types:
                    r.append(CheckResult.fail("pyg.reverse_store_present", f"bidirectional {et} has no reverse store {rev}", **kw))
                else:
                    ei_r = getattr(hd[rev], "edge_index", None)
                    if ei is None or ei_r is None:
                        continue
                    a = set(zip(ei[0].tolist(), ei[1].tolist()))
                    b = set(zip(ei_r[1].tolist(), ei_r[0].tolist()))
                    if a != b:
                        r.append(CheckResult.fail("pyg.reverse_store_mirrors", f"{rev} does not mirror {et} ({len(a ^ b)} mismatching edges)", **kw))
        if spec is not None and et[0] == et[2]:
            espec = next((e for e in spec.edges if e.forward == et), None)
            if espec is not None and espec.bidirectional and ei is not None and ei.numel():
                sym, k = _is_symmetric(ei, sizes.get(et[0], 0))
                if not sym:
                    r.append(CheckResult.fail("pyg.edges_symmetric", f"{et}: {k} edges lack their reverse", **kw))
    # isolated nodes per type (degree over all stores)
    deg: Dict[str, Any] = {nt: torch.zeros(sizes[nt], dtype=torch.long) for nt in node_types if sizes.get(nt, 0)}
    for et in edge_types:
        ei = getattr(hd[et], "edge_index", None)
        if ei is None or ei.numel() == 0:
            continue
        if et[0] in deg:
            deg[et[0]] += torch.bincount(ei[0], minlength=sizes[et[0]])[: sizes[et[0]]]
        if et[2] in deg:
            deg[et[2]] += torch.bincount(ei[1], minlength=sizes[et[2]])[: sizes[et[2]]]
    iso = {nt: int((d == 0).sum()) for nt, d in deg.items() if int((d == 0).sum())}
    if iso:
        only_adj = spec is not None and all(e.rel in {"adjacent_zone", "adjacent_surface"} for e in spec.edges)
        r.append((CheckResult.warn if only_adj else CheckResult.fail)("pyg.isolated_nodes", f"isolated nodes per type: {iso}", **kw))
    y = getattr(hd, "y", None)
    if y is not None:
        r += _finite_report(y.float() if not torch.is_floating_point(y) else y, "y", kw)
    mid_attr = getattr(hd, "model_id", None)
    if model_id is not None and mid_attr is not None and str(mid_attr) != str(model_id):
        r.append(CheckResult.fail("pyg.model_id_matches", f"HeteroData.model_id={mid_attr!r} but expected {model_id!r}", **kw))
    if not any(x.status.value == "FAIL" for x in r):
        r.append(CheckResult.ok("pyg.graph_valid", "nodes " + ", ".join(f"{nt}={sizes[nt]}" for nt in node_types) + f"; {len(edge_types)} edge stores", **kw))
    return r


def check_graph(obj: Any, spec: Optional[ConfigSpec], **kw: Any) -> List[CheckResult]:
    Data, HeteroData = _pyg_types()
    if Data is None:
        return [CheckResult.skip("pyg.available", "torch_geometric not importable", config=spec.name if spec else None, scope=SCOPE)]
    if isinstance(obj, HeteroData):
        return check_hetero_data(obj, spec, model_id=kw.get("model_id"))
    if isinstance(obj, Data):
        return check_homo_data(obj, spec, **kw)
    return [CheckResult.fail("pyg.graph_type", f"unexpected object {type(obj).__name__}", config=spec.name if spec else None, scope=SCOPE)]


# --------------------------------------------------------------------------
# collections
# --------------------------------------------------------------------------

def check_graph_collection(graphs: Sequence[Any], spec: Optional[ConfigSpec], *, model_ids: Optional[Sequence[str]] = None, min_graphs_for_variance: int = 20) -> List[CheckResult]:
    """Feature-dimension stability, constant / all-zero feature columns across graphs."""
    torch = _torch()
    Data, HeteroData = _pyg_types()
    kw = dict(config=spec.name if spec else None, scope=SCOPE)
    if torch is None or Data is None:
        return [CheckResult.skip("pyg.available", "torch/torch_geometric not importable", **kw)]
    r: List[CheckResult] = []
    if not graphs:
        return [CheckResult.fail("pyg.collection_nonempty", "no graphs", **kw)]
    kinds = Counter("hetero" if isinstance(g, HeteroData) else "homo" if isinstance(g, Data) else type(g).__name__ for g in graphs)
    if len(kinds) > 1:
        r.append(CheckResult.fail("pyg.collection_single_kind", f"mixed graph kinds {dict(kinds)}", **kw))
        return r
    kind = next(iter(kinds))
    if model_ids is not None:
        dup = [m for m, c in Counter(map(str, model_ids)).items() if c > 1]
        if dup:
            r.append(CheckResult.fail("pyg.collection_model_ids_unique", f"duplicate model ids {examples(dup)}", **kw))

    # per node type: x dim per graph, column-wise min/max/nunique for constant detection
    dims: Dict[str, Counter] = defaultdict(Counter)
    edims: Dict[str, Counter] = defaultdict(Counter)
    col_min: Dict[str, Any] = {}
    col_max: Dict[str, Any] = {}
    col_nonzero: Dict[str, Any] = {}
    n_nodes: Dict[str, int] = Counter()
    for g in graphs:
        stores = [(nt, getattr(g[nt], "x", None)) for nt in g.node_types] if kind == "hetero" else [("node", getattr(g, "x", None))]
        for nt, x in stores:
            if x is None or x.numel() == 0:
                if x is not None:
                    dims[nt][int(x.size(-1)) if x.dim() == 2 else -1] += 1
                continue
            dims[nt][int(x.size(1))] += 1
            n_nodes[nt] += int(x.size(0))
            mn, mx = x.min(dim=0).values, x.max(dim=0).values
            nz = (x != 0).any(dim=0)
            if nt not in col_min or col_min[nt].numel() != mn.numel():
                col_min[nt], col_max[nt], col_nonzero[nt] = mn.clone(), mx.clone(), nz.clone()
            else:
                col_min[nt] = torch.minimum(col_min[nt], mn)
                col_max[nt] = torch.maximum(col_max[nt], mx)
                col_nonzero[nt] |= nz
        estores = [(str(et), getattr(g[et], "edge_attr", None)) for et in g.edge_types] if kind == "hetero" else [("edge", getattr(g, "edge_attr", None))]
        for et, ea in estores:
            if ea is not None and ea.dim() == 2:
                edims[et][int(ea.size(1))] += 1
    for nt, c in dims.items():
        if len(c) > 1:
            r.append(CheckResult.fail("pyg.feature_dim_stable", f"'{nt}': x feature dimension varies across graphs {dict(c)} — batching will fail / features misaligned", **kw))
        else:
            r.append(CheckResult.ok("pyg.feature_dim_stable", f"'{nt}': F={next(iter(c))} in all {sum(c.values())} graphs", **kw))
    for et, c in edims.items():
        if len(c) > 1:
            r.append(CheckResult.fail("pyg.edge_attr_dim_stable", f"{et}: edge_attr dimension varies across graphs {dict(c)}", **kw))
    if len(graphs) >= min_graphs_for_variance:
        for nt in col_min:
            const = (col_min[nt] == col_max[nt]).nonzero().flatten().tolist()
            zero = (~col_nonzero[nt]).nonzero().flatten().tolist()
            if zero:
                r.append(CheckResult.warn("pyg.feature_all_zero", f"'{nt}': {len(zero)} feature columns are 0 in every node of every graph (NaN→0 fill or unused category?): idx {zero[:15]}{' …' if len(zero) > 15 else ''}", details={"columns": zero}, **kw))
            nonzero_const = [i for i in const if i not in set(zero)]
            if nonzero_const:
                r.append(CheckResult.warn("pyg.feature_constant", f"'{nt}': {len(nonzero_const)} feature columns constant (non-zero) across all graphs: idx {nonzero_const[:15]}", **kw))
            big = (col_max[nt].abs() > 1e6) | (col_min[nt].abs() > 1e6)
            if bool(big.any()):
                r.append(CheckResult.warn("pyg.feature_scale", f"'{nt}': columns with |x|>1e6 (unnormalised / unit error?): idx {big.nonzero().flatten().tolist()[:10]}", **kw))
    return r


# --------------------------------------------------------------------------
# training_ready.pt packages
# --------------------------------------------------------------------------

def _sha256_ids(ids: Iterable[Any]) -> str:
    """Same recipe as ``beg.datasets.tools.stable_hash`` applied to the id list."""
    payload = json.dumps([str(x) for x in ids], sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_training_ready_payload(payload: Dict[str, Any], spec: Optional[ConfigSpec] = None, *, meta: Optional[Dict[str, Any]] = None, expected_n_graphs: Optional[int] = None, check_graphs: bool = True, max_graphs_checked: Optional[int] = None) -> List[CheckResult]:
    torch = _torch()
    cfg_name = str(payload.get("config_name", spec.name if spec else ""))
    kw = dict(config=cfg_name, scope=SCOPE)
    r: List[CheckResult] = []
    for key in ("dataset_ids", "graphs", "split", "labels_raw", "label", "config_signature_hash"):
        if key not in payload:
            r.append(CheckResult.fail("pyg.tr_keys", f"training_ready payload lacks '{key}'", **kw))
    if any(x.failed for x in r):
        return r
    ids = [str(x) for x in payload["dataset_ids"]]
    graphs = payload["graphs"]
    split = payload["split"]
    labels = payload["labels_raw"]
    n = len(ids)
    if len(graphs) != n:
        r.append(CheckResult.fail("pyg.tr_lengths", f"{n} dataset_ids but {len(graphs)} graphs", **kw))
    if len(labels.get("by_index", [])) != n:
        r.append(CheckResult.fail("pyg.tr_lengths", f"{n} dataset_ids but {len(labels.get('by_index', []))} labels", **kw))
    if len(set(ids)) != n:
        r.append(CheckResult.fail("pyg.tr_ids_unique", f"dataset_ids has {n - len(set(ids))} duplicates", **kw))
    if expected_n_graphs is not None and n != expected_n_graphs:
        r.append(CheckResult.fail("pyg.tr_n_graphs", f"{n} graphs, expected {expected_n_graphs}", **kw))
    by_id = labels.get("by_id", {})
    if set(by_id) != set(ids):
        r.append(CheckResult.fail("pyg.tr_labels_by_id", f"labels_raw.by_id keys ≠ dataset_ids ({len(set(by_id) ^ set(ids))} differ)", **kw))
    vals = [float(v) for v in labels.get("by_index", [])]
    bad = [v for v in vals if not math.isfinite(v)]
    nonpos = [v for v in vals if math.isfinite(v) and v <= 0]
    if bad:
        r.append(CheckResult.fail("pyg.tr_labels_finite", f"{len(bad)} non-finite labels", **kw))
    if nonpos:
        r.append(CheckResult.fail("pyg.tr_labels_positive", f"{len(nonpos)} labels ≤ 0 (EUI must be positive)", **kw))
    # splits
    idx = {s: [int(i) for i in split.get(f"{s}_idx", [])] for s in ("train", "val", "test")}
    sid = {s: [str(i) for i in split.get(f"{s}_ids", [])] for s in ("train", "val", "test")}
    all_idx = idx["train"] + idx["val"] + idx["test"]
    if len(set(all_idx)) != len(all_idx):
        r.append(CheckResult.fail("pyg.tr_split_disjoint", "split indices overlap", **kw))
    if set(all_idx) != set(range(n)):
        r.append(CheckResult.fail("pyg.tr_split_covers", f"split indices cover {len(set(all_idx))}/{n} graphs", **kw))
    for s in ("train", "val", "test"):
        if [ids[i] for i in idx[s] if i < n] != sid[s]:
            r.append(CheckResult.fail("pyg.tr_split_ids_match_idx", f"{s}_ids ≠ dataset_ids[{s}_idx]", **kw))
        sha_key = f"{s}_ids_sha"
        if sha_key in split and split[sha_key] != _sha256_ids(sid[s]):
            r.append(CheckResult.fail("pyg.tr_split_sha", f"{sha_key} does not match stable_hash({s}_ids) — split ids were altered after hashing", **kw))
    if not idx["train"] or not idx["val"] or not idx["test"]:
        r.append(CheckResult.fail("pyg.tr_split_nonempty", f"empty split: sizes {[len(idx[s]) for s in ('train', 'val', 'test')]}", **kw))
    # label normalisation meta
    lm = payload.get("label", {})
    if idx["train"] and vals and "train_mean_raw" in lm:
        tr = [vals[i] for i in idx["train"] if i < len(vals)]
        mean = sum(tr) / len(tr)
        var = sum((v - mean) ** 2 for v in tr) / max(len(tr) - 1, 1)
        std = math.sqrt(var)
        if abs(mean - float(lm["train_mean_raw"])) > 1e-3 * max(1.0, abs(mean)):
            r.append(CheckResult.fail("pyg.tr_label_stats", f"label.train_mean_raw={lm['train_mean_raw']} but recomputed train mean={mean:.4f}", **kw))
        if "train_std_raw" in lm and abs(std - float(lm["train_std_raw"])) > 2e-2 * max(1.0, std):
            r.append(CheckResult.warn("pyg.tr_label_stats", f"label.train_std_raw={lm['train_std_raw']} vs recomputed sample std={std:.4f} (ddof convention?)", **kw))
        if float(lm.get("train_std_raw", 1.0)) <= 0:
            r.append(CheckResult.fail("pyg.tr_label_stats", "train_std_raw ≤ 0", **kw))
    # graphs vs ids / y
    if check_graphs and torch is not None and graphs:
        Data, HeteroData = _pyg_types()
        sel = range(n) if max_graphs_checked is None else range(min(n, max_graphs_checked))
        mism, ybad, ydenorm_bad = 0, 0, 0
        for i in sel:
            g = graphs[i]
            gm = getattr(g, "model_id", None)
            if gm is not None and str(gm) != ids[i]:
                mism += 1
            y = getattr(g, "y", None)
            if y is not None:
                yv = float(y.flatten()[0])
                if not math.isfinite(yv):
                    ybad += 1
                elif "train_mean_raw" in lm and "train_std_raw" in lm and lm.get("space", "").startswith("zscore"):
                    raw = yv * float(lm["train_std_raw"]) + float(lm["train_mean_raw"])
                    if abs(raw - vals[i]) > 1e-2 * max(1.0, abs(vals[i])):
                        ydenorm_bad += 1
        if mism:
            r.append(CheckResult.fail("pyg.tr_graph_model_ids", f"{mism} graphs whose model_id ≠ dataset_ids[i]", **kw))
        if ybad:
            r.append(CheckResult.fail("pyg.tr_graph_y_finite", f"{ybad} graphs with non-finite y", **kw))
        if ydenorm_bad:
            r.append(CheckResult.fail("pyg.tr_graph_y_denorm", f"{ydenorm_bad} graphs whose de-normalised y ≠ labels_raw", **kw))
        r += check_graph_collection([graphs[i] for i in sel], spec, model_ids=[ids[i] for i in sel])
        for i in list(sel)[: min(len(sel), 25)]:
            r += check_graph(graphs[i], spec, model_id=ids[i])
    # meta.json agreement
    if meta:
        for key in ("config_name", "graph_kind", "config_signature_hash"):
            if key in meta and key in payload and str(meta[key]) != str(payload[key]):
                r.append(CheckResult.fail("pyg.tr_meta_matches", f"meta.json {key}={meta[key]!r} ≠ payload {payload[key]!r}", **kw))
        if "n_graphs" in meta and int(meta["n_graphs"]) != n:
            r.append(CheckResult.fail("pyg.tr_meta_matches", f"meta.json n_graphs={meta['n_graphs']} ≠ {n} dataset_ids in payload", **kw))
        ss = meta.get("split_sizes", {})
        for s in ("train", "val", "test"):
            if s in ss and int(ss[s]) != len(idx[s]):
                r.append(CheckResult.fail("pyg.tr_meta_matches", f"meta split_sizes.{s}={ss[s]} ≠ {len(idx[s])}", **kw))
    if not any(x.failed for x in r):
        r.append(CheckResult.ok("pyg.tr_valid", f"{n} graphs, split {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}", **kw))
    return r


def check_training_ready_alignment(payloads: Dict[str, Dict[str, Any]]) -> List[CheckResult]:
    """All configurations must share dataset_ids and split assignment for paired comparison."""
    r: List[CheckResult] = []
    if len(payloads) < 2:
        return r
    ref_name, ref = next(iter(payloads.items()))
    ref_ids = [str(x) for x in ref["dataset_ids"]]
    for name, p in payloads.items():
        if name == ref_name:
            continue
        ids = [str(x) for x in p["dataset_ids"]]
        if set(ids) != set(ref_ids):
            r.append(CheckResult.fail("pyg.tr_ids_aligned", f"{name}: dataset_ids differ from {ref_name} ({len(set(ids) ^ set(ref_ids))} ids)", config=name, scope=SCOPE))
        elif ids != ref_ids:
            r.append(CheckResult.warn("pyg.tr_ids_aligned", f"{name}: same ids as {ref_name} but different order", config=name, scope=SCOPE))
        for s in ("train", "val", "test"):
            a = set(map(str, p["split"].get(f"{s}_ids", [])))
            b = set(map(str, ref["split"].get(f"{s}_ids", [])))
            if a != b:
                r.append(CheckResult.fail("pyg.tr_split_aligned", f"{name}: {s} split differs from {ref_name} ({len(a ^ b)} ids) — configurations are not compared on identical splits", config=name, scope=SCOPE))
        la, lb = p.get("labels_raw", {}).get("by_id", {}), ref.get("labels_raw", {}).get("by_id", {})
        diff = [k for k in la if k in lb and abs(float(la[k]) - float(lb[k])) > 1e-9]
        if diff:
            r.append(CheckResult.fail("pyg.tr_labels_aligned", f"{name}: {len(diff)} labels differ from {ref_name}", config=name, scope=SCOPE))
    if not r:
        r.append(CheckResult.ok("pyg.tr_aligned", f"{len(payloads)} training packages share ids, splits and labels", scope=SCOPE))
    return r
