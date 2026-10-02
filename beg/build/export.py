"""Writing per-model PyG graphs to disk, and the manifests that index them.

The layout downstream depends on::

    outputs/pyg_graphs/{homo,hetero}/<Cn_full_name>/<model_id>.pt
    outputs/pyg_graphs/graphs_manifest_homo.parquet
    outputs/pyg_graphs/graphs_manifest_hetero.parquet

``BEG_03_Datasets`` reads the manifests, not the directories, so the manifests
are the contract. They are rebuilt from the files on disk, so they describe only
graphs that exist, including after an interrupted run.

Both graph types go through :func:`export_graphs`; only the per-model builder differs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

#: Subdirectory of ``pyg_graphs`` for each graph type.
GRAPH_KIND_DIRS = {"homo": "homo", "hetero": "hetero"}


def safe_artifact_stem(model_id: Any) -> str:
    """A filename stem that cannot escape its directory."""
    return str(model_id).replace(os.sep, "_").replace("/", "_")


def filter_df_by_model_id(df: pd.DataFrame | None, model_id: Any) -> pd.DataFrame:
    """The rows of one model, or an empty frame shaped like ``df``.

    Scans the whole frame; in a loop over models use :func:`group_by_model`.
    """
    if df is None or not hasattr(df, "columns") or df.empty:
        return pd.DataFrame() if df is None else df.iloc[0:0].copy()
    if "model_id" not in df.columns:
        return df.copy()
    return df.loc[df["model_id"].astype(str) == str(model_id)].copy()


def group_by_model(df: pd.DataFrame | None) -> tuple[dict, pd.DataFrame]:
    """Index a frame by ``model_id`` in one pass. Returns ``(groups, empty_slice)``.

    ``groups`` maps each model id, as a string, to its rows. ``empty_slice`` is a
    zero-row frame with the same columns, for a model the frame has no rows for.
    """
    if df is None or not hasattr(df, "columns") or df.empty:
        empty = pd.DataFrame() if df is None else df.iloc[0:0]
        return {}, empty
    if "model_id" not in df.columns:
        # Not per-model data; every model sees the whole frame.
        return {}, df
    keys = df["model_id"].astype(str)
    return {str(k): g for k, g in df.groupby(keys, sort=False)}, df.iloc[0:0]


def slice_for_model(grouped: dict, empty: pd.DataFrame, model_id: Any) -> pd.DataFrame:
    """One model's rows from a :func:`group_by_model` result."""
    if not grouped:
        return empty
    return grouped.get(str(model_id), empty)


def prune_graph_dir(
    out_dir: Path, target_model_ids: Sequence[str] | set[str], wipe_all: bool = False
) -> int:
    """Remove the ``.pt`` files the current run will not produce (all if ``wipe_all``).

    The manifest is built from the directory, so this keeps it to the current
    model set. Returns how many files were removed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.pt"))

    if wipe_all:
        for path in existing:
            path.unlink()
        return len(existing)

    allowed = {safe_artifact_stem(mid) for mid in target_model_ids}
    removed = 0
    for path in existing:
        if path.stem not in allowed:
            path.unlink()
            removed += 1
    return removed


def edge_table_name(edge_spec: Mapping) -> str | None:
    """The relation an edge spec reads from, or None for a plain foreign key."""
    edge_src = edge_spec.get("edge_src")
    if not edge_src or (isinstance(edge_src, str) and edge_src.lower() == "fk"):
        return None
    return edge_src.replace(".sql", "") if isinstance(edge_src, str) else str(edge_src)


def expected_edge_feature_cols(
    edges_dict: Mapping[str, pd.DataFrame], edges_list: Sequence[dict]
) -> list[str]:
    """The numeric edge-feature columns a configuration carries, in order.

    Fixed once per configuration from the models that have edges, so every graph
    has the same edge-feature width and batches collate: a model without edges
    gets ``edge_attr`` of shape ``(0, D)``, not ``(0, 0)``.

    Homogeneous graphs have one edge store, so this is a flat list. The
    heterogeneous equivalent is :func:`expected_edge_feature_cols_by_type`.
    """
    edge_specs = {str(e.get("edge_src")): e for e in edges_list if e.get("edge_src")}
    seen: set[str] = set()
    ordered: list[str] = []

    for edge_table, df in edges_dict.items():
        if df is None or df.empty:
            continue
        spec = edge_specs.get(str(edge_table), {})
        ignore = {"model_id", "src_id", "tgt_id", "src", "tgt"}
        if spec.get("src_col"):
            ignore.add(str(spec["src_col"]))
        if spec.get("tgt_col"):
            ignore.add(str(spec["tgt_col"]))
        for col in df.columns:
            if col in ignore or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if col not in seen:
                seen.add(col)
                ordered.append(col)
    return ordered


def expected_edge_feature_cols_by_type(
    edges_dict: Mapping[str, pd.DataFrame],
    edges_list: Sequence[dict],
    model_id_col: str = "model_id",
) -> dict[tuple[str, str, str], list[str]]:
    """The same contract as :func:`expected_edge_feature_cols`, per edge type.

    A heterogeneous graph has one edge store per ``(src_cat, rel, tgt_cat)``
    triple, and each needs its own fixed width. An edge type with no rows in any
    model maps to an empty list, which pins its width at zero.
    """
    expected: dict[tuple[str, str, str], list[str]] = {}

    for spec in edges_list:
        table = edge_table_name(spec)
        if table is None:
            continue

        etype = (str(spec.get("src_cat")), str(spec.get("rel", "to")), str(spec.get("tgt_cat")))
        df = edges_dict.get(table)
        if df is None or df.empty:
            expected[etype] = []
            continue

        ignore = {model_id_col, "src_id", "tgt_id", "src", "tgt"}
        if spec.get("src_col"):
            ignore.add(str(spec["src_col"]))
        if spec.get("tgt_col"):
            ignore.add(str(spec["tgt_col"]))

        expected[etype] = [
            c for c in df.columns
            if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
        ]
    return expected


def config_specs(configs_summary: pd.DataFrame, config_name: str) -> dict:
    """Node tables and edge specs for one configuration, for either graph type."""
    from beg.build.frames import get_het_specs

    return get_het_specs(configs_summary, config_name)


def homo_build_one(
    *, config_name, model_id, nodes_all, edges_all, specs, edge_feature_cols_expected
):
    """Adapter: build one homogeneous graph from frames already sliced to it.

    ``edge_feature_cols_expected`` is a flat list here and a dict keyed by edge
    type in :func:`hetero_build_one`.

    ``specs["node_tables"]`` doubles as the node *order*: the homogeneous builder
    merges node tables in the order the configuration declares them.
    """
    from beg.build.pyg_homo import build_homo_data_for_model_numeric

    return build_homo_data_for_model_numeric(
        config_name=config_name,
        model_id=str(model_id),
        nodes_dict=nodes_all,
        edges_dict=edges_all,
        node_table_order=specs["node_tables"],
        edges_list=specs["edges_list"],
        keep_raw_dfs=False,
        edge_feature_cols_expected=edge_feature_cols_expected,
    )


def hetero_build_one(
    *, config_name, model_id, nodes_all, edges_all, specs, edge_feature_cols_expected
):
    """Adapter: build one heterogeneous graph from frames already sliced to it.

    ``filter_model_id=False``: the caller has already sliced the frames with
    :func:`group_by_model`.
    """
    from beg.build.pyg_hetero import build_hetero_data_for_model_numeric

    return build_hetero_data_for_model_numeric(
        config_name=config_name,
        model_id=str(model_id),
        nodes_dict=nodes_all,
        edges_dict=edges_all,
        node_tables=specs["node_tables"],
        edges_list=specs["edges_list"],
        keep_raw_dfs=False,
        edge_feature_cols_expected=edge_feature_cols_expected,
        filter_model_id=False,
    )


@dataclass
class ExportResult:
    """What one export pass wrote, per configuration."""

    kind: str = "homo"
    rows: list = field(repr=False, default_factory=list)
    target_model_ids: dict = field(repr=False, default_factory=dict)

    @property
    def saved(self) -> int:
        return sum(r["saved"] for r in self.rows)

    @property
    def skipped(self) -> int:
        return sum(r["skipped_existing"] for r in self.rows)

    def summary(self) -> pd.DataFrame:
        """One row per configuration: models, written, reused, pruned."""
        if not self.rows:
            return pd.DataFrame(
                columns=[
                    "config_name", "config_full_name", "model_ids",
                    "saved", "skipped_existing", "removed_stale", "edge_feature_dim",
                ]
            )
        return (
            pd.DataFrame(self.rows).sort_values("config_full_name").reset_index(drop=True)
        )


def export_graphs(
    *,
    kind: str,
    configs: Sequence[str],
    data: Mapping[str, dict],
    full_name_of: Mapping[str, str],
    specs_of: Callable[[str], dict],
    build_one: Callable[..., Any],
    graphs_root: Path,
    keep_model_ids: set[str] | None = None,
    force_rebuild: bool = False,
    progress: bool = True,
) -> ExportResult:
    """Build and save one ``.pt`` per model, for each configuration.

    ``build_one`` is the per-model adapter, :func:`homo_build_one` or
    :func:`hetero_build_one`.

    ``keep_model_ids`` (lower-case) restricts the export to those models, normally
    the ones with a label.

    An existing ``.pt`` is reused unless ``force_rebuild`` is set, so an
    interrupted export resumes where it stopped.
    """
    import torch

    if kind not in GRAPH_KIND_DIRS:
        raise ValueError(f"kind must be one of {sorted(GRAPH_KIND_DIRS)}, got {kind!r}")

    out_root = Path(graphs_root) / GRAPH_KIND_DIRS[kind]
    result = ExportResult(kind=kind)

    iterator = configs
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(configs, desc=f"Exporting {kind} graphs", unit="config")
        except ImportError:
            pass

    for cfg in iterator:
        cfg_full = str(full_name_of[cfg])
        payload = data[cfg]

        model_ids = [str(m) for m in (payload.get("model_ids") or [])]
        if keep_model_ids is not None:
            model_ids = [m for m in model_ids if m.strip().lower() in keep_model_ids]

        nodes_all = payload.get("nodes") or {}
        edges_all = payload.get("edges") or {}
        specs = specs_of(cfg)
        edges_list = specs["edges_list"]
        # Homogeneous graphs have one edge store and take a flat column list;
        # heterogeneous graphs need one list per edge type.
        edge_cols = (
            expected_edge_feature_cols(edges_all, edges_list)
            if kind == "homo"
            else expected_edge_feature_cols_by_type(edges_all, edges_list)
        )

        out_dir = out_root / cfg_full
        removed = prune_graph_dir(out_dir, model_ids, wipe_all=force_rebuild)
        result.target_model_ids[cfg_full] = set(model_ids)

        # Index every frame by model once.
        nodes_idx = {t: group_by_model(df) for t, df in nodes_all.items()}
        edges_idx = {t: group_by_model(df) for t, df in edges_all.items()}

        saved = skipped = 0
        for mid in model_ids:
            out_path = out_dir / f"{safe_artifact_stem(mid)}.pt"
            if out_path.exists() and not force_rebuild:
                skipped += 1
                continue

            graph = build_one(
                config_name=cfg,
                model_id=str(mid),
                nodes_all={t: slice_for_model(g, e, mid) for t, (g, e) in nodes_idx.items()},
                edges_all={t: slice_for_model(g, e, mid) for t, (g, e) in edges_idx.items()},
                specs=specs,
                edge_feature_cols_expected=edge_cols,
            )
            torch.save(graph, out_path)
            saved += 1

        result.rows.append(
            {
                "config_name": cfg,
                "config_full_name": cfg_full,
                "model_ids": len(model_ids),
                "saved": saved,
                "skipped_existing": skipped,
                "removed_stale": removed,
                # For homo this is the edge feature width; for hetero, the number
                # of edge types, each with its own width.
                "edge_feature_dim": len(edge_cols),
            }
        )

    return result


def build_manifest(graphs_root: Path, kind: str) -> pd.DataFrame:
    """Index every ``.pt`` on disk for one graph type.

    Built from the directory, so it describes what exists, also after an
    interrupted run.
    """
    root = Path(graphs_root) / GRAPH_KIND_DIRS[kind]
    rows = []
    if root.exists():
        for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for path in sorted(cfg_dir.glob("*.pt")):
                rows.append(
                    {
                        "config_full_name": cfg_dir.name,
                        "config_id": cfg_dir.name.split("_", 1)[0],
                        "model_id": path.stem,
                        "graph_type": kind,
                        "path": path.relative_to(Path(graphs_root)).as_posix(),
                        "bytes": path.stat().st_size,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["config_full_name", "config_id", "model_id", "graph_type", "path", "bytes"],
    )


def write_manifests(graphs_root: Path) -> dict[str, Path]:
    """Write both manifests next to the graph directories. Returns their paths."""
    graphs_root = Path(graphs_root)
    graphs_root.mkdir(parents=True, exist_ok=True)

    written = {}
    for kind in GRAPH_KIND_DIRS:
        frame = build_manifest(graphs_root, kind)
        path = graphs_root / f"graphs_manifest_{kind}.parquet"
        frame.to_parquet(path, index=False)
        written[kind] = path
    return written


def manifest_summary(graphs_root: Path) -> pd.DataFrame:
    """Per configuration: how many graphs the manifests index, and their size."""
    frames = [build_manifest(graphs_root, kind) for kind in GRAPH_KIND_DIRS]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame(columns=["config_full_name", "graph_type", "model_id", "bytes"])

    if combined.empty:
        return combined

    return (
        combined.groupby(["graph_type", "config_full_name"])
        .agg(graphs=("model_id", "count"), megabytes=("bytes", lambda s: round(s.sum() / 1e6, 1)))
        .reset_index()
        .sort_values(["graph_type", "config_full_name"])
        .reset_index(drop=True)
    )


def reuse_existing_export(
    graphs_root: Path,
    config,
    keep_model_ids: set[str],
) -> dict | None:
    """Describe a configuration already complete on disk, or return ``None``.

    Complete means the directory holds a graph for every model in
    ``keep_model_ids``; the caller then skips the database fetch. Otherwise the
    result is ``None`` and the configuration is built.

    Returns the summary row and model ids ``build_and_export`` would have
    collected, so a skipped configuration still appears in the report.
    """
    kind = "homo" if config.graph_type == "homo" else "hetero"
    cfg_dir = Path(graphs_root) / GRAPH_KIND_DIRS[kind] / config.full_name
    if not cfg_dir.is_dir():
        return None

    stems = sorted(p.stem for p in cfg_dir.glob("*.pt"))
    if not stems:
        return None

    on_disk = {s.strip().lower() for s in stems}
    wanted = {str(m).strip().lower() for m in keep_model_ids}
    if not wanted <= on_disk:
        return None

    kept = {s for s in stems if s.strip().lower() in wanted}

    # The summary's edge feature width is not in the file names; read one graph.
    import torch

    sample = torch.load(cfg_dir / f"{sorted(kept)[0]}.pt", map_location="cpu", weights_only=False)
    if kind == "homo":
        edge_attr = getattr(sample, "edge_attr", None)
        edge_feature_dim = int(edge_attr.shape[1]) if edge_attr is not None and edge_attr.dim() == 2 else 0
    else:
        edge_feature_dim = len(sample.edge_types)

    return {
        "row": {
            "config_name": config.slug,
            "config_full_name": config.full_name,
            "model_ids": len(kept),
            "saved": 0,
            "skipped_existing": len(kept),
            "removed_stale": 0,
            "edge_feature_dim": edge_feature_dim,
        },
        "kind": kind,
        "model_ids": kept,
    }


def build_and_export(
    *,
    configs: Sequence[str],
    configs_dir: Path,
    engine,
    graphs_root: Path,
    keep_model_ids: set[str] | None = None,
    max_graphs: int | None = None,
    rebuild_transforms: bool = False,
    transform_mode: str = "tables",
    force_rebuild: bool = False,
    progress: bool = True,
):
    """Fetch and export one configuration at a time, holding only one in memory.

    Each configuration's frames are released before the next is read. Transforms
    whose outputs already exist are reused unless ``rebuild_transforms`` is set.

    A configuration whose graphs are already complete on disk skips the fetch
    (see :func:`reuse_existing_export`) and contributes no rows to
    ``configs_summary``, which is empty when every configuration is reused.
    Callers needing the specs on a reuse run must fetch them themselves.

    Returns ``(results_by_kind, configs_summary, model_ids_by_full_name)``.
    """
    import gc

    from beg.config.naming import resolve
    from beg.pipeline.dataframes import build_graph_dataframes

    results: dict[str, ExportResult] = {
        "homo": ExportResult(kind="homo"),
        "hetero": ExportResult(kind="hetero"),
    }
    summaries = []
    model_ids: dict[str, set] = {}

    iterator = list(configs)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Building configurations", unit="config")
        except ImportError:
            pass

    for cfg in iterator:
        config = resolve(cfg)
        if config.derived_from:
            continue  # C2 and C7 are produced by attaching a weather vector later

        # Reuse a configuration complete on disk. Not with `max_graphs`: which
        # models the database would return is not knowable from the disk.
        if not force_rebuild and max_graphs is None and keep_model_ids is not None:
            done = reuse_existing_export(graphs_root, config, keep_model_ids)
            if done is not None:
                results[done["kind"]].rows.append(done["row"])
                results[done["kind"]].target_model_ids[config.full_name] = done["model_ids"]
                model_ids[config.full_name] = done["model_ids"]
                continue

        out = build_graph_dataframes(
            configs_dir=configs_dir,
            config_names=[config.id],
            engine=engine,
            rebuild_transforms=rebuild_transforms,
            max_graphs=max_graphs,
            transform_mode=transform_mode,
            verbose=False,
        )
        summary, data = out["configs_summary"], out["data"]
        summaries.append(summary)

        kind = "homo" if config.graph_type == "homo" else "hetero"
        one = export_graphs(
            kind=kind,
            configs=[config.slug],
            data=data,
            full_name_of={config.slug: config.full_name},
            specs_of=partial_specs(summary),
            build_one=homo_build_one if kind == "homo" else hetero_build_one,
            graphs_root=graphs_root,
            keep_model_ids=keep_model_ids,
            force_rebuild=force_rebuild,
            progress=False,
        )
        results[kind].rows.extend(one.rows)
        results[kind].target_model_ids.update(one.target_model_ids)
        model_ids.update(one.target_model_ids)

        # Release this configuration's frames before reading the next.
        del out, data
        gc.collect()

    import pandas as _pd

    combined = _pd.concat(summaries, ignore_index=True) if summaries else _pd.DataFrame()
    return results, combined, model_ids


def partial_specs(configs_summary: pd.DataFrame):
    """A ``specs_of`` callable bound to one ``configs_summary``."""
    from functools import partial

    return partial(config_specs, configs_summary)
