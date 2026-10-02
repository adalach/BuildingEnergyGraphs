from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupedSplitResult:
    dataset_ids: tuple[str, ...]
    idx_train: tuple[int, ...]
    idx_val: tuple[int, ...]
    idx_test: tuple[int, ...]
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    seed: int
    method: str
    group_column: str
    n_groups: int
    n_bins: int
    score: float
    objective_breakdown: dict[str, float]
    group_assignments_df: pd.DataFrame
    split_summary_df: pd.DataFrame


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_id_to_city_key(model_id: str) -> str:
    parts = str(model_id).strip().lower().rsplit("_", 1)
    return parts[0] if len(parts) == 2 else str(model_id).strip().lower()


def load_city_country_lookup_from_epw(epw_dir: Path) -> dict[str, str]:
    epw_dir = Path(epw_dir)
    if not epw_dir.exists():
        raise FileNotFoundError(f"Missing EPW directory: {epw_dir}")

    city_country: dict[str, str] = {}
    for epw_path in sorted(epw_dir.glob("*.epw")):
        with epw_path.open("r", encoding="utf-8", errors="ignore") as handle:
            header = handle.readline().strip()
        parts = [part.strip() for part in header.split(",")]
        if len(parts) < 4 or parts[0].upper() != "LOCATION":
            continue
        city_country[epw_path.stem.strip().lower()] = parts[3] or "unknown"

    if not city_country:
        raise RuntimeError(f"No EPW city-country rows could be parsed from {epw_dir}")
    return city_country


def resolve_city_key_by_id(
    model_ids: Iterable[str],
    *,
    city_key_by_id: Optional[Mapping[str, str]] = None,
) -> list[str]:
    normalized_ids = [str(mid).strip().lower() for mid in model_ids]
    if city_key_by_id is None:
        return [model_id_to_city_key(mid) for mid in normalized_ids]

    missing_ids = [mid for mid in normalized_ids if mid not in city_key_by_id]
    if missing_ids:
        raise RuntimeError(f"Missing city_key mapping for model ids, sample: {missing_ids[:10]}")
    return [str(city_key_by_id[mid]).strip().lower() for mid in normalized_ids]


def compute_graph_common_ids(
    manifest_frames: Sequence[pd.DataFrame],
    *,
    cfg_col: str = "_cfg",
    model_id_col: str = "_mid",
) -> set[str]:
    cfg_to_ids: Dict[str, set[str]] = {}
    for frame in manifest_frames:
        for cfg, sub in frame.groupby(cfg_col):
            cfg_to_ids.setdefault(str(cfg), set()).update(sub[model_id_col].astype(str).tolist())
    if not cfg_to_ids:
        raise ValueError("No configs found while computing graph-common ids.")
    return set.intersection(*(cfg_to_ids[cfg] for cfg in sorted(cfg_to_ids)))


def build_label_scope(
    labels_df: pd.DataFrame,
    *,
    label_col: str,
    graph_common_ids: Iterable[str],
    city_key_by_id: Optional[Mapping[str, str]] = None,
) -> tuple[dict[str, float], list[str], dict[str, Any], pd.DataFrame]:
    series = labels_df.set_index("building_id")[label_col].dropna()
    series.index = series.index.astype(str).str.strip().str.lower()

    graph_common_ids = {str(x).strip().lower() for x in graph_common_ids}
    keep = series[series.index.isin(graph_common_ids)]
    dataset_ids = sorted(keep.index.tolist())

    if not dataset_ids:
        raise RuntimeError("No dataset ids remain after intersecting labels with graph-common ids.")

    label_map = {str(mid): float(keep[mid]) for mid in dataset_ids}
    city_keys = resolve_city_key_by_id(dataset_ids, city_key_by_id=city_key_by_id)
    city_df = (
        pd.DataFrame(
            {
                "model_id": dataset_ids,
                "city_key": city_keys,
                label_col: [label_map[mid] for mid in dataset_ids],
            }
        )
        .sort_values(["city_key", "model_id"])
        .reset_index(drop=True)
    )

    meta = {
        "label_source": {"label_col": str(label_col)},
        "graph_scope": {"n_graph_common_ids": int(len(graph_common_ids))},
        "labels": {
            "n_labels_raw_dropna_global": int(len(labels_df[label_col].dropna())),
            "n_labels_kept_after_graph_match": int(len(dataset_ids)),
        },
    }
    return label_map, dataset_ids, meta, city_df


def summarize_label_tail(
    label_map_raw: Mapping[str, float],
    *,
    label_col: str,
    top_n: int = 20,
    city_key_by_id: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    city_keys = resolve_city_key_by_id(label_map_raw.keys(), city_key_by_id=city_key_by_id)
    rows = [
        {
            "model_id": str(mid),
            "city_key": city_key,
            label_col: float(value),
        }
        for (mid, value), city_key in zip(label_map_raw.items(), city_keys)
    ]
    return pd.DataFrame(rows).sort_values(label_col, ascending=False).head(int(top_n)).reset_index(drop=True)


def _quantile_bins(values: pd.Series, *, n_bins: int) -> tuple[pd.Series, int]:
    effective_bins = max(2, min(int(n_bins), int(values.nunique())))
    try:
        binned = pd.qcut(values, q=effective_bins, labels=False, duplicates="drop")
    except ValueError:
        binned = pd.cut(values, bins=effective_bins, labels=False, duplicates="drop")
    binned = pd.Series(binned, index=values.index).fillna(0).astype(int)
    return binned, int(binned.nunique())


def make_random_model_split(
    dataset_ids: Sequence[str],
    label_map_raw: Mapping[str, float],
    *,
    seed: int = 1234,
    train_fraction: float = 0.70,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
    city_key_by_id: Optional[Mapping[str, str]] = None,
    city_country_lookup: Optional[Mapping[str, str]] = None,
    representative_size_by_id: Optional[Mapping[str, float]] = None,
    representative_size_name: str = "component_num_nodes",
) -> GroupedSplitResult:
    total_fraction = float(train_fraction) + float(val_fraction) + float(test_fraction)
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("train/val/test fractions must sum to 1.0")

    ids = [str(mid) for mid in dataset_ids]
    if len(ids) < 3:
        raise RuntimeError("Random split requires at least three samples.")

    rng = np.random.default_rng(int(seed))
    shuffled = list(ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(float(train_fraction) * n_total))
    n_val = int(round(float(val_fraction) * n_total))
    n_train = max(1, min(n_train, n_total - 2))
    n_val = max(1, min(n_val, n_total - n_train - 1))
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_set = set(shuffled[:n_train])
    val_set = set(shuffled[n_train : n_train + n_val])
    test_set = set(shuffled[n_train + n_val :])
    if not train_set or not val_set or not test_set:
        raise RuntimeError("Random split failed to populate all train/val/test splits.")

    train_ids = tuple(mid for mid in ids if mid in train_set)
    val_ids = tuple(mid for mid in ids if mid in val_set)
    test_ids = tuple(mid for mid in ids if mid in test_set)
    index_lookup = {str(mid): i for i, mid in enumerate(ids)}
    idx_train = tuple(index_lookup[mid] for mid in train_ids)
    idx_val = tuple(index_lookup[mid] for mid in val_ids)
    idx_test = tuple(index_lookup[mid] for mid in test_ids)

    split_summary_df = summarize_split(
        dataset_ids=ids,
        label_map_raw=label_map_raw,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        city_key_by_id=city_key_by_id,
    )

    city_keys = resolve_city_key_by_id(ids, city_key_by_id=city_key_by_id)
    split_lookup = {
        **{str(mid): "train" for mid in train_ids},
        **{str(mid): "val" for mid in val_ids},
        **{str(mid): "test" for mid in test_ids},
    }
    detail_df = pd.DataFrame(
        {
            "model_id": ids,
            "city_key": city_keys,
            "label_raw": [float(label_map_raw[mid]) for mid in ids],
            "split": [split_lookup[str(mid)] for mid in ids],
        }
    )
    if city_country_lookup is not None:
        detail_df["country"] = detail_df["city_key"].map(lambda city: str(city_country_lookup.get(str(city), "unknown")))
    else:
        detail_df["country"] = "unknown"

    if representative_size_by_id is not None:
        detail_df[representative_size_name] = [float(representative_size_by_id[str(mid)]) for mid in ids]
    else:
        detail_df[representative_size_name] = np.nan

    city_totals = (
        detail_df.groupby("city_key", as_index=False)
        .agg(
            city_total_models=("model_id", "size"),
            country=("country", lambda s: s.iloc[0]),
        )
    )
    city_totals["city_size_bin"], _ = _quantile_bins(city_totals["city_total_models"], n_bins=6)
    city_size_bin_lookup = dict(zip(city_totals["city_key"], city_totals["city_size_bin"]))
    city_total_lookup = dict(zip(city_totals["city_key"], city_totals["city_total_models"]))

    group_assignments_df = (
        detail_df.groupby(["city_key", "country", "split"], as_index=False)
        .agg(
            n_models=("model_id", "size"),
            label_mean=("label_raw", "mean"),
            label_median=("label_raw", "median"),
            representative_size_mean=(representative_size_name, "mean"),
            representative_size_median=(representative_size_name, "median"),
        )
        .assign(
            city_size_bin=lambda d: d["city_key"].map(lambda city: int(city_size_bin_lookup.get(str(city), -1))),
            city_total_models=lambda d: d["city_key"].map(lambda city: int(city_total_lookup.get(str(city), 0))),
        )
        .sort_values(["city_key", "split"])
        .reset_index(drop=True)
    )

    return GroupedSplitResult(
        dataset_ids=tuple(ids),
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        seed=int(seed),
        method="random_model_id_split",
        group_column="model_id",
        n_groups=int(len(ids)),
        n_bins=0,
        score=0.0,
        objective_breakdown={},
        group_assignments_df=group_assignments_df,
        split_summary_df=split_summary_df,
    )


def summarize_split(
    *,
    dataset_ids: Sequence[str],
    label_map_raw: Mapping[str, float],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    city_key_by_id: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    split_lookup = {
        **{str(mid): "train" for mid in train_ids},
        **{str(mid): "val" for mid in val_ids},
        **{str(mid): "test" for mid in test_ids},
    }
    rows = []
    total_n = max(len(dataset_ids), 1)
    for split_name in ("train", "val", "test"):
        split_ids = [mid for mid in dataset_ids if split_lookup.get(str(mid)) == split_name]
        values = np.array([float(label_map_raw[mid]) for mid in split_ids], dtype=float)
        cities = set(resolve_city_key_by_id(split_ids, city_key_by_id=city_key_by_id))
        rows.append(
            {
                "split": split_name,
                "n_samples": int(len(split_ids)),
                "share_pct": 100.0 * len(split_ids) / total_n,
                "n_cities": int(len(cities)),
                "label_mean_raw": float(values.mean()) if len(values) else np.nan,
                "label_std_raw": float(values.std(ddof=0)) if len(values) else np.nan,
                "label_median_raw": float(np.median(values)) if len(values) else np.nan,
                "label_min_raw": float(values.min()) if len(values) else np.nan,
                "label_max_raw": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def split_hashes(split: Mapping[str, Sequence[Any]]) -> dict[str, str]:
    return {
        "train_ids_sha": stable_hash([str(x) for x in split["train_ids"]]),
        "val_ids_sha": stable_hash([str(x) for x in split["val_ids"]]),
        "test_ids_sha": stable_hash([str(x) for x in split["test_ids"]]),
    }


def build_split_alignment_summary(config_payloads: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    cfgs = sorted(config_payloads)
    if not cfgs:
        return pd.DataFrame()

    ref_cfg = cfgs[0]
    ref_ids = [str(x) for x in config_payloads[ref_cfg]["dataset_ids"]]
    ref_split = config_payloads[ref_cfg]["split"]

    rows = []
    for cfg in cfgs:
        payload = config_payloads[cfg]
        ids = [str(x) for x in payload["dataset_ids"]]
        split = payload["split"]
        rows.append(
            {
                "config": cfg,
                "n_ids": int(len(ids)),
                "matches_ref_ids": ids == ref_ids,
                "matches_ref_train_ids": [str(x) for x in split["train_ids"]] == [str(x) for x in ref_split["train_ids"]],
                "matches_ref_val_ids": [str(x) for x in split["val_ids"]] == [str(x) for x in ref_split["val_ids"]],
                "matches_ref_test_ids": [str(x) for x in split["test_ids"]] == [str(x) for x in ref_split["test_ids"]],
                **split_hashes(split),
            }
        )
    return pd.DataFrame(rows)


def build_city_split_summary(
    split: Mapping[str, Sequence[str]],
    *,
    city_key_by_id: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    rows = []
    for split_name in ("train", "val", "test"):
        split_ids = [str(mid) for mid in split[f"{split_name}_ids"]]
        city_keys = resolve_city_key_by_id(split_ids, city_key_by_id=city_key_by_id)
        for mid, city_key in zip(split_ids, city_keys):
            rows.append({"model_id": mid, "city_key": city_key, "split": split_name})
    city_df = pd.DataFrame(rows)
    summary = (
        city_df.groupby(["city_key", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for split_name in ("train", "val", "test"):
        if split_name not in summary.columns:
            summary[split_name] = 0
    summary["total"] = summary[["train", "val", "test"]].sum(axis=1)
    summary["single_split_only"] = (
        (summary["train"] > 0).astype(int)
        + (summary["val"] > 0).astype(int)
        + (summary["test"] > 0).astype(int)
    ) == 1
    return summary.sort_values(["single_split_only", "total", "city_key"], ascending=[False, False, True]).reset_index(drop=True)


def build_country_split_summary(group_assignments_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"country", "city_key", "split", "n_models"}
    if group_assignments_df.empty or not required_cols.issubset(group_assignments_df.columns):
        return pd.DataFrame()

    grouped = (
        group_assignments_df.groupby(["country", "split"], as_index=False)
        .agg(
            n_cities=("city_key", "size"),
            n_models=("n_models", "sum"),
        )
    )
    city_wide = grouped.pivot(index="country", columns="split", values="n_cities").fillna(0.0)
    model_wide = grouped.pivot(index="country", columns="split", values="n_models").fillna(0.0)

    rows = []
    for country in sorted(set(city_wide.index.tolist()) | set(model_wide.index.tolist())):
        row = {"country": country}
        n_splits_present = 0
        for split_name in ("train", "val", "test"):
            n_cities = float(city_wide.loc[country, split_name]) if split_name in city_wide.columns and country in city_wide.index else 0.0
            n_models = float(model_wide.loc[country, split_name]) if split_name in model_wide.columns and country in model_wide.index else 0.0
            row[f"{split_name}_cities"] = int(n_cities)
            row[f"{split_name}_models"] = int(n_models)
            if n_cities > 0:
                n_splits_present += 1
        country_rows = group_assignments_df.loc[group_assignments_df["country"] == country].copy()
        row["total_cities"] = int(country_rows["city_key"].nunique())
        row["total_models"] = int(country_rows["n_models"].sum())
        row["n_splits_present"] = int(n_splits_present)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["total_models", "country"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_representative_size_split_summary(
    detail_df: pd.DataFrame,
    *,
    size_col: str = "num_nodes",
) -> pd.DataFrame:
    if detail_df.empty or size_col not in detail_df.columns:
        return pd.DataFrame()

    rows = []
    for split_name in ("train", "val", "test"):
        values = detail_df.loc[detail_df["split"] == split_name, size_col].to_numpy(dtype=float)
        rows.append(
            {
                "split": split_name,
                "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=0)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "p90": float(np.quantile(values, 0.90)) if len(values) else np.nan,
                "p99": float(np.quantile(values, 0.99)) if len(values) else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_label_split_summary(
    label_map_raw: Mapping[str, float],
    split: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    rows = []
    for split_name in ("train", "val", "test"):
        values = np.array([float(label_map_raw[mid]) for mid in split[f"{split_name}_ids"]], dtype=float)
        rows.append(
            {
                "split": split_name,
                "n": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=0)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )

    # Two-sample Kolmogorov-Smirnov test of train against val and test. The KS rows
    # hold the statistic (largest gap between the empirical CDFs) in `mean` and its
    # p-value in `std`.
    from scipy.stats import ks_2samp

    train_values = np.array([float(label_map_raw[mid]) for mid in split["train_ids"]], dtype=float)
    for other_name in ("val", "test"):
        other_values = np.array(
            [float(label_map_raw[mid]) for mid in split[f"{other_name}_ids"]], dtype=float
        )
        ks = ks_2samp(train_values, other_values)
        rows.append(
            {
                "split": f"KS_train_vs_{other_name}",
                "n": np.nan,
                "mean": float(ks.statistic),
                "std": float(ks.pvalue),
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        )

    return pd.DataFrame(rows)


def graph_size_row(graph: Any, graph_kind: str) -> tuple[int, int]:
    """The node and edge count of one graph, whichever kind it is."""
    if graph_kind == "homo":
        num_nodes = int(getattr(graph, "num_nodes", 0) or 0)
        edge_index = getattr(graph, "edge_index", None)
        num_edges = int(edge_index.size(1)) if edge_index is not None and edge_index.numel() > 0 else 0
        return num_nodes, num_edges

    num_nodes = int(sum(graph[nt].num_nodes for nt in graph.node_types))
    num_edges = int(
        sum(
            graph[et].edge_index.size(1)
            for et in graph.edge_types
            if getattr(graph[et], "edge_index", None) is not None
        )
    )
    return num_nodes, num_edges


def graph_size_detail(
    graphs: Sequence[Any],
    dataset_ids: Sequence[str],
    split: Mapping[str, Sequence[str]],
    *,
    graph_kind: str,
) -> pd.DataFrame:
    """Per-graph node and edge counts for one configuration.

    Measured while the configuration is in memory and kept as a small frame, so
    the diagnostics never need every configuration's graphs at once.
    """
    split_lookup = {
        **{str(mid): "train" for mid in split["train_ids"]},
        **{str(mid): "val" for mid in split["val_ids"]},
        **{str(mid): "test" for mid in split["test_ids"]},
    }
    rows = []
    for mid, graph in zip(dataset_ids, graphs, strict=True):
        num_nodes, num_edges = graph_size_row(graph, graph_kind)
        rows.append(
            {
                "model_id": str(mid),
                "split": split_lookup.get(str(mid), "unknown"),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
            }
        )
    return pd.DataFrame(rows)


def build_graph_size_summary(config_payloads: Mapping[str, Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Node and edge statistics per configuration, and the per-graph detail.

    A payload carries either its graphs or, as :func:`export_all_configs` produces,
    a precomputed ``size_detail`` frame.
    """
    summary_rows = []
    detail_tables: dict[str, pd.DataFrame] = {}

    for cfg, payload in sorted(config_payloads.items()):
        detail_df = payload.get("size_detail")
        if detail_df is None:
            detail_df = graph_size_detail(
                payload["graphs"],
                payload["dataset_ids"],
                payload["split"],
                graph_kind=payload["graph_kind"],
            )
        detail_tables[cfg] = detail_df
        summary_rows.append(
            {
                "config": cfg,
                "graph_kind": payload["graph_kind"],
                "n_graphs": int(len(detail_df)),
                "nodes_min": int(detail_df["num_nodes"].min()),
                "nodes_mean": float(detail_df["num_nodes"].mean()),
                "nodes_max": int(detail_df["num_nodes"].max()),
                "edges_min": int(detail_df["num_edges"].min()),
                "edges_mean": float(detail_df["num_edges"].mean()),
                "edges_max": int(detail_df["num_edges"].max()),
            }
        )

    return pd.DataFrame(summary_rows), detail_tables


def write_diagnostics_bundle(
    *,
    output_dir: Path,
    config_payloads: Mapping[str, Mapping[str, Any]],
    label_map_raw: Mapping[str, float],
    split: Mapping[str, Sequence[str]],
    label_col: str,
    label_scope_meta: Optional[Mapping[str, Any]] = None,
    group_assignments_df: Optional[pd.DataFrame] = None,
    representative_size_config: Optional[str] = None,
    city_key_by_id: Optional[Mapping[str, str]] = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_alignment_df = build_split_alignment_summary(config_payloads)
    city_split_df = build_city_split_summary(split, city_key_by_id=city_key_by_id)
    label_split_df = build_label_split_summary(label_map_raw, split)
    graph_size_summary_df, graph_size_detail = build_graph_size_summary(config_payloads)
    country_split_df = build_country_split_summary(
        group_assignments_df if group_assignments_df is not None else pd.DataFrame()
    )
    representative_size_split_df = pd.DataFrame()
    if representative_size_config and representative_size_config in graph_size_detail:
        representative_size_split_df = build_representative_size_split_summary(
            graph_size_detail[representative_size_config],
            size_col="num_nodes",
        )

    artifacts = {
        "split_alignment": output_dir / "split_alignment_summary.csv",
        "city_split": output_dir / "city_split_summary.csv",
        "label_split": output_dir / "label_split_summary.csv",
        "graph_size_summary": output_dir / "graph_size_summary.csv",
        "diagnostics_meta": output_dir / "dataset_diagnostics_meta.json",
    }
    if not country_split_df.empty:
        artifacts["country_split"] = output_dir / "country_split_summary.csv"
    if not representative_size_split_df.empty:
        artifacts["representative_size_split"] = output_dir / "representative_size_split_summary.csv"

    split_alignment_df.to_csv(artifacts["split_alignment"], index=False)
    city_split_df.to_csv(artifacts["city_split"], index=False)
    label_split_df.to_csv(artifacts["label_split"], index=False)
    graph_size_summary_df.to_csv(artifacts["graph_size_summary"], index=False)
    if not country_split_df.empty:
        country_split_df.to_csv(artifacts["country_split"], index=False)
    if not representative_size_split_df.empty:
        representative_size_split_df.to_csv(artifacts["representative_size_split"], index=False)

    for cfg, detail_df in graph_size_detail.items():
        detail_df.to_csv(output_dir / f"graph_size_detail_{cfg}.csv", index=False)

    diag_meta = {
        "label_col": str(label_col),
        "label_scope_meta": dict(label_scope_meta or {}),
        "split_hashes": split_hashes(split),
        "configs": sorted(config_payloads),
        "n_graphs_total": int(len(next(iter(config_payloads.values()))["dataset_ids"])) if config_payloads else 0,
        "representative_size_config": representative_size_config,
    }
    artifacts["diagnostics_meta"].write_text(json.dumps(diag_meta, indent=2), encoding="utf-8")
    return artifacts


def graph_schema_signature(graph: Any) -> dict[str, Any]:
    try:
        from torch_geometric.data import Data, HeteroData
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch_geometric is required to inspect graph schemas.") from exc

    if isinstance(graph, HeteroData):
        return {
            "kind": "hetero",
            "node_types": list(graph.node_types),
            "edge_types": [list(et) for et in graph.edge_types],
            "node_x_dims": {
                nt: int(graph[nt].x.size(-1)) if getattr(graph[nt], "x", None) is not None else 0
                for nt in graph.node_types
            },
            "edge_attr_dims": {
                str(et): int(graph[et].edge_attr.size(-1)) if getattr(graph[et], "edge_attr", None) is not None else 0
                for et in graph.edge_types
            },
        }

    if isinstance(graph, Data):
        x = getattr(graph, "x", None)
        edge_attr = getattr(graph, "edge_attr", None)
        return {
            "kind": "homo",
            "x_dim": int(x.size(-1)) if x is not None else 0,
            "edge_attr_dim": int(edge_attr.size(-1)) if edge_attr is not None else 0,
        }

    return {"kind": type(graph).__name__}


def prune_stale_dataset_exports(datasets_dir: Path, active_configs: Sequence[str]) -> list[str]:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    active = {str(cfg) for cfg in active_configs}
    removed = []
    for child in datasets_dir.iterdir():
        if child.is_dir() and child.name not in active:
            import shutil

            shutil.rmtree(child)
            removed.append(child.name)
    return removed


def config_signature_for(
    *,
    cfg_name: str,
    sample_graph: Any,
    split: Mapping[str, Any],
    label_meta: Mapping[str, Any],
    n_graphs: int,
) -> tuple[dict[str, Any], str]:
    """The identity of a training-ready package, and its hash.

    Computed from *one* graph plus the split and label metadata: every graph in a
    configuration shares one schema, so a rerun can check a package without
    loading it.

    ``BEG_04_Train`` hashes this into ``dataset_fingerprint``, so changing what
    goes in here invalidates every completed run.
    """
    schema = graph_schema_signature(sample_graph)
    signature = {
        "config_name": str(cfg_name),
        "graph_kind": str(schema["kind"]),
        "graph_schema": schema,
        "split": {
            "method": str(split.get("method", "unknown")),
            "group_column": split.get("group_column"),
            "n_groups": split.get("n_groups"),
            "n_bins": split.get("n_bins"),
            "seed": int(split["seed"]),
            "objective_breakdown": split.get("objective_breakdown"),
            "objective_weights": split.get("objective_weights"),
            "representative_size_config": split.get("representative_size_config"),
            "representative_size_name": split.get("representative_size_name"),
            **split_hashes(split),
        },
        "label": dict(label_meta),
        "n_graphs": int(n_graphs),
    }
    return signature, stable_hash(signature)


def export_training_ready_package(
    *,
    cfg_name: str,
    graphs: Sequence[Any],
    split: Mapping[str, Any],
    label_map_raw: Mapping[str, float],
    label_meta: Mapping[str, Any],
    output_dir: Path,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> tuple[Path, Path]:
    import datetime

    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = output_dir / str(cfg_name)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = [str(x) for x in split["dataset_ids"]]
    split_block = {
        "train_idx": [int(i) for i in split["idx_train"]],
        "val_idx": [int(i) for i in split["idx_val"]],
        "test_idx": [int(i) for i in split["idx_test"]],
        "train_ids": [str(x) for x in split["train_ids"]],
        "val_ids": [str(x) for x in split["val_ids"]],
        "test_ids": [str(x) for x in split["test_ids"]],
        "seed": int(split["seed"]),
        "method": str(split.get("method", "unknown")),
        "group_column": split.get("group_column"),
        "n_groups": split.get("n_groups"),
        "n_bins": split.get("n_bins"),
        "score": split.get("score"),
        "objective_breakdown": split.get("objective_breakdown"),
        "objective_weights": split.get("objective_weights"),
        "representative_size_config": split.get("representative_size_config"),
        "representative_size_name": split.get("representative_size_name"),
        **split_hashes(split),
    }

    config_signature, config_signature_hash = config_signature_for(
        cfg_name=cfg_name,
        sample_graph=graphs[0],
        split=split,
        label_meta=label_meta,
        n_graphs=len(dataset_ids),
    )
    graph_kind = str(config_signature["graph_kind"])

    payload = {
        "schema_version": "beg-training-ready-v4",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config_name": str(cfg_name),
        "graph_kind": graph_kind,
        "config_signature_hash": config_signature_hash,
        "config_signature": config_signature,
        "dataset_ids": dataset_ids,
        "graphs": list(graphs),
        "split": split_block,
        "label": dict(label_meta),
        "labels_raw": {
            "by_index": [float(label_map_raw[mid]) for mid in dataset_ids],
            "by_id": {mid: float(label_map_raw[mid]) for mid in dataset_ids},
            "space": "raw",
        },
        "extra_meta": dict(extra_meta or {}),
    }

    pt_path = cfg_dir / "training_ready.pt"
    meta_path = cfg_dir / "training_ready.meta.json"
    tmp_path = pt_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(pt_path)

    human_meta = {
        "schema_version": payload["schema_version"],
        "created_at": payload["created_at"],
        "config_name": payload["config_name"],
        "graph_kind": payload["graph_kind"],
        "config_signature_hash": config_signature_hash,
        "n_graphs": len(dataset_ids),
        "split_sizes": {
            "train": len(split_block["train_ids"]),
            "val": len(split_block["val_ids"]),
            "test": len(split_block["test_ids"]),
        },
        "split": {
            "method": split_block["method"],
            "group_column": split_block["group_column"],
            "n_groups": split_block["n_groups"],
            "n_bins": split_block["n_bins"],
            "score": split_block["score"],
            "objective_breakdown": split_block.get("objective_breakdown"),
            "objective_weights": split_block.get("objective_weights"),
            "representative_size_config": split_block.get("representative_size_config"),
            "representative_size_name": split_block.get("representative_size_name"),
            **split_hashes(split),
        },
        "label": dict(label_meta),
        "notes": "Single source of truth export. Overwritten on each run.",
    }
    meta_path.write_text(json.dumps(human_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return pt_path, meta_path


def split_result_to_dict(
    result: GroupedSplitResult,
    *,
    label_col: str,
    representative_size_config: Optional[str] = None,
    representative_size_name: Optional[str] = None,
) -> dict[str, Any]:
    """The split as the packaging and diagnostics stages expect it.

    ``GroupedSplitResult`` is the split as computed; this is the split as
    recorded, including the three id hashes that ``BEG_04_Train`` fingerprints.
    """
    split = {
        "dataset_ids": list(result.dataset_ids),
        "idx_train": list(result.idx_train),
        "idx_val": list(result.idx_val),
        "idx_test": list(result.idx_test),
        "train_ids": list(result.train_ids),
        "val_ids": list(result.val_ids),
        "test_ids": list(result.test_ids),
        "seed": int(result.seed),
        "method": str(result.method),
        "group_column": str(result.group_column),
        "n_groups": int(result.n_groups),
        "n_bins": int(result.n_bins),
        "score": float(result.score),
        "objective_breakdown": dict(result.objective_breakdown or {}),
        "objective_weights": {},
        "representative_size_config": representative_size_config,
        "representative_size_name": representative_size_name,
        "label_col": str(label_col),
    }
    split.update(split_hashes(split))
    return split


def train_label_normalisation(
    split: Mapping[str, Sequence[str]],
    label_map_raw: Mapping[str, float],
    *,
    label_col: str,
) -> dict[str, Any]:
    """Z-score parameters taken from the training split alone; a zero std becomes 1."""
    y_train = np.array([float(label_map_raw[str(mid)]) for mid in split["train_ids"]], dtype=float)
    if not len(y_train):
        raise RuntimeError("Cannot normalise labels: the training split is empty.")

    mean = float(y_train.mean())
    std = float(y_train.std(ddof=0))
    if std == 0.0:
        std = 1.0

    return {
        "label_col": str(label_col),
        "train_mean_raw": mean,
        "train_std_raw": std,
        "space": "zscore(train_only_raw)",
        "denorm_formula": "y_raw = y_z * train_std_raw + train_mean_raw",
    }


def attach_target(graph: Any, model_id: str, label_map_raw: Mapping[str, float], label_meta: Mapping[str, Any]):
    """Give one graph its normalised target and its model id, in place."""
    import torch

    value = float(label_map_raw[str(model_id)])
    z = (value - float(label_meta["train_mean_raw"])) / float(label_meta["train_std_raw"])
    graph.y = torch.tensor([z], dtype=torch.float32)
    graph.model_id = str(model_id)
    return graph


def load_graph_manifests(graphs_root: Path) -> dict[str, pd.DataFrame]:
    """Read both graph manifests, with model ids folded to lower case.

    ``BEG_02_Graphs`` names each file after the model as the database spells it
    (``Lodz_0011``), while :func:`build_label_scope` indexes lower-case ids; see
    "Model identifiers differ in case between artefacts" in ``docs/limitations.md``.

    Every id downstream of this function is lower case; the original spelling
    remains in ``path``, which names the file on disk.
    """
    graphs_root = Path(graphs_root)
    manifests = {}
    for kind in ("homo", "hetero"):
        path = graphs_root / f"graphs_manifest_{kind}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {kind} manifest: {path}. Run BEG_02_Graphs first."
            )
        frame = pd.read_parquet(path)
        frame["model_id"] = frame["model_id"].astype(str).str.strip().str.lower()
        manifests[kind] = frame
    return manifests


def manifest_paths_for_config(
    manifest_df: pd.DataFrame,
    graphs_root: Path,
    cfg_full_name: str,
) -> dict[str, Path]:
    """``model_id -> absolute path`` for one configuration, keyed lower case.

    Manifest paths are relative to ``graphs_root``; this joins them to it.
    """
    subset = manifest_df.loc[manifest_df["config_full_name"].astype(str) == str(cfg_full_name)]
    return {
        str(row.model_id).strip().lower(): Path(graphs_root) / str(row.path)
        for row in subset.itertuples(index=False)
    }


def representative_size_lookup(
    manifest_df: pd.DataFrame,
    graphs_root: Path,
    cfg_full_name: str,
    *,
    model_ids: Optional[Sequence[str]] = None,
    progress: bool = True,
) -> dict[str, int]:
    """Node count per model for one configuration, read from its graph files.

    The split diagnostics use it to compare building size across train, val and
    test; sizes are only comparable within one configuration.
    """
    import torch

    paths = manifest_paths_for_config(manifest_df, graphs_root, cfg_full_name)
    if not paths:
        raise RuntimeError(f"No manifest rows for representative size config: {cfg_full_name}")

    wanted = [str(m).strip().lower() for m in (model_ids if model_ids is not None else paths)]
    missing = [m for m in wanted if m not in paths]
    if missing:
        raise RuntimeError(
            f"{cfg_full_name} has no graph for {len(missing)} model(s), first: {missing[:5]}"
        )

    iterator = wanted
    if progress:
        try:
            import sys

            from tqdm import tqdm

            # About 20 updates in all, stepped by iteration count: notebooks are
            # stored with their outputs, one stream chunk per tqdm refresh.
            iterator = tqdm(
                wanted,
                desc=f"Sizing {cfg_full_name}",
                unit="graph",
                file=sys.stdout,
                miniters=max(1, len(wanted) // 20),
                mininterval=0,
            )
        except ImportError:
            pass

    sizes = {}
    for model_id in iterator:
        graph = torch.load(paths[model_id], map_location="cpu", weights_only=False)
        sizes[model_id] = int(getattr(graph, "num_nodes", 0) or 0)
    return sizes


def export_all_configs(
    *,
    graphs_root: Path,
    output_dir: Path,
    manifests: Mapping[str, pd.DataFrame],
    configs: Sequence[str],
    dataset_ids: Sequence[str],
    split: Mapping[str, Any],
    label_map_raw: Mapping[str, float],
    label_meta: Mapping[str, Any],
    extra_meta: Optional[Mapping[str, Any]] = None,
    force_rebuild: bool = False,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Package every configuration, holding one in memory at a time.

    Each configuration is loaded, given its targets, written, measured and
    released before the next is read. What is kept per configuration is its
    dataset ids, its split and a ``size_detail`` frame of per-graph node and edge
    counts, which is all the diagnostics need.

    A configuration whose package is current by :func:`config_signature_for`
    is skipped, unless ``force_rebuild`` is set.

    Returns ``config_payloads`` keyed by configuration full name, ready for
    :func:`write_diagnostics_bundle`.
    """
    import gc

    import torch

    from beg.config.naming import resolve

    graphs_root, output_dir = Path(graphs_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_ids = [str(m) for m in dataset_ids]

    iterator = list(configs)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Packaging configurations", unit="config")
        except ImportError:
            pass

    payloads: dict[str, dict[str, Any]] = {}

    for cfg in iterator:
        config = resolve(cfg)
        cfg_full = config.full_name
        kind = "homo" if config.graph_type == "homo" else "hetero"

        paths = manifest_paths_for_config(manifests[kind], graphs_root, cfg_full)
        missing = [mid for mid in dataset_ids if mid not in paths]
        if missing:
            raise RuntimeError(
                f"{cfg_full} has no graph for {len(missing)} of {len(dataset_ids)} models "
                f"(first: {missing[:5]}). Rerun BEG_02_Graphs before packaging."
            )

        cfg_dir = output_dir / cfg_full
        sizes_path = cfg_dir / "graph_sizes.csv"
        meta_path = cfg_dir / "training_ready.meta.json"
        package_path = cfg_dir / "training_ready.pt"

        # `paths` is bound as a default so the closure keeps this iteration's paths.
        def load(model_id: str, _paths=paths):
            return torch.load(_paths[model_id], map_location="cpu", weights_only=False)

        if not force_rebuild and package_path.exists() and meta_path.exists() and sizes_path.exists():
            _, expected_hash = config_signature_for(
                cfg_name=cfg_full,
                sample_graph=load(dataset_ids[0]),
                split=split,
                label_meta=label_meta,
                n_graphs=len(dataset_ids),
            )
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if stored.get("config_signature_hash") == expected_hash:
                payloads[cfg_full] = {
                    "dataset_ids": list(dataset_ids),
                    "split": split,
                    "graph_kind": str(stored.get("graph_kind", kind)),
                    "size_detail": pd.read_csv(sizes_path),
                    "package_path": package_path,
                    "reused": True,
                }
                continue

        graphs = [attach_target(load(mid), mid, label_map_raw, label_meta) for mid in dataset_ids]

        detail_df = graph_size_detail(graphs, dataset_ids, split, graph_kind=kind)

        export_training_ready_package(
            cfg_name=cfg_full,
            graphs=graphs,
            split=split,
            label_map_raw=label_map_raw,
            label_meta=label_meta,
            output_dir=output_dir,
            extra_meta=extra_meta,
        )
        cfg_dir.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(sizes_path, index=False)

        payloads[cfg_full] = {
            "dataset_ids": list(dataset_ids),
            "split": split,
            "graph_kind": kind,
            "size_detail": detail_df,
            "package_path": package_path,
            "reused": False,
        }

        del graphs
        gc.collect()

    return payloads


def export_summary(payloads: Mapping[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per configuration: what was written, and how big its graphs are."""
    rows = []
    for cfg, payload in sorted(payloads.items()):
        detail = payload["size_detail"]
        rows.append(
            {
                "config": cfg,
                "graph_kind": payload["graph_kind"],
                "n_graphs": int(len(detail)),
                "nodes_mean": round(float(detail["num_nodes"].mean()), 1),
                "edges_mean": round(float(detail["num_edges"].mean()), 1),
                "package_mb": round(Path(payload["package_path"]).stat().st_size / 1e6, 1),
                "reused": bool(payload["reused"]),
            }
        )
    return pd.DataFrame(rows)
