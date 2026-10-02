"""Split-quality and graph-size figures for ``BEG_03_Datasets``.

All ten configurations share one split, so whatever is wrong with it is wrong
with every result in the paper. The figures check for a label distribution that
differs between train and test, a city or country present in only one split, a
size distribution that does not carry across, and a label predictable from graph
size alone.

The frames these draw from are built by :mod:`beg.datasets.tools` and hold node
and edge counts per graph, not the graphs themselves.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from beg.config.naming import resolve, sort_key

#: One colour per split, used by every figure here so they can be read together.
SPLIT_COLOURS = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}

#: Homogeneous and heterogeneous graphs differ enough in size to be worth
#: separating by eye wherever configurations are compared.
GRAPH_KIND_COLOURS = {"homo": "#72b7b2", "hetero": "#b279a2"}

SPLITS = ("train", "val", "test")


def build_label_detail(
    label_map_raw: Mapping[str, float],
    split: Mapping[str, Sequence[str]],
    *,
    label_col: str,
    city_key_by_id: Optional[Mapping[str, str]] = None,
    city_country_lookup: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """One row per model: its label, split, city and country."""
    from beg.datasets.tools import resolve_city_key_by_id

    split_lookup = {
        **{str(mid): "train" for mid in split["train_ids"]},
        **{str(mid): "val" for mid in split["val_ids"]},
        **{str(mid): "test" for mid in split["test_ids"]},
    }
    model_ids = [str(m) for m in label_map_raw]
    frame = pd.DataFrame(
        {
            "model_id": model_ids,
            label_col: [float(label_map_raw[m]) for m in model_ids],
            "split": [split_lookup.get(m, "unknown") for m in model_ids],
            "city_key": resolve_city_key_by_id(model_ids, city_key_by_id=city_key_by_id),
        }
    )
    if city_country_lookup is not None:
        frame["country"] = frame["city_key"].map(city_country_lookup)
    return frame.sort_values(["split", label_col, "model_id"]).reset_index(drop=True)


def build_graph_detail(
    config_payloads: Mapping[str, Mapping[str, Any]],
    label_map_raw: Mapping[str, float],
    *,
    label_col: str,
    city_key_by_id: Optional[Mapping[str, str]] = None,
    city_country_lookup: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Every graph's node and edge count, across every configuration."""
    from beg.datasets.tools import graph_size_detail, resolve_city_key_by_id

    config_order = sorted(config_payloads, key=sort_key)
    frames = []
    for cfg in config_order:
        payload = config_payloads[cfg]
        detail = payload.get("size_detail")
        if detail is None:
            detail = graph_size_detail(
                payload["graphs"], payload["dataset_ids"], payload["split"],
                graph_kind=payload["graph_kind"],
            )
        detail = detail.copy()
        detail["config"] = cfg
        detail["graph_kind"] = payload["graph_kind"]
        detail[label_col] = detail["model_id"].map(lambda m: float(label_map_raw[str(m)]))
        detail["city_key"] = resolve_city_key_by_id(detail["model_id"], city_key_by_id=city_key_by_id)
        if city_country_lookup is not None:
            detail["country"] = detail["city_key"].map(city_country_lookup)
        frames.append(detail)

    combined = pd.concat(frames, ignore_index=True)
    combined["config"] = pd.Categorical(combined["config"], categories=config_order, ordered=True)
    return combined


def build_city_assignment(
    city_split_df: pd.DataFrame,
    group_assignments_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per-city split composition, and whether a city spans more than one split.

    The split is random over models, so cities are expected to span splits.
    """
    frame = city_split_df.copy()
    present = (frame[list(SPLITS)] > 0).sum(axis=1)
    frame["n_splits_present"] = present
    frame["spans_splits"] = present > 1
    frame["dominant_split"] = frame[list(SPLITS)].idxmax(axis=1).where(
        frame[list(SPLITS)].max(axis=1) > 0, "unknown"
    )

    if group_assignments_df is not None and not group_assignments_df.empty:
        meta = group_assignments_df.groupby("city_key", as_index=False).agg(
            country=("country", "first"), n_models=("n_models", "sum"),
        )
        frame = frame.merge(meta, on="city_key", how="left")
    return frame


def build_config_correlations(graph_detail_df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    """How strongly the label tracks graph size, per configuration.

    A high correlation would mean a model could score well by counting nodes,
    which would make the comparison between representations say less than it
    appears to.
    """
    rows = []
    for cfg, sub in graph_detail_df.groupby("config", observed=True):
        rows.append(
            {
                "config": str(cfg),
                "graph_kind": str(sub["graph_kind"].iloc[0]),
                "label_vs_nodes_spearman": float(sub[label_col].corr(sub["num_nodes"], method="spearman")),
                "label_vs_edges_spearman": float(sub[label_col].corr(sub["num_edges"], method="spearman")),
            }
        )
    return pd.DataFrame(rows).sort_values("config", key=lambda s: s.map(sort_key)).reset_index(drop=True)


def plot_label_split_distributions(label_detail_df: pd.DataFrame, *, label_col: str):
    """Is the target distributed the same way in train, val and test?"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    value_cap = float(label_detail_df[label_col].quantile(0.995))
    bins = np.linspace(float(label_detail_df[label_col].min()), value_cap, 45)
    tail_notes = []

    for split_name in SPLITS:
        values = label_detail_df.loc[label_detail_df["split"] == split_name, label_col].to_numpy(dtype=float)
        if not len(values):
            continue
        # The top of this distribution is long and thin. Clipping into the last
        # bin keeps the body legible without hiding how much was clipped.
        clipped = np.minimum(values, value_cap)
        tail_count = int((values > value_cap).sum())
        axes[0, 0].hist(
            clipped, bins=bins, weights=np.ones(len(clipped)) / len(values),
            alpha=0.45, label=f"{split_name} (n={len(values)})", color=SPLIT_COLOURS[split_name],
        )
        tail_notes.append(f"{split_name} above p99.5: {tail_count} ({tail_count / len(values):.2%})")

    axes[0, 0].axvline(value_cap, linestyle="--", color="#444444", linewidth=1.5,
                       label="overflow threshold (global p99.5)")
    axes[0, 0].set_title("Label histogram, tail clipped into the last bin")
    axes[0, 0].set_xlabel(label_col)
    axes[0, 0].set_ylabel("Share of split samples")
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[0, 0].text(
        0.98, 0.98, "\n".join(tail_notes), transform=axes[0, 0].transAxes,
        ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    for split_name in SPLITS:
        values = np.sort(label_detail_df.loc[label_detail_df["split"] == split_name, label_col].to_numpy())
        if not len(values):
            continue
        axes[0, 1].plot(values, np.arange(1, len(values) + 1) / len(values),
                        linewidth=2, label=split_name, color=SPLIT_COLOURS[split_name])
    axes[0, 1].set_title("Empirical CDF by split")
    axes[0, 1].set_xlabel(label_col)
    axes[0, 1].set_ylabel("CDF")
    axes[0, 1].legend(frameon=False)

    box_data = [label_detail_df.loc[label_detail_df["split"] == s, label_col].to_numpy() for s in SPLITS]
    box = axes[1, 0].boxplot(box_data, patch_artist=True, tick_labels=list(SPLITS), showfliers=False)
    for patch, split_name in zip(box["boxes"], SPLITS, strict=True):
        patch.set_facecolor(SPLIT_COLOURS[split_name])
        patch.set_alpha(0.6)
    axes[1, 0].set_title("Label by split")
    axes[1, 0].set_ylabel(label_col)

    ranked = label_detail_df.sort_values(label_col, ascending=False).reset_index(drop=True)
    axes[1, 1].plot(np.arange(1, len(ranked) + 1), ranked[label_col], color="#444444", linewidth=2)
    axes[1, 1].axvline(max(int(0.01 * len(ranked)), 1), linestyle="--", color="#aa3377",
                       linewidth=1.5, label="top 1% cutoff")
    axes[1, 1].set_title("Rank-ordered label tail")
    axes[1, 1].set_xlabel("Rank (1 = highest label)")
    axes[1, 1].set_ylabel(label_col)
    axes[1, 1].legend(frameon=False)

    fig.tight_layout()
    return fig


def plot_city_split_balance(label_detail_df: pd.DataFrame, city_assignment_df: pd.DataFrame):
    """How models and cities are spread across the three splits."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    counts = (
        label_detail_df.groupby("split")
        .agg(n_samples=("model_id", "size"), n_cities=("city_key", "nunique"))
        .reindex(list(SPLITS))
    )
    for ax, column, title in (
        (axes[0, 0], "n_samples", "Samples per split"),
        (axes[0, 1], "n_cities", "Cities per split"),
    ):
        ax.bar(counts.index, counts[column], color=[SPLIT_COLOURS[s] for s in counts.index], alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel(column)
        for i, value in enumerate(counts[column]):
            ax.text(i, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)

    top_cities = city_assignment_df.sort_values("total", ascending=False).head(30).iloc[::-1]
    left = np.zeros(len(top_cities), dtype=float)
    for split_name in SPLITS:
        values = top_cities[split_name].to_numpy(dtype=float)
        axes[1, 0].barh(top_cities["city_key"], values, left=left,
                        color=SPLIT_COLOURS[split_name], alpha=0.88, label=split_name)
        left += values
    axes[1, 0].set_title("Largest 30 cities, by split composition")
    axes[1, 0].set_xlabel("models per city")
    axes[1, 0].legend(frameon=False)

    overlap = city_assignment_df["n_splits_present"].value_counts().reindex([1, 2, 3], fill_value=0).sort_index()
    axes[1, 1].bar(overlap.index.astype(str), overlap.to_numpy(dtype=float),
                   color=["#4c78a8", "#f58518", "#54a24b"], alpha=0.9)
    axes[1, 1].set_title("Cities present in 1, 2, or 3 splits")
    axes[1, 1].set_xlabel("number of splits per city")
    axes[1, 1].set_ylabel("n_cities")
    for i, value in enumerate(overlap.to_numpy(dtype=float)):
        axes[1, 1].text(i, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def plot_country_and_size_balance(
    country_split_df: pd.DataFrame,
    representative_size_detail_df: pd.DataFrame,
    *,
    representative_size_config: str,
    group_assignments_df: Optional[pd.DataFrame] = None,
):
    """Country coverage, and whether building size carries across the splits."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    if group_assignments_df is not None and not group_assignments_df.empty:
        presence = (
            group_assignments_df.groupby("split")["country"].nunique().reindex(list(SPLITS)).fillna(0)
        )
        axes[0, 0].bar(presence.index, presence.to_numpy(),
                       color=[SPLIT_COLOURS[s] for s in presence.index], alpha=0.85)
        for i, value in enumerate(presence.to_numpy()):
            axes[0, 0].text(i, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_title("Countries represented per split")
    axes[0, 0].set_ylabel("n_countries")

    top = country_split_df.head(12)
    positions = np.arange(len(top))
    train = top["train_models"].to_numpy(dtype=float)
    val = top["val_models"].to_numpy(dtype=float)
    test = top["test_models"].to_numpy(dtype=float)
    axes[0, 1].bar(positions, train, color=SPLIT_COLOURS["train"], label="train")
    axes[0, 1].bar(positions, val, bottom=train, color=SPLIT_COLOURS["val"], label="val")
    axes[0, 1].bar(positions, test, bottom=train + val, color=SPLIT_COLOURS["test"], label="test")
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels(top["country"], rotation=45, ha="right")
    axes[0, 1].set_title("Top countries by model count")
    axes[0, 1].set_ylabel("n_models")
    axes[0, 1].legend(frameon=False)

    for split_name in SPLITS:
        values = np.sort(
            representative_size_detail_df.loc[
                representative_size_detail_df["split"] == split_name, "num_nodes"
            ].to_numpy()
        )
        if not len(values):
            continue
        axes[1, 0].plot(values, np.arange(1, len(values) + 1) / len(values), linewidth=2,
                        label=f"{split_name} (n={len(values)})", color=SPLIT_COLOURS[split_name])
    axes[1, 0].set_title(f"Building-size CDF ({representative_size_config})")
    axes[1, 0].set_xlabel("num_nodes")
    axes[1, 0].set_ylabel("CDF")
    axes[1, 0].legend(frameon=False)

    box_data = [
        representative_size_detail_df.loc[
            representative_size_detail_df["split"] == s, "num_nodes"
        ].to_numpy()
        for s in SPLITS
    ]
    box = axes[1, 1].boxplot(box_data, patch_artist=True, tick_labels=list(SPLITS), showfliers=False)
    for patch, split_name in zip(box["boxes"], SPLITS, strict=True):
        patch.set_facecolor(SPLIT_COLOURS[split_name])
        patch.set_alpha(0.6)
    axes[1, 1].set_title("Building size by split")
    axes[1, 1].set_ylabel("num_nodes")

    fig.tight_layout()
    return fig


def plot_graph_config_overview(
    graph_size_summary_df: pd.DataFrame,
    graph_detail_df: pd.DataFrame,
    config_corr_df: pd.DataFrame,
):
    """How the ten configurations differ in size, and how size relates to the label."""
    import matplotlib.pyplot as plt

    config_order = sorted(graph_size_summary_df["config"].tolist(), key=sort_key)
    config_ids = [resolve(cfg).id for cfg in config_order]
    kind_of = dict(
        zip(graph_size_summary_df["config"], graph_size_summary_df["graph_kind"], strict=True)
    )

    def boxplot_by_config(ax, value_col, title, yscale=None):
        grouped = [graph_detail_df.loc[graph_detail_df["config"] == cfg, value_col].to_numpy()
                   for cfg in config_order]
        box = ax.boxplot(grouped, patch_artist=True, showfliers=False)
        for patch, cfg in zip(box["boxes"], config_order, strict=True):
            patch.set_facecolor(GRAPH_KIND_COLOURS[kind_of[cfg]])
            patch.set_alpha(0.65)
        ax.set_xticks(range(1, len(config_order) + 1))
        ax.set_xticklabels(config_ids)
        ax.set_title(title)
        ax.set_ylabel(value_col)
        if yscale:
            ax.set_yscale(yscale)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Configurations differing only in weather features have identical sizes: one point, one label
    ordered = graph_size_summary_df.sort_values("config", key=lambda s: s.map(sort_key))
    points: dict[tuple[float, float], list[str]] = {}
    for _, row in ordered.iterrows():
        position = (round(float(row["nodes_mean"]), 3), round(float(row["edges_mean"]), 3))
        points.setdefault(position, []).append(row["config"])
    for (nodes, edges), configs in points.items():
        axes[0, 0].scatter(nodes, edges, s=120, color=GRAPH_KIND_COLOURS[kind_of[configs[0]]], alpha=0.9)
        axes[0, 0].annotate("/".join(resolve(cfg).id for cfg in configs), (nodes, edges),
                            xytext=(8, 4), textcoords="offset points", fontsize=10)
    axes[0, 0].set_title("Mean node and edge counts by configuration")
    axes[0, 0].set_xlabel("mean nodes")
    axes[0, 0].set_ylabel("mean edges")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")

    boxplot_by_config(axes[0, 1], "num_nodes", "Node-count distribution by configuration", yscale="log")
    boxplot_by_config(axes[1, 0], "num_edges", "Edge-count distribution by configuration", yscale="log")

    x = np.arange(len(config_corr_df))
    width = 0.38
    axes[1, 1].bar(x - width / 2, config_corr_df["label_vs_nodes_spearman"], width=width,
                   label="label vs nodes", color="#6baed6")
    axes[1, 1].bar(x + width / 2, config_corr_df["label_vs_edges_spearman"], width=width,
                   label="label vs edges", color="#fd8d3c")
    axes[1, 1].axhline(0.0, color="#444444", linewidth=1)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([resolve(cfg).id for cfg in config_corr_df["config"]])
    axes[1, 1].set_title("Spearman correlation of label with graph size")
    axes[1, 1].set_ylabel("Spearman rho")
    axes[1, 1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    return fig


def plot_label_vs_graph_size(
    graph_detail_df: pd.DataFrame,
    *,
    label_col: str,
    configs: Sequence[str],
):
    """Label against graph size as hexbin density, one panel per configuration."""
    import matplotlib.pyplot as plt

    present = [cfg for cfg in configs if (graph_detail_df["config"] == cfg).any()]
    if not present:
        raise ValueError("None of the requested configurations are in graph_detail_df.")

    ncols = 3
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5 * nrows), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    max_label = float(graph_detail_df[label_col].max())
    # Zone and component graphs differ tenfold in size, so each row gets its own node range
    row_max_nodes: dict[int, float] = {}
    for index, cfg in enumerate(present):
        nodes = float(graph_detail_df.loc[graph_detail_df["config"] == cfg, "num_nodes"].max())
        row_max_nodes[index // ncols] = max(row_max_nodes.get(index // ncols, 0.0), nodes)

    hexbins = []
    for index, (ax, cfg) in enumerate(zip(axes, present, strict=False)):
        sub = graph_detail_df.loc[graph_detail_df["config"] == cfg]
        hexbins.append(ax.hexbin(sub["num_nodes"], sub[label_col], gridsize=35, mincnt=1, cmap="viridis"))
        ax.set_title(cfg, fontsize=10)
        ax.set_xlabel("num_nodes")
        ax.set_xlim(0, row_max_nodes[index // ncols] * 1.02)
        ax.set_ylim(0, max_label * 1.02)
    for ax in axes[::ncols]:
        ax.set_ylabel(label_col)
    for ax in axes[len(present):]:
        ax.axis("off")

    # One colour scale for every panel, so the colour bar reads the same everywhere
    top = max(float(np.max(h.get_array())) for h in hexbins)
    for h in hexbins:
        h.set_clim(1, top)
    fig.colorbar(hexbins[0], ax=axes.tolist(), shrink=0.9).set_label("graph count")
    return fig


def eui_distribution_summary(eui: pd.Series) -> pd.DataFrame:
    """The target's distribution in the numbers the manuscript quotes."""
    eui = pd.to_numeric(eui, errors="coerce")
    if eui.isna().any():
        raise ValueError(f"{int(eui.isna().sum())} missing or non-numeric EUI value(s)")
    rows = [
        ("models", f"{len(eui):,}"),
        ("minimum", f"{eui.min():.1f}"),
        ("5th percentile", f"{eui.quantile(0.05):.1f}"),
        ("mean", f"{eui.mean():.1f}"),
        ("median", f"{eui.median():.1f}"),
        ("95th percentile", f"{eui.quantile(0.95):.1f}"),
        ("maximum", f"{eui.max():.1f}"),
        ("standard deviation", f"{eui.std(ddof=1):.1f}"),
    ]
    return pd.DataFrame(rows, columns=["statistic", "kWh/(m2 y)"])


def plot_eui_distribution(eui: pd.Series, *, bins: int = 35):
    """Density of the prediction target over every model in the dataset.

    The style is applied through a context, so it does not restyle later figures
    in the kernel.
    """
    import matplotlib.pyplot as plt

    eui = pd.to_numeric(eui, errors="coerce").dropna()
    p05, median, p95 = (float(eui.quantile(q)) for q in (0.05, 0.5, 0.95))
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        ax.hist(eui, bins=bins, density=True, color="#9ec1d8", edgecolor="#74818b", alpha=0.7)
        ax.axvline(p05, color="#74818b", linestyle="--", linewidth=1.5, label=f"5th percentile ({p05:.1f})")
        ax.axvline(median, color="#405c6d", linewidth=2.0, label=f"Median ({median:.1f})")
        ax.axvline(p95, color="#74818b", linestyle="--", linewidth=1.5, label=f"95th percentile ({p95:.1f})")
        ax.set_xlabel("Total EUI [kWh/(m²·y)]", fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.55)
        ax.legend(loc="upper right", fontsize=12, frameon=True)
        fig.tight_layout()
    return fig
