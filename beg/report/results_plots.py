"""The paper's results figures, drawn from the verified runs.

Five figures answer four questions: how the ten configurations compare across
the whole search, how they compare at their best, what weather buys, and whether
the selected runs actually trained well rather than merely scoring well. A sixth
shows the computational cost.

Zone and component configurations sit in separate panels, or are set apart by a
band or a divider where they share an axis. They differ by an order of magnitude
in graph size and, in the homogeneous case, in achievable accuracy, and C1
against C6 is a granularity change, not a ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from beg.config.naming import resolve, sort_key
from beg.report import style as S

#: Read by the cost figure and the notebooks.
FIGURE_FACECOLOR = S.FIGURE_FACECOLOR


def split_panels(configs: Sequence[str]) -> tuple[list[str], list[str]]:
    """Zone configurations and component configurations, in order."""
    ordered = sorted(configs, key=sort_key)
    zone = [c for c in ordered if resolve(c).level == "zone"]
    component = [c for c in ordered if resolve(c).level == "component"]
    return zone, component

def _new_figure(nrows: int, ncols: int, figsize, **kwargs):
    """A figure and axes on exactly white, for the manuscript."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=S.FIGURE_FACECOLOR, **kwargs)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(S.FIGURE_FACECOLOR)
    return fig, axes

def _kind_of(runs: pd.DataFrame, config_name: str) -> str:
    rows = runs.loc[runs["config_name"] == config_name, "graph_kind"]
    return str(rows.iloc[0]) if len(rows) else "hetero"


def _kind_legend(ax, *, labels=None, frameon: bool = True, **kwargs):
    """The one legend these figures need: which colour is which graph type.

    Homogeneous first, as published: it is the simpler representation and the
    figures read left to right from simple to elaborate.
    """
    from matplotlib.patches import Patch

    names = labels or S.KIND_SHORT_LABEL
    ax.legend(
        handles=[
            Patch(facecolor=S.KIND_FILL[kind], edgecolor=S.KIND_EDGE[kind], label=names[kind])
            for kind in ("homo", "hetero")
        ],
        frameon=frameon,
        loc="upper right",
        **kwargs,
    )


def _draw_kind_box(ax, values, position, kind, *, width=0.62):
    """One box in the published style: pale opaque fill, deep edge and median."""
    edge, fill = S.KIND_EDGE[kind], S.KIND_FILL[kind]
    box = ax.boxplot(
        values,
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": edge, "linewidth": 1.7},
        whiskerprops={"color": edge, "linewidth": 1.1},
        capprops={"color": edge, "linewidth": 1.1},
    )
    box["boxes"][0].set(facecolor=fill, edgecolor=edge, alpha=1.0, linewidth=1.3)
    return box


def plot_all_runs_boxplot(runs: pd.DataFrame, *, metric: str = "test_mse_z"):
    """Every run's test error, per configuration.

    Shows how consistent each configuration is across the search, which the
    selected-run table cannot.
    """
    frame = runs.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[metric])

    zone, component = split_panels(frame["config_name"].unique())
    with S.ALL_RUNS.context():
        fig, (left, right) = _new_figure(1, 2, S.ALL_RUNS.figsize, sharey=True)

        def panel(ax, configs, title):
            for index, config in enumerate(configs):
                values = frame.loc[frame["config_name"] == config, metric].to_numpy(float)
                _draw_kind_box(ax, values, index, _kind_of(frame, config))
            ax.set_xticks(np.arange(len(configs)))
            ax.set_xticklabels([resolve(c).id for c in configs])
            ax.set_title(title)
            ax.grid(axis="y", color=S.GRID_COLOUR, linewidth=0.8, alpha=0.48)
            ax.set_axisbelow(True)

        panel(left, zone, "zone level")
        left.set_ylabel(metric)
        panel(right, component, "component level")
        _kind_legend(right)
        fig.tight_layout()
    fig.beg_style = S.ALL_RUNS
    return fig


def plot_all_runs_scatter(runs: pd.DataFrame, *, metric: str = "test_mse_z", seed: int = 0):
    """The same runs as points, so the shape of each cluster is visible.

    Shows whether a configuration's runs form one tight group or two, as when a
    model family diverges on some variants and not others.
    """
    frame = runs.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[metric])

    zone, component = split_panels(frame["config_name"].unique())
    rng = np.random.default_rng(seed)
    with S.ALL_RUNS.context():
        fig, (left, right) = _new_figure(1, 2, S.ALL_RUNS.figsize, sharey=True)

        def panel(ax, configs, title, jitter=0.22):
            for index, config in enumerate(configs):
                subset = frame[frame["config_name"] == config]
                if subset.empty:
                    continue
                kind = _kind_of(frame, config)
                x = index + (rng.random(len(subset)) - 0.5) * 2 * jitter
                ax.scatter(
                    x,
                    subset[metric].to_numpy(float),
                    s=28,
                    alpha=0.55,
                    facecolor=S.KIND_FILL[kind],
                    edgecolor=S.KIND_EDGE[kind],
                    linewidths=0.6,
                    marker=S.KIND_MARKER[kind],
                )
            ax.set_xticks(np.arange(len(configs)))
            ax.set_xticklabels([resolve(c).id for c in configs])
            ax.set_title(title)
            ax.grid(axis="y", color=S.GRID_COLOUR, linewidth=0.8, alpha=0.48)
            ax.set_axisbelow(True)

        panel(left, zone, "zone level")
        left.set_ylabel(metric)
        panel(right, component, "component level")
        _kind_legend(right)
        fig.tight_layout()
    fig.beg_style = S.ALL_RUNS
    return fig


def plot_weather_effect(runs: pd.DataFrame, *, metric: str = "best_val_mse_z", seed: int = 20260716):
    """Every run, grouped by how weather enters the graph.

    Weather is the x-axis grouping and colour carries only the graph type, so the
    two questions the figure answers stay separable. Within a group the matching
    zone and component configurations sit side by side on one y-axis; a grey band
    runs behind the component columns and through their labels.

    Plotted on validation error: this is the comparison the configurations were
    designed to support, and reading it off the test set would turn a design
    question into a second model selection.
    """
    from matplotlib.patches import Rectangle

    frame = runs.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[metric])

    grouped = S.configs_by_weather()
    available = {c for c in frame["config_name"].unique()}
    grouped = {
        weather: [c for c in configs if c.full_name in available or c.id in available]
        for weather, configs in grouped.items()
    }
    grouped = {w: c for w, c in grouped.items() if c}

    flat = [config for configs in grouped.values() for config in configs]
    positions = np.arange(len(flat), dtype=float)
    group_positions: dict[str, list[float]] = {}
    cursor = 0
    for weather, configs in grouped.items():
        group_positions[weather] = positions[cursor:cursor + len(configs)].tolist()
        cursor += len(configs)

    rng = np.random.default_rng(seed)

    with S.WEATHER.context():
        fig, ax = _new_figure(1, 1, S.WEATHER.figsize)

        # The component band sits behind everything and reaches down through the
        # granularity labels. Opaque box fills keep it from tinting the boxes.
        for config, position in zip(flat, positions, strict=True):
            if config.level == "component":
                ax.add_patch(Rectangle(
                    (position - 0.5, -0.18), 1.0, 1.18,
                    transform=ax.get_xaxis_transform(),
                    facecolor=S.COMPONENT_BACKGROUND, edgecolor="none",
                    clip_on=False, zorder=0,
                ))

        for weather, configs in grouped.items():
            for config, position in zip(configs, group_positions[weather], strict=True):
                subset = frame.loc[frame["config_name"].isin({config.full_name, config.id})]
                values = subset[metric].to_numpy(float)
                if len(values) == 0:
                    raise RuntimeError(f"no stable runs available for {config.id}")
                kind = config.graph_type
                _draw_kind_box(ax, values, position, kind)
                ax.scatter(
                    np.full(len(values), position) + rng.normal(0.0, 0.075, size=len(values)),
                    values, s=18, color=S.KIND_EDGE[kind], alpha=0.38, linewidths=0, zorder=3,
                )

        ax.set_xticks(positions, [c.id for c in flat], fontsize=S.WEATHER_TICK)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.grid(axis="y", color=S.GRID_COLOUR, linewidth=0.8, alpha=0.48)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(flat) - 0.5)

        # A boxed caption per weather group, above the axis, and the granularity
        # of each configuration below it.
        for weather, members in group_positions.items():
            start, width = members[0] - 0.5, len(members)
            ax.add_patch(Rectangle(
                (start, 1.0), width, 0.12, transform=ax.get_xaxis_transform(),
                facecolor="white", edgecolor=S.GROUP_LABEL_EDGE, linewidth=1.0,
                clip_on=False, zorder=5,
            ))
            ax.text(float(np.mean(members)), 1.06, S.WEATHER_LABEL[weather],
                    transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=S.WEATHER_TICK, zorder=6)
        ax.text(0.5, 1.19, "Weather representation", transform=ax.transAxes,
                ha="center", va="center", fontsize=S.WEATHER_TICK)
        for config, position in zip(flat, positions, strict=True):
            ax.text(position, -0.145, config.level, transform=ax.get_xaxis_transform(),
                    ha="center", va="center", fontsize=S.WEATHER_TICK, zorder=4)

        ax.set_ylabel("Best validation MSE (z-score)")
        ax.set_xlabel("Granularity", fontsize=S.WEATHER_TICK, labelpad=24)
        ax.set_ylim(bottom=0.0, top=max(0.5, float(frame[metric].max()) * 1.08))
        _kind_legend(ax, title="Graph representation", title_fontsize=S.WEATHER_LEGEND,
                     fontsize=S.WEATHER_LEGEND, bbox_to_anchor=(0.99, 0.99),
                     bbox_transform=ax.transAxes)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.78, bottom=0.29)
    fig.beg_style = S.WEATHER
    return fig

def _grid_figure(*, share_x: bool = True, share_y: bool = True):
    """The 2x5 panel grid both large figures use.

    Rows are zone then component, columns are the five weather-by-graph-type
    combinations, and a narrow spacer column separates the two homogeneous
    columns from the three heterogeneous ones so the eye does not read the row
    as a single sequence.

    The prediction grid shares both scales, because the panels are only
    comparable on one square range. The curve grid shares only the vertical
    scale, since each run stopped at its own epoch and places its own ticks.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=S.GRID.figsize, facecolor=S.FIGURE_FACECOLOR)
    spec = fig.add_gridspec(2, 6, width_ratios=S.GRID_WIDTH_RATIOS, hspace=0.16, wspace=0.08)
    axes = np.empty((2, 5), dtype=object)
    first = None
    for row in range(2):
        for column in range(5):
            ax = fig.add_subplot(
                spec[row, S.GRID_COLUMNS[column]],
                sharex=first if share_x else None,
                sharey=first if share_y else None,
            )
            ax.set_facecolor(S.FIGURE_FACECOLOR)
            first = first or ax
            axes[row, column] = ax
    return fig, axes


def _grid_headers(fig, axes, *, xlabel: Optional[str] = None) -> None:
    """The two section headers, and the shared x label when the figure has one."""
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.14)
    spans = {
        "Homogeneous graphs": (
            axes[0, S.HOMOGENEOUS_COLUMNS[0]].get_position().x0,
            axes[0, S.HOMOGENEOUS_COLUMNS[1]].get_position().x1,
        ),
        "Heterogeneous graphs": (
            axes[0, S.HETEROGENEOUS_COLUMNS[0]].get_position().x0,
            axes[0, S.HETEROGENEOUS_COLUMNS[1]].get_position().x1,
        ),
    }
    for text, (left, right) in spans.items():
        centre = (left + right) / 2
        fig.text(centre, 0.925, text, ha="center", va="center", fontsize=S.GRID_SECTION_HEADER)
        if xlabel:
            fig.text(centre, 0.055, xlabel, ha="center", va="center", fontsize=S.GRID_AXIS_LABEL)


def _square_limits(observed_max: float, *, divisions: int = 3):
    """A shared square range from zero to a round number above the data.

    Energy use intensity has a floor at zero, and the panels are only comparable
    on one range, so the axes start there and end at the next round step above
    the largest prediction.
    """
    if not np.isfinite(observed_max) or observed_max <= 0:
        return 0.0, 1.0, np.arange(divisions + 1) / divisions

    target = observed_max / divisions
    magnitude = 10.0 ** np.floor(np.log10(target))
    step = 10 * magnitude
    for multiple in (1, 2, 2.5, 3, 4, 5, 10):
        candidate = multiple * magnitude
        if candidate >= target:
            step = candidate
            break
    return 0.0, float(step * divisions), np.arange(divisions + 1) * step


def _row_for(rows: pd.DataFrame, config) -> Optional[pd.Series]:
    match = rows.loc[rows["config_name"].isin({config.full_name, config.id})]
    return None if match.empty else match.iloc[0]


def plot_training_curves(best: pd.DataFrame, runs_dir: Path):
    """Training and validation loss for each selected run, in the published grid.

    A selected run that scored well while still improving, or while diverging
    from its training loss, is a different claim from one that converged. The
    dashed marker is the epoch whose weights were kept.
    """
    from matplotlib.lines import Line2D

    from beg.report.results import load_history

    grid = S.config_grid()
    with S.GRID.context():
        fig, axes = _grid_figure(share_x=False)
        for row_index, row in enumerate(grid):
            for column_index, config in enumerate(row):
                ax = axes[row_index, column_index]
                entry = _row_for(best, config)
                if entry is None:
                    ax.set_axis_off()
                    continue
                history = load_history(runs_dir, entry["run_dir"])
                ax.plot(history["epoch"], history["train_mse_z"], linewidth=2.0,
                        color=S.TRAIN_COLOUR, label="Train")
                ax.plot(history["epoch"], history["val_mse_z"], linewidth=2.0,
                        color=S.VALIDATION_COLOUR, label="Validation")
                best_epoch = pd.to_numeric(pd.Series([entry.get("best_val_epoch")]),
                                           errors="coerce").iloc[0]
                if pd.notna(best_epoch):
                    ax.axvline(float(best_epoch), linestyle="--", color=S.VALIDATION_COLOUR,
                               linewidth=1.4, alpha=0.7)
                # Linear and capped, as published, so the flat tail after
                # convergence stays legible.
                ax.set_ylim(0.0, S.CURVE_YMAX)
                ax.set_yticks(S.CURVE_YTICKS)
                # Three ticks only: the start, the epoch whose weights were
                # kept, and where training stopped. The last is dropped when it
                # would collide with the best-epoch label, which happens
                # whenever a run stopped soon after its best epoch.
                last_epoch = float(history["epoch"].max())
                ticks = [0.0, last_epoch]
                if pd.notna(best_epoch):
                    marked = float(best_epoch)
                    if abs(last_epoch - marked) < 0.08 * max(last_epoch, 1.0):
                        ticks = [0.0, marked]
                    else:
                        ticks = [0.0, marked, last_epoch]
                ax.set_xticks(sorted(set(ticks)))
                ax.grid(axis="both", alpha=0.22)
                ax.set_axisbelow(True)
                ax.text(0.50, 0.93, config.id, transform=ax.transAxes, ha="center",
                        va="top", fontsize=S.GRID_PANEL_LABEL)
                if row_index == 0:
                    ax.set_title(S.GRID_COLUMN_LABELS[column_index], pad=20,
                                 fontsize=S.GRID_COLUMN_TITLE)
                # Both rows keep their epoch ticks: the runs are different
                # lengths, so the top row's axis is not the bottom row's.
                if column_index in (0, 2):
                    ax.set_ylabel(f"{S.GRID_ROW_TITLE[S.GRID_ROW_LEVELS[row_index]]} MSE_z",
                                  fontsize=S.GRID_AXIS_LABEL, labelpad=8)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        # No shared x label: the space goes to the legend, as in the published
        # figure.
        _grid_headers(fig, axes, xlabel=None)
        fig.legend(
            handles=[
                Line2D([0], [0], color=S.TRAIN_COLOUR, linewidth=2.4, label="Train"),
                Line2D([0], [0], color=S.VALIDATION_COLOUR, linewidth=2.4, label="Validation"),
                Line2D([0], [0], color=S.VALIDATION_COLOUR, linewidth=1.6, linestyle="--",
                       alpha=0.7, label="Best validation epoch"),
            ],
            loc="lower center", ncol=3, frameon=True, fontsize=S.GRID_LEGEND,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.beg_style = S.GRID
    return fig


def plot_test_predictions(best: pd.DataFrame, runs_dir: Path, *,
                          target_column: Optional[str] = None):
    """Predicted against measured EUI for each selected run, in the published grid.

    The scatter shows what R2 hides: a model accurate in the middle of the range
    and flat in the tail scores well and is not useful for the buildings that
    matter most. Points are drawn faint and rasterised, so the vector copy stays
    small enough to open.
    """
    from beg.report.results import load_predictions

    def columns_of(predictions):
        truth = target_column or next(
            (c for c in ("y_true_raw", "target_raw", "y_true", "target") if c in predictions.columns), None)
        pred = next(
            (c for c in ("y_pred_raw", "prediction_raw", "y_pred", "prediction") if c in predictions.columns), None)
        if truth is None or pred is None:
            raise KeyError("test_predictions.csv has no recognisable truth/prediction columns; "
                           f"found {list(predictions.columns)}")
        return truth, pred

    grid = S.config_grid()
    drawn: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for row_index, row in enumerate(grid):
        for column_index, config in enumerate(row):
            entry = _row_for(best, config)
            if entry is None:
                continue
            predictions = load_predictions(runs_dir, entry["run_dir"])
            truth_col, pred_col = columns_of(predictions)
            drawn[(row_index, column_index)] = (
                pd.to_numeric(predictions[truth_col], errors="coerce").to_numpy(float),
                pd.to_numeric(predictions[pred_col], errors="coerce").to_numpy(float),
            )
    if not drawn:
        raise RuntimeError("no selected runs have predictions to plot")

    observed = float(max(np.nanmax(np.concatenate(pair)) for pair in drawn.values()))
    lower, upper, ticks = _square_limits(observed)

    with S.GRID.context():
        fig, axes = _grid_figure()
        for row_index, row in enumerate(grid):
            for column_index, config in enumerate(row):
                ax = axes[row_index, column_index]
                pair = drawn.get((row_index, column_index))
                if pair is None:
                    ax.set_axis_off()
                    continue
                truth, prediction = pair
                ax.scatter(truth, prediction, s=18, color=S.TRAIN_COLOUR, alpha=0.16,
                           linewidths=0, rasterized=True)
                ax.plot([lower, upper], [lower, upper], color=S.VALIDATION_COLOUR,
                        linewidth=1.6, linestyle="--", alpha=0.85)
                ax.set_xlim(lower, upper)
                ax.set_ylim(lower, upper)
                ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
                ax.set_yticks(ticks)
                ax.grid(axis="both", alpha=0.22)
                ax.set_axisbelow(True)
                ax.tick_params(axis="both", pad=6)
                ax.tick_params(axis="x", labelsize=16)
                labels = ax.get_xticklabels()
                if labels:
                    labels[0].set_ha("left")
                    labels[-1].set_ha("right")
                ax.text(0.50, 0.93, config.id, transform=ax.transAxes, ha="center",
                        va="top", fontsize=S.GRID_PANEL_LABEL)
                if row_index == 0:
                    ax.set_title(S.GRID_COLUMN_LABELS[column_index], pad=20,
                                 fontsize=S.GRID_COLUMN_TITLE)
                    ax.tick_params(axis="x", labelbottom=False)
                if column_index in (0, 2):
                    ax.set_ylabel(f"{S.GRID_ROW_TITLE[S.GRID_ROW_LEVELS[row_index]]} pred. EUI",
                                  fontsize=S.GRID_AXIS_LABEL, labelpad=8)
                else:
                    ax.tick_params(axis="y", labelleft=False)

        _grid_headers(fig, axes, xlabel="True EUI [kWh/(m²·y)]")
    fig.beg_style = S.GRID
    return fig


def save_figure(fig, path: Path, *, style=None, dpi: Optional[int] = None,
                formats: Optional[Sequence[str]] = None) -> list[Path]:
    """Write a figure as the manuscript needs it: 350 dpi plus a vector copy.

    Each ``plot_*`` function tags its figure with the style it was drawn in, and
    the export settings come from that style. Pass ``dpi`` or ``formats`` to
    override for a draft.
    """
    return S.save(fig, path, style=style or getattr(fig, "beg_style", None),
                  dpi=dpi, formats=formats)

# ---------------------------------------------------------------------------
# Computational cost
# ---------------------------------------------------------------------------

#: Colours for the cost figure. The graph-type pair is the same one every other
#: results figure uses; only the simulation panels add colours of their own.
ENERGYPLUS_FILL, ENERGYPLUS_EDGE = S.ENERGYPLUS_FILL, S.ENERGYPLUS_EDGE
MEDIAN_COLOUR, PERCENTILE_COLOUR = S.MEDIAN_COLOUR, S.PERCENTILE_COLOUR
KIND_EDGE = S.KIND_EDGE
KIND_FILL = S.KIND_FILL

#: Typography for the cost figure, from beg.report.style. Applied through
#: rc_context, so the settings do not leak into figures drawn after it.
COST_FIGURE_RC = {"font.family": "DejaVu Sans", **S.COST.rc}


def _style_cost_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=S.GRID_COLOUR, linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)


def _cost_histogram(ax, values, *, title, xlabel) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 29)
    ax.hist(values, bins=bins, color=ENERGYPLUS_FILL, edgecolor=ENERGYPLUS_EDGE, linewidth=0.65, alpha=0.82)
    ax.set_xscale("log")
    ax.axvline(np.median(values), color=MEDIAN_COLOUR, linewidth=1.5, zorder=4)
    for q in (5, 95):
        ax.axvline(np.percentile(values, q), color=PERCENTILE_COLOUR, linewidth=1.2, linestyle="--", zorder=4)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    _style_cost_axis(ax)


def _granularity_divider(ax, configs: Sequence[str]) -> None:
    """Separate zone from component configurations, wherever the boundary is.

    Derived from the configurations' levels, so it stays in the right place when
    the set is narrowed or reordered.
    """
    levels = [resolve(c).level for c in configs]
    boundary = next((i for i in range(1, len(levels)) if levels[i] != levels[i - 1]), None)
    if boundary is None:
        return
    ax.axvline(boundary + 0.5, color="#A9A9A9", linewidth=0.8, zorder=1)
    for label, lo, hi in (("Zone-level", 1, boundary), ("Component-level", boundary + 1, len(configs))):
        ax.text((lo + hi) / 2, -0.15, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.3, clip_on=False)


def plot_computational_cost(
    simulation: pd.DataFrame,
    runs: pd.DataFrame,
    packages: pd.DataFrame,
    best: pd.DataFrame,
):
    """What producing the dataset cost, against what the surrogates cost.

    Four panels: EnergyPlus wall-clock time and SQL output size per simulated
    model, then total search time and model package size per configuration. The
    selected run of each configuration is marked in the last panel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    configs = sorted(runs["config_name"].unique(), key=sort_key)
    positions = np.arange(1, len(configs) + 1)
    tick_labels = [resolve(c).id for c in configs]
    kinds = [resolve(c).graph_type for c in configs]

    with plt.rc_context(COST_FIGURE_RC):
        fig, axes = _new_figure(2, 2, (7.35, 5.65))

        _cost_histogram(axes[0, 0], simulation["simulation_time_s"],
                        title="(a) EnergyPlus simulation time",
                        xlabel="Elapsed time per simulation [s] (log scale)")
        _cost_histogram(axes[0, 1], simulation["sql_size_mib"],
                        title="(b) EnergyPlus SQL result file size",
                        xlabel="SQL result size per simulation [MiB] (log scale)")

        hours = (
            runs.assign(train_seconds=pd.to_numeric(runs["train_seconds"], errors="coerce"))
            .groupby("config_name")["train_seconds"].sum()
            .reindex(configs) / 3600.0
        )
        ax = axes[1, 0]
        ax.bar(positions, hours.to_numpy(), color=[KIND_FILL[k] for k in kinds],
               edgecolor=[KIND_EDGE[k] for k in kinds], linewidth=0.8, alpha=0.95)
        ax.set_xticks(positions, tick_labels)
        ax.set_title("(c) Full hyperparameter search time", loc="left", fontsize=10.5)
        ax.set_ylabel("Cumulative training time [h]")
        ax.legend(handles=[Patch(facecolor=KIND_FILL[k], edgecolor=KIND_EDGE[k], label=name)
                           for k, name in (("homo", "Homogeneous"), ("hetero", "Heterogeneous"))],
                  loc="upper left", frameon=False, fontsize=8.0)
        _granularity_divider(ax, configs)
        _style_cost_axis(ax)

        ax = axes[1, 1]
        selected = dict(zip(best["config_name"], best["run_dir"], strict=True))
        size_of = dict(zip(packages["run_dir"], packages["package_mib"], strict=True))
        rng = np.random.default_rng(20260726)
        for position, config, kind in zip(positions, configs, kinds, strict=True):
            sizes = packages.loc[packages["config_name"] == config, "package_mib"].to_numpy(float)
            ax.scatter(position + rng.uniform(-0.2, 0.2, len(sizes)), sizes, s=7,
                       color=KIND_FILL[kind], edgecolors="none", alpha=0.28, zorder=2)
            ax.boxplot([sizes], positions=[position], widths=0.48, patch_artist=True, showfliers=False,
                       medianprops={"color": MEDIAN_COLOUR, "linewidth": 1.4},
                       boxprops={"facecolor": KIND_FILL[kind], "edgecolor": KIND_EDGE[kind],
                                 "linewidth": 0.9, "alpha": 0.55},
                       whiskerprops={"color": KIND_EDGE[kind], "linewidth": 0.8},
                       capprops={"color": KIND_EDGE[kind], "linewidth": 0.8})
            # One colour for the marker and its legend entry, so the legend
            # describes the mark actually drawn.
            ax.scatter(position, size_of[selected[config]], marker="D", s=28, color=MEDIAN_COLOUR,
                       edgecolors="white", linewidths=0.55, zorder=5)
        ax.set_xticks(positions, tick_labels)
        ax.set_ylim(bottom=0)
        ax.set_title("(d) Deployable model package sizes", loc="left", fontsize=10.5)
        ax.set_ylabel("Model package size [MiB]")
        ax.legend(handles=[Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=MEDIAN_COLOUR,
                                  markeredgecolor="white", markersize=5.5, label="Selected model")],
                  loc="upper left", frameon=False, fontsize=8.0)
        _granularity_divider(ax, configs)
        _style_cost_axis(ax)

        fig.subplots_adjust(left=0.095, right=0.985, bottom=0.13, top=0.965, wspace=0.24, hspace=0.42)
    return fig
