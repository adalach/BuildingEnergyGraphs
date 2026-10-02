"""The figure style the published results use.

Every results figure in the manuscript uses the same small vocabulary, and this
module is the one place it is written down. Colour encodes the graph
representation and nothing else: heterogeneous is blue, homogeneous is orange,
in a deep edge colour with a pale fill. Weather is shown by position in the
layout, so the figures stay readable in greyscale.

Figure sizes and font sizes are recorded per figure, as they produced the
published appearance. The cost figure is authored at final print width with 9 pt
text, while the large grids are authored oversized with proportionally large
text and scaled down by LaTeX. Changing either changes the printed result.

The rc settings are applied through ``matplotlib.rc_context``, so they do not
leak into later figures in the same process, including a reader's own plots in a
notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from beg.config.naming import CONFIGS, ConfigId

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

#: Edge colour per graph type. The only thing colour encodes in these figures.
KIND_EDGE = {"hetero": "#1b6ca8", "homo": "#d95f02"}

#: Box and marker fills, pale enough to carry a label but distinct in greyscale.
KIND_FILL = {"hetero": "#b9d6ea", "homo": "#f3c7a6"}

#: A second channel for the same distinction, so colour is never load-bearing.
KIND_MARKER = {"hetero": "o", "homo": "s"}


#: The short forms the manuscript's figures use, where space is tight.
KIND_SHORT_LABEL = {"hetero": "hetero", "homo": "homo"}

#: Training curves. Matplotlib's tab:blue and tab:orange, as published.
TRAIN_COLOUR = "#1f77b4"
VALIDATION_COLOUR = "#ff7f0e"

#: The band behind component columns, separating them from zone columns.
COMPONENT_BACKGROUND = "#ececec"

#: Grid lines, and the outline of the group labels above the axis.
GRID_COLOUR = "#c7c7c7"
GROUP_LABEL_EDGE = "#4a4a4a"

#: Simulation cost panels.
ENERGYPLUS_FILL, ENERGYPLUS_EDGE = "#9EC3D9", "#58788C"
MEDIAN_COLOUR, PERCENTILE_COLOUR = "#263F4D", "#707B83"

#: Every figure is written on exactly white, for the manuscript.
FIGURE_FACECOLOR = "#ffffff"

# ---------------------------------------------------------------------------
# Layout, derived from the configuration definitions
# ---------------------------------------------------------------------------

#: Weather variants in the order the figures group them.
WEATHER_ORDER = ("no_weather", "graph_weather", "building_weather", "external_weather")

#: Labels for those groups, as printed.
WEATHER_LABEL = {
    "no_weather": "No weather",
    "graph_weather": "Graph weather",
    "building_weather": "Building weather",
    "external_weather": "External weather",
}

#: Column headings of the two large grids. The third column is heterogeneous
#: without weather, which is a different claim from the first column's
#: homogeneous without weather, so it is named in full.
GRID_COLUMN_LABELS = (
    "No weather",
    "Graph weather",
    "Hetero no weather",
    "Building weather",
    "External weather",
)

#: (graph_type, weather_variant) per column of the large grids.
GRID_COLUMN_KEYS = (
    ("homo", "no_weather"),
    ("homo", "graph_weather"),
    ("hetero", "no_weather"),
    ("hetero", "building_weather"),
    ("hetero", "external_weather"),
)

#: Row order of the large grids.
GRID_ROW_LEVELS = ("zone", "component")

#: Axis-label form: singular and capitalised, as the published panels read.
GRID_ROW_TITLE = {"zone": "Zone", "component": "Component"}

#: Gridspec columns the five panels occupy. Column 2 is a spacer that separates
#: the homogeneous pair from the heterogeneous triple.
GRID_COLUMNS = (0, 1, 3, 4, 5)
GRID_WIDTH_RATIOS = (1.0, 1.0, 0.32, 1.0, 1.0, 1.0)

#: Which panels carry the two section headers.
HOMOGENEOUS_COLUMNS = (0, 1)
HETEROGENEOUS_COLUMNS = (2, 4)


def _config_at(level: str, graph_type: str, weather_variant: str) -> ConfigId | None:
    for config in CONFIGS:
        if (config.level, config.graph_type, config.weather_variant) == (
            level,
            graph_type,
            weather_variant,
        ):
            return config
    return None


def config_grid() -> list[list[ConfigId]]:
    """The ten configurations as the published grids arrange them.

    Rows are zone then component; columns follow :data:`GRID_COLUMN_KEYS`. Built
    from :mod:`beg.config.naming`, so the figures follow any change to the
    configuration set.
    """
    grid = []
    for level in GRID_ROW_LEVELS:
        row = []
        for graph_type, weather in GRID_COLUMN_KEYS:
            config = _config_at(level, graph_type, weather)
            if config is None:
                raise LookupError(
                    f"no configuration for {level}/{graph_type}/{weather}; the "
                    "published grid assumes all ten exist"
                )
            row.append(config)
        grid.append(row)
    return grid


def configs_by_weather() -> dict[str, list[ConfigId]]:
    """Configurations grouped by weather variant, zone before component.

    This is the x-axis grouping of the validation figure: each weather
    representation is one group, and within it the matching zone and component
    configurations sit side by side on a shared y-axis.
    """
    grouped: dict[str, list[ConfigId]] = {}
    for weather in WEATHER_ORDER:
        members = [c for c in CONFIGS if c.weather_variant == weather]
        zone = sorted((c for c in members if c.level == "zone"), key=lambda c: int(c.id[1:]))
        component = sorted(
            (c for c in members if c.level == "component"), key=lambda c: int(c.id[1:])
        )
        if zone or component:
            grouped[weather] = zone + component
    return grouped


# ---------------------------------------------------------------------------
# Per-figure geometry and typography
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureStyle:
    """One figure's published size, typography and export settings."""

    figsize: tuple[float, float]
    rc: dict[str, object] = field(default_factory=dict)
    dpi: int = 350
    formats: tuple[str, ...] = ("png", "pdf")

    def context(self):
        """An rc_context for this figure, leaving the process settings alone."""
        import matplotlib.pyplot as plt

        return plt.rc_context({"font.family": "DejaVu Sans", **self.rc})


#: Drawn at final print width, with text already at its printed size.
COST = FigureStyle(
    figsize=(7.35, 5.65),
    rc={
        "font.size": 9.2,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    },
)

#: The two large grids: authored oversized, scaled down by the manuscript.
GRID = FigureStyle(
    figsize=(24.0, 10.5),
    rc={"axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18},
    formats=("png", "svg"),
)

#: The grouped validation figure, one wide axis.
WEATHER = FigureStyle(
    figsize=(16.0, 6.4),
    rc={"axes.labelsize": 17, "ytick.labelsize": 17},
)

#: The two all-run summaries, a pair of panels each.
ALL_RUNS = FigureStyle(
    figsize=(12.4, 4.5),
    rc={"axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 10},
)

#: Font sizes inside the large grids, set per artist because the panel label and
#: the column title differ.
GRID_PANEL_LABEL = 28
GRID_COLUMN_TITLE = 22
GRID_AXIS_LABEL = 22
GRID_SECTION_HEADER = 24
GRID_LEGEND = 22

#: Training curves are drawn on a linear axis capped just above the worst
#: converged run, so the flat tail that matters stays legible.
CURVE_YMAX = 1.2
CURVE_YTICKS = (0.0, 0.4, 0.8, 1.2)

#: Font sizes inside the grouped validation figure.
WEATHER_TICK = 17
WEATHER_LEGEND = 16


def show(fig, *, width_px: int = 1400) -> None:
    """Display ``fig`` in a notebook at screen resolution, then close it.

    Files written by :func:`save` keep their print resolution. The notebook embeds
    one PNG at most ``width_px`` wide, so a stored notebook stays small enough to
    read on GitHub.
    """
    import io

    import matplotlib.pyplot as plt
    from IPython.display import Image, display

    dpi = min(float(fig.dpi), width_px / fig.get_figwidth())
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=FIGURE_FACECOLOR, edgecolor="none")
    plt.close(fig)
    display(Image(data=buffer.getvalue()))


def save(fig, path, *, style: FigureStyle | None = None, dpi=None, formats=None) -> list:
    """Write a figure on exactly white, in each requested format.

    Defaults come from the figure's own style, so a caller that passes nothing
    gets what the manuscript uses: 350 dpi and a vector copy beside the raster.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_dpi = dpi if dpi is not None else (style.dpi if style else 350)
    resolved_formats = formats if formats is not None else (style.formats if style else ("png", "pdf"))

    written = []
    for suffix in resolved_formats:
        target = path.with_suffix(f".{suffix}")
        fig.savefig(target, dpi=resolved_dpi, bbox_inches="tight",
                    facecolor=FIGURE_FACECOLOR, edgecolor="none")
        written.append(target)
    return written
