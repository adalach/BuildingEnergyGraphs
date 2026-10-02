#!/usr/bin/env python3
"""The dataset figures in the manuscript: building size, zone count and envelope.

Footprints are the union of ground-level Room floor faces and thermal-zone counts
the number of Honeybee Rooms in a model, both read from the HBJSON models.
Envelope U-values, the perimeter-zone flag, the building types and the construction
periods come from the dataset's own ``model_summary.csv``.

Both inputs are part of the published BEM4AI dataset: unpack ``models/hbjson.zip``
into ``data/hbjson/`` and copy ``summaries/model_summary.csv`` into
``data/summaries/``. Parsing all 13,260 HBJSON models takes a while; ``--workers``
runs it in parallel and ``--statistics-input`` reuses an earlier pass.

    python paper_assets/figures/dataset_creation_figures.py --workers 8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

# Non-interactive backend, for batch execution
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from beg.settings import data_dir, display_path, output_dir

DEFAULT_MODELS_DIR = data_dir() / "hbjson"
#: Envelope U-values and the perimeter-zone flag, one row per model.
DEFAULT_MODEL_SUMMARY = data_dir() / "summaries" / "model_summary.csv"
DEFAULT_FIGURE = output_dir() / "paper" / "figures" / "footprint_area_and_total_zones_distributions.png"
DEFAULT_U_VALUE_FIGURE = output_dir() / "paper" / "figures" / "u_value_distributions_abcd_density.png"
DEFAULT_STATISTICS_CSV = output_dir() / "paper" / "tables" / "dataset_creation_statistics.csv"
DEFAULT_SUMMARY_JSON = output_dir() / "paper" / "tables" / "dataset_creation_summary.json"
#: Font families in order of preference; matplotlib uses the first one installed.
FONT_FAMILIES = ["TUM Neue Helvetica", "Helvetica Neue", "Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"]
FONT = {"font.family": "sans-serif", "font.sans-serif": FONT_FAMILIES}
#: Building-type codes in ``model_summary.csv`` and their names in Building Stock EU28.
BUILDING_TYPES = {
    "multifamily": "Multifamily houses",
    "apartment_block": "Apartment blocks",
    "sf_terraced": "Single family–Terraced houses",
    "trade": "Trade",
    "offices": "Offices",
    "education": "Education",
    "hotels_restaurants": "Hotels and Restaurants",
    "health": "Health",
}
SECTORS = {"res": "Residential", "nonres": "Service"}
BEFORE_1970 = ("Before 1945", "1945 - 1969")


def _program_type_identifier(room: dict) -> str:
    program_type = room.get("properties", {}).get("energy", {}).get("program_type")
    if isinstance(program_type, str):
        return program_type
    if isinstance(program_type, dict):
        return str(program_type.get("identifier", ""))
    return ""


def _floor_polygon(face: dict) -> tuple[float, Polygon] | None:
    boundary = face.get("geometry", {}).get("boundary", [])
    if len(boundary) < 3:
        return None
    heights = [float(point[2]) for point in boundary]
    # Some regenerated HBJSONs do not persist Honeybee's Floor face type.
    # Their floor and ceiling faces remain detectable from horizontal geometry.
    if max(heights) - min(heights) > 1e-6:
        return None
    polygon = Polygon([(float(point[0]), float(point[1])) for point in boundary])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        return None
    height = float(np.median(heights))
    return height, polygon


def _model_statistics(path: Path) -> dict:
    model = json.loads(path.read_text())
    rooms = model.get("rooms", [])
    floor_polygons = [
        floor
        for room in rooms
        for face in room.get("faces", [])
        if (floor := _floor_polygon(face)) is not None
    ]
    if not floor_polygons:
        raise ValueError(f"No valid floor face found in {path}")

    ground_z = min(height for height, _ in floor_polygons)
    ground_polygons = [
        polygon
        for height, polygon in floor_polygons
        if np.isclose(height, ground_z, rtol=0.0, atol=1e-6)
    ]
    ground_zone_count = sum(
        any(
            floor is not None and np.isclose(floor[0], ground_z, rtol=0.0, atol=1e-6)
            for floor in (_floor_polygon(face) for face in room.get("faces", []))
        )
        for room in rooms
    )
    program_types = Counter(_program_type_identifier(room) for room in rooms)
    program_type = program_types.most_common(1)[0][0] if program_types else ""

    return {
        "model_id": path.stem,
        "footprint_area_m2": float(unary_union(ground_polygons).area),
        "total_zones": len(rooms),
        "ground_level_zone_count": ground_zone_count,
        "has_multiple_ground_level_zones": ground_zone_count > 1,
        "program_type": program_type,
    }


def _usage_class(program_type: str) -> str:
    text = program_type.lower()
    if any(token in text for token in ("apartment", "multifamily", "singlefamily")):
        return "Residential"
    return "Service"


def _create_figure(statistics: pd.DataFrame, output_path: Path) -> None:
    # The paper figure's two-panel layout, colour, grid and density
    # normalisation, on a linear footprint-area axis.
    plt.style.use(["seaborn-v0_8-whitegrid", FONT])
    figure, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=200)
    fill_colour = "#9ec1d8"
    edge_colour = "#74818b"

    axes[0].hist(
        statistics["footprint_area_m2"],
        bins=35,
        density=True,
        color=fill_colour,
        edgecolor=edge_colour,
        alpha=0.7,
    )
    axes[0].set_title("Footprint area distribution", fontsize=19)
    axes[0].set_xlabel("Footprint area [m²]", fontsize=16)
    axes[0].set_ylabel("Density", fontsize=16)

    axes[1].hist(
        statistics["total_zones"],
        bins=35,
        density=True,
        color=fill_colour,
        edgecolor=edge_colour,
        alpha=0.7,
    )
    axes[1].set_title("Total zone count distribution", fontsize=19)
    axes[1].set_xlabel("Total number of zones (zones/floor × floors)", fontsize=16)
    axes[1].set_ylabel("Density", fontsize=16)

    for axis in axes:
        axis.tick_params(labelsize=14)
        axis.grid(True, alpha=0.55)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close(figure)


def _load_u_values(path: Path, model_ids: pd.Series) -> pd.DataFrame:
    columns = [
        "model_id",
        "wall_u_value",
        "roof_u_value",
        "floor_u_value",
        "window_u_value",
    ]
    values = pd.read_csv(path, usecols=columns, low_memory=False)
    if values["model_id"].duplicated().any():
        raise ValueError(f"Duplicate model IDs in {path}")
    values = values[values["model_id"].isin(set(model_ids))].copy()
    if len(values) != len(model_ids):
        raise ValueError(
            f"U-value table contains {len(values):,} of {len(model_ids):,} model IDs"
        )
    return values


def _create_u_value_figure(u_values: pd.DataFrame, output_path: Path) -> None:
    columns = [
        ("wall_u_value", "(a)", "Wall U-value"),
        ("roof_u_value", "(b)", "Roof U-value"),
        ("floor_u_value", "(c)", "Floor U-value"),
        ("window_u_value", "(d)", "Window U-value"),
    ]
    style = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.25",
        "axes.labelcolor": "0.25",
        "xtick.color": "0.25",
        "ytick.color": "0.25",
        "grid.color": "0.85",
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "font.size": 12,
        **FONT,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(12, 6.8),
            gridspec_kw={"hspace": 0.55, "wspace": 0.25},
        )
        for axis, (column, panel, title) in zip(axes.flatten(), columns):
            series = pd.to_numeric(u_values[column], errors="coerce").dropna()
            series = series[(series > 0) & (series < 20)]
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(length=0, labelsize=11)
            axis.grid(True, alpha=0.6)
            axis.hist(
                series,
                bins=60,
                density=True,
                alpha=0.25,
                edgecolor="0.2",
                linewidth=0.8,
            )
            axis.text(
                0.02,
                1.08,
                f"{panel} {title}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=14,
                color="black",
                clip_on=False,
            )
            axis.set_xlabel("U-value [W/m²K]", fontsize=12)
            axis.set_ylabel("Density", fontsize=12)
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
        plt.close(figure)


def _summary(statistics: pd.DataFrame) -> dict:
    grouped = statistics.assign(usage_class=statistics["program_type"].map(_usage_class))
    usage = {}
    for name, group in grouped.groupby("usage_class", sort=True):
        area = group["footprint_area_m2"]
        usage[name] = {
            "count": int(len(group)),
            "share_percent": round(100 * len(group) / len(grouped), 1),
            "mean_footprint_area_m2": round(float(area.mean()), 1),
            "median_footprint_area_m2": round(float(area.median()), 1),
            "sd_footprint_area_m2": round(float(area.std(ddof=1)), 1),
        }
    return {
        "model_count": int(len(statistics)),
        "footprint_area_m2": {
            "mean": round(float(statistics["footprint_area_m2"].mean()), 1),
            "median": round(float(statistics["footprint_area_m2"].median()), 1),
            "sd": round(float(statistics["footprint_area_m2"].std(ddof=1)), 1),
        },
        "total_zones": {
            "mean": round(float(statistics["total_zones"].mean()), 1),
            "median": round(float(statistics["total_zones"].median()), 1),
            "sd": round(float(statistics["total_zones"].std(ddof=1)), 1),
            "min": int(statistics["total_zones"].min()),
            "max": int(statistics["total_zones"].max()),
        },
        "multiple_ground_level_zones": {
            "count": int(statistics["has_multiple_ground_level_zones"].sum()),
            "share_percent": round(
                100 * float(statistics["has_multiple_ground_level_zones"].mean()), 1
            ),
        },
        "perimeter_zone_decomposition": {
            "count": int(statistics["has_offset_perimeter_zones"].sum()),
            "share_percent": round(
                100 * float(statistics["has_offset_perimeter_zones"].mean()), 1
            ),
            "source": "model_summary.csv, column perimeter",
        },
        "usage_classes": usage,
        "program_type_counts": {
            name: int(count)
            for name, count in statistics["program_type"].value_counts().items()
        },
    }


def _stock_summary(path: Path, model_ids: pd.Series) -> dict:
    """Sectors, building types and construction periods of the models."""
    models = pd.read_csv(path, usecols=["model_id", "sector", "built_c", "built_age"])
    models = models[models["model_id"].isin(set(model_ids))]
    if len(models) != len(model_ids):
        raise ValueError(f"{path} contains {len(models):,} of {len(model_ids):,} model IDs")
    unknown = set(models["sector"]) - set(SECTORS)
    if unknown:
        raise ValueError(f"Unknown sector codes in {path}: {sorted(unknown)}")
    sectors = {}
    for code, name in SECTORS.items():
        group = models[models["sector"] == code]
        types = group["built_c"].value_counts()
        sectors[name] = {
            "count": int(len(group)),
            "share_percent": round(100 * len(group) / len(models), 1),
            "building_types": {
                BUILDING_TYPES.get(building_type, building_type): {
                    "count": int(count),
                    "share_of_sector_percent": round(100 * count / len(group), 1),
                }
                for building_type, count in types.items()
            },
        }
    before_1970 = int(models["built_age"].isin(BEFORE_1970).sum())
    return {
        "sectors": sectors,
        "construction_periods": {
            "before_1970": {
                "count": before_1970,
                "share_percent": round(100 * before_1970 / len(models), 1),
            },
            "counts": {name: int(count) for name, count in models["built_age"].value_counts().items()},
        },
    }


def _resolved_font() -> str:
    from matplotlib import font_manager

    path = font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILIES))
    return font_manager.get_font(path).family_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--perimeter-csv",
        type=Path,
        default=DEFAULT_MODEL_SUMMARY,
        help="The dataset's model_summary.csv: perimeter-zone flag, sector, building type and period.",
    )
    parser.add_argument(
        "--statistics-input",
        type=Path,
        help="Reuse a previously generated per-model statistics CSV instead of parsing HBJSONs.",
    )
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--u-values-csv", type=Path, default=DEFAULT_MODEL_SUMMARY)
    parser.add_argument("--u-value-figure", type=Path, default=DEFAULT_U_VALUE_FIGURE)
    parser.add_argument("--statistics-csv", type=Path, default=DEFAULT_STATISTICS_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", type=int, default=0, help="Zero-based model offset.")
    parser.add_argument("--stop", type=int, help="Exclusive model offset.")
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Write statistics only; useful when processing the source set in batches.",
    )
    parser.add_argument(
        "--skip-u-value-figure",
        action="store_true",
        help="Do not write the four-panel envelope U-value figure.",
    )
    args = parser.parse_args()

    if args.statistics_input:
        statistics = pd.read_csv(args.statistics_input)
        statistics = statistics.rename(
            columns={
                "has_perimeter_zone_processing": "has_multiple_ground_level_zones"
            }
        )
    else:
        paths = sorted(args.models_dir.glob("*.hbjson"))
        if not paths:
            raise FileNotFoundError(f"No HBJSON models found under {args.models_dir}")
        paths = paths[args.start : args.stop]
        if args.workers == 1:
            rows = []
            for index, path in enumerate(paths, start=1):
                rows.append(_model_statistics(path))
                if index % 500 == 0 or index == len(paths):
                    print(f"Parsed {index:,}/{len(paths):,} HBJSON models", flush=True)
        else:
            # Submit bounded batches to keep memory use flat; the complete HBJSON
            # set is about 5 GB.
            rows = []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for start in range(0, len(paths), 64):
                    rows.extend(executor.map(_model_statistics, paths[start : start + 64]))
                    if len(rows) % 500 < 64 or len(rows) == len(paths):
                        print(f"Parsed {len(rows):,}/{len(paths):,} HBJSON models", flush=True)
        statistics = pd.DataFrame(rows)
    statistics = statistics.sort_values("model_id")
    statistics = statistics.drop(columns=["has_offset_perimeter_zones"], errors="ignore")

    perimeter = pd.read_csv(args.perimeter_csv, usecols=["model_id", "perimeter"])
    if perimeter["model_id"].duplicated().any():
        raise ValueError(f"Duplicate model IDs in {args.perimeter_csv}")
    perimeter["has_offset_perimeter_zones"] = perimeter["perimeter"].eq("yes")
    statistics = statistics.merge(
        perimeter[["model_id", "has_offset_perimeter_zones"]],
        on="model_id",
        how="left",
        validate="one_to_one",
    )
    if statistics["has_offset_perimeter_zones"].isna().any():
        missing = int(statistics["has_offset_perimeter_zones"].isna().sum())
        raise ValueError(
            f"{missing:,} HBJSON model IDs are absent from {args.perimeter_csv}"
        )
    statistics["has_offset_perimeter_zones"] = statistics[
        "has_offset_perimeter_zones"
    ].astype(bool)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.u_value_figure.parent.mkdir(parents=True, exist_ok=True)
    args.statistics_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(args.statistics_csv, index=False)
    summary = _summary(statistics)
    summary.update(_stock_summary(args.perimeter_csv, statistics["model_id"]))
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if not args.skip_figure:
        _create_figure(statistics, args.figure)
    if not args.skip_u_value_figure:
        _create_u_value_figure(
            _load_u_values(args.u_values_csv, statistics["model_id"]),
            args.u_value_figure,
        )

    print(f"Models: {len(statistics):,}")
    print(f"Font: {_resolved_font()}")
    if not args.skip_figure:
        print(f"Figure: {display_path(args.figure)}")
    if not args.skip_u_value_figure:
        print(f"U-value figure: {display_path(args.u_value_figure)}")
    print(f"Statistics: {display_path(args.statistics_csv)}")
    print(f"Summary: {display_path(args.summary_json)}")


if __name__ == "__main__":
    main()
