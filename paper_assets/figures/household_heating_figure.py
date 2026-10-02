#!/usr/bin/env python3
"""The European household space-heating comparison in the manuscript's Related Work.

Plots the 2023 climate-corrected residential space-heating consumption per floor
area that ODYSSEE-MURE reports for 24 European countries, converted from koe/m2
to kWh/m2. It sets the scale of the energy use this pipeline predicts against
national averages. The source values ship in ``paper_assets/data/`` and are cited
in the manuscript's figure caption.

    python paper_assets/figures/household_heating_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from beg.settings import display_path, output_dir, repo_root

DEFAULT_INPUT = repo_root() / "paper_assets" / "data" / "odyssee_household_heating_consumption_2023.csv"
DEFAULT_OUTPUT = output_dir() / "paper" / "figures" / "european_household_heating_consumption_2023.png"

#: Eurostat defines one tonne of oil equivalent as 41.868 GJ, so one koe is
#: 41.868 MJ, which is 11.63 kWh.
KWH_PER_KOE = 11.63

FIGURE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 8.5,
}


def load_heating(path: Path) -> pd.DataFrame:
    """Countries and their consumption in kWh/(m2 y), lowest first."""
    frame = pd.read_csv(path)
    frame["kwh_per_m2"] = frame["heating_consumption_koe_per_m2"] * KWH_PER_KOE
    return frame.sort_values("kwh_per_m2").reset_index(drop=True)


def plot_heating(frame: pd.DataFrame):
    with plt.rc_context(FIGURE_RC):
        fig, ax = plt.subplots(figsize=(11.7, 4.25), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        bars = ax.bar(frame["country"], frame["kwh_per_m2"], color="#9EC3D9", edgecolor="#58788C",
                      linewidth=0.65, width=0.78)
        ax.set_ylabel("Residential space-heating consumption\n[kWh/(m²·y)]")
        ax.set_ylim(0, max(190.0, float(frame["kwh_per_m2"].max()) * 1.15))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=48, length=0)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        for bar, value in zip(bars, frame["kwh_per_m2"], strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2.2, f"{value:.1f}", ha="center",
                    va="bottom", fontsize=7.0, color="#263F4D", rotation=90)
        fig.subplots_adjust(left=0.075, right=0.995, top=0.97, bottom=0.29)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG; a PDF is written beside it")
    args = parser.parse_args()

    fig = plot_heating(load_heating(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in (args.output, args.output.with_suffix(".pdf")):
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
        print(f"wrote {display_path(path)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
