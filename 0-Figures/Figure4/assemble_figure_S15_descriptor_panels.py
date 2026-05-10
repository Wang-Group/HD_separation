"""Assemble Figure S15 descriptor-space interpretation panels.

Figure S15 uses the current machine-learning interpretation descriptors:
weighted electronegativity, weighted H adsorption free energy, and weighted
work function. The BO samples are overlaid on descriptor maps reconstructed
from the primary-screening dataset.

Run from `0-Figures/Figure4`:

    python assemble_figure_S15_descriptor_panels.py

The script reads the current 561-sample primary-screening feature matrix and
the curated BO table. It does not modify source data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import descriptor_joint_panel_helpers as joint
import plot_top3_shap_descriptor_bo_overlay as top3


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DESCRIPTOR_COLS = [
    "weighted_electronegativity",
    "weighted_FE_ads_avg",
    "weighted_work_function",
]

LABELS = {
    "weighted_electronegativity": "Weighted Electronegativity",
    "weighted_FE_ads_avg": "Weighted H Adsorption Free Energy (eV)",
    "weighted_work_function": "Weighted Work Function (eV)",
}

PAIRS = [
    ("weighted_electronegativity", "weighted_FE_ads_avg"),
    ("weighted_electronegativity", "weighted_work_function"),
    ("weighted_FE_ads_avg", "weighted_work_function"),
]

ROW_CONFIGS = [
    ("A", "contour", "global"),
    ("B", "points_only", "global"),
    ("C", "contour", "local"),
    ("D", "points_only", "local"),
]


def descriptor_limits(initial_df, bo_df, scope):
    source = (
        pd.concat([initial_df[DESCRIPTOR_COLS], bo_df[DESCRIPTOR_COLS]], ignore_index=True)
        if scope == "global"
        else bo_df[DESCRIPTOR_COLS]
    )
    lims = {}
    min_pad = {
        "weighted_electronegativity": 0.04,
        "weighted_FE_ads_avg": 0.035,
        "weighted_work_function": 0.08,
    }
    for col in DESCRIPTOR_COLS:
        lo = float(source[col].min())
        hi = float(source[col].max())
        span = hi - lo
        pad = max(0.075 * span, min_pad[col])
        lims[col] = (lo - pad, hi + pad)
    return lims


def draw_single_legend(fig):
    handles = [
        Patch(facecolor="none", edgecolor="none", label="H/D Separation Coefficient")
    ]
    handles.extend(
        Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(joint.BIN_COLORS, joint.BIN_LABELS)
    )
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=6,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        prop={"family": "Arial", "weight": "bold", "size": 13},
        handlelength=1.65,
        columnspacing=1.15,
        handletextpad=0.50,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(1.25)


def draw_row_label(fig, axes_this_row, letter):
    row_bbox = mtransforms.Bbox.union([ax.get_position() for ax in axes_this_row])
    fig.text(
        0.018,
        row_bbox.y1 + 0.006,
        letter,
        va="bottom",
        ha="left",
        fontsize=28,
        fontweight="bold",
        family="Arial",
        color="black",
    )


def draw_s15(initial_df, bo_df):
    """Draw the complete four-row Figure S15."""
    joint.LABELS.update(LABELS)

    fig = plt.figure(figsize=(16.4, 17.1), dpi=300)
    outer = fig.add_gridspec(
        4,
        3,
        left=0.075,
        right=0.990,
        bottom=0.045,
        top=0.945,
        hspace=0.42,
        wspace=0.36,
    )

    global_lims = descriptor_limits(initial_df, bo_df, "global")
    local_lims = descriptor_limits(initial_df, bo_df, "local")

    for row_idx, (letter, mode, scope) in enumerate(ROW_CONFIGS):
        lims = global_lims if scope == "global" else local_lims
        axes_this_row = []
        for col_idx, (x_col, y_col) in enumerate(PAIRS):
            axes_this_row.extend(
                joint.draw_joint_panel(
                    fig,
                    outer[row_idx, col_idx],
                    initial_df,
                    bo_df,
                    x_col,
                    y_col,
                    lims,
                    mode,
                )
            )
        draw_row_label(fig, axes_this_row, letter)

    draw_single_legend(fig)
    save_figure(fig, "figure_S15_descriptor_space_interpretation_BO_outcomes")


def save_figure(fig, output_stem):
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    joint.configure_matplotlib()
    initial_df, bo_df = top3.build_tables()
    initial_df = initial_df.dropna(subset=["separation_coefficient"])
    bo_df = bo_df.dropna(subset=["separation_coefficient"])

    draw_s15(initial_df, bo_df)


if __name__ == "__main__":
    main()
