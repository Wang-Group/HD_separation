"""Assemble Figure S14 descriptor-analysis panels.

This script reuses the Figure 4C descriptor plotting helpers in
`plot_figure4C_descriptor_joint_marginals.py` and generates the four-row
Figure S14 descriptor-analysis summary.

Run from `0-Figures/Figure4`:

    python assemble_figure_S14_descriptor_panels.py

The script reads the curated BO descriptor table and the current 561-sample
primary-screening feature matrix. It does not modify source data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

import plot_figure4C_descriptor_joint_marginals as base


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ROW_CONFIGS = [
    (
        "A",
        "contour",
        "global",
    ),
    (
        "B",
        "points_only",
        "global",
    ),
    (
        "C",
        "contour",
        "local",
    ),
    (
        "D",
        "points_only",
        "local",
    ),
]

def draw_s14_legend(fig, y_anchor):
    """Draw one row-level separation-coefficient legend."""
    handles = [
        Patch(facecolor="none", edgecolor="none", label="H/D Separation Coefficient")
    ]
    handles.extend(
        Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(base.BIN_COLORS, base.BIN_LABELS)
    )
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=6,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        prop={"family": "Arial", "weight": "bold", "size": 12},
        handlelength=1.55,
        columnspacing=1.05,
        handletextpad=0.45,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(1.2)


def row_lims(row_kind, global_lims):
    if row_kind == "global":
        return global_lims
    if row_kind == "local":
        return base.LOCAL_LIMS
    raise ValueError(f"Unknown row kind: {row_kind}")


def draw_four_row_s14(initial_df, bo_df, global_lims):
    """Draw the complete four-row S14 figure."""
    fig = plt.figure(figsize=(15.4, 20.2), dpi=300)
    outer = fig.add_gridspec(
        4,
        3,
        left=0.085,
        right=0.985,
        bottom=0.040,
        top=0.955,
        hspace=0.92,
        wspace=0.43,
    )

    row_axes = []
    for row_idx, (letter, mode, lim_kind) in enumerate(ROW_CONFIGS):
        lims = row_lims(lim_kind, global_lims)
        axes_this_row = []
        for col_idx, (x_col, y_col) in enumerate(base.PAIRS):
            axes_this_row.extend(
                base.draw_joint_panel(
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
        row_axes.append(axes_this_row)

        row_bbox = mtransforms.Bbox.union([ax.get_position() for ax in axes_this_row])
        y_pos = row_bbox.y1 + 0.012
        fig.text(
            0.012,
            y_pos,
            f"{letter}",
            va="center",
            ha="left",
            fontsize=32,
            fontweight="bold",
            family="Arial",
        )

    for axes_this_row in row_axes:
        row_bbox = mtransforms.Bbox.union([ax.get_position() for ax in axes_this_row])
        draw_s14_legend(fig, row_bbox.y1 + 0.019)

    save_figure(fig, "figure_S14_four_panel_BO_descriptor_analysis_rows")


def save_figure(fig, output_stem):
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    base.configure_matplotlib()
    initial_df = base.build_initial_descriptor_table()
    bo_df = base.load_bo_descriptor_table()
    global_lims = base.global_limits(initial_df, bo_df)

    draw_four_row_s14(initial_df, bo_df, global_lims)


if __name__ == "__main__":
    main()
