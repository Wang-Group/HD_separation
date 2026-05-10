"""Draw the three-panel descriptor map used as the bottom of Figure 4.

This script intentionally reuses the current Figure 4 descriptor helpers and
current machine-learning projection table. It draws only the global contour
view: primary-screening samples define the descriptor landscape, and BO samples
are overlaid as filled circles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import descriptor_joint_panel_helpers as joint


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_TABLE = ROOT / "0-Figures" / "Figure3" / "data" / "figure3_top_from_ml_dataset.csv"
BO_TABLE = FIGURE_DIR / "curated" / "figure4_bo_complete_112.csv"
EH_EXTRA = ROOT / "hd_separation_ml" / "EH_extra.xlsx"
S15_INITIAL_PROJECTION = OUTPUT_DIR / "top3_shap_initial_descriptor_projection_table.csv"
S15_BO_PROJECTION = OUTPUT_DIR / "top3_shap_bo_descriptor_projection_table.csv"

DESCRIPTOR_COLS = [
    "weighted_electronegativity",
    "weighted_FE_ads_avg",
    "weighted_work_function",
]

LABELS = {
    "weighted_electronegativity": "Weighted Electronegativity",
    "weighted_FE_ads_avg": "Weighted H Adsorption Energy (eV)",
    "weighted_work_function": "Weighted Work Function (eV)",
}

PAIRS = [
    ("weighted_electronegativity", "weighted_FE_ads_avg"),
    ("weighted_electronegativity", "weighted_work_function"),
    ("weighted_FE_ads_avg", "weighted_work_function"),
]

AXIS_LABEL_SIZE = 14.4
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 15.6


def descriptor_limits(initial_df, bo_df):
    lims = {}
    min_pad = {
        "weighted_electronegativity": 0.04,
        "weighted_FE_ads_avg": 0.035,
        "weighted_work_function": 0.08,
    }
    for col in DESCRIPTOR_COLS:
        values = list(initial_df[col].dropna()) + list(bo_df[col].dropna())
        lo = min(values)
        hi = max(values)
        span = hi - lo
        lims[col] = (lo - max(0.075 * span, min_pad[col]), hi + max(0.075 * span, min_pad[col]))
    return lims


def load_element_descriptors():
    raw = pd.read_excel(EH_EXTRA)
    element_col = raw.columns[0]
    desc = raw.rename(
        columns={
            element_col: "element",
            "delta_EH": "H_adsorption",
            "Work function (eV)": "work_function",
        }
    )
    desc = desc[
        [
            "element",
            "electronegativity",
            "H_adsorption",
            "work_function",
        ]
    ].drop_duplicates("element")
    return desc.set_index("element")


def weighted_descriptor(df, element1_col, element2_col, frac1, frac2, mapping, descriptor):
    v1 = mapping.reindex(df[element1_col])[descriptor].to_numpy(float)
    v2 = mapping.reindex(df[element2_col])[descriptor].to_numpy(float)
    return frac1.to_numpy(float) * v1 + frac2.to_numpy(float) * v2


def build_tables():
    # Prefer the projection tables used by Figure S15 so the manuscript Figure 4C
    # bottom panel and SI descriptor-space panels are visually identical.
    if S15_INITIAL_PROJECTION.exists() and S15_BO_PROJECTION.exists():
        initial = pd.read_csv(S15_INITIAL_PROJECTION)
        bo = pd.read_csv(S15_BO_PROJECTION)
        return initial, bo

    mapping = load_element_descriptors()

    initial = pd.read_csv(INITIAL_TABLE)[
        ["sample", "element1", "element2", "separation_coefficient"]
    ].copy()
    init_frac1 = pd.Series(
        np.where(initial["element1"].eq(initial["element2"]), 1.0, 6.0 / 7.0),
        index=initial.index,
    )
    init_frac2 = pd.Series(
        np.where(initial["element1"].eq(initial["element2"]), 0.0, 1.0 / 7.0),
        index=initial.index,
    )
    initial["weighted_electronegativity"] = weighted_descriptor(
        initial, "element1", "element2", init_frac1, init_frac2, mapping, "electronegativity"
    )
    initial["weighted_FE_ads_avg"] = weighted_descriptor(
        initial, "element1", "element2", init_frac1, init_frac2, mapping, "H_adsorption"
    )
    initial["weighted_work_function"] = weighted_descriptor(
        initial, "element1", "element2", init_frac1, init_frac2, mapping, "work_function"
    )

    bo = pd.read_csv(BO_TABLE)
    if "sample" not in bo.columns:
        bo["sample"] = bo["system"].astype(str) + bo["row_id"].astype(int).astype(str)
    bo = bo[
        [
            "sample",
            "system",
            "substance1",
            "substance2",
            "volume1",
            "volume2",
            "separation_coefficient",
        ]
    ].copy()
    total = bo["volume1"].astype(float) + bo["volume2"].astype(float)
    frac1 = bo["volume1"].astype(float) / total
    frac2 = bo["volume2"].astype(float) / total
    bo["weighted_electronegativity"] = weighted_descriptor(
        bo, "substance1", "substance2", frac1, frac2, mapping, "electronegativity"
    )
    bo["weighted_FE_ads_avg"] = weighted_descriptor(
        bo, "substance1", "substance2", frac1, frac2, mapping, "H_adsorption"
    )
    bo["weighted_work_function"] = weighted_descriptor(
        bo, "substance1", "substance2", frac1, frac2, mapping, "work_function"
    )

    initial.to_csv(OUTPUT_DIR / "figure4C_initial_weighted_descriptor_table.csv", index=False)
    bo.to_csv(OUTPUT_DIR / "figure4C_bo_weighted_descriptor_table.csv", index=False)
    return initial, bo


def add_legend(fig):
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
        bbox_to_anchor=(0.5, 0.992),
        ncol=6,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        prop={"family": "Arial", "weight": "bold", "size": LEGEND_FONT_SIZE},
        handlelength=1.65,
        columnspacing=1.15,
        handletextpad=0.50,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(1.25)


def draw_figure(initial_df, bo_df, with_legend):
    joint.LABELS.update(LABELS)
    lims = descriptor_limits(initial_df, bo_df)

    fig = plt.figure(figsize=(16.4, 4.25), dpi=300)
    outer = fig.add_gridspec(
        1,
        3,
        left=0.075,
        right=0.990,
        bottom=0.170,
        top=0.860 if with_legend else 0.965,
        wspace=0.288,
    )

    for col_idx, (x_col, y_col) in enumerate(PAIRS):
        _, ax_scatter, _ = joint.draw_joint_panel(
            fig,
            outer[0, col_idx],
            initial_df,
            bo_df,
            x_col,
            y_col,
            lims,
            mode="contour",
        )
        ax_scatter.xaxis.label.set_size(AXIS_LABEL_SIZE)
        ax_scatter.yaxis.label.set_size(AXIS_LABEL_SIZE)
        for tick in ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels():
            tick.set_fontsize(TICK_LABEL_SIZE)

    if with_legend:
        add_legend(fig)

    suffix = "with_legend" if with_legend else "no_legend"
    png_path = OUTPUT_DIR / f"figure4C_bottom_descriptor_maps_{suffix}.png"
    pdf_path = OUTPUT_DIR / f"figure4C_bottom_descriptor_maps_{suffix}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    joint.configure_matplotlib()
    initial_df, bo_df = build_tables()
    initial_df = initial_df.dropna(subset=DESCRIPTOR_COLS + ["separation_coefficient"])
    bo_df = bo_df.dropna(subset=DESCRIPTOR_COLS + ["separation_coefficient"])
    draw_figure(initial_df, bo_df, with_legend=False)
    draw_figure(initial_df, bo_df, with_legend=True)


if __name__ == "__main__":
    main()
