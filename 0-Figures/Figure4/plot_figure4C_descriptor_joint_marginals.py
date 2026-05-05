from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = FIGURE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INITIAL_FEATURE_MATRIX = (
    ROOT
    / "hd_separation_ml"
    / "outputs"
    / "feature_selection_561_rs_from_eis_no_eis_fitting"
    / "feature_matrix_561_no_eis_fitting.csv"
)
BO_DESCRIPTOR_TABLE = (
    FIGURE_DIR / "curated" / "figure4_bo_weighted_ml_top3_descriptors.csv"
)

BIN_EDGES = np.array([2, 4, 6, 8, 10, 12], dtype=float)
BIN_LABELS = [
    f"{int(BIN_EDGES[i])}-{int(BIN_EDGES[i + 1])}"
    for i in range(len(BIN_EDGES) - 1)
]
BIN_COLORS = ["#AE5CFA", "#AAC0FC", "#F1E2AD", "#F19FA0", "#F35656"]

LABELS = {
    "weighted_electronegativity": "Weighted Electronegativity",
    "weighted_FE_ads_avg": "Weighted H Adsorption Energy (eV)",
    "weighted_delta_EOH": "Weighted OH Adsorption Energy (eV)",
}

PAIRS = [
    ("weighted_electronegativity", "weighted_FE_ads_avg"),
    ("weighted_electronegativity", "weighted_delta_EOH"),
    ("weighted_delta_EOH", "weighted_FE_ads_avg"),
]

LOCAL_LIMS = {
    "weighted_electronegativity": (1.58, 2.22),
    "weighted_FE_ads_avg": (-0.70, -0.30),
    "weighted_delta_EOH": (-2.58, -0.88),
}

# Same BO-centered EN/OH windows as LOCAL_LIMS, but with a broader H adsorption
# axis to show where the BO samples sit within a wider H-binding landscape.
LOCAL_H_WIDE_LIMS = {
    "weighted_electronegativity": (1.58, 2.22),
    "weighted_FE_ads_avg": (-0.90, 0.15),
    "weighted_delta_EOH": (-2.58, -0.88),
}

INIT_HIST = "#d9dee5"
INIT_POINT_SIZE = 20
BO_POINT_SIZE = 42
MASK_THRESHOLD = 0.055
KNN_NEIGHBORS = 36
CONTOUR_ALPHA = 0.30


def configure_matplotlib():
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.linewidth"] = 2.2
    mpl.rcParams["xtick.major.width"] = 1.0
    mpl.rcParams["ytick.major.width"] = 1.0
    mpl.rcParams["xtick.major.size"] = 5
    mpl.rcParams["ytick.major.size"] = 5


def build_initial_descriptor_table():
    df = pd.read_csv(INITIAL_FEATURE_MATRIX)
    required = [
        "sample",
        "element1",
        "element2",
        "separation_coefficient",
        "FE_ads_avg_element1",
        "FE_ads_avg_element2",
        "delta_EOH_element1",
        "delta_EOH_element2",
        "electronegativity_element1",
        "electronegativity_element2",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required initial columns: {missing}")

    frac1 = np.where(df["element1"].eq(df["element2"]), 1.0, 6.0 / 7.0)
    frac2 = np.where(df["element1"].eq(df["element2"]), 0.0, 1.0 / 7.0)

    out = df[["sample", "element1", "element2", "separation_coefficient"]].copy()
    out["weighted_electronegativity"] = (
        frac1 * df["electronegativity_element1"]
        + frac2 * df["electronegativity_element2"]
    )
    out["weighted_FE_ads_avg"] = (
        frac1 * df["FE_ads_avg_element1"] + frac2 * df["FE_ads_avg_element2"]
    )
    out["weighted_delta_EOH"] = (
        frac1 * df["delta_EOH_element1"] + frac2 * df["delta_EOH_element2"]
    )
    return out.dropna(
        subset=[
            "separation_coefficient",
            "weighted_electronegativity",
            "weighted_FE_ads_avg",
            "weighted_delta_EOH",
        ]
    )


def load_bo_descriptor_table():
    df = pd.read_csv(BO_DESCRIPTOR_TABLE)
    required = [
        "sample",
        "system",
        "separation_coefficient",
        "weighted_electronegativity",
        "weighted_FE_ads_avg",
        "weighted_delta_EOH",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required BO columns: {missing}")
    return df.dropna(subset=required)


def separation_bin_index(values):
    idx = np.digitize(values, BIN_EDGES, right=False) - 1
    return np.clip(idx, 0, len(BIN_COLORS) - 1)


def colors_for_separation(values):
    return np.asarray(BIN_COLORS, dtype=object)[separation_bin_index(values)]


def style_tick_labels(ax, size=10):
    ax.tick_params(axis="both", which="both", labelsize=size, colors="black")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_color("black")


def global_limits(initial_df, bo_df):
    lims = {}
    merged = pd.concat(
        [
            initial_df[["weighted_electronegativity", "weighted_FE_ads_avg", "weighted_delta_EOH"]],
            bo_df[["weighted_electronegativity", "weighted_FE_ads_avg", "weighted_delta_EOH"]],
        ],
        ignore_index=True,
    )
    for col in merged.columns:
        lo = float(merged[col].min())
        hi = float(merged[col].max())
        pad = 0.04 * (hi - lo)
        lims[col] = (lo - pad, hi + pad)
    return lims


def draw_masked_knn_contour(ax, initial_df, x_col, y_col, xlim, ylim):
    train = initial_df.dropna(subset=[x_col, y_col, "separation_coefficient"])
    if len(train) < 3:
        return

    grid_x = np.linspace(xlim[0], xlim[1], 220)
    grid_y = np.linspace(ylim[0], ylim[1], 220)
    xx, yy = np.meshgrid(grid_x, grid_y)

    points = train[[x_col, y_col]].to_numpy(float)
    values = train["separation_coefficient"].to_numpy(float)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    knn = KNeighborsRegressor(
        n_neighbors=min(KNN_NEIGHBORS, len(train)),
        weights="distance",
    )
    knn.fit(points, values)
    zz = knn.predict(grid_points).reshape(xx.shape)

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    points_norm = np.c_[
        (points[:, 0] - xlim[0]) / x_span,
        (points[:, 1] - ylim[0]) / y_span,
    ]
    grid_norm = np.c_[
        (grid_points[:, 0] - xlim[0]) / x_span,
        (grid_points[:, 1] - ylim[0]) / y_span,
    ]
    nearest = NearestNeighbors(n_neighbors=1).fit(points_norm)
    dist = nearest.kneighbors(grid_norm, return_distance=True)[0].reshape(xx.shape)
    zz = np.ma.masked_where(dist > MASK_THRESHOLD, zz)

    cmap = ListedColormap(BIN_COLORS)
    norm = BoundaryNorm(BIN_EDGES, cmap.N, clip=True)
    ax.contourf(
        xx,
        yy,
        zz,
        levels=BIN_EDGES,
        cmap=cmap,
        norm=norm,
        alpha=CONTOUR_ALPHA,
        zorder=0,
    )
    ax.contour(
        xx,
        yy,
        zz,
        levels=BIN_EDGES,
        colors="#777777",
        linewidths=0.35,
        alpha=0.30,
        zorder=1,
    )


def setup_joint_axes(fig, subspec):
    gs = subspec.subgridspec(
        2,
        2,
        height_ratios=[0.70, 4.00],
        width_ratios=[4.00, 0.80],
        hspace=0.03,
        wspace=0.03,
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[1, 0], sharex=ax_histx)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")
    return ax_histx, ax_scatter, ax_histy


def filter_window(df, x_col, y_col, xlim, ylim):
    return df[df[x_col].between(*xlim) & df[y_col].between(*ylim)].copy()


def draw_marginals(ax_histx, ax_histy, initial_local, bo_local, x_col, y_col, xlim, ylim):
    bins_x = np.linspace(xlim[0], xlim[1], 28)
    bins_y = np.linspace(ylim[0], ylim[1], 24)

    ax_histx.hist(
        initial_local[x_col],
        bins=bins_x,
        color=INIT_HIST,
        edgecolor="none",
        alpha=0.55,
        zorder=0,
    )
    ax_histy.hist(
        initial_local[y_col],
        bins=bins_y,
        orientation="horizontal",
        color=INIT_HIST,
        edgecolor="none",
        alpha=0.55,
        zorder=0,
    )

    bin_idx = separation_bin_index(bo_local["separation_coefficient"].to_numpy(float))
    bo_x_groups = [
        bo_local.loc[bin_idx == idx, x_col].to_numpy(float)
        for idx in range(len(BIN_COLORS))
    ]
    bo_y_groups = [
        bo_local.loc[bin_idx == idx, y_col].to_numpy(float)
        for idx in range(len(BIN_COLORS))
    ]
    ax_histx.hist(
        bo_x_groups,
        bins=bins_x,
        stacked=True,
        color=BIN_COLORS,
        edgecolor="black",
        linewidth=0.55,
        alpha=0.84,
        zorder=3,
    )
    ax_histy.hist(
        bo_y_groups,
        bins=bins_y,
        orientation="horizontal",
        stacked=True,
        color=BIN_COLORS,
        edgecolor="black",
        linewidth=0.55,
        alpha=0.84,
        zorder=3,
    )

    ax_histx.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_histx.tick_params(axis="y", left=False, labelleft=False)
    ax_histy.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_histy.tick_params(axis="y", left=False, labelleft=False)

    for side in ["left", "right", "top"]:
        ax_histx.spines[side].set_visible(False)
    ax_histx.spines["bottom"].set_visible(True)
    ax_histx.spines["bottom"].set_linewidth(0.9)

    for side in ["right", "top", "bottom"]:
        ax_histy.spines[side].set_visible(False)
    ax_histy.spines["left"].set_visible(True)
    ax_histy.spines["left"].set_linewidth(0.9)


def draw_joint_panel(fig, subspec, initial_df, bo_df, x_col, y_col, lims, mode):
    ax_histx, ax_scatter, ax_histy = setup_joint_axes(fig, subspec)

    xlim = lims[x_col]
    ylim = lims[y_col]
    initial_local = filter_window(initial_df, x_col, y_col, xlim, ylim)
    bo_local = filter_window(bo_df, x_col, y_col, xlim, ylim)

    if mode == "contour":
        draw_masked_knn_contour(ax_scatter, initial_df, x_col, y_col, xlim, ylim)
        ax_scatter.scatter(
            initial_local[x_col],
            initial_local[y_col],
            s=INIT_POINT_SIZE,
            marker="o",
            facecolors="none",
            edgecolors="#b8c2cb",
            linewidths=0.65,
            alpha=0.45,
            zorder=2,
        )
    elif mode == "points_only":
        initial_edge = colors_for_separation(
            initial_local["separation_coefficient"].to_numpy(float)
        )
        ax_scatter.scatter(
            initial_local[x_col],
            initial_local[y_col],
            s=24,
            marker="o",
            facecolors="none",
            edgecolors=initial_edge,
            linewidths=0.75,
            alpha=0.58,
            zorder=2,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    bo_sep = bo_local["separation_coefficient"].to_numpy(float)
    bo_bins = separation_bin_index(bo_sep)
    for idx, color in enumerate(BIN_COLORS):
        keep = bo_bins == idx
        ax_scatter.scatter(
            bo_local.loc[keep, x_col],
            bo_local.loc[keep, y_col],
            s=BO_POINT_SIZE,
            marker="o",
            facecolors=color,
            edgecolors="black",
            linewidths=0.75,
            alpha=0.82,
            zorder=4,
        )

    ax_scatter.set_xlim(*xlim)
    ax_scatter.set_ylim(*ylim)
    ax_scatter.set_xlabel(LABELS[x_col], fontsize=12, fontweight="bold")
    ax_scatter.set_ylabel(LABELS[y_col], fontsize=12, fontweight="bold")
    ax_scatter.grid(True, color="#c9c9c9", linewidth=0.35, alpha=0.28)
    style_tick_labels(ax_scatter)

    draw_marginals(ax_histx, ax_histy, initial_local, bo_local, x_col, y_col, xlim, ylim)

    for side in ["left", "right", "top", "bottom"]:
        ax_scatter.spines[side].set_linewidth(2.0)


def draw_legend(fig):
    handles = [Patch(facecolor="none", edgecolor="none", label="Separation Coefficient")]
    handles.extend(
        Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(BIN_COLORS, BIN_LABELS)
    )
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        prop={"family": "Arial", "weight": "bold", "size": 12},
        handlelength=1.7,
        columnspacing=1.2,
        handletextpad=0.55,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(1.2)


def draw_figure(initial_df, bo_df, lims, mode, output_stem):
    fig = plt.figure(figsize=(15.0, 4.45), dpi=300)
    outer = fig.add_gridspec(
        1, 3, left=0.055, right=0.99, bottom=0.13, top=0.84, wspace=0.32
    )
    for panel_idx, (x_col, y_col) in enumerate(PAIRS):
        draw_joint_panel(fig, outer[panel_idx], initial_df, bo_df, x_col, y_col, lims, mode)

    draw_legend(fig)

    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    configure_matplotlib()
    initial_df = build_initial_descriptor_table()
    bo_df = load_bo_descriptor_table()
    global_lims = global_limits(initial_df, bo_df)

    draw_figure(
        initial_df,
        bo_df,
        LOCAL_LIMS,
        "contour",
        "figure4C_local_BO_descriptor_joint_marginals_initial_contour_same_bins_circles_old_panel_aspect_masked_strict055",
    )
    draw_figure(
        initial_df,
        bo_df,
        LOCAL_H_WIDE_LIMS,
        "contour",
        "figure4C_local_BO_descriptor_joint_marginals_Hwide_initial_contour_same_bins_circles_masked_strict055",
    )
    draw_figure(
        initial_df,
        bo_df,
        global_lims,
        "contour",
        "figure4C_global_descriptor_joint_marginals_initial_contour_same_bins_circles_masked_strict055",
    )
    draw_figure(
        initial_df,
        bo_df,
        global_lims,
        "points_only",
        "figure4C_global_descriptor_joint_marginals_points_only_initial_edgecolor_bo_filled",
    )


if __name__ == "__main__":
    main()
