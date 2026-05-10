from pathlib import Path

"""Overlay BO samples on the top SHAP descriptor spaces.

The descriptor spaces are taken from the current seed-42 10-feature H/D
separation model: electronegativity, H adsorption energy, and work function.
The weighted version is intended for composition-ratio trends in BO, whereas
the metal1/metal2 version is a closer but more discrete view of the ML input
semantics.
"""

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

FEATURE_MATRIX = (
    ROOT
    / "hd_separation_ml"
    / "outputs"
    / "feature_selection_561_rs_from_eis_no_eis_fitting"
    / "feature_matrix_561_no_eis_fitting.csv"
)
BO_TABLE = FIGURE_DIR / "curated" / "figure4_bo_complete_112.csv"
EH_EXTRA = ROOT / "hd_separation_ml" / "EH_extra.xlsx"

BIN_EDGES = np.array([2, 4, 6, 8, 10, 12], dtype=float)
BIN_LABELS = [f"{int(BIN_EDGES[i])}-{int(BIN_EDGES[i + 1])}" for i in range(len(BIN_EDGES) - 1)]
BIN_COLORS = ["#AE5CFA", "#AAC0FC", "#F1E2AD", "#F19FA0", "#F35656"]

KNN_NEIGHBORS = 36
MASK_THRESHOLD = 0.055


def configure_matplotlib():
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.linewidth"] = 2.0
    mpl.rcParams["xtick.major.width"] = 1.0
    mpl.rcParams["ytick.major.width"] = 1.0


def separation_bin_index(values):
    idx = np.digitize(values, BIN_EDGES, right=False) - 1
    return np.clip(idx, 0, len(BIN_COLORS) - 1)


def colors_for_separation(values):
    return np.asarray(BIN_COLORS, dtype=object)[separation_bin_index(values)]


def style_axis(ax, label_size=12, tick_size=10):
    ax.tick_params(axis="both", labelsize=tick_size, colors="black")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontweight("bold")
        tick.set_color("black")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontsize(label_size)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def build_element_map(initial):
    frames = []
    for element_col, suffix in [("element1", "element1"), ("element2", "element2")]:
        frames.append(
            initial[
                [
                    element_col,
                    f"FE_ads_avg_{suffix}",
                    f"electronegativity_{suffix}",
                ]
            ].rename(
                columns={
                    element_col: "element",
                    f"FE_ads_avg_{suffix}": "FE_ads_avg",
                    f"electronegativity_{suffix}": "electronegativity",
                }
            )
        )
    mapping = pd.concat(frames, ignore_index=True).groupby("element").median(numeric_only=True)
    eh = pd.read_excel(EH_EXTRA)
    element_col = eh.columns[0]
    wf = (
        eh[[element_col, "Work function (eV)"]]
        .dropna()
        .rename(columns={element_col: "element", "Work function (eV)": "work_function"})
        .set_index("element")["work_function"]
    )
    common = mapping.index.intersection(wf.index)
    mapping.loc[common, "work_function"] = wf.loc[common].astype(float)
    return mapping


def weighted_value(element1, element2, frac1, frac2, mapping, descriptor):
    v1 = mapping.loc[element1, descriptor].to_numpy(float)
    v2 = mapping.loc[element2, descriptor].to_numpy(float)
    return frac1.to_numpy(float) * v1 + frac2.to_numpy(float) * v2


def build_tables():
    initial_raw = pd.read_csv(FEATURE_MATRIX)
    mapping = build_element_map(initial_raw)

    init = initial_raw[["sample", "element1", "element2", "separation_coefficient"]].copy()
    init_frac1 = pd.Series(np.where(init["element1"].eq(init["element2"]), 1.0, 6.0 / 7.0), index=init.index)
    init_frac2 = pd.Series(np.where(init["element1"].eq(init["element2"]), 0.0, 1.0 / 7.0), index=init.index)
    init["weighted_electronegativity"] = weighted_value(init["element1"], init["element2"], init_frac1, init_frac2, mapping, "electronegativity")
    init["weighted_FE_ads_avg"] = weighted_value(init["element1"], init["element2"], init_frac1, init_frac2, mapping, "FE_ads_avg")
    init["weighted_work_function"] = weighted_value(init["element1"], init["element2"], init_frac1, init_frac2, mapping, "work_function")

    init["metal1_electronegativity"] = mapping.loc[init["element1"], "electronegativity"].to_numpy(float)
    init["metal1_FE_ads_avg"] = mapping.loc[init["element1"], "FE_ads_avg"].to_numpy(float)
    init["metal2_work_function"] = mapping.loc[init["element2"], "work_function"].to_numpy(float)
    init["source"] = "initial"

    bo_raw = pd.read_csv(BO_TABLE)
    if "sample" not in bo_raw.columns:
        bo_raw["sample"] = bo_raw["system"].astype(str) + bo_raw["row_id"].astype(int).astype(str)
    bo = bo_raw[["sample", "system", "substance1", "substance2", "volume1", "volume2", "separation_coefficient"]].copy()
    total = bo["volume1"].astype(float) + bo["volume2"].astype(float)
    bo["frac1"] = bo["volume1"].astype(float) / total
    bo["frac2"] = bo["volume2"].astype(float) / total
    bo["weighted_electronegativity"] = weighted_value(bo["substance1"], bo["substance2"], bo["frac1"], bo["frac2"], mapping, "electronegativity")
    bo["weighted_FE_ads_avg"] = weighted_value(bo["substance1"], bo["substance2"], bo["frac1"], bo["frac2"], mapping, "FE_ads_avg")
    bo["weighted_work_function"] = weighted_value(bo["substance1"], bo["substance2"], bo["frac1"], bo["frac2"], mapping, "work_function")

    major_is_1 = bo["volume1"].astype(float).ge(bo["volume2"].astype(float))
    bo["metal1"] = np.where(major_is_1, bo["substance1"], bo["substance2"])
    bo["metal2"] = np.where(major_is_1, bo["substance2"], bo["substance1"])
    bo["metal1_electronegativity"] = mapping.loc[bo["metal1"], "electronegativity"].to_numpy(float)
    bo["metal1_FE_ads_avg"] = mapping.loc[bo["metal1"], "FE_ads_avg"].to_numpy(float)
    bo["metal2_work_function"] = mapping.loc[bo["metal2"], "work_function"].to_numpy(float)
    bo["source"] = "BO"

    init.to_csv(OUTPUT_DIR / "top3_shap_initial_descriptor_projection_table.csv", index=False)
    bo.to_csv(OUTPUT_DIR / "top3_shap_bo_descriptor_projection_table.csv", index=False)
    return init, bo


def masked_knn_contour(ax, initial, x_col, y_col, xlim, ylim):
    train = initial.dropna(subset=[x_col, y_col, "separation_coefficient"])
    points = train[[x_col, y_col]].to_numpy(float)
    values = train["separation_coefficient"].to_numpy(float)
    if len(train) < 4:
        return
    gx = np.linspace(xlim[0], xlim[1], 200)
    gy = np.linspace(ylim[0], ylim[1], 200)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.c_[xx.ravel(), yy.ravel()]
    model = KNeighborsRegressor(n_neighbors=min(KNN_NEIGHBORS, len(train)), weights="distance")
    model.fit(points, values)
    zz = model.predict(grid).reshape(xx.shape)

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    points_norm = np.c_[(points[:, 0] - xlim[0]) / x_span, (points[:, 1] - ylim[0]) / y_span]
    grid_norm = np.c_[(grid[:, 0] - xlim[0]) / x_span, (grid[:, 1] - ylim[0]) / y_span]
    dist = NearestNeighbors(n_neighbors=1).fit(points_norm).kneighbors(grid_norm, return_distance=True)[0].reshape(xx.shape)
    zz = np.ma.masked_where(dist > MASK_THRESHOLD, zz)

    cmap = ListedColormap(BIN_COLORS)
    norm = BoundaryNorm(BIN_EDGES, cmap.N, clip=True)
    ax.contourf(xx, yy, zz, levels=BIN_EDGES, cmap=cmap, norm=norm, alpha=0.34, zorder=0)
    ax.contour(xx, yy, zz, levels=BIN_EDGES, colors="#777777", linewidths=0.35, alpha=0.35, zorder=1)


def limits(initial, bo, x_col, y_col):
    merged = pd.concat([initial[[x_col, y_col]], bo[[x_col, y_col]]], ignore_index=True)
    out = []
    for col in [x_col, y_col]:
        lo = float(merged[col].min())
        hi = float(merged[col].max())
        pad = 0.045 * (hi - lo)
        out.append((lo - pad, hi + pad))
    return out


def draw_legend(fig):
    handles = [Patch(facecolor="none", edgecolor="none", label="Separation Coefficient")]
    handles.extend(Patch(facecolor=color, edgecolor="black", label=label) for color, label in zip(BIN_COLORS, BIN_LABELS))
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        prop={"family": "Arial", "weight": "bold", "size": 10},
        handlelength=1.7,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.3,
    )
    legend.get_frame().set_linewidth(1.1)


def draw_map(initial, bo, specs, output_stem):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.45), dpi=300)
    bo_colors = colors_for_separation(bo["separation_coefficient"].to_numpy(float))
    init_colors = colors_for_separation(initial["separation_coefficient"].to_numpy(float))

    for ax, (x_col, y_col, x_label, y_label) in zip(axes, specs):
        xlim, ylim = limits(initial, bo, x_col, y_col)
        masked_knn_contour(ax, initial, x_col, y_col, xlim, ylim)
        ax.scatter(
            initial[x_col],
            initial[y_col],
            s=18,
            marker="o",
            facecolors="none",
            edgecolors=init_colors,
            linewidths=0.55,
            alpha=0.36,
            zorder=2,
        )
        ax.scatter(
            bo[x_col],
            bo[y_col],
            s=46,
            marker="o",
            c=bo_colors,
            edgecolors="black",
            linewidths=0.7,
            alpha=0.82,
            zorder=4,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, color="#c9c9c9", linewidth=0.35, alpha=0.25)
        style_axis(ax)

    draw_legend(fig)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.15, top=0.84, wspace=0.34)
    png = OUTPUT_DIR / f"{output_stem}.png"
    pdf = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def summarize(init, bo):
    rows = []
    cols = [
        "weighted_electronegativity",
        "weighted_FE_ads_avg",
        "weighted_work_function",
        "metal1_electronegativity",
        "metal1_FE_ads_avg",
        "metal2_work_function",
    ]
    for col in cols:
        lo = init[col].quantile(0.05)
        hi = init[col].quantile(0.95)
        rows.append(
            {
                "descriptor": col,
                "initial_p05": lo,
                "initial_p95": hi,
                "bo_inside_initial_5_95_fraction": bo[col].between(lo, hi).mean(),
                "bo_min": bo[col].min(),
                "bo_max": bo[col].max(),
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "top3_shap_bo_initial_overlap_summary.csv", index=False)


def main():
    configure_matplotlib()
    init, bo = build_tables()
    summarize(init, bo)
    weighted_specs = [
        ("weighted_electronegativity", "weighted_FE_ads_avg", "Weighted Electronegativity", "Weighted H Adsorption Energy (eV)"),
        ("weighted_electronegativity", "weighted_work_function", "Weighted Electronegativity", "Weighted Work Function (eV)"),
        ("weighted_FE_ads_avg", "weighted_work_function", "Weighted H Adsorption Energy (eV)", "Weighted Work Function (eV)"),
    ]
    ml_style_specs = [
        ("metal1_electronegativity", "metal1_FE_ads_avg", "Electronegativity of Metal 1", "H Adsorption Energy of Metal 1 (eV)"),
        ("metal1_electronegativity", "metal2_work_function", "Electronegativity of Metal 1", "Work Function of Metal 2 (eV)"),
        ("metal1_FE_ads_avg", "metal2_work_function", "H Adsorption Energy of Metal 1 (eV)", "Work Function of Metal 2 (eV)"),
    ]
    draw_map(init, bo, weighted_specs, "top3_shap_weighted_descriptor_bo_overlay")
    draw_map(init, bo, ml_style_specs, "top3_shap_metal1_metal2_descriptor_bo_overlay")


if __name__ == "__main__":
    main()
