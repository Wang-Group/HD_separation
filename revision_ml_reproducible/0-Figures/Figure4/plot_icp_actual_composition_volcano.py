from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "hd_separation_ml"))
from icp_master import MASTER, load_bo_composition
BO_TABLE = FIGURE_DIR / "curated" / "figure4_bo_complete_112.csv"
FEATURE_MATRIX = (
    ROOT
    / "hd_separation_ml"
    / "outputs"
    / "feature_selection_561_rs_from_eis_no_eis_fitting"
    / "feature_matrix_561_no_eis_fitting.csv"
)
DEFAULT_ICP = MASTER
OUTPUT_DIR = FIGURE_DIR / "outputs" / "icp_actual_composition"

SYSTEM_ORDER = ["Co-Mo", "Co-V", "Co-Ni", "Fe-Ni"]
SYSTEM_COLORS = {
    "Co-Mo": "#d04a02",
    "Co-V": "#2a7f62",
    "Co-Ni": "#377eb8",
    "Fe-Ni": "#8b5fbf",
}


def configure_matplotlib() -> None:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["axes.linewidth"] = 1.4
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def normalize_icp_sample(raw_name: object) -> str | None:
    text = str(raw_name).strip().replace(" ", "")
    match = re.match(r"^([A-Z][a-z]?)([A-Z][a-z]?)-?(\d+)$", text)
    if not match:
        return None
    first, second, row_id = match.groups()
    return f"{first}-{second}{int(row_id)}"


def load_icp_table(path: Path) -> pd.DataFrame:
    with pd.ExcelFile(path) as book:
        is_master = "Samples" in book.sheet_names
    if is_master:
        return load_bo_composition(path)
    raw = pd.read_excel(path)
    if raw.shape[1] < 7:
        raise ValueError(f"ICP-OES table should have at least 7 columns; found {raw.shape[1]}.")

    cols = list(raw.columns)
    n1_col = next((col for col in cols if str(col).strip().lower().startswith("n1")), None)
    n2_col = next((col for col in cols if str(col).strip().lower().startswith("n2")), None)
    icp = pd.DataFrame(
        {
            "icp_raw_sample": raw.iloc[:, 0].astype(str).str.strip(),
            "icp_m1_mass_ug_ml": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "icp_m2_mass_ug_ml": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "icp_m1_mmol_ml": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "icp_m2_mmol_ml": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
            "icp_c1_c2_molar_ratio": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
            "icp_w1_w2_mass_ratio": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
        }
    )
    if n1_col is not None and n2_col is not None:
        icp["icp_n1_umol"] = pd.to_numeric(raw[n1_col], errors="coerce")
        icp["icp_n2_umol"] = pd.to_numeric(raw[n2_col], errors="coerce")
    else:
        icp["icp_n1_umol"] = np.nan
        icp["icp_n2_umol"] = np.nan
    icp["icp_total_metal_umol"] = icp["icp_n1_umol"] + icp["icp_n2_umol"]
    icp["icp_n1_n2_molar_ratio"] = icp["icp_n1_umol"] / icp["icp_n2_umol"]
    icp["icp_deposited_m1_m2_molar_ratio"] = icp["icp_n1_n2_molar_ratio"].where(
        icp["icp_n1_n2_molar_ratio"].replace([np.inf, -np.inf], np.nan).notna(),
        icp["icp_c1_c2_molar_ratio"],
    )
    icp["icp_ratio_source"] = np.where(
        icp["icp_n1_n2_molar_ratio"].replace([np.inf, -np.inf], np.nan).notna(),
        "n1/n2 amount",
        "c1/c2 concentration",
    )
    icp["sample"] = icp["icp_raw_sample"].map(normalize_icp_sample)
    icp["icp_source_columns"] = " | ".join(str(c) for c in cols)
    ratio = icp["icp_deposited_m1_m2_molar_ratio"].astype(float)
    icp["actual_frac1_from_icp"] = ratio / (1.0 + ratio)
    icp.loc[~np.isfinite(icp["actual_frac1_from_icp"]), "actual_frac1_from_icp"] = np.nan
    icp["actual_frac1_from_icp_amount"] = icp["icp_n1_umol"] / icp["icp_total_metal_umol"]
    return icp


def load_bo_table() -> pd.DataFrame:
    bo = pd.read_csv(BO_TABLE)
    if "sample" not in bo.columns:
        bo["sample"] = bo["system"].astype(str) + bo["row_id"].astype(int).astype(str)
    total = bo["volume1"].astype(float) + bo["volume2"].astype(float)
    bo["nominal_frac1_from_precursor"] = bo["volume1"].astype(float) / total
    bo["nominal_volume_ratio_v1_v2"] = bo["volume1"].astype(float) / bo["volume2"].astype(float)
    return bo


def build_element_electronegativity_map() -> pd.Series:
    feature_matrix = pd.read_csv(FEATURE_MATRIX)
    parts = []
    for element_col, en_col in [
        ("element1", "electronegativity_element1"),
        ("element2", "electronegativity_element2"),
    ]:
        parts.append(
            feature_matrix[[element_col, en_col]]
            .rename(columns={element_col: "element", en_col: "electronegativity"})
            .dropna()
        )
    element_en = pd.concat(parts, ignore_index=True)
    return element_en.groupby("element")["electronegativity"].median()


def add_weighted_electronegativity(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    element_en = build_element_electronegativity_map()
    out["electronegativity_substance1"] = out["substance1"].map(element_en)
    out["electronegativity_substance2"] = out["substance2"].map(element_en)
    out["nominal_weighted_electronegativity"] = (
        out["nominal_frac1_from_precursor"].astype(float) * out["electronegativity_substance1"].astype(float)
        + (1.0 - out["nominal_frac1_from_precursor"].astype(float))
        * out["electronegativity_substance2"].astype(float)
    )
    out["actual_weighted_electronegativity"] = (
        out["actual_frac1_from_icp"].astype(float) * out["electronegativity_substance1"].astype(float)
        + (1.0 - out["actual_frac1_from_icp"].astype(float))
        * out["electronegativity_substance2"].astype(float)
    )
    return out


def quadratic_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4 or np.unique(np.round(x, 8)).size < 3:
        return {
            "quad_a": np.nan,
            "quad_b": np.nan,
            "quad_c": np.nan,
            "quad_r2": np.nan,
            "quad_vertex_x": np.nan,
            "quad_vertex_y": np.nan,
            "linear_r2": np.nan,
            "delta_r2_quad_minus_linear": np.nan,
        }

    quad = np.polyfit(x, y, 2)
    quad_pred = np.polyval(quad, x)
    linear = np.polyfit(x, y, 1)
    linear_pred = np.polyval(linear, x)

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        quad_r2 = np.nan
        linear_r2 = np.nan
    else:
        quad_r2 = 1.0 - float(np.sum((y - quad_pred) ** 2)) / ss_tot
        linear_r2 = 1.0 - float(np.sum((y - linear_pred) ** 2)) / ss_tot

    a, b, c = [float(v) for v in quad]
    if abs(a) > 1e-12:
        vertex_x = -b / (2.0 * a)
        vertex_y = float(np.polyval(quad, vertex_x))
    else:
        vertex_x = np.nan
        vertex_y = np.nan

    return {
        "quad_a": a,
        "quad_b": b,
        "quad_c": c,
        "quad_r2": quad_r2,
        "quad_vertex_x": vertex_x,
        "quad_vertex_y": vertex_y,
        "linear_r2": linear_r2,
        "delta_r2_quad_minus_linear": quad_r2 - linear_r2,
    }


def add_fit_line(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    color: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    data = frame[[x_col, "separation_coefficient"]].dropna()
    if len(data) < 4 or data[x_col].nunique() < 3:
        return
    x = data[x_col].to_numpy(float)
    y = data["separation_coefficient"].to_numpy(float)
    coefs = np.polyfit(x, y, 2)
    if xlim is None:
        xx = np.linspace(max(0.0, np.nanmin(x) - 0.03), min(1.0, np.nanmax(x) + 0.03), 240)
    else:
        xx = np.linspace(xlim[0], xlim[1], 240)
    yy = np.polyval(coefs, xx)
    ax.plot(xx, yy, color=color, lw=2.0, alpha=0.95, zorder=4)


def padded_extent(values: pd.Series, frac: float = 0.08) -> tuple[float, float]:
    arr = values.to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (0.0, 1.0)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if np.isclose(lo, hi):
        return (lo - 0.02, hi + 0.02)
    pad = frac * (hi - lo)
    return (lo - pad, hi + pad)


def draw_volcano_comparison(merged: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(SYSTEM_ORDER),
        2,
        figsize=(8.2, 10.0),
        dpi=300,
        sharey=True,
        constrained_layout=True,
    )
    x_specs = [
        ("nominal_frac1_from_precursor", "Nominal M1 fraction"),
        ("actual_frac1_from_icp", "ICP actual M1 fraction"),
    ]

    for row_idx, system in enumerate(SYSTEM_ORDER):
        sub = merged.loc[merged["system"].eq(system)].copy()
        color = SYSTEM_COLORS[system]
        for col_idx, (x_col, x_label) in enumerate(x_specs):
            ax = axes[row_idx, col_idx]
            plot_data = sub.dropna(subset=[x_col, "separation_coefficient"])
            ax.scatter(
                plot_data[x_col],
                plot_data["separation_coefficient"],
                s=44,
                facecolor=color,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.86,
                zorder=5,
            )
            add_fit_line(ax, plot_data, x_col, color="#111111")

            if not plot_data.empty:
                best = plot_data.loc[plot_data["separation_coefficient"].idxmax()]
                ax.scatter(
                    [best[x_col]],
                    [best["separation_coefficient"]],
                    marker="*",
                    s=145,
                    facecolor="#f03b20",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=7,
                )
                ax.annotate(
                    str(best["sample"]),
                    xy=(best[x_col], best["separation_coefficient"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="black",
                )

            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(2.0, 12.2)
            ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
            ax.tick_params(axis="both", labelsize=8.5)
            ax.set_xlabel(x_label, fontsize=9.5)
            if col_idx == 0:
                ax.set_ylabel(f"{system}\nH/D separation coefficient", fontsize=9.5)
            else:
                ax.set_ylabel("")
            ax.text(
                0.02,
                0.94,
                f"n = {len(plot_data)}",
                transform=ax.transAxes,
                fontsize=8.0,
                ha="left",
                va="top",
            )

    fig.suptitle(
        "BO composition-volcano check with ICP-OES actual molar ratios",
        fontsize=12.5,
        fontweight="bold",
    )
    png = output_dir / "icp_actual_composition_volcano_nominal_vs_actual.png"
    pdf = output_dir / "icp_actual_composition_volcano_nominal_vs_actual.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def draw_electronegativity_comparison(merged: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        len(SYSTEM_ORDER),
        2,
        figsize=(8.2, 10.0),
        dpi=300,
        sharey=True,
        constrained_layout=True,
    )
    x_specs = [
        ("nominal_weighted_electronegativity", "Precursor-weighted electronegativity"),
        ("actual_weighted_electronegativity", "ICP-weighted electronegativity"),
    ]

    for row_idx, system in enumerate(SYSTEM_ORDER):
        sub = merged.loc[merged["system"].eq(system)].copy()
        color = SYSTEM_COLORS[system]
        shared_xlim = padded_extent(
            pd.concat(
                [
                    sub["nominal_weighted_electronegativity"],
                    sub["actual_weighted_electronegativity"],
                ],
                ignore_index=True,
            )
        )
        for col_idx, (x_col, x_label) in enumerate(x_specs):
            ax = axes[row_idx, col_idx]
            plot_data = sub.dropna(subset=[x_col, "separation_coefficient"])
            ax.scatter(
                plot_data[x_col],
                plot_data["separation_coefficient"],
                s=44,
                facecolor=color,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.86,
                zorder=5,
            )
            add_fit_line(ax, plot_data, x_col, color="#111111", xlim=padded_extent(plot_data[x_col]))

            if not plot_data.empty:
                best = plot_data.loc[plot_data["separation_coefficient"].idxmax()]
                ax.scatter(
                    [best[x_col]],
                    [best["separation_coefficient"]],
                    marker="*",
                    s=145,
                    facecolor="#f03b20",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=7,
                )
                ax.annotate(
                    str(best["sample"]),
                    xy=(best[x_col], best["separation_coefficient"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="black",
                )

            ax.set_xlim(*shared_xlim)
            ax.set_ylim(2.0, 12.2)
            ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
            ax.tick_params(axis="both", labelsize=8.5)
            ax.set_xlabel(x_label, fontsize=9.5)
            if col_idx == 0:
                ax.set_ylabel(f"{system}\nH/D separation coefficient", fontsize=9.5)
            else:
                ax.set_ylabel("")
            ax.text(
                0.02,
                0.94,
                f"n = {len(plot_data)}",
                transform=ax.transAxes,
                fontsize=8.0,
                ha="left",
                va="top",
            )

    fig.suptitle(
        "Electronegativity-volcano check with ICP-OES actual composition weights",
        fontsize=12.5,
        fontweight="bold",
    )
    png = output_dir / "icp_weighted_electronegativity_volcano_nominal_vs_actual.png"
    pdf = output_dir / "icp_weighted_electronegativity_volcano_nominal_vs_actual.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def draw_electronegativity_overall(merged: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7), dpi=300, sharey=True, constrained_layout=True)
    x_specs = [
        ("nominal_weighted_electronegativity", "Precursor-weighted electronegativity"),
        ("actual_weighted_electronegativity", "ICP-weighted electronegativity"),
    ]
    shared_xlim = padded_extent(
        pd.concat(
            [
                merged["nominal_weighted_electronegativity"],
                merged["actual_weighted_electronegativity"],
            ],
            ignore_index=True,
        ),
        frac=0.04,
    )

    for ax, (x_col, x_label) in zip(axes, x_specs):
        for system in SYSTEM_ORDER:
            sub = merged.loc[merged["system"].eq(system)].dropna(subset=[x_col, "separation_coefficient"])
            ax.scatter(
                sub[x_col],
                sub["separation_coefficient"],
                s=38,
                facecolor=SYSTEM_COLORS[system],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.84,
                label=system,
                zorder=5,
            )
        add_fit_line(ax, merged, x_col, color="#111111", xlim=padded_extent(merged[x_col], frac=0.04))
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(2.0, 12.2)
        ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.set_xlabel(x_label, fontsize=9.5)
    axes[0].set_ylabel("H/D separation coefficient", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Overall BO electronegativity trend, 88 ICP-matched samples", fontsize=12.0, fontweight="bold")
    png = output_dir / "icp_weighted_electronegativity_volcano_overall.png"
    pdf = output_dir / "icp_weighted_electronegativity_volcano_overall.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def draw_electronegativity_shift_overlay(merged: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300, constrained_layout=True)
    x_nom = "nominal_weighted_electronegativity"
    x_act = "actual_weighted_electronegativity"
    y_col = "separation_coefficient"
    shared_xlim = padded_extent(pd.concat([merged[x_nom], merged[x_act]], ignore_index=True), frac=0.05)

    for system in SYSTEM_ORDER:
        sub = merged.loc[merged["system"].eq(system)].dropna(subset=[x_nom, x_act, y_col]).copy()
        color = SYSTEM_COLORS[system]
        for _, row in sub.iterrows():
            ax.plot(
                [row[x_nom], row[x_act]],
                [row[y_col], row[y_col]],
                color=color,
                lw=0.8,
                alpha=0.28,
                zorder=2,
            )
        ax.scatter(
            sub[x_nom],
            sub[y_col],
            s=36,
            facecolor="white",
            edgecolor=color,
            linewidth=1.1,
            alpha=0.9,
            label=f"{system} precursor",
            zorder=4,
        )
        ax.scatter(
            sub[x_act],
            sub[y_col],
            s=38,
            facecolor=color,
            edgecolor="black",
            linewidth=0.55,
            alpha=0.85,
            label=f"{system} ICP",
            zorder=5,
        )

    add_fit_line(ax, merged.rename(columns={x_nom: "x_fit"}), "x_fit", color="#6b7280", xlim=padded_extent(merged[x_nom], frac=0.04))
    add_fit_line(ax, merged.rename(columns={x_act: "x_fit"}), "x_fit", color="#111111", xlim=padded_extent(merged[x_act], frac=0.04))

    ax.set_xlim(*shared_xlim)
    ax.set_ylim(2.0, 12.2)
    ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=8.8)
    ax.set_xlabel("Weighted electronegativity", fontsize=10.0)
    ax.set_ylabel("H/D separation coefficient", fontsize=10.0)
    ax.set_title("ICP-weighted points overlaid on the precursor-weighted electronegativity plot", fontsize=11.0, fontweight="bold")

    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markersize=5.8, label="Precursor-weighted"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#808080", markeredgecolor="black", markersize=5.8, label="ICP-weighted"),
        mpl.lines.Line2D([0], [0], color="#6b7280", lw=2.0, label="Precursor fit"),
        mpl.lines.Line2D([0], [0], color="#111111", lw=2.0, label="ICP fit"),
    ]
    system_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=SYSTEM_COLORS[s], markeredgecolor="black", markersize=5.8, label=s)
        for s in SYSTEM_ORDER
    ]
    ax.legend(handles=legend_handles + system_handles, frameon=False, fontsize=7.4, loc="lower right", ncol=2)

    png = output_dir / "icp_weighted_electronegativity_overlay_on_precursor_plot.png"
    pdf = output_dir / "icp_weighted_electronegativity_overlay_on_precursor_plot.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_electronegativity_shift_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEM_ORDER + ["all_matched_BO"]:
        sub = merged.copy() if system == "all_matched_BO" else merged.loc[merged["system"].eq(system)].copy()
        delta = sub["actual_weighted_electronegativity"] - sub["nominal_weighted_electronegativity"]
        rows.append(
            {
                "system": system,
                "matched_icp_points": len(sub),
                "mean_actual_minus_nominal_en": float(delta.mean()),
                "median_actual_minus_nominal_en": float(delta.median()),
                "mean_abs_shift_en": float(delta.abs().mean()),
                "max_abs_shift_en": float(delta.abs().max()),
                "points_shifted_lower_en": int((delta < 0).sum()),
                "points_shifted_higher_en": int((delta > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_deposition_calibration(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEM_ORDER:
        sub = merged.loc[merged["system"].eq(system)].dropna(
            subset=["nominal_volume_ratio_v1_v2", "icp_deposited_m1_m2_molar_ratio"]
        )
        nominal_ratio = sub["nominal_volume_ratio_v1_v2"].astype(float)
        actual_ratio = sub["icp_deposited_m1_m2_molar_ratio"].astype(float)
        valid = nominal_ratio.gt(0) & actual_ratio.gt(0)
        nominal_ratio = nominal_ratio.loc[valid]
        actual_ratio = actual_ratio.loc[valid]
        x = np.log(nominal_ratio.to_numpy(float))
        y = np.log(actual_ratio.to_numpy(float))
        if len(x) < 4 or np.unique(np.round(x, 8)).size < 3:
            slope = np.nan
            intercept = np.nan
            r2 = np.nan
        else:
            slope, intercept = np.polyfit(x, y, 1)
            pred = intercept + slope * x
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = np.nan if ss_tot <= 0 else 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot
        enrichment_at_equal_precursor = float(np.exp(intercept)) if np.isfinite(intercept) else np.nan
        if np.isfinite(intercept) and np.isfinite(slope) and abs(slope) > 1e-12:
            nominal_ratio_for_equal_deposit = float(np.exp(-intercept / slope))
            nominal_frac_for_equal_deposit = nominal_ratio_for_equal_deposit / (1.0 + nominal_ratio_for_equal_deposit)
        else:
            nominal_ratio_for_equal_deposit = np.nan
            nominal_frac_for_equal_deposit = np.nan
        rows.append(
            {
                "system": system,
                "matched_icp_points": len(nominal_ratio),
                "log_actual_ratio_intercept": float(intercept),
                "log_actual_ratio_slope": float(slope),
                "log_ratio_fit_r2": float(r2),
                "predicted_actual_m1_m2_at_equal_precursor": enrichment_at_equal_precursor,
                "observed_nominal_ratio_min": float(nominal_ratio.min()) if len(nominal_ratio) else np.nan,
                "observed_nominal_ratio_max": float(nominal_ratio.max()) if len(nominal_ratio) else np.nan,
                "observed_actual_ratio_min": float(actual_ratio.min()) if len(actual_ratio) else np.nan,
                "observed_actual_ratio_max": float(actual_ratio.max()) if len(actual_ratio) else np.nan,
                "nominal_ratio_needed_for_predicted_actual_m1_m2_1": nominal_ratio_for_equal_deposit,
                "nominal_frac1_needed_for_predicted_actual_frac1_0p5": nominal_frac_for_equal_deposit,
            }
        )
    return pd.DataFrame(rows)


def predict_actual_ratio_from_calibration(nominal_ratio: pd.Series, slope: float, intercept: float) -> pd.Series:
    ratio = nominal_ratio.astype(float)
    predicted = np.exp(intercept + slope * np.log(ratio))
    return pd.Series(predicted, index=nominal_ratio.index)


def draw_deposition_calibration(merged: pd.DataFrame, calibration: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.5), dpi=300, constrained_layout=True)
    for ax, system in zip(axes.ravel(), SYSTEM_ORDER):
        sub = merged.loc[merged["system"].eq(system)].dropna(
            subset=["nominal_frac1_from_precursor", "actual_frac1_from_icp", "nominal_volume_ratio_v1_v2"]
        )
        color = SYSTEM_COLORS[system]
        ax.scatter(
            sub["nominal_frac1_from_precursor"],
            sub["actual_frac1_from_icp"],
            s=42,
            facecolor=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.86,
            zorder=5,
        )
        ax.plot([0, 1], [0, 1], color="#8a8f98", lw=1.2, ls="--", zorder=1)

        row = calibration.loc[calibration["system"].eq(system)].iloc[0]
        if np.isfinite(row["log_actual_ratio_slope"]):
            lo = max(0.01, float(sub["nominal_frac1_from_precursor"].min()) - 0.03)
            hi = min(0.99, float(sub["nominal_frac1_from_precursor"].max()) + 0.03)
            x_frac = np.linspace(lo, hi, 240)
            x_ratio = x_frac / (1.0 - x_frac)
            y_ratio = np.exp(row["log_actual_ratio_intercept"] + row["log_actual_ratio_slope"] * np.log(x_ratio))
            y_frac = y_ratio / (1.0 + y_ratio)
            ax.plot(x_frac, y_frac, color="#111111", lw=2.0, zorder=4)
            ax.text(
                0.04,
                0.92,
                f"K@1:1 = {row['predicted_actual_m1_m2_at_equal_precursor']:.2f}\nR2 = {row['log_ratio_fit_r2']:.2f}",
                transform=ax.transAxes,
                fontsize=8.0,
                ha="left",
                va="top",
            )

        ax.set_title(system, fontsize=10.5, fontweight="bold")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
        ax.tick_params(axis="both", labelsize=8.0)
        ax.set_xlabel("Precursor M1 fraction", fontsize=9.0)
        ax.set_ylabel("ICP actual M1 fraction", fontsize=9.0)

    fig.suptitle("Deposition-bias calibration from ICP-OES", fontsize=12.0, fontweight="bold")
    png = output_dir / "icp_deposition_bias_calibration_actual_vs_precursor.png"
    pdf = output_dir / "icp_deposition_bias_calibration_actual_vs_precursor.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_deposition_calibrated_all_bo(bo: pd.DataFrame, icp: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    out = bo.merge(icp, on="sample", how="left")
    out["deposition_corrected_c1_c2_ratio"] = out["icp_deposited_m1_m2_molar_ratio"].astype(float)
    out["composition_source"] = np.where(
        out["deposition_corrected_c1_c2_ratio"].notna(),
        "ICP measured",
        "log-ratio calibration from ICP",
    )
    for _, row in calibration.iterrows():
        system = row["system"]
        mask = out["system"].eq(system) & out["deposition_corrected_c1_c2_ratio"].isna()
        if mask.any() and np.isfinite(row["log_actual_ratio_slope"]):
            out.loc[mask, "deposition_corrected_c1_c2_ratio"] = predict_actual_ratio_from_calibration(
                out.loc[mask, "nominal_volume_ratio_v1_v2"],
                row["log_actual_ratio_slope"],
                row["log_actual_ratio_intercept"],
            )
    ratio = out["deposition_corrected_c1_c2_ratio"].astype(float)
    out["deposition_corrected_frac1"] = ratio / (1.0 + ratio)
    element_en = build_element_electronegativity_map()
    out["electronegativity_substance1"] = out["substance1"].map(element_en)
    out["electronegativity_substance2"] = out["substance2"].map(element_en)
    out["deposition_corrected_weighted_electronegativity"] = (
        out["deposition_corrected_frac1"].astype(float) * out["electronegativity_substance1"].astype(float)
        + (1.0 - out["deposition_corrected_frac1"].astype(float))
        * out["electronegativity_substance2"].astype(float)
    )
    return out


def draw_deposition_corrected_electronegativity_volcano(
    deposition_corrected_bo: pd.DataFrame,
    output_dir: Path,
) -> None:
    x_col = "deposition_corrected_weighted_electronegativity"
    y_col = "separation_coefficient"
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300, constrained_layout=True)

    for system in SYSTEM_ORDER:
        sub = deposition_corrected_bo.loc[deposition_corrected_bo["system"].eq(system)].dropna(
            subset=[x_col, y_col, "composition_source"]
        )
        color = SYSTEM_COLORS[system]
        measured = sub.loc[sub["composition_source"].eq("ICP measured")]
        estimated = sub.loc[~sub["composition_source"].eq("ICP measured")]
        ax.scatter(
            measured[x_col],
            measured[y_col],
            s=42,
            marker="o",
            facecolor=color,
            edgecolor="black",
            linewidth=0.65,
            alpha=0.86,
            zorder=5,
        )
        ax.scatter(
            estimated[x_col],
            estimated[y_col],
            s=50,
            marker="s",
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.72,
            zorder=6,
        )

    fit_data = deposition_corrected_bo.dropna(subset=[x_col, y_col])
    add_fit_line(ax, fit_data, x_col, color="#111111", xlim=padded_extent(fit_data[x_col], frac=0.04))
    fit = quadratic_fit(fit_data[x_col].to_numpy(float), fit_data[y_col].to_numpy(float))
    if np.isfinite(fit["quad_vertex_x"]):
        ax.axvline(fit["quad_vertex_x"], color="#111111", lw=1.0, ls=":", alpha=0.65, zorder=3)
        ax.text(
            0.03,
            0.95,
            f"vertex = {fit['quad_vertex_x']:.3f}\nR2 = {fit['quad_r2']:.2f}",
            transform=ax.transAxes,
            fontsize=8.3,
            ha="left",
            va="top",
        )

    ax.set_xlim(*padded_extent(fit_data[x_col], frac=0.05))
    ax.set_ylim(2.0, 12.2)
    ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=8.8)
    ax.set_xlabel("Deposited-composition-weighted electronegativity", fontsize=10.0)
    ax.set_ylabel("H/D separation coefficient", fontsize=10.0)
    ax.set_title("Unified volcano using deposited-composition weighting", fontsize=11.0, fontweight="bold")

    source_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#808080",
            markeredgecolor="black",
            markersize=5.8,
            label="ICP measured composition",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#808080",
            markeredgecolor="black",
            markersize=5.8,
            label="calibration-derived composition",
        ),
    ]
    system_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SYSTEM_COLORS[system],
            markeredgecolor="black",
            markersize=5.8,
            label=system,
        )
        for system in SYSTEM_ORDER
    ]
    ax.legend(handles=source_handles + system_handles, frameon=False, fontsize=7.4, loc="lower right", ncol=2)

    png = output_dir / "deposition_corrected_weighted_electronegativity_volcano_all_112.png"
    pdf = output_dir / "deposition_corrected_weighted_electronegativity_volcano_all_112.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_deposition_corrected_electronegativity_summary(
    deposition_corrected_bo: pd.DataFrame,
) -> pd.DataFrame:
    x_col = "deposition_corrected_weighted_electronegativity"
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [
        ("all_112_deposition_corrected", deposition_corrected_bo.copy())
    ]
    groups.extend(
        (system, deposition_corrected_bo.loc[deposition_corrected_bo["system"].eq(system)].copy())
        for system in SYSTEM_ORDER
    )
    for label, sub in groups:
        fit = quadratic_fit(
            sub[x_col].to_numpy(float),
            sub["separation_coefficient"].to_numpy(float),
        )
        best = sub.loc[sub["separation_coefficient"].idxmax()]
        row = {
            "group": label,
            "n": len(sub),
            "icp_measured_points": int(sub["composition_source"].eq("ICP measured").sum()),
            "calibration_derived_points": int((~sub["composition_source"].eq("ICP measured")).sum()),
            "best_sample": best["sample"],
            "best_separation_coefficient": float(best["separation_coefficient"]),
            "best_deposition_corrected_weighted_electronegativity": float(best[x_col]),
            "observed_en_min": float(sub[x_col].min()),
            "observed_en_max": float(sub[x_col].max()),
        }
        for key, value in fit.items():
            row[key] = value
        row["volcano_like_fit"] = volcano_like_in_observed_domain(fit, sub[x_col])
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(merged: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEM_ORDER:
        sub = merged.loc[merged["system"].eq(system)].copy()
        row = {
            "system": system,
            "matched_icp_points": len(sub),
            "missing_icp_points": int(missing["system"].eq(system).sum()),
        }
        if not sub.empty:
            best = sub.loc[sub["separation_coefficient"].idxmax()]
            row.update(
                {
                    "best_sample_matched": best["sample"],
                    "best_separation_coefficient_matched": float(best["separation_coefficient"]),
                    "best_nominal_frac1": float(best["nominal_frac1_from_precursor"]),
                    "best_actual_frac1": float(best["actual_frac1_from_icp"]),
                    "nominal_actual_frac1_pearson_r": float(
                        sub[["nominal_frac1_from_precursor", "actual_frac1_from_icp"]].corr().iloc[0, 1]
                    ),
                }
            )
        else:
            row.update(
                {
                    "best_sample_matched": np.nan,
                    "best_separation_coefficient_matched": np.nan,
                    "best_nominal_frac1": np.nan,
                    "best_actual_frac1": np.nan,
                    "nominal_actual_frac1_pearson_r": np.nan,
                }
            )

        for prefix, x_col in [
            ("nominal", "nominal_frac1_from_precursor"),
            ("actual_icp", "actual_frac1_from_icp"),
        ]:
            fit = quadratic_fit(
                sub[x_col].to_numpy(float),
                sub["separation_coefficient"].to_numpy(float),
            )
            for key, value in fit.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_volcano_like_fit"] = bool(
                np.isfinite(fit["quad_a"])
                and fit["quad_a"] < 0
                and np.isfinite(fit["quad_vertex_x"])
                and 0.05 <= fit["quad_vertex_x"] <= 0.95
            )
        rows.append(row)
    return pd.DataFrame(rows)


def volcano_like_in_observed_domain(fit: dict[str, float], x: pd.Series) -> bool:
    values = x.dropna().to_numpy(float)
    if len(values) == 0:
        return False
    return bool(
        np.isfinite(fit["quad_a"])
        and fit["quad_a"] < 0
        and np.isfinite(fit["quad_vertex_x"])
        and float(np.nanmin(values)) <= fit["quad_vertex_x"] <= float(np.nanmax(values))
    )


def build_electronegativity_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [(system, merged.loc[merged["system"].eq(system)].copy()) for system in SYSTEM_ORDER]
    groups.append(("all_matched_BO", merged.copy()))

    for system, sub in groups:
        row = {"system": system, "matched_icp_points": len(sub)}
        if not sub.empty:
            best = sub.loc[sub["separation_coefficient"].idxmax()]
            row.update(
                {
                    "best_sample": best["sample"],
                    "best_separation_coefficient": float(best["separation_coefficient"]),
                    "best_nominal_weighted_electronegativity": float(best["nominal_weighted_electronegativity"]),
                    "best_actual_weighted_electronegativity": float(best["actual_weighted_electronegativity"]),
                    "nominal_actual_weighted_electronegativity_pearson_r": float(
                        sub[
                            [
                                "nominal_weighted_electronegativity",
                                "actual_weighted_electronegativity",
                            ]
                        ]
                        .corr()
                        .iloc[0, 1]
                    ),
                }
            )
        else:
            row.update(
                {
                    "best_sample": np.nan,
                    "best_separation_coefficient": np.nan,
                    "best_nominal_weighted_electronegativity": np.nan,
                    "best_actual_weighted_electronegativity": np.nan,
                    "nominal_actual_weighted_electronegativity_pearson_r": np.nan,
                }
            )

        for prefix, x_col in [
            ("nominal_en", "nominal_weighted_electronegativity"),
            ("actual_icp_en", "actual_weighted_electronegativity"),
        ]:
            fit = quadratic_fit(
                sub[x_col].to_numpy(float),
                sub["separation_coefficient"].to_numpy(float),
            )
            for key, value in fit.items():
                row[f"{prefix}_{key}"] = value
            row[f"{prefix}_observed_min"] = float(sub[x_col].min()) if len(sub) else np.nan
            row[f"{prefix}_observed_max"] = float(sub[x_col].max()) if len(sub) else np.nan
            row[f"{prefix}_volcano_like_fit"] = volcano_like_in_observed_domain(fit, sub[x_col])
        rows.append(row)
    return pd.DataFrame(rows)


def add_global_quadratic_residual(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    residual_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    out[residual_col] = np.nan
    valid = out[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 4 or valid[x_col].nunique() < 3:
        return out
    coefs = np.polyfit(valid[x_col].to_numpy(float), valid[y_col].to_numpy(float), 2)
    pred = np.polyval(coefs, out[x_col].to_numpy(float))
    out[residual_col] = out[y_col].astype(float) - pred
    return out


def build_loading_effect_summary(matched: pd.DataFrame) -> pd.DataFrame:
    x_col = "actual_weighted_electronegativity"
    y_col = "separation_coefficient"
    residual_col = "actual_en_global_quad_residual"
    with_residual = add_global_quadratic_residual(matched, x_col, y_col, residual_col)
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("all_icp_matched", with_residual.copy())]
    groups.extend((system, with_residual.loc[with_residual["system"].eq(system)].copy()) for system in SYSTEM_ORDER)

    for label, sub in groups:
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["icp_total_metal_umol", y_col, residual_col]
        )
        sub = sub.loc[sub["icp_total_metal_umol"].gt(0)].copy()
        row = {
            "group": label,
            "n": len(sub),
            "total_metal_umol_min": float(sub["icp_total_metal_umol"].min()) if len(sub) else np.nan,
            "total_metal_umol_median": float(sub["icp_total_metal_umol"].median()) if len(sub) else np.nan,
            "total_metal_umol_max": float(sub["icp_total_metal_umol"].max()) if len(sub) else np.nan,
        }
        if len(sub) >= 4:
            sub["log10_total_metal_umol"] = np.log10(sub["icp_total_metal_umol"].astype(float))
            low_cut = float(sub["icp_total_metal_umol"].quantile(0.25))
            high_cut = float(sub["icp_total_metal_umol"].quantile(0.75))
            low = sub.loc[sub["icp_total_metal_umol"].le(low_cut)]
            high = sub.loc[sub["icp_total_metal_umol"].ge(high_cut)]
            row.update(
                {
                    "spearman_log_total_vs_separation": float(
                        sub["log10_total_metal_umol"].corr(sub[y_col], method="spearman")
                    ),
                    "spearman_log_total_vs_en_residual": float(
                        sub["log10_total_metal_umol"].corr(sub[residual_col], method="spearman")
                    ),
                    "pearson_log_total_vs_en_residual": float(
                        sub["log10_total_metal_umol"].corr(sub[residual_col], method="pearson")
                    ),
                    "low_loading_q25_umol": low_cut,
                    "high_loading_q75_umol": high_cut,
                    "mean_separation_low_loading_q25": float(low[y_col].mean()),
                    "mean_separation_high_loading_q75": float(high[y_col].mean()),
                    "mean_residual_low_loading_q25": float(low[residual_col].mean()),
                    "mean_residual_high_loading_q75": float(high[residual_col].mean()),
                }
            )
        else:
            row.update(
                {
                    "spearman_log_total_vs_separation": np.nan,
                    "spearman_log_total_vs_en_residual": np.nan,
                    "pearson_log_total_vs_en_residual": np.nan,
                    "low_loading_q25_umol": np.nan,
                    "high_loading_q75_umol": np.nan,
                    "mean_separation_low_loading_q25": np.nan,
                    "mean_separation_high_loading_q75": np.nan,
                    "mean_residual_low_loading_q25": np.nan,
                    "mean_residual_high_loading_q75": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def draw_loading_effect(matched: pd.DataFrame, output_dir: Path) -> None:
    x_col = "actual_weighted_electronegativity"
    y_col = "separation_coefficient"
    residual_col = "actual_en_global_quad_residual"
    data = add_global_quadratic_residual(matched, x_col, y_col, residual_col)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[x_col, y_col, residual_col, "icp_total_metal_umol"]
    )
    data = data.loc[data["icp_total_metal_umol"].gt(0)].copy()
    data["log10_total_metal_umol"] = np.log10(data["icp_total_metal_umol"].astype(float))
    lo = float(data["log10_total_metal_umol"].min())
    hi = float(data["log10_total_metal_umol"].max())
    if np.isclose(lo, hi):
        data["marker_size"] = 58.0
    else:
        data["marker_size"] = 28.0 + 92.0 * (data["log10_total_metal_umol"] - lo) / (hi - lo)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), dpi=300, constrained_layout=True)
    ax = axes[0]
    for system in SYSTEM_ORDER:
        sub = data.loc[data["system"].eq(system)]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=sub["marker_size"],
            facecolor=SYSTEM_COLORS[system],
            edgecolor="black",
            linewidth=0.65,
            alpha=0.82,
            label=system,
            zorder=5,
        )
    add_fit_line(ax, data, x_col, color="#111111", xlim=padded_extent(data[x_col], frac=0.04))
    ax.set_xlim(*padded_extent(data[x_col], frac=0.05))
    ax.set_ylim(2.0, 12.2)
    ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=8.3)
    ax.set_xlabel("ICP-weighted electronegativity", fontsize=9.3)
    ax.set_ylabel("H/D separation coefficient", fontsize=9.3)
    ax.set_title("Point size = ICP total metal amount", fontsize=10.0, fontweight="bold")

    ax = axes[1]
    for system in SYSTEM_ORDER:
        sub = data.loc[data["system"].eq(system)]
        ax.scatter(
            sub["log10_total_metal_umol"],
            sub[residual_col],
            s=44,
            facecolor=SYSTEM_COLORS[system],
            edgecolor="black",
            linewidth=0.65,
            alpha=0.82,
            label=system,
            zorder=5,
        )
    valid = data[["log10_total_metal_umol", residual_col]].dropna()
    if len(valid) >= 4 and valid["log10_total_metal_umol"].nunique() > 1:
        slope, intercept = np.polyfit(
            valid["log10_total_metal_umol"].to_numpy(float),
            valid[residual_col].to_numpy(float),
            1,
        )
        xx = np.linspace(valid["log10_total_metal_umol"].min(), valid["log10_total_metal_umol"].max(), 160)
        ax.plot(xx, intercept + slope * xx, color="#111111", lw=1.8, zorder=4)
    ax.axhline(0.0, color="#8a8f98", lw=1.0, ls="--", zorder=2)
    ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
    ax.tick_params(axis="both", labelsize=8.3)
    ax.set_xlabel("log10(total ICP metal amount, umol)", fontsize=9.3)
    ax.set_ylabel("Residual from electronegativity volcano", fontsize=9.3)
    ax.set_title("Loading proxy vs volcano residual", fontsize=10.0, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.4, loc="best")

    png = output_dir / "icp_total_metal_loading_proxy_effect.png"
    pdf = output_dir / "icp_total_metal_loading_proxy_effect.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def add_loading_adjusted_effective_electronegativity(
    matched: pd.DataFrame,
    l0_umol: float,
    carbon_electronegativity: float = 2.55,
) -> pd.DataFrame:
    out = matched.copy()
    loading = out["icp_total_metal_umol"].astype(float)
    f_metal = loading / (loading + l0_umol)
    out["loading_adjusted_metal_fraction_proxy"] = f_metal
    out["loading_adjusted_effective_electronegativity"] = (
        f_metal * out["actual_weighted_electronegativity"].astype(float)
        + (1.0 - f_metal) * carbon_electronegativity
    )
    out["loading_adjusted_l0_umol"] = l0_umol
    out["loading_adjusted_carbon_electronegativity"] = carbon_electronegativity
    return out


def build_loading_adjusted_sensitivity(
    matched: pd.DataFrame,
    carbon_electronegativity: float = 2.55,
) -> pd.DataFrame:
    data = matched.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "actual_weighted_electronegativity",
            "separation_coefficient",
            "icp_total_metal_umol",
        ]
    )
    data = data.loc[data["icp_total_metal_umol"].gt(0)].copy()
    y = data["separation_coefficient"].to_numpy(float)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    rows = []

    grid = np.concatenate(
        [
            np.array([0.0], dtype=float),
            np.linspace(0.1, 5.0, 20),
            np.linspace(6.0, 100.0, 48),
            np.linspace(110.0, 500.0, 40),
        ]
    )
    for l0_umol in grid:
        if l0_umol == 0:
            x = data["actual_weighted_electronegativity"].to_numpy(float)
            model = "ICP metal-only"
        else:
            loading = data["icp_total_metal_umol"].to_numpy(float)
            f_metal = loading / (loading + l0_umol)
            x = (
                f_metal * data["actual_weighted_electronegativity"].to_numpy(float)
                + (1.0 - f_metal) * carbon_electronegativity
            )
            model = "loading-adjusted effective"
        coefs = np.polyfit(x, y, 2)
        pred = np.polyval(coefs, x)
        r2 = np.nan if ss_tot <= 0 else 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot
        vertex = -coefs[1] / (2.0 * coefs[0]) if abs(coefs[0]) > 1e-12 else np.nan
        rows.append(
            {
                "model": model,
                "l0_umol": float(l0_umol),
                "carbon_electronegativity": carbon_electronegativity if l0_umol > 0 else np.nan,
                "n": len(data),
                "quad_r2": r2,
                "quad_a": float(coefs[0]),
                "quad_b": float(coefs[1]),
                "quad_c": float(coefs[2]),
                "quad_vertex_x": float(vertex),
            }
        )
    return pd.DataFrame(rows).sort_values("quad_r2", ascending=False)


def build_three_electronegativity_comparison_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, x_col in [
        ("precursor-weighted", "nominal_weighted_electronegativity"),
        ("ICP/deposited-weighted", "actual_weighted_electronegativity"),
        ("loading-adjusted effective", "loading_adjusted_effective_electronegativity"),
    ]:
        fit_data = comparison.dropna(subset=[x_col, "separation_coefficient"])
        fit = quadratic_fit(
            fit_data[x_col].to_numpy(float),
            fit_data["separation_coefficient"].to_numpy(float),
        )
        rows.append(
            {
                "weighting": label,
                "n": len(fit_data),
                "x_min": float(fit_data[x_col].min()),
                "x_max": float(fit_data[x_col].max()),
                "quad_r2": fit["quad_r2"],
                "quad_vertex_x": fit["quad_vertex_x"],
                "quad_vertex_y": fit["quad_vertex_y"],
                "quad_a": fit["quad_a"],
                "linear_r2": fit["linear_r2"],
                "l0_umol": float(fit_data["loading_adjusted_l0_umol"].iloc[0])
                if label == "loading-adjusted effective"
                else np.nan,
                "carbon_electronegativity": float(
                    fit_data["loading_adjusted_carbon_electronegativity"].iloc[0]
                )
                if label == "loading-adjusted effective"
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def draw_three_electronegativity_comparison(
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    x_specs = [
        ("nominal_weighted_electronegativity", "Precursor-weighted"),
        ("actual_weighted_electronegativity", "ICP/deposited-weighted"),
        ("loading_adjusted_effective_electronegativity", "Loading-adjusted effective"),
    ]
    shared_xlim = padded_extent(
        pd.concat([comparison[col] for col, _ in x_specs], ignore_index=True),
        frac=0.045,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), dpi=300, sharey=True, constrained_layout=True)

    for ax, (x_col, title) in zip(axes, x_specs):
        fit_data = comparison.dropna(subset=[x_col, "separation_coefficient"])
        for system in SYSTEM_ORDER:
            sub = fit_data.loc[fit_data["system"].eq(system)]
            ax.scatter(
                sub[x_col],
                sub["separation_coefficient"],
                s=38,
                facecolor=SYSTEM_COLORS[system],
                edgecolor="black",
                linewidth=0.65,
                alpha=0.84,
                label=system,
                zorder=5,
            )
        add_fit_line(ax, fit_data, x_col, color="#111111", xlim=padded_extent(fit_data[x_col], frac=0.04))
        fit = quadratic_fit(
            fit_data[x_col].to_numpy(float),
            fit_data["separation_coefficient"].to_numpy(float),
        )
        ax.text(
            0.04,
            0.94,
            f"R2 = {fit['quad_r2']:.3f}\nvertex = {fit['quad_vertex_x']:.3f}",
            transform=ax.transAxes,
            fontsize=7.8,
            ha="left",
            va="top",
        )
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(2.0, 12.2)
        ax.grid(True, color="#e4e6eb", linewidth=0.7, zorder=0)
        ax.tick_params(axis="both", labelsize=8.2)
        ax.set_xlabel("Weighted electronegativity", fontsize=9.4)
    axes[0].set_ylabel("H/D separation coefficient", fontsize=9.4)
    axes[-1].legend(frameon=False, fontsize=7.1, loc="lower right")
    fig.suptitle("Comparison of three electronegativity weighting schemes", fontsize=12.0, fontweight="bold")

    png = output_dir / "electronegativity_three_weighting_comparison.png"
    pdf = output_dir / "electronegativity_three_weighting_comparison.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge ICP-OES actual molar ratios with Figure 4 BO data and test composition-volcano trends."
    )
    parser.add_argument("--icp", type=Path, default=DEFAULT_ICP, help="Path to ICP-OES xlsx file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bo = load_bo_table()
    icp = load_icp_table(args.icp)
    merged = bo.merge(icp, on="sample", how="left", indicator=True)
    missing = merged.loc[merged["_merge"].ne("both")].copy()
    matched = add_weighted_electronegativity(merged.loc[merged["_merge"].eq("both")].copy())

    matched.to_csv(args.output_dir / "bo_icp_actual_composition_merged.csv", index=False)
    missing[
        [
            "sample",
            "system",
            "row_id",
            "substance1",
            "volume1",
            "substance2",
            "volume2",
            "separation_coefficient",
            "nominal_frac1_from_precursor",
        ]
    ].to_csv(args.output_dir / "bo_missing_icp_matches.csv", index=False)

    unused_icp = icp.loc[~icp["sample"].isin(bo["sample"])].copy()
    unused_icp.to_csv(args.output_dir / "icp_rows_not_in_figure4_bo.csv", index=False)

    summary = build_summary(matched, missing)
    summary.to_csv(args.output_dir / "icp_actual_composition_volcano_summary.csv", index=False)
    electronegativity_summary = build_electronegativity_summary(matched)
    electronegativity_summary.to_csv(
        args.output_dir / "icp_weighted_electronegativity_volcano_summary.csv",
        index=False,
    )
    electronegativity_shift_summary = build_electronegativity_shift_summary(matched)
    electronegativity_shift_summary.to_csv(
        args.output_dir / "icp_weighted_electronegativity_shift_summary.csv",
        index=False,
    )
    deposition_calibration = fit_deposition_calibration(matched)
    deposition_calibration.to_csv(
        args.output_dir / "icp_deposition_bias_log_ratio_calibration.csv",
        index=False,
    )
    deposition_corrected_bo = build_deposition_calibrated_all_bo(bo, icp, deposition_calibration)
    deposition_corrected_bo.to_csv(
        args.output_dir / "bo_deposition_calibrated_composition_all_112.csv",
        index=False,
    )
    deposition_corrected_en_summary = build_deposition_corrected_electronegativity_summary(
        deposition_corrected_bo
    )
    deposition_corrected_en_summary.to_csv(
        args.output_dir / "deposition_corrected_weighted_electronegativity_volcano_summary.csv",
        index=False,
    )
    loading_effect_summary = build_loading_effect_summary(matched)
    loading_effect_summary.to_csv(
        args.output_dir / "icp_total_metal_loading_effect_summary.csv",
        index=False,
    )
    loading_adjusted_sensitivity = build_loading_adjusted_sensitivity(matched)
    loading_adjusted_sensitivity.to_csv(
        args.output_dir / "loading_adjusted_effective_electronegativity_sensitivity.csv",
        index=False,
    )
    best_loading_adjusted = loading_adjusted_sensitivity.loc[
        loading_adjusted_sensitivity["l0_umol"].gt(0)
    ].iloc[0]
    three_way_comparison = add_loading_adjusted_effective_electronegativity(
        matched,
        l0_umol=float(best_loading_adjusted["l0_umol"]),
        carbon_electronegativity=float(best_loading_adjusted["carbon_electronegativity"]),
    )
    three_way_comparison.to_csv(
        args.output_dir / "electronegativity_three_weighting_comparison_table.csv",
        index=False,
    )
    three_way_summary = build_three_electronegativity_comparison_summary(three_way_comparison)
    three_way_summary.to_csv(
        args.output_dir / "electronegativity_three_weighting_comparison_summary.csv",
        index=False,
    )
    draw_volcano_comparison(matched, args.output_dir)
    draw_electronegativity_comparison(matched, args.output_dir)
    draw_electronegativity_overall(matched, args.output_dir)
    draw_electronegativity_shift_overlay(matched, args.output_dir)
    draw_deposition_calibration(matched, deposition_calibration, args.output_dir)
    draw_deposition_corrected_electronegativity_volcano(deposition_corrected_bo, args.output_dir)
    draw_loading_effect(matched, args.output_dir)
    draw_three_electronegativity_comparison(three_way_comparison, args.output_dir)

    print(f"Matched ICP rows used for Figure 4 BO: {len(matched)}")
    print(f"BO rows missing ICP: {len(missing)}")
    print(f"ICP rows not used in Figure 4 BO: {len(unused_icp)}")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nWeighted electronegativity summary:")
    print(electronegativity_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nWeighted electronegativity shift summary:")
    print(electronegativity_shift_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nDeposition-bias log-ratio calibration:")
    print(deposition_calibration.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nDeposition-corrected weighted electronegativity volcano:")
    print(deposition_corrected_en_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nICP total metal loading effect:")
    print(loading_effect_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nThree electronegativity weighting comparison:")
    print(three_way_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nBest loading-adjusted sensitivity row:")
    print(best_loading_adjusted.to_frame().T.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
