from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs" / "right_censored_middle_lsv"
LSV_DIR = ROOT / "hd_separation_ml" / "data_extraction" / "all_data" / "LSV"

CENSORING_CSV = DATA_DIR / "figure1_this_work_eta10_censoring.csv"
THIS_WORK_CSV = DATA_DIR / "figure1_bottom_this_work_from_all_data.csv"
SUMMARY_CSV = OUTPUT_DIR / "right_censored_middle_lsv_summary.csv"
OVERLAY_PNG = OUTPUT_DIR / "right_censored_middle_lsv_overlay.png"
OVERLAY_PDF = OUTPUT_DIR / "right_censored_middle_lsv_overlay.pdf"
MULTIPANEL_PNG = OUTPUT_DIR / "right_censored_middle_lsv_multipanel.png"
MULTIPANEL_PDF = OUTPUT_DIR / "right_censored_middle_lsv_multipanel.pdf"

RHE_OFFSET_V = 0.9268
TARGET_J_MA_CM2 = -10.0
CAP_OVERPOTENTIAL_MV = 550.0

SOURCE_COLORS = {
    "initial_screening_561": "#7A7A7A",
    "bo_figure4_four_system_112": "#D62728",
    "bo_extra_co_fe_36": "#2563EB",
}
SOURCE_LABELS = {
    "initial_screening_561": "Initial screening",
    "bo_figure4_four_system_112": "BO four-system",
    "bo_extra_co_fe_36": "Co-Fe BO",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
        }
    )


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_lsv(sample: str, area_cm2: float) -> pd.DataFrame:
    frame = pd.read_csv(LSV_DIR / f"{sample}.txt", sep="\t", encoding="utf-8-sig")
    out = pd.DataFrame(
        {
            "E_raw_V": to_num(frame["Potential(V)"]),
            "I_A": to_num(frame["Current(A)"]),
        }
    ).dropna()
    out["E_RHE_V"] = out["E_raw_V"] + RHE_OFFSET_V
    out["j_mA_cm2"] = out["I_A"] * 1000.0 / area_cm2
    return out.reset_index(drop=True)


def load_middle_right_censored() -> pd.DataFrame:
    censoring = pd.read_csv(CENSORING_CSV)
    metadata = pd.read_csv(THIS_WORK_CSV)[
        ["sample", "LSV_geometry_area_cm2", "target_current_A_raw"]
    ]
    points = censoring.merge(metadata, on="sample", how="left", validate="one_to_one")
    points["plot_x_overpotential_mV"] = to_num(points["plot_x_overpotential_mV"])
    points["right_censored_eta_lower_bound_mV"] = to_num(
        points["right_censored_eta_lower_bound_mV"]
    )
    points["j_at_most_negative_E_mA_cm2"] = to_num(points["j_at_most_negative_E_mA_cm2"])
    points["separation_coefficient"] = to_num(points["separation_coefficient"])
    points = points[
        points["eta10_status"].eq("right_censored_non_reacher")
        & points["plot_x_overpotential_mV"].lt(CAP_OVERPOTENTIAL_MV)
    ].copy()
    return points.sort_values(["plot_x_overpotential_mV", "sample"])


def summarize_curves(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in points.iterrows():
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        endpoint = frame.iloc[-1]
        rows.append(
            {
                "sample": row["sample"],
                "source_type": row["source_type"],
                "eta10_status": row["eta10_status"],
                "right_censored_eta_lower_bound_mV": row[
                    "right_censored_eta_lower_bound_mV"
                ],
                "plot_x_overpotential_mV": row["plot_x_overpotential_mV"],
                "separation_coefficient": row["separation_coefficient"],
                "E_start_RHE_V": frame["E_RHE_V"].iloc[0],
                "E_end_RHE_V": endpoint["E_RHE_V"],
                "j_end_mA_cm2": endpoint["j_mA_cm2"],
                "j_min_mA_cm2": frame["j_mA_cm2"].min(),
                "j_max_mA_cm2": frame["j_mA_cm2"].max(),
                "n_lsv_points": len(frame),
            }
        )
    return pd.DataFrame(rows)


def plot_overlay(points: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    for _, row in points.iterrows():
        source = row["source_type"]
        color = SOURCE_COLORS.get(source, "#4B5563")
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        zorder = 4 if source != "initial_screening_561" else 3
        ax.plot(
            frame["E_RHE_V"],
            frame["j_mA_cm2"],
            color=color,
            alpha=0.78,
            linewidth=1.6,
            zorder=zorder,
        )
        ax.scatter(
            [frame["E_RHE_V"].iloc[-1]],
            [frame["j_mA_cm2"].iloc[-1]],
            marker="o",
            s=52,
            facecolors="white",
            edgecolors=color,
            linewidths=1.4,
            zorder=zorder + 1,
        )

    ax.axhline(TARGET_J_MA_CM2, color="black", linestyle="--", linewidth=1.15)
    ax.axvline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    ax.set_xlim(-0.39, 0.018)
    ax.set_ylim(-13.5, 2)
    ax.set_xlabel("Potential vs RHE (V)", fontweight="bold")
    ax.set_ylabel("Current density (mA cm$^{-2}$)", fontweight="bold")
    ax.set_title("Middle right-censored LSVs: eta10 > bound", fontweight="bold")
    ax.grid(True, color="#E5E7EB", linewidth=0.75)

    handles = [
        Line2D([0], [0], color=color, linewidth=2.4, label=SOURCE_LABELS[source])
        for source, color in SOURCE_COLORS.items()
        if source in set(points["source_type"])
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7.5,
                markerfacecolor="white",
                markeredgecolor="black",
                label="Measured endpoint",
            ),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.15, label="-10 mA cm$^{-2}$"),
        ]
    )
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(OVERLAY_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OVERLAY_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_multipanel(points: pd.DataFrame) -> None:
    cols = 4
    rows = int(np.ceil(len(points) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14.5, rows * 2.65), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, points.iterrows()):
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        color = SOURCE_COLORS.get(row["source_type"], "#4B5563")
        endpoint = frame.iloc[-1]
        ax.plot(frame["E_RHE_V"], frame["j_mA_cm2"], color=color, linewidth=1.55)
        ax.scatter(
            [endpoint["E_RHE_V"]],
            [endpoint["j_mA_cm2"]],
            marker="o",
            s=42,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=4,
        )
        ax.axhline(TARGET_J_MA_CM2, color="black", linestyle="--", linewidth=0.85)
        ax.axvline(endpoint["E_RHE_V"], color=color, linestyle=":", linewidth=0.8)
        ax.set_xlim(frame["E_RHE_V"].min() - 0.012, 0.018)
        ax.set_ylim(min(frame["j_mA_cm2"].min() - 1.2, -12.5), 1.5)
        ax.set_title(
            f"{row['sample']}  >{row['plot_x_overpotential_mV']:.0f} mV",
            fontsize=9.7,
            fontweight="bold",
            pad=3,
        )
        ax.text(
            0.03,
            0.08,
            f"j_end={endpoint['j_mA_cm2']:.2f}",
            transform=ax.transAxes,
            fontsize=8.2,
        )
        ax.tick_params(axis="both", labelsize=8, length=3)
        ax.grid(True, color="#EBEEF2", linewidth=0.6)

    for ax in axes[len(points) :]:
        ax.axis("off")

    fig.supxlabel("Potential vs RHE (V)", y=0.012, fontsize=13, fontweight="bold")
    fig.supylabel("Current density (mA cm$^{-2}$)", x=0.006, fontsize=13, fontweight="bold")
    fig.suptitle(
        "This-work middle right-censored LSVs",
        y=0.998,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.98))
    fig.savefig(MULTIPANEL_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(MULTIPANEL_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    points = load_middle_right_censored()
    summary = summarize_curves(points)
    summary.to_csv(SUMMARY_CSV, index=False)
    plot_multipanel(points)

    print(f"Middle right-censored samples: {len(points)}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {MULTIPANEL_PNG}")


if __name__ == "__main__":
    main()
