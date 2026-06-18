from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs" / "eta_below_10_lsv"
LSV_DIR = ROOT / "hd_separation_ml" / "data_extraction" / "all_data" / "LSV"

CENSORING_CSV = DATA_DIR / "figure1_this_work_eta10_censoring.csv"
THIS_WORK_CSV = DATA_DIR / "figure1_bottom_this_work_from_all_data.csv"
SUMMARY_CSV = OUTPUT_DIR / "eta_below_10_lsv_summary.csv"
NEAR_TARGET_CSV = OUTPUT_DIR / "eta_below_10_near_target_eta_estimates.csv"
OVERLAY_PNG = OUTPUT_DIR / "eta_below_10_lsv_overlay.png"
OVERLAY_PDF = OUTPUT_DIR / "eta_below_10_lsv_overlay.pdf"
MULTIPANEL_PNG = OUTPUT_DIR / "eta_below_10_lsv_multipanel.png"
MULTIPANEL_PDF = OUTPUT_DIR / "eta_below_10_lsv_multipanel.pdf"
REVIEW_PNG = OUTPUT_DIR / "eta_below_10_near_target_review_lsv.png"
REVIEW_PDF = OUTPUT_DIR / "eta_below_10_near_target_review_lsv.pdf"
CURRENT_NON_HER_SUMMARY_CSV = OUTPUT_DIR / "current_eta_below_10_non_her_lsv_summary.csv"
CURRENT_NON_HER_PNG = OUTPUT_DIR / "current_eta_below_10_non_her_lsv_multipanel.png"
CURRENT_NON_HER_PDF = OUTPUT_DIR / "current_eta_below_10_non_her_lsv_multipanel.pdf"

RHE_OFFSET_V = 0.9268
TARGET_J_MA_CM2 = -10.0
ETA_THRESHOLD_MV = 10.0
NEAR_TARGET_TOLERANCE_MA_CM2 = 2.0
MAX_ESTIMATE_DELTA_MA_CM2 = 10.0
BORDERLINE_RECOVERED_SAMPLES = {"Co-V8", "Co-V11", "Co-V20", "Co-V22"}
TURNING_POINT_ESTIMATE_SAMPLES = {
    "Fe-Pt",
    "Pd-Pd",
    "Y-Pd",
    "Co-Pt",
    "W-Pd",
}
CHARGING_CURRENT_FORCED_CAP_SAMPLES = {"Sc-Pb", "Cd-Y", "W-Pb", "Ce-Pb"}
REVIEW_SAMPLES = [
    "Co-V10",
    "Co-V12",
    "Co-V8",
    "Co-V24",
    "Co-V20",
    "Co-V13",
    "Co-V11",
    "Co-V14",
    "Co-V22",
    "Fe-Pt",
    "Pd-Pd",
    "Y-Pd",
    "Co-Pt",
    "Sc-Pd",
    "W-Pd",
    "Co-Fe29",
    "Co-Fe33",
    "Sc-Pb",
    "Cd-Y",
    "W-Pb",
    "Ce-Pb",
]

SOURCE_COLORS = {
    "initial_screening_561": "#7A7A7A",
    "bo_figure4_four_system_112": "#D62728",
    "bo_extra_co_fe_36": "#2563EB",
}
SOURCE_LABELS = {
    "initial_screening_561": "Initial screening",
    "bo_figure4_four_system_112": "Co-V BO",
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


def load_eta_below_10() -> pd.DataFrame:
    censoring = pd.read_csv(CENSORING_CSV)
    metadata = pd.read_csv(THIS_WORK_CSV)[
        [
            "sample",
            "LSV_geometry_area_cm2",
            "target_current_A_raw",
            "eta10_mV_vs_RHE_full_lsv",
        ]
    ]
    points = censoring.merge(metadata, on="sample", how="left", validate="one_to_one")
    points["eta10_or_censor_bound_uncapped_mV"] = to_num(
        points["eta10_or_censor_bound_uncapped_mV"]
    )
    points = points[
        points["eta10_or_censor_bound_uncapped_mV"] < ETA_THRESHOLD_MV
    ].copy()
    return points.sort_values(["separation_coefficient", "sample"], ascending=[False, True])


def load_current_non_her_lower_bound() -> pd.DataFrame:
    points = load_eta_below_10()
    return points[points["eta10_status"].eq("left_censored_already_active")].copy()


def summarize_curves(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in points.iterrows():
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        closest_idx = int((frame["j_mA_cm2"] - TARGET_J_MA_CM2).abs().idxmin())
        closest = frame.loc[closest_idx]
        closest_eta_mV = -float(closest["E_RHE_V"]) * 1000.0
        closest_delta_mA_cm2 = abs(float(closest["j_mA_cm2"]) - TARGET_J_MA_CM2)
        is_near_target = closest_delta_mA_cm2 <= NEAR_TARGET_TOLERANCE_MA_CM2
        is_borderline_recovered = row["sample"] in BORDERLINE_RECOVERED_SAMPLES
        is_turning_point_estimate = row["sample"] in TURNING_POINT_ESTIMATE_SAMPLES
        is_charging_current_forced_cap = row["sample"] in CHARGING_CURRENT_FORCED_CAP_SAMPLES
        estimate_delta_ok = closest_delta_mA_cm2 < MAX_ESTIMATE_DELTA_MA_CM2
        has_reportable_eta_estimate = (
            not is_charging_current_forced_cap
            and estimate_delta_ok
            and closest_eta_mV >= ETA_THRESHOLD_MV
            and (is_near_target or is_borderline_recovered or is_turning_point_estimate)
        )
        if is_charging_current_forced_cap:
            recommended_figure1_x_mV = 550.0
            recommendation = "charging_current_forced_cap"
        elif is_turning_point_estimate and has_reportable_eta_estimate:
            recommended_figure1_x_mV = closest_eta_mV
            recommendation = "turning_point_eta_estimate"
        elif has_reportable_eta_estimate:
            recommended_figure1_x_mV = closest_eta_mV
            recommendation = (
                "borderline_recovered_near_target_estimate"
                if is_borderline_recovered and not is_near_target
                else "recovered_near_target_estimate"
            )
        else:
            recommended_figure1_x_mV = row["plot_x_overpotential_mV"]
            recommendation = "keep_lower_bound_non_her_current"
        rows.append(
            {
                "sample": row["sample"],
                "source_type": row["source_type"],
                "eta10_status": row["eta10_status"],
                "eta10_or_censor_bound_uncapped_mV": row[
                    "eta10_or_censor_bound_uncapped_mV"
                ],
                "plot_x_overpotential_mV": row["plot_x_overpotential_mV"],
                "separation_coefficient": row["separation_coefficient"],
                "legacy_nearest_point_eta10_mV": row["eta10_mV_vs_RHE_full_lsv"],
                "closest_to_target_eta_mV": closest_eta_mV,
                "closest_to_target_E_RHE_V": closest["E_RHE_V"],
                "closest_to_target_j_mA_cm2": closest["j_mA_cm2"],
                "closest_to_target_abs_delta_mA_cm2": closest_delta_mA_cm2,
                "closest_to_target_index": closest_idx,
                "within_2mA_of_target": is_near_target,
                "borderline_recovered_sample": is_borderline_recovered,
                "turning_point_estimate_sample": is_turning_point_estimate,
                "charging_current_forced_cap": is_charging_current_forced_cap,
                "has_reportable_near_target_eta_estimate": has_reportable_eta_estimate,
                "recommended_figure1_x_mV": recommended_figure1_x_mV,
                "eta10_recovery_recommendation": recommendation,
                "E_start_RHE_V": frame["E_RHE_V"].iloc[0],
                "E_end_RHE_V": frame["E_RHE_V"].iloc[-1],
                "j_start_mA_cm2": frame["j_mA_cm2"].iloc[0],
                "j_min_mA_cm2": frame["j_mA_cm2"].min(),
                "j_max_mA_cm2": frame["j_mA_cm2"].max(),
                "n_lsv_points": len(frame),
            }
        )
    return pd.DataFrame(rows)


def plot_overlay(points: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.5))
    full_ax, zoom_ax = axes

    for _, row in points.iterrows():
        source = row["source_type"]
        color = SOURCE_COLORS.get(source, "#4B5563")
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        zorder = 4 if source != "initial_screening_561" else 3
        alpha = 0.82 if source != "initial_screening_561" else 0.36
        width = 1.5 if source != "initial_screening_561" else 1.0
        for ax in axes:
            ax.plot(
                frame["E_RHE_V"],
                frame["j_mA_cm2"],
                color=color,
                alpha=alpha,
                linewidth=width,
                zorder=zorder,
            )
            ax.scatter(
                [frame["E_RHE_V"].iloc[0]],
                [frame["j_mA_cm2"].iloc[0]],
                s=10,
                color=color,
                alpha=min(alpha + 0.15, 1.0),
                zorder=zorder + 1,
            )

    for ax in axes:
        ax.axhline(
            TARGET_J_MA_CM2,
            color="black",
            linestyle="--",
            linewidth=1.1,
            zorder=2,
        )
        ax.axvline(0.0, color="#555555", linestyle=":", linewidth=0.9, zorder=2)
        ax.set_xlabel("Potential vs RHE (V)", fontweight="bold")
        ax.grid(True, color="#E5E7EB", linewidth=0.75)

    full_ax.set_xlim(-0.59, 0.02)
    full_ax.set_ylim(-370, 8)
    full_ax.set_ylabel("Current density (mA cm$^{-2}$)", fontweight="bold")
    full_ax.set_title("Full LSV traces, eta/bound < 10 mV", fontweight="bold")

    zoom_ax.set_xlim(-0.035, 0.012)
    zoom_ax.set_ylim(-65, 5)
    zoom_ax.set_title("Zoom near target current", fontweight="bold")

    handles = [
        Line2D([0], [0], color=color, linewidth=2.4, label=SOURCE_LABELS[source])
        for source, color in SOURCE_COLORS.items()
        if source in set(points["source_type"])
    ]
    handles.extend(
        [
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.1, label="-10 mA cm$^{-2}$"),
            Line2D([0], [0], color="#555555", linestyle=":", linewidth=0.9, label="0 V vs RHE"),
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
        fontsize=10.5,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OVERLAY_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OVERLAY_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_multipanel(points: pd.DataFrame) -> None:
    cols = 5
    rows = int(np.ceil(len(points) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.35), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, points.iterrows()):
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        color = SOURCE_COLORS.get(row["source_type"], "#4B5563")
        ax.plot(frame["E_RHE_V"], frame["j_mA_cm2"], color=color, linewidth=1.5)
        ax.scatter(
            [frame["E_RHE_V"].iloc[0]],
            [frame["j_mA_cm2"].iloc[0]],
            s=15,
            color="#F59E0B",
            edgecolors="black",
            linewidths=0.3,
            zorder=4,
        )
        ax.axhline(TARGET_J_MA_CM2, color="black", linestyle="--", linewidth=0.85)
        ax.axvline(0.0, color="#555555", linestyle=":", linewidth=0.75)
        ax.set_xlim(frame["E_RHE_V"].min() - 0.01, 0.018)
        y_min = min(frame["j_mA_cm2"].min() * 1.04, TARGET_J_MA_CM2 - 5)
        ax.set_ylim(y_min, 6)
        ax.set_title(
            f"{row['sample']}  sep={row['separation_coefficient']:.2f}",
            fontsize=9.5,
            fontweight="bold",
            pad=3,
        )
        ax.tick_params(axis="both", labelsize=8, length=3)
        ax.grid(True, color="#EBEEF2", linewidth=0.6)

    for ax in axes[len(points) :]:
        ax.axis("off")

    fig.supxlabel("Potential vs RHE (V)", y=0.012, fontsize=13, fontweight="bold")
    fig.supylabel("Current density (mA cm$^{-2}$)", x=0.005, fontsize=13, fontweight="bold")
    fig.suptitle(
        "This-work LSVs with eta/bound < 10 mV",
        y=0.998,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.018, 0.028, 1, 0.982))
    fig.savefig(MULTIPANEL_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(MULTIPANEL_PDF, bbox_inches="tight")
    plt.close(fig)


def save_near_target_audit(summary: pd.DataFrame) -> pd.DataFrame:
    audit = summary.sort_values(
        ["closest_to_target_abs_delta_mA_cm2", "sample"],
        ascending=[True, True],
    ).copy()
    audit.to_csv(NEAR_TARGET_CSV, index=False)
    return audit


def plot_review_samples(points: pd.DataFrame) -> None:
    review = points[points["sample"].isin(REVIEW_SAMPLES)].copy()
    review["review_order"] = review["sample"].map(
        {sample: idx for idx, sample in enumerate(REVIEW_SAMPLES)}
    )
    review = review.sort_values("review_order")
    if review.empty:
        return

    cols = 4
    rows = int(np.ceil(len(review) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16.2, rows * 2.8), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, review.iterrows()):
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        closest_idx = int((frame["j_mA_cm2"] - TARGET_J_MA_CM2).abs().idxmin())
        closest = frame.loc[closest_idx]
        closest_eta_mV = -float(closest["E_RHE_V"]) * 1000.0
        closest_delta_mA_cm2 = abs(float(closest["j_mA_cm2"]) - TARGET_J_MA_CM2)
        near_target = closest_delta_mA_cm2 <= NEAR_TARGET_TOLERANCE_MA_CM2
        borderline = row["sample"] in BORDERLINE_RECOVERED_SAMPLES
        turning_point = row["sample"] in TURNING_POINT_ESTIMATE_SAMPLES
        forced_cap = row["sample"] in CHARGING_CURRENT_FORCED_CAP_SAMPLES
        estimate_delta_ok = closest_delta_mA_cm2 < MAX_ESTIMATE_DELTA_MA_CM2
        color = (
            "#111827"
            if forced_cap
            else (
                "#7C3AED"
                if turning_point and estimate_delta_ok
                else ("#D62728" if estimate_delta_ok and (near_target or borderline) else "#6B7280")
            )
        )

        ax.plot(frame["E_RHE_V"], frame["j_mA_cm2"], color="#2563EB", linewidth=1.45)
        ax.scatter(
            [closest["E_RHE_V"]],
            [closest["j_mA_cm2"]],
            s=42,
            color=color,
            edgecolors="black",
            linewidths=0.45,
            zorder=4,
        )
        ax.axhline(TARGET_J_MA_CM2, color="black", linestyle="--", linewidth=0.85)
        ax.axvline(closest["E_RHE_V"], color=color, linestyle=":", linewidth=0.85)
        ax.set_xlim(-0.095, 0.015)
        y_low = min(float(frame.loc[frame["E_RHE_V"].between(-0.10, 0.02), "j_mA_cm2"].min()) - 4, -28)
        ax.set_ylim(y_low, 2)
        ax.grid(True, color="#E5E7EB", linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8, length=3)
        ax.set_title(
            f"{row['sample']}  eta~{closest_eta_mV:.1f} mV\n"
            f"j={closest['j_mA_cm2']:.2f}, delta={closest_delta_mA_cm2:.2f}",
            fontsize=9.0,
            fontweight="bold",
            pad=3,
        )

    for ax in axes[len(review) :]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="#2563EB", linewidth=2.0, label="LSV"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#D62728",
            markeredgecolor="black",
            markersize=6,
            label="Recovered eta estimate",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#7C3AED",
            markeredgecolor="black",
            markersize=6,
            label="Turning-point eta estimate",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6B7280",
            markeredgecolor="black",
            markersize=6,
            label="Closest point outside tolerance",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111827",
            markeredgecolor="black",
            markersize=6,
            label="Charging current, forced cap",
        ),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label="-10 mA cm$^{-2}$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=True,
        bbox_to_anchor=(0.5, 1.015),
        fontsize=10,
        columnspacing=1.0,
        handlelength=1.6,
    )
    fig.supxlabel("Potential vs RHE (V)", y=0.012, fontsize=13, fontweight="bold")
    fig.supylabel("Current density (mA cm$^{-2}$)", x=0.006, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.95))
    fig.savefig(REVIEW_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(REVIEW_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_current_non_her_multipanel(points: pd.DataFrame) -> None:
    cols = 5
    rows = int(np.ceil(len(points) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.45), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, points.iterrows()):
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        closest_idx = int((frame["j_mA_cm2"] - TARGET_J_MA_CM2).abs().idxmin())
        closest = frame.loc[closest_idx]
        closest_eta_mV = -float(closest["E_RHE_V"]) * 1000.0
        closest_delta_mA_cm2 = abs(float(closest["j_mA_cm2"]) - TARGET_J_MA_CM2)
        color = SOURCE_COLORS.get(row["source_type"], "#4B5563")

        ax.plot(frame["E_RHE_V"], frame["j_mA_cm2"], color=color, linewidth=1.45)
        ax.scatter(
            [frame["E_RHE_V"].iloc[0]],
            [frame["j_mA_cm2"].iloc[0]],
            s=18,
            color="#F59E0B",
            edgecolors="black",
            linewidths=0.35,
            zorder=4,
        )
        ax.scatter(
            [closest["E_RHE_V"]],
            [closest["j_mA_cm2"]],
            s=34,
            color="#6B7280",
            edgecolors="black",
            linewidths=0.45,
            zorder=5,
        )
        ax.axhline(TARGET_J_MA_CM2, color="black", linestyle="--", linewidth=0.85)
        ax.axvline(0.0, color="#555555", linestyle=":", linewidth=0.75)
        ax.axvline(closest["E_RHE_V"], color="#6B7280", linestyle=":", linewidth=0.7)
        ax.set_xlim(frame["E_RHE_V"].min() - 0.01, 0.018)
        y_min = min(frame["j_mA_cm2"].min() * 1.04, TARGET_J_MA_CM2 - 5)
        ax.set_ylim(y_min, 6)
        ax.set_title(
            f"{row['sample']}  HD={row['separation_coefficient']:.2f}\n"
            f"closest j={closest['j_mA_cm2']:.1f}, eta~{closest_eta_mV:.1f} mV, "
            f"delta={closest_delta_mA_cm2:.1f}",
            fontsize=8.7,
            fontweight="bold",
            pad=3,
        )
        ax.tick_params(axis="both", labelsize=7.8, length=3)
        ax.grid(True, color="#EBEEF2", linewidth=0.6)

    for ax in axes[len(points) :]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label="-10 mA cm$^{-2}$"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#F59E0B",
            markeredgecolor="black",
            markersize=5.5,
            label="Scan start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6B7280",
            markeredgecolor="black",
            markersize=5.5,
            label="Closest point to -10",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.045),
        fontsize=10,
        columnspacing=1.0,
        handlelength=1.6,
    )
    fig.supxlabel("Potential vs RHE (V)", y=0.012, fontsize=13, fontweight="bold")
    fig.supylabel("Current density (mA cm$^{-2}$)", x=0.005, fontsize=13, fontweight="bold")
    fig.suptitle(
        "Current eta < 10 mV lower-bound / non-HER-current LSVs",
        y=0.988,
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.018, 0.028, 1, 0.935))
    fig.savefig(CURRENT_NON_HER_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(CURRENT_NON_HER_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    points = load_eta_below_10()
    current_non_her = load_current_non_her_lower_bound()
    summary = summarize_curves(points)
    current_non_her_summary = summarize_curves(current_non_her)
    current_non_her_summary.to_csv(CURRENT_NON_HER_SUMMARY_CSV, index=False)
    near_target = summary.sort_values(
        ["closest_to_target_abs_delta_mA_cm2", "sample"],
        ascending=[True, True],
    )
    plot_current_non_her_multipanel(current_non_her)

    print(f"Samples with eta/bound < {ETA_THRESHOLD_MV:g} mV: {len(points)}")
    print(f"Current eta < {ETA_THRESHOLD_MV:g} mV lower-bound samples: {len(current_non_her)}")
    print(
        "Within 2 mA cm^-2 of -10 mA cm^-2: "
        f"{int(near_target['within_2mA_of_target'].sum())}"
    )
    print(
        "Reportable near-target eta estimates >= 10 mV: "
        f"{int(near_target['has_reportable_near_target_eta_estimate'].sum())}"
    )
    print(
        "Charging-current forced-cap samples: "
        f"{int(near_target['charging_current_forced_cap'].sum())}"
    )
    print(f"Saved: {CURRENT_NON_HER_SUMMARY_CSV}")
    print(f"Saved: {CURRENT_NON_HER_PNG}")


if __name__ == "__main__":
    main()
