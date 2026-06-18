from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs" / "abnormal_eta10_lsv_audit"
LSV_DIR = ROOT / "hd_separation_ml" / "data_extraction" / "all_data" / "LSV"

THIS_WORK_CSV = DATA_DIR / "figure1_bottom_this_work_from_all_data.csv"
CENSORING_CSV = DATA_DIR / "figure1_this_work_eta10_censoring.csv"
LSV_AUDIT_CSV = HERE / "outputs" / "lsv_target_failures" / "lsv_target_reach_audit_all_709.csv"

SUMMARY_CSV = OUTPUT_DIR / "abnormal_eta10_lsv_audit_summary.csv"
ENDPOINT_GATE_PNG = OUTPUT_DIR / "abnormal_eta10_endpoint_gate_lsv.png"
ENDPOINT_GATE_PDF = OUTPUT_DIR / "abnormal_eta10_endpoint_gate_lsv.pdf"
START_ACTIVE_PNG = OUTPUT_DIR / "abnormal_eta10_start_active_lsv.png"
START_ACTIVE_PDF = OUTPUT_DIR / "abnormal_eta10_start_active_lsv.pdf"
TURNING_POINT_PNG = OUTPUT_DIR / "abnormal_eta10_turning_point_estimates_lsv.png"
TURNING_POINT_PDF = OUTPUT_DIR / "abnormal_eta10_turning_point_estimates_lsv.pdf"

RHE_OFFSET_V = 0.9268
TARGET_J_MA_CM2 = -10.0
ENDPOINT_NEAR_TARGET_TOLERANCE_MA_CM2 = 2.0
MIN_OVERPOTENTIAL_MV = 10.0
CAP_OVERPOTENTIAL_MV = 550.0

STATUS_LABELS = {
    "endpoint_gate_endpoint_bound": "endpoint bound",
    "endpoint_gate_cap": "endpoint cap",
    "lower_bound_non_her": "10 mV lower bound",
    "near_target_estimate": "near-target estimate",
    "turning_point_estimate": "turning-point estimate",
}

STATUS_COLORS = {
    "endpoint_gate_endpoint_bound": "#0F766E",
    "endpoint_gate_cap": "#111827",
    "lower_bound_non_her": "#6B7280",
    "near_target_estimate": "#D62728",
    "turning_point_estimate": "#7C3AED",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
        }
    )


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


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


def load_abnormal_rows() -> pd.DataFrame:
    this_work = pd.read_csv(THIS_WORK_CSV)
    censoring = pd.read_csv(CENSORING_CSV)
    audit = pd.read_csv(LSV_AUDIT_CSV)

    audit["endpoint_does_not_reach_target"] = to_bool(
        audit["endpoint_does_not_reach_target"]
    )
    audit["never_reaches_target_full_trace"] = to_bool(
        audit["never_reaches_target_full_trace"]
    )
    audit["has_target_crossing_full_lsv"] = to_bool(
        audit["has_target_crossing_full_lsv"]
    )

    cols = [
        "sample",
        "source_type",
        "LSV_geometry_area_cm2",
        "separation_coefficient",
    ]
    rows = (
        this_work[cols]
        .merge(
            audit[
                [
                    "sample",
                    "endpoint_does_not_reach_target",
                    "never_reaches_target_full_trace",
                    "has_target_crossing_full_lsv",
                    "j_at_most_negative_E_mA_cm2",
                    "minimum_j_mA_cm2",
                    "nearest_j_to_target_mA_cm2",
                    "E_at_nearest_j_raw_V",
                    "most_negative_E_vs_RHE_V",
                    "table_plot_x_overpotential_mV",
                ]
            ],
            on="sample",
            how="left",
            validate="one_to_one",
        )
        .merge(
            censoring[
                [
                    "sample",
                    "eta10_status",
                    "plot_x_overpotential_mV",
                    "nearest_to_target_eta_estimate_mV",
                    "nearest_to_target_abs_delta_mA_cm2",
                ]
            ],
            on="sample",
            how="left",
            validate="one_to_one",
        )
    )

    endpoint_gate = rows[
        rows["endpoint_does_not_reach_target"]
        & ~rows["never_reaches_target_full_trace"]
    ].copy()
    endpoint_delta = (
        to_num(endpoint_gate["j_at_most_negative_E_mA_cm2"]) - TARGET_J_MA_CM2
    ).abs()
    endpoint_eta_mV = -to_num(endpoint_gate["most_negative_E_vs_RHE_V"]) * 1000.0
    endpoint_gate["audit_family"] = "endpoint_gate_fail"
    endpoint_gate["extraction_rule"] = np.where(
        endpoint_delta <= ENDPOINT_NEAR_TARGET_TOLERANCE_MA_CM2,
        "endpoint_gate_endpoint_bound",
        "endpoint_gate_cap",
    )
    endpoint_gate["assigned_eta_mV"] = np.where(
        endpoint_gate["extraction_rule"].eq("endpoint_gate_endpoint_bound"),
        endpoint_eta_mV,
        CAP_OVERPOTENTIAL_MV,
    )
    endpoint_gate["assigned_plot_x_mV"] = pd.Series(
        endpoint_gate["assigned_eta_mV"], index=endpoint_gate.index
    ).clip(lower=MIN_OVERPOTENTIAL_MV, upper=CAP_OVERPOTENTIAL_MV)
    endpoint_gate["endpoint_delta_to_target_mA_cm2"] = endpoint_delta

    start_active_statuses = {
        "left_censored_already_active": "lower_bound_non_her",
        "left_censored_near_target_estimate": "near_target_estimate",
        "left_censored_turning_point_estimate": "turning_point_estimate",
    }
    start_active = rows[rows["eta10_status"].isin(start_active_statuses)].copy()
    start_active["audit_family"] = "start_active_or_no_strict_crossing"
    start_active["extraction_rule"] = start_active["eta10_status"].map(
        start_active_statuses
    )
    start_active["assigned_eta_mV"] = start_active["plot_x_overpotential_mV"]
    start_active["assigned_plot_x_mV"] = start_active["plot_x_overpotential_mV"]
    start_active["endpoint_delta_to_target_mA_cm2"] = (
        to_num(start_active["j_at_most_negative_E_mA_cm2"]) - TARGET_J_MA_CM2
    ).abs()

    out = pd.concat([endpoint_gate, start_active], ignore_index=True)
    out["assigned_E_RHE_V"] = -to_num(out["assigned_plot_x_mV"]) / 1000.0
    out["nearest_E_RHE_V"] = to_num(out["E_at_nearest_j_raw_V"]) + RHE_OFFSET_V
    out["nearest_delta_to_target_mA_cm2"] = (
        to_num(out["nearest_j_to_target_mA_cm2"]) - TARGET_J_MA_CM2
    ).abs()
    out["extraction_rule_label"] = out["extraction_rule"].map(STATUS_LABELS)
    return out.sort_values(
        ["audit_family", "extraction_rule", "separation_coefficient", "sample"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def panel_title(row: pd.Series) -> str:
    assigned = float(row["assigned_plot_x_mV"])
    endpoint_j = float(row["j_at_most_negative_E_mA_cm2"])
    nearest_j = float(row["nearest_j_to_target_mA_cm2"])
    return (
        f"{row['sample']}  HD={row['separation_coefficient']:.2f}\n"
        f"{row['extraction_rule_label']}, x={assigned:.1f} mV; "
        f"end j={endpoint_j:.1f}; near j={nearest_j:.1f}"
    )


def plot_group(rows: pd.DataFrame, output_png: Path, output_pdf: Path, title: str) -> None:
    if rows.empty:
        return

    cols = 5
    n_rows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(
        n_rows,
        cols,
        figsize=(18.4, n_rows * 2.55),
        sharex=False,
        sharey=False,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, rows.iterrows()):
        color = STATUS_COLORS.get(row["extraction_rule"], "#6B7280")
        frame = read_lsv(row["sample"], float(row["LSV_geometry_area_cm2"]))
        assigned_e = float(row["assigned_E_RHE_V"])
        nearest_e = float(row["nearest_E_RHE_V"])

        ax.plot(frame["E_RHE_V"], frame["j_mA_cm2"], color=color, linewidth=1.45)
        ax.scatter(
            [frame["E_RHE_V"].iloc[0]],
            [frame["j_mA_cm2"].iloc[0]],
            s=16,
            color="#F59E0B",
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )
        ax.scatter(
            [frame["E_RHE_V"].iloc[-1]],
            [frame["j_mA_cm2"].iloc[-1]],
            s=24,
            marker="s",
            color="#111827",
            edgecolors="black",
            linewidths=0.35,
            zorder=5,
        )
        ax.scatter(
            [nearest_e],
            [row["nearest_j_to_target_mA_cm2"]],
            s=30,
            color="#FFFFFF",
            edgecolors=color,
            linewidths=1.0,
            zorder=6,
        )
        ax.axhline(TARGET_J_MA_CM2, color="black", linestyle="--", linewidth=0.85)
        ax.axvline(assigned_e, color=color, linestyle="-", linewidth=0.95)
        ax.axvline(nearest_e, color="#6B7280", linestyle=":", linewidth=0.75)
        ax.axvline(0.0, color="#555555", linestyle=":", linewidth=0.6)
        ax.set_xlim(frame["E_RHE_V"].min() - 0.012, 0.018)
        y_min = min(float(frame["j_mA_cm2"].min()) * 1.04, TARGET_J_MA_CM2 - 5)
        y_max = max(5.0, float(frame["j_mA_cm2"].max()) + 3.0)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, color="#EBEEF2", linewidth=0.6)
        ax.tick_params(axis="both", labelsize=7.8, length=3)
        ax.set_title(panel_title(row), fontsize=8.4, fontweight="bold", pad=3)

    for ax in axes[len(rows) :]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label="-10 mA cm$^{-2}$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F59E0B", markeredgecolor="black", markersize=5.5, label="Scan start"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#111827", markeredgecolor="black", markersize=5.5, label="Endpoint"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#6B7280", markersize=5.5, label="Closest point to -10"),
        Line2D([0], [0], color="#6B7280", linestyle=":", linewidth=0.9, label="Closest-point E"),
    ]
    for rule in STATUS_LABELS:
        if rule in set(rows["extraction_rule"]):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=STATUS_COLORS[rule],
                    linewidth=1.35,
                    label=STATUS_LABELS[rule],
                )
            )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 7),
        frameon=True,
        bbox_to_anchor=(0.5, 1.018),
        fontsize=9.2,
        columnspacing=0.8,
        handlelength=1.35,
    )
    fig.supxlabel("Potential vs RHE (V)", y=0.01, fontsize=13, fontweight="bold")
    fig.supylabel("Current density (mA cm$^{-2}$)", x=0.005, fontsize=13, fontweight="bold")
    fig.suptitle(title, y=0.992, fontsize=15.5, fontweight="bold")
    fig.tight_layout(rect=(0.018, 0.03, 1, 0.955))
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    abnormal = load_abnormal_rows()
    abnormal.to_csv(SUMMARY_CSV, index=False)

    endpoint_gate = abnormal[abnormal["audit_family"].eq("endpoint_gate_fail")].copy()
    start_active = abnormal[
        abnormal["audit_family"].eq("start_active_or_no_strict_crossing")
    ].copy()

    plot_group(
        endpoint_gate,
        ENDPOINT_GATE_PNG,
        ENDPOINT_GATE_PDF,
        "Abnormal eta10: endpoint gate fails but target is reached elsewhere",
    )
    plot_group(
        start_active,
        START_ACTIVE_PNG,
        START_ACTIVE_PDF,
        "Abnormal eta10: endpoint reaches target but strict crossing is unreliable",
    )
    plot_group(
        start_active[start_active["extraction_rule"].eq("turning_point_estimate")]
        .sort_values("assigned_plot_x_mV")
        .reset_index(drop=True),
        TURNING_POINT_PNG,
        TURNING_POINT_PDF,
        "Recovered turning-point eta10 estimates",
    )

    print(f"Saved: {SUMMARY_CSV}")
    print(f"Endpoint-gate abnormal samples: {len(endpoint_gate)}")
    print(f"Start-active/no-strict-crossing abnormal samples: {len(start_active)}")
    print(f"Saved: {ENDPOINT_GATE_PNG}")
    print(f"Saved: {START_ACTIVE_PNG}")
    print(f"Saved: {TURNING_POINT_PNG}")


if __name__ == "__main__":
    main()
