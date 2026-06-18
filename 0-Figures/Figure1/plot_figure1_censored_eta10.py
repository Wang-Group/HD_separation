from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "outputs"
LSV_DIR = ROOT / "hd_separation_ml" / "data_extraction" / "all_data" / "LSV"

THIS_WORK_CSV = DATA_DIR / "figure1_bottom_this_work_from_all_data.csv"
LITERATURE_CSV = DATA_DIR / "figure1_bottom_literature_benchmarks.csv"
LSV_AUDIT_CSV = OUTPUT_DIR / "lsv_target_failures" / "lsv_target_reach_audit_all_709.csv"

CLASSIFIED_CSV = DATA_DIR / "figure1_this_work_eta10_censoring.csv"
THIS_WORK_FRONT_CSV = OUTPUT_DIR / "this_work_pareto_front_points_censored_eta10.csv"
PRE_FRONT_CSV = OUTPUT_DIR / "figure1_pre_pareto_front_censored_eta10.csv"
GLOBAL_FRONT_CSV = OUTPUT_DIR / "figure1_global_pareto_front_censored_eta10.csv"
MAIN_PNG = OUTPUT_DIR / "pareto_plot_v2_censored_eta10.png"
MAIN_PDF = OUTPUT_DIR / "pareto_plot_v2_censored_eta10.pdf"
MAIN_NO_TEXT_PNG = OUTPUT_DIR / "pareto_plot_v2_censored_eta10_no_text.png"
MAIN_NO_TEXT_PDF = OUTPUT_DIR / "pareto_plot_v2_censored_eta10_no_text.pdf"

RHE_OFFSET_V = 0.9268
TARGET_J_MA_CM2 = -10.0
RIGHT_CENSORED_NEAR_TARGET_TOLERANCE_MA_CM2 = 2.0
LEFT_CENSORED_NEAR_TARGET_TOLERANCE_MA_CM2 = 2.0
LEFT_CENSORED_MAX_ESTIMATE_DELTA_MA_CM2 = 10.0
MIN_OVERPOTENTIAL_MV = 10.0
CAP_OVERPOTENTIAL_MV = 550.0
EARLY_CROSSING_WINDOW_V = 0.020
EARLY_CROSSING_MAX_ETA_MV = 20.0
EXTRAPOLATED_LEFT_SAMPLES = {"Co-Fe29", "Co-Fe33"}
LEFT_CENSORED_BORDERLINE_RECOVERED_SAMPLES = {"Co-V8", "Co-V11", "Co-V20", "Co-V22"}
LEFT_CENSORED_TURNING_POINT_ESTIMATE_SAMPLES = {
    "Fe-Pt",
    "Pd-Pd",
    "Y-Pd",
    "Co-Pt",
    "W-Pd",
}
LEFT_CENSORED_FORCED_CAP_SAMPLES = {"Sc-Pb", "Cd-Y", "W-Pb", "Ce-Pb"}
EXCLUDED_LITERATURE_FRONT_POINTS = {
    ("literature_reference", "33", "33", 550.0, 16.5),
}


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_pure_metal_sample(series: pd.Series) -> pd.Series:
    metals = series.astype(str).str.strip().str.extract(r"^([A-Z][a-z]?)-([A-Z][a-z]?)$")
    return metals[0].notna() & metals[0].eq(metals[1])


def max_condition_current_density_mA(condition: pd.Series) -> pd.Series:
    matches = condition.astype(str).str.extractall(r"(\d+(?:\.\d+)?)\s*mA")
    if matches.empty:
        return pd.Series(np.nan, index=condition.index, dtype=float)

    current_density = pd.to_numeric(matches[0], errors="coerce")
    return current_density.groupby(level=0).max().reindex(condition.index)


def pareto_frontier_xy(points: pd.DataFrame) -> pd.DataFrame:
    """Return points that minimize x and maximize y."""
    sorted_points = points.sort_values(["x", "y"], ascending=[True, False]).reset_index(drop=True)
    best_y = -np.inf
    keep = []

    for _, row in sorted_points.iterrows():
        if row["y"] > best_y:
            keep.append(True)
            best_y = row["y"]
        else:
            keep.append(False)

    return sorted_points.loc[keep].reset_index(drop=True)


def enclosure_polygon(front: pd.DataFrame, x_ref: float, y_ref: float) -> np.ndarray:
    front = front.sort_values("x", ascending=True).reset_index(drop=True)
    x0 = float(front.loc[0, "x"])
    y0 = float(front.loc[0, "y"])
    y_last = float(front.loc[len(front) - 1, "y"])

    vertices = [(x0, y_ref), (x0, y0)]
    vertices.extend(list(zip(front["x"], front["y"])))
    vertices.extend([(x_ref, y_last), (x_ref, y_ref)])
    return np.array(vertices, dtype=float)


def legend_name(label: str) -> str:
    label = str(label).strip()
    if label.lower() == "this work":
        return "Ours"
    return label


def read_lsv_potentials(path: Path) -> pd.Series:
    for skiprows in (0, 1):
        frame = pd.read_csv(path, sep="\t", encoding="utf-8-sig", skiprows=skiprows)
        if "Potential(V)" in frame.columns:
            return to_num(frame["Potential(V)"]).dropna()
        if "E(V)" in frame.columns:
            return to_num(frame["E(V)"]).dropna()
    raise ValueError(f"Unrecognized LSV columns in {path}")


def read_lsv(sample: str, area_cm2: float) -> pd.DataFrame:
    frame = pd.read_csv(LSV_DIR / f"{sample}.txt", sep="\t", encoding="utf-8-sig")
    out = pd.DataFrame(
        {
            "E_raw_V": to_num(frame["Potential(V)"]),
            "I_A": to_num(frame["Current(A)"]),
        }
    ).dropna()
    out["E_RHE_V"] = out["E_raw_V"] + RHE_OFFSET_V
    out["eta_mV"] = -out["E_RHE_V"] * 1000.0
    out["j_mA_cm2"] = out["I_A"] * 1000.0 / area_cm2
    return out.reset_index(drop=True)


def estimate_turning_point(sample: str, area_cm2: float) -> dict[str, float]:
    frame = read_lsv(sample, area_cm2)
    smooth_j = (
        frame["j_mA_cm2"]
        .rolling(7, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    eta = frame["eta_mV"].to_numpy()
    j = frame["j_mA_cm2"].to_numpy()
    candidates = []

    for idx in range(3, len(frame) - 3):
        if eta[idx] < MIN_OVERPOTENTIAL_MV:
            continue
        window = smooth_j[idx - 3 : idx + 4]
        if smooth_j[idx] >= np.nanmax(window) - 1e-12:
            candidates.append(idx)

    if not candidates:
        candidates = np.flatnonzero(eta >= MIN_OVERPOTENTIAL_MV).tolist()
    if not candidates:
        return {
            "turning_point_eta_estimate_mV": np.nan,
            "turning_point_j_mA_cm2": np.nan,
            "turning_point_abs_delta_mA_cm2": np.nan,
        }

    candidate_frame = pd.DataFrame(
        {
            "idx": candidates,
            "eta_mV": eta[candidates],
            "j_mA_cm2": j[candidates],
        }
    )
    candidate_frame["delta_mA_cm2"] = (
        candidate_frame["j_mA_cm2"] - TARGET_J_MA_CM2
    ).abs()
    best_delta = candidate_frame["delta_mA_cm2"].min()
    near_best = candidate_frame[
        candidate_frame["delta_mA_cm2"].le(best_delta + 0.1)
    ]
    chosen = near_best.sort_values("eta_mV", ascending=False).iloc[0]
    return {
        "turning_point_eta_estimate_mV": float(chosen["eta_mV"]),
        "turning_point_j_mA_cm2": float(chosen["j_mA_cm2"]),
        "turning_point_abs_delta_mA_cm2": float(chosen["delta_mA_cm2"]),
    }


def add_lsv_voltage_ranges(points: pd.DataFrame) -> pd.DataFrame:
    starts = {}
    ends = {}

    for sample in points["sample"].astype(str):
        potentials = read_lsv_potentials(LSV_DIR / f"{sample}.txt")
        starts[sample] = float(potentials.max())
        ends[sample] = float(potentials.min())

    out = points.copy()
    out["lsv_start_E_raw_V"] = out["sample"].map(starts)
    out["lsv_end_E_raw_V"] = out["sample"].map(ends)
    out["left_censored_eta_upper_bound_mV"] = -(
        out["lsv_start_E_raw_V"] + RHE_OFFSET_V
    ) * 1000.0
    out["right_censored_eta_lower_bound_mV"] = -(
        out["lsv_end_E_raw_V"] + RHE_OFFSET_V
    ) * 1000.0
    return out


def load_this_work() -> pd.DataFrame:
    points = pd.read_csv(THIS_WORK_CSV)
    audit = pd.read_csv(LSV_AUDIT_CSV)

    audit["has_target_crossing_full_lsv"] = to_bool(audit["has_target_crossing_full_lsv"])
    audit["never_reaches_target_full_trace"] = to_bool(audit["never_reaches_target_full_trace"])

    cols = [
        "sample",
        "has_target_crossing_full_lsv",
        "never_reaches_target_full_trace",
        "n_target_crossings_full_lsv",
        "eta_cross_worst_mV_vs_RHE",
        "E_cross_worst_raw_V",
        "most_negative_E_raw_V",
        "most_negative_E_vs_RHE_V",
        "j_at_most_negative_E_mA_cm2",
        "minimum_j_mA_cm2",
        "nearest_j_to_target_mA_cm2",
        "E_at_nearest_j_raw_V",
    ]
    points = points.merge(audit[cols], on="sample", how="left", validate="one_to_one")
    points = add_lsv_voltage_ranges(points)

    points["eta10_exact_mV"] = to_num(points["eta_cross_worst_mV_vs_RHE"])
    points["nearest_j_to_target_mA_cm2"] = to_num(points["nearest_j_to_target_mA_cm2"])
    points["E_at_nearest_j_raw_V"] = to_num(points["E_at_nearest_j_raw_V"])
    points["nearest_to_target_eta_estimate_mV"] = -(
        points["E_at_nearest_j_raw_V"] + RHE_OFFSET_V
    ) * 1000.0
    points["nearest_to_target_abs_delta_mA_cm2"] = (
        points["nearest_j_to_target_mA_cm2"] - TARGET_J_MA_CM2
    ).abs()
    points["turning_point_eta_estimate_mV"] = np.nan
    points["turning_point_j_mA_cm2"] = np.nan
    points["turning_point_abs_delta_mA_cm2"] = np.nan
    turning_mask = points["sample"].isin(LEFT_CENSORED_TURNING_POINT_ESTIMATE_SAMPLES)
    for idx, row in points.loc[turning_mask].iterrows():
        estimate = estimate_turning_point(
            str(row["sample"]),
            float(row["LSV_geometry_area_cm2"]),
        )
        for key, value in estimate.items():
            points.loc[idx, key] = value
    points["eta_cross_raw_V"] = (
        -points["eta10_exact_mV"] / 1000.0 - RHE_OFFSET_V
    )
    early_crossing_distance_to_start_mV = (
        points["lsv_start_E_raw_V"] - points["eta_cross_raw_V"]
    ) * 1000.0
    points["is_very_early_crossing"] = (
        points["has_target_crossing_full_lsv"]
        & points["eta10_exact_mV"].notna()
        & (points["eta10_exact_mV"].abs() <= EARLY_CROSSING_MAX_ETA_MV)
        & (early_crossing_distance_to_start_mV >= 0.0)
        & (early_crossing_distance_to_start_mV <= EARLY_CROSSING_WINDOW_V * 1000.0)
    )
    points["is_extrapolated_left_crossing"] = (
        points["is_very_early_crossing"]
        | points["sample"].isin(EXTRAPOLATED_LEFT_SAMPLES)
    )

    exact = points["has_target_crossing_full_lsv"] & ~points["is_extrapolated_left_crossing"]
    right_censored = ~exact & points["never_reaches_target_full_trace"]
    left_censored = ~exact & ~points["never_reaches_target_full_trace"]
    most_negative_j = to_num(points["minimum_j_mA_cm2"]).fillna(
        to_num(points["j_at_most_negative_E_mA_cm2"])
    )
    right_censored_near_target = right_censored & (
        most_negative_j <= TARGET_J_MA_CM2 + RIGHT_CENSORED_NEAR_TARGET_TOLERANCE_MA_CM2
    )
    right_censored_assigned_to_cap = right_censored & ~right_censored_near_target
    left_forced_cap = left_censored & points["sample"].isin(LEFT_CENSORED_FORCED_CAP_SAMPLES)
    left_estimate_delta_ok = points["nearest_to_target_abs_delta_mA_cm2"].lt(
        LEFT_CENSORED_MAX_ESTIMATE_DELTA_MA_CM2
    )
    left_near_target_estimate = (
        left_censored
        & ~left_forced_cap
        & left_estimate_delta_ok
        & points["nearest_to_target_eta_estimate_mV"].ge(MIN_OVERPOTENTIAL_MV)
        & (
            points["nearest_to_target_abs_delta_mA_cm2"].le(
                LEFT_CENSORED_NEAR_TARGET_TOLERANCE_MA_CM2
            )
            | points["sample"].isin(LEFT_CENSORED_BORDERLINE_RECOVERED_SAMPLES)
        )
    )
    left_turning_point_estimate = (
        left_censored
        & ~left_forced_cap
        & ~left_near_target_estimate
        & points["sample"].isin(LEFT_CENSORED_TURNING_POINT_ESTIMATE_SAMPLES)
        & points["turning_point_eta_estimate_mV"].ge(MIN_OVERPOTENTIAL_MV)
        & points["turning_point_abs_delta_mA_cm2"].lt(
            LEFT_CENSORED_MAX_ESTIMATE_DELTA_MA_CM2
        )
    )
    left_remaining_at_lower_bound = (
        left_censored
        & ~left_near_target_estimate
        & ~left_turning_point_estimate
        & ~left_forced_cap
    )

    points["eta10_status"] = "exact_crossing"
    points.loc[right_censored, "eta10_status"] = "right_censored_non_reacher"
    points.loc[left_remaining_at_lower_bound, "eta10_status"] = "left_censored_already_active"
    points.loc[left_near_target_estimate, "eta10_status"] = "left_censored_near_target_estimate"
    points.loc[left_turning_point_estimate, "eta10_status"] = "left_censored_turning_point_estimate"
    points.loc[left_forced_cap, "eta10_status"] = "left_censored_forced_to_cap"

    points["x"] = np.nan
    points.loc[exact, "x"] = points.loc[exact, "eta10_exact_mV"]
    points.loc[left_near_target_estimate, "x"] = points.loc[
        left_near_target_estimate, "nearest_to_target_eta_estimate_mV"
    ]
    points.loc[left_turning_point_estimate, "x"] = points.loc[
        left_turning_point_estimate, "turning_point_eta_estimate_mV"
    ]
    points.loc[right_censored_near_target, "x"] = points.loc[
        right_censored_near_target, "right_censored_eta_lower_bound_mV"
    ]
    points.loc[right_censored_assigned_to_cap, "x"] = CAP_OVERPOTENTIAL_MV
    points.loc[left_forced_cap, "x"] = CAP_OVERPOTENTIAL_MV
    points.loc[left_remaining_at_lower_bound, "x"] = points.loc[
        left_remaining_at_lower_bound, "left_censored_eta_upper_bound_mV"
    ]
    points["x_uncapped_mV"] = points["x"].copy()
    points.loc[left_remaining_at_lower_bound, "x"] = MIN_OVERPOTENTIAL_MV
    points["x"] = points["x"].clip(lower=MIN_OVERPOTENTIAL_MV, upper=CAP_OVERPOTENTIAL_MV)
    points["y"] = to_num(points["separation_coefficient"])
    points["label"] = "This work"
    points["display_label"] = points["sample"]
    points["is_pure_metal"] = is_pure_metal_sample(points["sample"])
    points["is_exact_eta10"] = exact
    points["is_eta10_pareto_eligible"] = exact | left_near_target_estimate | left_turning_point_estimate
    points["is_right_censored_eta10"] = right_censored
    points["is_right_censored_near_target"] = right_censored_near_target
    points["is_right_censored_assigned_to_cap"] = right_censored_assigned_to_cap
    points["is_left_censored_eta10"] = left_censored
    points["is_left_censored_near_target_estimate"] = left_near_target_estimate
    points["is_left_censored_turning_point_estimate"] = left_turning_point_estimate
    points["is_left_censored_forced_to_cap"] = left_forced_cap
    points["is_eta10_below_10mV_excluded_from_pareto"] = left_remaining_at_lower_bound
    points["is_extrapolated_left_censored"] = points["is_left_censored_eta10"] & points[
        "is_extrapolated_left_crossing"
    ].fillna(False)
    return points.dropna(subset=["x", "y"]).reset_index(drop=True)


def load_literature() -> pd.DataFrame:
    literature = pd.read_csv(LITERATURE_CSV)
    if "measurement_condition" not in literature.columns:
        literature["measurement_condition"] = "50mA cm-2"
    condition = literature["measurement_condition"].fillna("").astype(str)
    condition_current_density_mA = max_condition_current_density_mA(condition)
    measured_at_front_condition = condition_current_density_mA.ge(50.0).fillna(False)
    out = pd.DataFrame(
        {
            "display_label": literature["reference_label"].astype(str),
            "label": literature["source_label"].astype(str),
            "source_type": literature["source_type"].astype(str),
            "measurement_condition": condition,
            "condition_current_density_mA": condition_current_density_mA,
            "measured_at_front_condition": measured_at_front_condition,
            "x": to_num(literature["plot_x_overpotential_mV"]).clip(
                lower=MIN_OVERPOTENTIAL_MV,
                upper=CAP_OVERPOTENTIAL_MV,
            ),
            "y": to_num(literature["separation_coefficient"]),
            "eta10_status": "literature_or_control",
            "is_exact_eta10": True,
        }
    )
    exclusion_keys = pd.Series(
        list(
            zip(
                out["source_type"],
                out["label"],
                out["display_label"],
                out["x"].round(6),
                out["y"].round(6),
            )
        ),
        index=out.index,
    )
    excluded = exclusion_keys.isin(
        {
            (
                source_type,
                source_label,
                reference_label,
                round(x, 6),
                round(y, 6),
            )
            for source_type, source_label, reference_label, x, y in EXCLUDED_LITERATURE_FRONT_POINTS
        }
    )
    out["include_in_pareto_front"] = (~excluded) & out["measured_at_front_condition"]
    return out.dropna(subset=["x", "y"]).reset_index(drop=True)


def save_classification(this_work: pd.DataFrame, front_ours: pd.DataFrame, front_all: pd.DataFrame) -> None:
    front_ours_ids = set(front_ours["sample"])
    front_all_ids = set(front_all.loc[front_all["label"].eq("This work"), "display_label"])

    export_cols = [
        "sample",
        "source_type",
        "eta10_status",
        "x_uncapped_mV",
        "x",
        "eta10_exact_mV",
        "left_censored_eta_upper_bound_mV",
        "right_censored_eta_lower_bound_mV",
        "nearest_to_target_eta_estimate_mV",
        "nearest_j_to_target_mA_cm2",
        "nearest_to_target_abs_delta_mA_cm2",
        "turning_point_eta_estimate_mV",
        "turning_point_j_mA_cm2",
        "turning_point_abs_delta_mA_cm2",
        "separation_coefficient",
        "lsv_start_E_raw_V",
        "lsv_end_E_raw_V",
        "j_at_most_negative_E_mA_cm2",
        "minimum_j_mA_cm2",
        "is_pure_metal",
        "is_right_censored_near_target",
        "is_right_censored_assigned_to_cap",
        "is_left_censored_near_target_estimate",
        "is_left_censored_turning_point_estimate",
        "is_left_censored_forced_to_cap",
        "is_eta10_pareto_eligible",
        "is_eta10_below_10mV_excluded_from_pareto",
        "is_extrapolated_left_censored",
    ]
    export = this_work[export_cols].rename(
        columns={
            "x": "plot_x_overpotential_mV",
            "x_uncapped_mV": "eta10_or_censor_bound_uncapped_mV",
        }
    )
    export["on_this_work_exact_pareto_front"] = export["sample"].isin(front_ours_ids)
    export["on_global_exact_pareto_front"] = export["sample"].isin(front_all_ids)
    export.to_csv(CLASSIFIED_CSV, index=False)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
        }
    )


def plot_main(
    literature: pd.DataFrame,
    this_work: pd.DataFrame,
    front_pre: pd.DataFrame,
    front_all: pd.DataFrame,
    output_png: Path = MAIN_PNG,
    output_pdf: Path = MAIN_PDF,
    show_text_annotations: bool = True,
) -> None:
    setup_style()

    data = pd.concat([literature, this_work], ignore_index=True)
    is_this = data["label"].eq("This work")
    front_is_pure_metal = front_all.get(
        "is_pure_metal", pd.Series(False, index=front_all.index)
    ).fillna(False).astype(bool)
    front_ours = front_all[front_all["label"].eq("This work") & ~front_is_pure_metal].copy()
    pareto_samples = set(front_ours["display_label"])

    c_pre = "blue"
    c_post = "red"
    y_ref = 2.3
    x_ref = CAP_OVERPOTENTIAL_MV

    poly_pre = enclosure_polygon(front_pre, x_ref, y_ref)
    poly_all = enclosure_polygon(front_all, x_ref, y_ref)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.add_patch(
        Polygon(poly_pre, closed=True, facecolor=to_rgba(c_pre, 0.045), edgecolor=c_pre, linewidth=2.5, zorder=1)
    )
    ax.add_patch(
        Polygon(poly_all, closed=True, facecolor=to_rgba(c_post, 0.045), edgecolor=c_post, linewidth=2.5, zorder=1)
    )

    palette = [
        "#2ca9df",
        "#f3b61f",
        "#6f2dbd",
        "#d64f1a",
        "#f5e84b",
        "#db7eb3",
        "#7e8ce8",
        "#9b5d16",
        "#2bd8c4",
        "#3447b5",
        "#c0c37b",
        "#df70b8",
        "#43e2b5",
        "#8ecf6b",
    ]
    other_labels = sorted(literature["label"].unique(), key=lambda x: str(x))
    style = {
        lbl: {"marker": "^", "color": palette[i % len(palette)]}
        for i, lbl in enumerate(other_labels)
    }

    for lbl in other_labels:
        sub = literature[literature["label"].eq(lbl)]
        sub_front_condition = sub[sub["measured_at_front_condition"]]
        sub_other = sub[~sub["measured_at_front_condition"]]
        ax.scatter(
            sub_front_condition["x"],
            sub_front_condition["y"],
            s=145,
            alpha=0.85,
            marker=style[lbl]["marker"],
            color=style[lbl]["color"],
            edgecolors="black",
            linewidths=0.9,
            label="_nolegend_",
            zorder=4,
        )
        ax.scatter(
            sub_other["x"],
            sub_other["y"],
            s=120,
            alpha=0.78,
            marker="s",
            color=style[lbl]["color"],
            edgecolors="black",
            linewidths=0.8,
            label="_nolegend_",
            zorder=4,
        )

    if show_text_annotations:
        square_condition_labels = literature[
            (~literature["measured_at_front_condition"])
            & literature["condition_current_density_mA"].notna()
            & literature["condition_current_density_mA"].lt(50.0)
            & literature["x"].lt(CAP_OVERPOTENTIAL_MV)
            & literature["y"].ge(8.0)
        ].copy()
        square_condition_offsets = {
            "28": (-18, 26),
            "32": (18, -24),
        }
        for _, row in square_condition_labels.iterrows():
            display_label = str(row["display_label"])
            dx, dy = square_condition_offsets.get(display_label, (16, 20))
            current_density = float(row["condition_current_density_mA"])
            ax.annotate(
                f"{current_density:g} mA cm$^{{-2}}$",
                (row["x"], row["y"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=11.5,
                ha="left" if dx >= 0 else "right",
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#4B5563",
                    "linewidth": 0.75,
                    "shrinkA": 2,
                    "shrinkB": 4,
                },
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=12,
            )

    eligible_nonpareto = this_work[
        this_work["is_eta10_pareto_eligible"] & ~this_work["sample"].isin(pareto_samples)
    ].copy()
    right_or_forced = this_work[
        this_work["is_right_censored_eta10"] | this_work["is_left_censored_forced_to_cap"]
    ]
    ours_other = pd.concat([eligible_nonpareto, right_or_forced], ignore_index=True)

    ax.scatter(
        ours_other["x"],
        ours_other["y"],
        marker="o",
        s=78,
        color="gray",
        alpha=0.48,
        edgecolors="black",
        linewidths=0.65,
        label="_nolegend_",
        zorder=3,
    )

    left = this_work[this_work["is_eta10_below_10mV_excluded_from_pareto"]]
    ax.scatter(
        left["x"],
        left["y"],
        marker="<",
        s=72,
        facecolors="none",
        edgecolors="black",
        linewidths=1.15,
        alpha=0.9,
        label="_nolegend_",
        zorder=5,
    )

    ax.plot(front_pre["x"], front_pre["y"], color=c_pre, linewidth=2.5, zorder=2)
    ax.plot(front_all["x"], front_all["y"], color=c_post, linewidth=2.5, zorder=2)
    ax.scatter(
        front_ours["x"],
        front_ours["y"],
        marker="*",
        s=370,
        color="red",
        edgecolors="black",
        linewidths=0.85,
        zorder=10,
        label="_nolegend_",
    )

    if show_text_annotations:
        front_label_positions = {
            "Co-V14": {"xytext": (29.0, 6.70), "ha": "left"},
            "Co-Fe27": {"xytext": (33.5, 8.52), "ha": "left"},
            "Co-V24": {"xytext": (4.0, 9.15), "ha": "left"},
            "Co-V12": {"xytext": (6.0, 10.45), "ha": "left"},
            "Co-V10": {"xytext": (62.0, 10.03), "ha": "left"},
            "Co-Mo": {"xytext": (82.0, 8.72), "ha": "left"},
            "Co-Mo16": {"xytext": (45.0, 10.70), "ha": "left"},
            "Co-Mo12": {"xytext": (35.0, 11.10), "ha": "left"},
            "Co-Mo7": {"xytext": (107.0, 11.42), "ha": "left"},
            "Co-Mo26": {"xytext": (118.0, 10.95), "ha": "left"},
            "Fe-Ni21": {"xytext": (153.0, 9.80), "ha": "left"},
            "Fe-Ni18": {"xytext": (196.0, 11.58), "ha": "left"},
            "Fe-Ni11": {"xytext": (220.0, 11.72), "ha": "left"},
            "Fe-Ni7": {"xytext": (249.0, 10.38), "ha": "left"},
        }
        for _, row in front_ours.iterrows():
            label = str(row["display_label"])
            label_spec = front_label_positions.get(label)
            if label_spec is None:
                textcoords = "offset points"
                xytext = (16, 18)
                ha = "left"
            else:
                textcoords = "data"
                xytext = label_spec["xytext"]
                ha = label_spec.get("ha", "left")
            ax.annotate(
                label,
                (row["x"], row["y"]),
                xytext=xytext,
                textcoords=textcoords,
                fontsize=10.3,
                ha=ha,
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#4B5563",
                    "linewidth": 0.72,
                    "shrinkA": 2,
                    "shrinkB": 4,
                },
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
                zorder=12,
            )

    ax.set_xlim(0, 560)
    ax.set_ylim(y_ref, 16.0)
    ax.set_xlabel("Overpotential (@ -10 mA/cm², mV; minimal mass-transport)", fontsize=23, fontweight="bold")
    ax.set_ylabel("Separation Coefficient (@ -50 mA/cm²)", fontsize=23, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20, width=1.35, length=6)

    legend_handles = []
    legend_labels = []
    for lbl in other_labels:
        pretty_lbl = legend_name(lbl)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style[lbl]["color"],
                linewidth=5.2,
                solid_capstyle="butt",
            )
        )
        legend_labels.append(pretty_lbl)

    legend_handles.extend(
        [
            Line2D([0], [0], marker="^", linestyle="None", markersize=8.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.0),
            Line2D([0], [0], marker="s", linestyle="None", markersize=8.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.0),
            Line2D([0], [0], marker="o", linestyle="None", markersize=8.5, markerfacecolor="gray", markeredgecolor="black", alpha=0.55),
            Line2D([0], [0], marker="<", linestyle="None", markersize=8.5, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.2),
            Line2D([0], [0], marker="*", linestyle="None", markersize=13, markerfacecolor="red", markeredgecolor="black"),
        ]
    )
    legend_labels.extend(
        [
            "Paper >= 50 mA cm$^{-2}$",
            "Paper < 50/unclear",
            "Ours other",
            "Ours with non-HER current",
            "Ours Pareto front",
        ]
    )
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.995),
        frameon=True,
        fontsize=15.0,
        ncol=int(np.ceil(len(legend_labels) / 2)),
        borderpad=0.35,
        borderaxespad=0.25,
        columnspacing=0.58,
        handlelength=0.95,
        handletextpad=0.35,
        labelspacing=0.38,
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    this_work = load_this_work()
    literature = load_literature()

    frontier_this_work = this_work[this_work["is_eta10_pareto_eligible"]].copy()
    literature_frontier = literature[literature["include_in_pareto_front"]].copy()
    frontier_this_work["pid"] = np.arange(len(frontier_this_work))
    literature["pid"] = np.arange(len(literature)) + len(frontier_this_work)
    literature_frontier["pid"] = np.arange(len(literature_frontier)) + len(frontier_this_work)

    front_this_work = pareto_frontier_xy(frontier_this_work)
    front_all = pareto_frontier_xy(
        pd.concat([literature_frontier, frontier_this_work], ignore_index=True)
    )
    front_pre = pareto_frontier_xy(literature_frontier)

    front_this_work[
        ["sample", "source_type", "eta10_status", "x", "y"]
    ].rename(
        columns={"x": "plot_x_overpotential_mV", "y": "separation_coefficient"}
    ).to_csv(THIS_WORK_FRONT_CSV, index=False)

    front_all[
        [
            "display_label",
            "label",
            "source_type",
            "eta10_status",
            "measurement_condition",
            "condition_current_density_mA",
            "x",
            "y",
        ]
    ].rename(
        columns={
            "display_label": "reference_label",
            "label": "source_label",
            "x": "plot_x_overpotential_mV",
            "y": "separation_coefficient",
        }
    ).to_csv(GLOBAL_FRONT_CSV, index=False)

    front_pre_export = front_pre.copy()
    front_pre_export["measurement_condition"] = front_pre_export.get(
        "measurement_condition", ""
    ).fillna("")
    if "is_pure_metal" not in front_pre_export.columns:
        front_pre_export["is_pure_metal"] = False
    else:
        front_pre_export["is_pure_metal"] = front_pre_export["is_pure_metal"].fillna(False)
    front_pre_export[
        [
            "display_label",
            "label",
            "source_type",
            "eta10_status",
            "measurement_condition",
            "condition_current_density_mA",
            "is_pure_metal",
            "x",
            "y",
        ]
    ].rename(
        columns={
            "display_label": "reference_label",
            "label": "source_label",
            "x": "plot_x_overpotential_mV",
            "y": "separation_coefficient",
        }
    ).to_csv(PRE_FRONT_CSV, index=False)

    save_classification(this_work, front_this_work, front_all)
    plot_main(literature, this_work, front_pre, front_all)
    plot_main(
        literature,
        this_work,
        front_pre,
        front_all,
        output_png=MAIN_NO_TEXT_PNG,
        output_pdf=MAIN_NO_TEXT_PDF,
        show_text_annotations=False,
    )

    print(f"This-work eta10 classes: {this_work['eta10_status'].value_counts().to_dict()}")
    print(f"This-work Pareto points: {len(front_this_work)}")
    print(f"Global Pareto points: {len(front_all)}")
    print(f"Saved: {MAIN_PNG}")
    print(f"Saved: {MAIN_NO_TEXT_PNG}")


if __name__ == "__main__":
    main()
