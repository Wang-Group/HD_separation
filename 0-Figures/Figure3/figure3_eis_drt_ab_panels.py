#!/usr/bin/env python3
"""Candidate Figure 3A/B EIS-DRT top panels.

Panel A maps total DRT relaxation resistance against a tau-resolved DRT window
near the strongest HD-associated relaxation signal. Panel B compares fitted
Nyquist examples selected to show distinct EIS behavior among high/low
HD-separation candidates.
"""

from __future__ import annotations

import math
import re
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    with contextlib.redirect_stdout(io.StringIO()):
        import cvxopt
        from pyDRTtools.runs import EIS_object, simple_run
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Run this script with the DRT conda env: "
        "C:\\Users\\yibinjiang\\anaconda3\\envs\\DRT\\python.exe"
    ) from exc

cvxopt.solvers.options["show_progress"] = False
FONT_SCALE = 1.2
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.weight": "bold",
        "font.size": 12 * FONT_SCALE,
        "axes.titlesize": 16 * FONT_SCALE,
        "axes.labelsize": 14 * FONT_SCALE,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": 12 * FONT_SCALE,
        "ytick.labelsize": 12 * FONT_SCALE,
        "svg.fonttype": "none",
    }
)


FIG3_DIR = Path(__file__).resolve().parent
ROOT = FIG3_DIR.parents[1]
EIS_DIR = ROOT / "hd_separation_ml" / "data_extraction" / "all_data" / "EIS"
DRT_DATA_DIR = FIG3_DIR / "data" / "eis_pydrttools_validation"
OUT_FIG = FIG3_DIR / "outputs" / "eis_pydrttools_validation"

FEATURES_FILE = DRT_DATA_DIR / "drt_features.csv"
DISTRIBUTIONS_FILE = DRT_DATA_DIR / "drt_distributions_long.csv"

MS_TAU_MIN = 8e-4
MS_TAU_MAX = 2e-2
NYQUIST_MAX_FREQ_HZ = 1_000.0
HD_CANDIDATE_POOL_SIZE = 30
HD_PLOT_SAMPLE_COUNT = 6
PANEL_B_SELECTION_EXCLUDED_SAMPLES = {"Mn-Y", "Co-V"}
PANEL_B_DROP_AFTER_SELECTION_SAMPLES = {"In-La", "Al-Al"}
PANEL_B_SAMPLE_REPLACEMENTS = {"Cu-Zn": "Fe-Ga", "Cu-Pb": "Fe-Ni"}
PANEL_B_TOP_LAYER_SAMPLES = {"Fe-Ni"}
SHOW_CURVE_LABELS = False
MAX_SELECTION_REL_RMSE = 0.18
BACKGROUND_EIS_SAMPLE_COUNT = 72
BACKGROUND_MAX_REL_RMSE = 0.12
BACKGROUND_MAX_NYQUIST_X_OHM = 190.0
BACKGROUND_MAX_NYQUIST_Y_OHM = 90.0
NYQUIST_XLIM_OHM = 190.0
NYQUIST_YLIM_OHM = 85.0

COLORS = {
    "top": "#0b7285",
    "bottom": "#c2410c",
    "fit": "#2f3437",
    "grid": "#e5e7eb",
}
HD_CMAP = "coolwarm"
SAMPLE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p", "*"]
SAMPLE_LINESTYLES = [
    "-",
    "--",
    ":",
    "-.",
    (0, (3, 1, 1, 1)),
    (0, (5, 1.5)),
    (0, (1, 1)),
    (0, (4, 1, 1, 1, 1, 1)),
    (0, (6, 2)),
    (0, (2, 1, 2, 3)),
    (0, (1, 1, 4, 1)),
    (0, (7, 1, 1, 1)),
]
CURVE_LABEL_FONTSIZE = 8.6 * FONT_SCALE

DIVERSE_BEHAVIOR_COLUMNS = [
    "max_fit_x_ohm",
    "max_fit_y_ohm",
    "arc_area_ohm2",
    "low_freq_fit_x_ohm",
    "low_freq_fit_y_ohm",
    "log10_peak_freq_hz",
]

PYDRT_SETTINGS = {
    "rbf_type": "PWL",
    "data_used": "Combined Re-Im Data",
    "induct_used": 1,
    "der_used": "1st order",
    "cv_type": "custom",
    "reg_param": 1e-3,
}

_FIT_CACHE: dict[str, pd.DataFrame] = {}
_SELECTION_CACHE: tuple[list[str], list[str], pd.DataFrame] | None = None


def hd_color_norm(data: pd.DataFrame) -> plt.Normalize:
    return plt.Normalize(
        vmin=float(data["separation_coefficient"].min()),
        vmax=float(data["separation_coefficient"].max()),
    )


def read_eis(path: Path) -> pd.DataFrame:
    rows: list[tuple[float, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        values = []
        for part in re.split(r"[\s,\t]+", line.strip()):
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                pass
        if len(values) >= 8:
            rows.append((values[0], values[4], values[5], values[6], values[7]))
        elif len(values) >= 5:
            rows.append((values[0], values[1], values[2], values[3], values[4]))
        elif len(values) >= 3:
            z_abs = math.hypot(values[1], values[2])
            phase = math.degrees(math.atan2(values[2], values[1]))
            rows.append((values[0], values[1], values[2], z_abs, phase))

    if not rows:
        raise ValueError(f"No numeric EIS rows found in {path}")
    df = pd.DataFrame(
        rows,
        columns=["freq_hz", "z_real_ohm", "z_imag_ohm", "z_abs_ohm", "phase_deg"],
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["freq_hz", "z_real_ohm", "z_imag_ohm"]
    )
    df = df[df["freq_hz"] > 0]
    return df.groupby("freq_hz", as_index=False).median(numeric_only=True).sort_values(
        "freq_hz", ascending=False
    )


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    xx = pd.Series(x).rank(method="average").to_numpy(float)
    yy = pd.Series(y).rank(method="average").to_numpy(float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    if ok.sum() < 3 or np.nanstd(xx[ok]) == 0 or np.nanstd(yy[ok]) == 0:
        return np.nan
    return float(np.corrcoef(xx[ok], yy[ok])[0, 1])


def add_fit_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0:
        return
    coef = np.polyfit(x[ok], y[ok], 1)
    xs = np.linspace(float(np.nanmin(x[ok])), float(np.nanmax(x[ok])), 120)
    ax.plot(xs, np.polyval(coef, xs), color=COLORS["fit"], linewidth=1.8, alpha=0.8)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)


def add_curve_label_legend(
    ax: plt.Axes, handles: list[Line2D], labels: list[str]
) -> None:
    legend = ax.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.045),
        ncol=2,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#d1d5db",
        fontsize=CURVE_LABEL_FONTSIZE,
        handlelength=0.95,
        handletextpad=0.22,
        columnspacing=0.42,
        borderaxespad=0.1,
        labelspacing=0.18,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")


def load_panel_a_data() -> pd.DataFrame:
    features = pd.read_csv(FEATURES_FILE)
    distributions = pd.read_csv(DISTRIBUTIONS_FILE)
    initial = features[features["dataset_subset"].eq("initial_screening_no_BO")].copy()
    tau_window = distributions[
        (distributions["tau_s"] >= MS_TAU_MIN)
        & (distributions["tau_s"] <= MS_TAU_MAX)
    ]
    ms_area = (
        tau_window.groupby("sample", as_index=False)["area_ohm"]
        .sum()
        .rename(columns={"area_ohm": "ms_tau_0p8_20ms_area_ohm"})
    )
    return initial.merge(ms_area, on="sample", how="left")


def plot_panel_a(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = data["drt_total_area_ohm"].to_numpy(float)
    y = data["ms_tau_0p8_20ms_area_ohm"].clip(lower=1e-2).to_numpy(float)
    hd = data["separation_coefficient"].to_numpy(float)
    norm = hd_color_norm(data)

    scatter = ax.scatter(
        x,
        y,
        c=hd,
        cmap=HD_CMAP,
        norm=norm,
        s=34,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total DRT Area (Ohm)")
    ax.set_ylabel("DRT Area at 0.8-20 ms (Ohm)")
    style_axis(ax)
    return scatter


def prepare_nyquist(sample: str) -> pd.DataFrame:
    df = read_eis(EIS_DIR / f"{sample}.txt")
    rs = float(df.loc[df["freq_hz"].idxmax(), "z_real_ohm"])
    out = df.copy()
    out["z_real_rs_corrected_ohm"] = out["z_real_ohm"] - rs
    out["neg_z_imag_ohm"] = -out["z_imag_ohm"]
    out = out[
        (out["z_real_rs_corrected_ohm"] >= 0)
        & (out["neg_z_imag_ohm"] >= 0)
    ]
    return out.sort_values("freq_hz", ascending=False)


def fit_eis_with_pydrttools(sample: str, group: str, hd: float) -> pd.DataFrame:
    if sample in _FIT_CACHE:
        cached = _FIT_CACHE[sample].copy()
        cached.insert(0, "group", group)
        cached.insert(2, "separation_coefficient", hd)
        return cached

    df = read_eis(EIS_DIR / f"{sample}.txt")
    freq = df["freq_hz"].to_numpy(float)
    z_re = df["z_real_ohm"].to_numpy(float)
    z_im = df["z_imag_ohm"].to_numpy(float)
    order = np.argsort(freq)[::-1]
    freq = freq[order]
    z_re = z_re[order]
    z_im = z_im[order]

    entry = EIS_object(freq, z_re, z_im)
    with contextlib.redirect_stdout(io.StringIO()):
        entry = simple_run(entry, **PYDRT_SETTINGS)

    fit_re = np.asarray(entry.mu_Z_re, dtype=float).ravel()
    fit_im = np.asarray(entry.mu_Z_im, dtype=float).ravel()
    raw_re = np.asarray(entry.b_re, dtype=float).ravel()
    raw_im = np.asarray(entry.b_im, dtype=float).ravel()
    # Use the fitted high-frequency real intercept for a common visual
    # translation of raw points and fitted curve.
    rs = float(fit_re[np.argmax(freq)])
    fitted = pd.DataFrame(
        {
            "sample": sample,
            "point_index": np.arange(len(freq)),
            "freq_hz": freq,
            "raw_z_real_rs_ohm": raw_re - rs,
            "raw_neg_z_imag_ohm": -raw_im,
            "raw_z_abs_ohm": np.hypot(raw_re, raw_im),
            "fit_z_real_rs_ohm": fit_re - rs,
            "fit_neg_z_imag_ohm": -fit_im,
            "fit_z_abs_ohm": np.hypot(fit_re, fit_im),
        }
    )
    _FIT_CACHE[sample] = fitted.copy()
    fitted.insert(0, "group", group)
    fitted.insert(2, "separation_coefficient", hd)
    return fitted


def build_fit_table(data: pd.DataFrame, top: list[str], bottom: list[str]) -> pd.DataFrame:
    rows = []
    for group, samples in [("High HD diverse", top), ("Low HD diverse", bottom)]:
        for sample in samples:
            hd = float(data.loc[data["sample"].eq(sample), "separation_coefficient"].iloc[0])
            rows.append(fit_eis_with_pydrttools(sample, group, hd))
    fits = pd.concat(rows, ignore_index=True)
    fits.to_csv(DRT_DATA_DIR / "figure3B_diverse_pydrttools_eis_fits.csv", index=False)
    return fits


def build_named_fit_table(data: pd.DataFrame, samples: list[str], group: str) -> pd.DataFrame:
    rows = []
    for sample in samples:
        hd = float(data.loc[data["sample"].eq(sample), "separation_coefficient"].iloc[0])
        rows.append(fit_eis_with_pydrttools(sample, group, hd))
    return pd.concat(rows, ignore_index=True)


def median_fit(fits: pd.DataFrame, group: str) -> pd.DataFrame:
    sub = fits[fits["group"].eq(group)]
    med = (
        sub.groupby("point_index", as_index=False)
        .agg(
            freq_hz=("freq_hz", "median"),
            x_median=("fit_z_real_rs_ohm", "median"),
            y_median=("fit_neg_z_imag_ohm", "median"),
            n=("sample", "nunique"),
        )
        .sort_values("point_index")
    )
    return med[med["n"] >= 5]


def candidate_extreme_pools(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric = "imag_loss_rel_rmse"
    candidates = data[~data["sample"].isin(PANEL_B_SELECTION_EXCLUDED_SAMPLES)].copy()
    top_pool = candidates.nlargest(HD_CANDIDATE_POOL_SIZE, "separation_coefficient").copy()
    bottom_pool = candidates.nsmallest(HD_CANDIDATE_POOL_SIZE, "separation_coefficient").copy()
    top_pool["candidate_group"] = "high_HD_candidate"
    bottom_pool["candidate_group"] = "low_HD_candidate"
    for pool in (top_pool, bottom_pool):
        pool["passes_fit_error_filter"] = pool[metric] <= MAX_SELECTION_REL_RMSE
    return top_pool, bottom_pool


def fit_candidate_pool(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in pool.itertuples(index=False):
        rows.append(
            fit_eis_with_pydrttools(
                sample=str(row.sample),
                group=str(row.candidate_group),
                hd=float(row.separation_coefficient),
            )
        )
    return pd.concat(rows, ignore_index=True)


def behavior_metrics_from_fits(fits: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    records = []
    low_freq_col = "freq_hz"
    for sample, df in fits.groupby("sample", sort=False):
        sub = df[df["freq_hz"] <= NYQUIST_MAX_FREQ_HZ].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["fit_z_real_rs_ohm", "fit_neg_z_imag_ohm", low_freq_col]
        )
        if len(sub) < 3:
            continue

        x = sub["fit_z_real_rs_ohm"].clip(lower=0).to_numpy(float)
        y = sub["fit_neg_z_imag_ohm"].clip(lower=0).to_numpy(float)
        sorted_xy = sub.assign(
            x_clip=sub["fit_z_real_rs_ohm"].clip(lower=0),
            y_clip=sub["fit_neg_z_imag_ohm"].clip(lower=0),
        ).sort_values("x_clip")
        peak_idx = sub["fit_neg_z_imag_ohm"].idxmax()
        low_idx = sub["freq_hz"].idxmin()
        records.append(
            {
                "sample": sample,
                "max_fit_x_ohm": float(np.nanmax(x)),
                "max_fit_y_ohm": float(np.nanmax(y)),
                "low_freq_fit_x_ohm": float(sub.loc[low_idx, "fit_z_real_rs_ohm"]),
                "low_freq_fit_y_ohm": float(sub.loc[low_idx, "fit_neg_z_imag_ohm"]),
                "peak_freq_hz": float(sub.loc[peak_idx, "freq_hz"]),
                "log10_peak_freq_hz": float(np.log10(sub.loc[peak_idx, "freq_hz"])),
                "arc_area_ohm2": float(
                    np.trapezoid(
                        sorted_xy["y_clip"].to_numpy(float),
                        sorted_xy["x_clip"].to_numpy(float),
                    )
                ),
            }
        )
    metrics = pd.DataFrame(records)
    return metrics.merge(
        data[
            [
                "sample",
                "separation_coefficient",
                "imag_loss_rel_rmse",
                "imag_loss_r2",
                "drt_total_area_ohm",
                "mid_tau_10_1kHz__area_ohm",
                "long_tau_0p1_10Hz__area_ohm",
            ]
        ],
        on="sample",
        how="left",
    )


def diverse_selection(pool_metrics: pd.DataFrame, count: int, prefer_high_hd: bool) -> list[str]:
    metric = "imag_loss_rel_rmse"
    if pool_metrics.empty:
        return []
    clean = pool_metrics[pool_metrics["passes_fit_error_filter"]].copy()
    if len(clean) < count:
        clean = pool_metrics.nsmallest(min(len(pool_metrics), max(count, len(clean))), metric).copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=DIVERSE_BEHAVIOR_COLUMNS)
    if len(clean) <= count:
        return clean.sort_values(
            ["separation_coefficient", metric],
            ascending=[not prefer_high_hd, True],
        )["sample"].tolist()

    values = clean[DIVERSE_BEHAVIOR_COLUMNS].to_numpy(float)
    med = np.nanmedian(values, axis=0)
    scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, np.nanstd(values, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    z = (values - med) / scale

    rmse = clean[metric].to_numpy(float)
    rmse_scale = np.nanstd(rmse)
    rmse_z = (rmse - np.nanmedian(rmse)) / (rmse_scale if rmse_scale > 1e-12 else 1.0)

    start_sort = clean.sort_values(
        [metric, "separation_coefficient"],
        ascending=[True, not prefer_high_hd],
    )
    selected_positions = [clean.index.get_loc(start_sort.index[0])]
    while len(selected_positions) < count:
        remaining = [i for i in range(len(clean)) if i not in selected_positions]
        selected_z = z[selected_positions]
        distances = []
        for pos in remaining:
            min_distance = float(np.min(np.linalg.norm(selected_z - z[pos], axis=1)))
            distances.append(min_distance - 0.08 * float(rmse_z[pos]))
        selected_positions.append(remaining[int(np.argmax(distances))])

    selected = clean.iloc[selected_positions].copy()
    selected["selection_order"] = range(1, len(selected) + 1)
    return selected["sample"].tolist()


def select_diverse_extreme_samples(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Pick fitted high/low HD examples that are clean and visually distinct."""
    global _SELECTION_CACHE
    if _SELECTION_CACHE is not None:
        return _SELECTION_CACHE[0], _SELECTION_CACHE[1]

    top_pool, bottom_pool = candidate_extreme_pools(data)
    top_fits = fit_candidate_pool(top_pool)
    bottom_fits = fit_candidate_pool(bottom_pool)
    top_metrics = behavior_metrics_from_fits(top_fits, data).merge(
        top_pool[["sample", "candidate_group", "passes_fit_error_filter"]],
        on="sample",
        how="left",
    )
    bottom_metrics = behavior_metrics_from_fits(bottom_fits, data).merge(
        bottom_pool[["sample", "candidate_group", "passes_fit_error_filter"]],
        on="sample",
        how="left",
    )

    top = diverse_selection(top_metrics, HD_PLOT_SAMPLE_COUNT, prefer_high_hd=True)
    bottom = diverse_selection(bottom_metrics, HD_PLOT_SAMPLE_COUNT, prefer_high_hd=False)
    top = [sample for sample in top if sample not in PANEL_B_DROP_AFTER_SELECTION_SAMPLES]
    bottom = [sample for sample in bottom if sample not in PANEL_B_DROP_AFTER_SELECTION_SAMPLES]
    for old_sample, new_sample in PANEL_B_SAMPLE_REPLACEMENTS.items():
        if old_sample in top and new_sample not in top:
            top[top.index(old_sample)] = new_sample
        if old_sample in bottom and new_sample not in bottom:
            bottom[bottom.index(old_sample)] = new_sample
    selection = pd.concat([top_metrics, bottom_metrics], ignore_index=True)
    selection["selected_for_panel_b"] = selection["sample"].isin(top + bottom)
    order = {sample: i + 1 for i, sample in enumerate(top + bottom)}
    selection["selection_order"] = selection["sample"].map(order)
    selection.sort_values(
        ["candidate_group", "selected_for_panel_b", "separation_coefficient"],
        ascending=[True, False, False],
    ).to_csv(DRT_DATA_DIR / "figure3B_diverse_selection_pool.csv", index=False)

    _SELECTION_CACHE = (top, bottom, selection)
    return top, bottom


def select_background_good_fit_samples(
    data: pd.DataFrame, exclude_samples: list[str]
) -> list[str]:
    """Stratified low-error samples for a faint EIS dataset background."""
    metric = "imag_loss_rel_rmse"
    exclude = set(exclude_samples)
    good = data[
        (data[metric] <= BACKGROUND_MAX_REL_RMSE)
        & (~data["sample"].isin(exclude))
    ].copy()
    good = good.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["sample", "separation_coefficient", metric]
    )
    if good.empty:
        good = data[~data["sample"].isin(exclude)].nsmallest(
            BACKGROUND_EIS_SAMPLE_COUNT, metric
        )
        return good["sample"].tolist()

    shape_records = []
    for sample in good["sample"]:
        try:
            nyq = prepare_nyquist(sample)
            nyq = nyq[nyq["freq_hz"] <= NYQUIST_MAX_FREQ_HZ]
            shape_records.append(
                {
                    "sample": sample,
                    "background_raw_max_x_ohm": float(
                        nyq["z_real_rs_corrected_ohm"].max()
                    ),
                    "background_raw_max_y_ohm": float(nyq["neg_z_imag_ohm"].max()),
                }
            )
        except Exception:
            continue
    shape = pd.DataFrame(shape_records)
    good = good.merge(shape, on="sample", how="inner")
    good = good[
        (good["background_raw_max_x_ohm"] <= BACKGROUND_MAX_NYQUIST_X_OHM)
        & (good["background_raw_max_y_ohm"] <= BACKGROUND_MAX_NYQUIST_Y_OHM)
    ].copy()
    if good.empty:
        good = data[~data["sample"].isin(exclude)].nsmallest(
            BACKGROUND_EIS_SAMPLE_COUNT, metric
        ).copy()
        good["background_raw_max_x_ohm"] = np.nan
        good["background_raw_max_y_ohm"] = np.nan

    n_bins = min(12, len(good), BACKGROUND_EIS_SAMPLE_COUNT)
    good["hd_bin"] = pd.qcut(
        good["separation_coefficient"].rank(method="first"),
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    per_bin = int(math.ceil(BACKGROUND_EIS_SAMPLE_COUNT / max(1, good["hd_bin"].nunique())))
    selected = (
        good.sort_values(["hd_bin", metric])
        .groupby("hd_bin", group_keys=False)
        .head(per_bin)
    )
    if len(selected) < BACKGROUND_EIS_SAMPLE_COUNT:
        selected = pd.concat(
            [
                selected,
                good[~good["sample"].isin(selected["sample"])]
                .sort_values(metric)
                .head(BACKGROUND_EIS_SAMPLE_COUNT - len(selected)),
            ],
            ignore_index=True,
        )
    selected = selected.head(BACKGROUND_EIS_SAMPLE_COUNT).copy()
    selected[
        [
            "sample",
            "separation_coefficient",
            metric,
            "imag_loss_r2",
            "drt_total_area_ohm",
            "background_raw_max_x_ohm",
            "background_raw_max_y_ohm",
        ]
    ].to_csv(DRT_DATA_DIR / "figure3B_background_good_fit_samples.csv", index=False)
    return selected["sample"].tolist()


def median_nyquist(samples: list[str]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        df = prepare_nyquist(sample)
        df["sample"] = sample
        rows.append(df)
    joined = pd.concat(rows, ignore_index=True)
    return (
        joined.groupby("freq_hz", as_index=False)
        .agg(
            x_median=("z_real_rs_corrected_ohm", "median"),
            y_median=("neg_z_imag_ohm", "median"),
        )
        .sort_values("freq_hz", ascending=False)
    )


def plot_panel_b(
    ax: plt.Axes, data: pd.DataFrame, show_curve_labels: bool = False
) -> tuple[list[str], list[str]]:
    top, bottom = select_diverse_extreme_samples(data)
    fits = build_fit_table(data, top, bottom)
    fits = fits[fits["freq_hz"] <= NYQUIST_MAX_FREQ_HZ].copy()
    norm = hd_color_norm(data)
    cmap = plt.get_cmap(HD_CMAP)
    sample_table = (
        fits[["sample", "group", "separation_coefficient"]]
        .drop_duplicates()
        .sort_values("separation_coefficient")
    )
    sample_styles = {
        row.sample: (
            SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)],
            SAMPLE_LINESTYLES[i % len(SAMPLE_LINESTYLES)],
        )
        for i, row in enumerate(sample_table.itertuples(index=False))
    }
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    for row in sample_table.itertuples(index=False):
        marker, linestyle = sample_styles[row.sample]
        color = cmap(norm(float(row.separation_coefficient)))
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4.8,
                markeredgecolor="#334155",
                markeredgewidth=0.25,
            )
        )
        legend_labels.append(f"{row.sample} {float(row.separation_coefficient):.1f}")

    plot_table = sample_table.assign(
        top_layer=sample_table["sample"].isin(PANEL_B_TOP_LAYER_SAMPLES)
    ).sort_values(["top_layer", "separation_coefficient"], ascending=[True, True])
    for row in plot_table.itertuples(index=False):
        group_fit = fits[fits["sample"].eq(row.sample)].sort_values(
            "freq_hz", ascending=False
        )
        color = cmap(norm(float(row.separation_coefficient)))
        marker, linestyle = sample_styles[row.sample]
        is_top_layer = row.sample in PANEL_B_TOP_LAYER_SAMPLES
        ax.plot(
            group_fit["fit_z_real_rs_ohm"],
            group_fit["fit_neg_z_imag_ohm"],
            color=color,
            alpha=0.60,
            linestyle=linestyle,
            linewidth=2.2,
            zorder=5 if is_top_layer else 2,
        )
        ax.scatter(
            group_fit["raw_z_real_rs_ohm"],
            group_fit["raw_neg_z_imag_ohm"],
            s=34,
            color=color,
            alpha=0.60,
            marker=marker,
            edgecolor="#334155",
            linewidth=0.35,
            zorder=6 if is_top_layer else 3,
        )

    if show_curve_labels or SHOW_CURVE_LABELS:
        add_curve_label_legend(ax, legend_handles, legend_labels)
    ax.set_xlabel("Z' - Rs (Ohm)")
    ax.set_ylabel("-Z'' (Ohm)")
    ax.set_xlim(left=-0.5, right=NYQUIST_XLIM_OHM)
    ax.set_ylim(bottom=-0.2, top=NYQUIST_YLIM_OHM)
    style_axis(ax)
    return top, bottom


def plot_panel_b_bode(ax: plt.Axes, data: pd.DataFrame) -> tuple[list[str], list[str]]:
    top, bottom = select_diverse_extreme_samples(data)
    fits = build_fit_table(data, top, bottom)
    fits = fits[fits["freq_hz"] <= NYQUIST_MAX_FREQ_HZ].copy()
    norm = hd_color_norm(data)
    cmap = plt.get_cmap(HD_CMAP)
    sample_table = (
        fits[["sample", "group", "separation_coefficient"]]
        .drop_duplicates()
        .sort_values("separation_coefficient")
    )
    sample_styles = {
        row.sample: (
            SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)],
            SAMPLE_LINESTYLES[i % len(SAMPLE_LINESTYLES)],
        )
        for i, row in enumerate(sample_table.itertuples(index=False))
    }
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    for row in sample_table.itertuples(index=False):
        marker, linestyle = sample_styles[row.sample]
        color = cmap(norm(float(row.separation_coefficient)))
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4.8,
                markeredgecolor="#334155",
                markeredgewidth=0.25,
            )
        )
        legend_labels.append(f"{row.sample} {float(row.separation_coefficient):.1f}")

    plot_table = sample_table.assign(
        top_layer=sample_table["sample"].isin(PANEL_B_TOP_LAYER_SAMPLES)
    ).sort_values(["top_layer", "separation_coefficient"], ascending=[True, True])
    for row in plot_table.itertuples(index=False):
        group_fit = fits[fits["sample"].eq(row.sample)].sort_values(
            "freq_hz", ascending=False
        )
        color = cmap(norm(float(row.separation_coefficient)))
        marker, linestyle = sample_styles[row.sample]
        is_top_layer = row.sample in PANEL_B_TOP_LAYER_SAMPLES
        ax.plot(
            group_fit["freq_hz"],
            group_fit["fit_z_abs_ohm"],
            color=color,
            alpha=0.60,
            linestyle=linestyle,
            linewidth=2.0,
            zorder=5 if is_top_layer else 2,
        )
        ax.scatter(
            group_fit["freq_hz"],
            group_fit["raw_z_abs_ohm"],
            s=32,
            color=color,
            alpha=0.60,
            marker=marker,
            edgecolor="#334155",
            linewidth=0.32,
            zorder=6 if is_top_layer else 3,
        )

    if SHOW_CURVE_LABELS:
        add_curve_label_legend(ax, legend_handles, legend_labels)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Z| (Ohm)")
    style_axis(ax)
    return top, bottom


def add_inset_colorbar(fig: plt.Figure, ax: plt.Axes, scatter) -> None:
    cax = inset_axes(ax, width="3.2%", height="38%", loc="upper right", borderpad=0.9)
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("Separation Coefficient", fontsize=12 * FONT_SCALE, fontweight="bold")
    cbar.ax.tick_params(labelsize=11 * FONT_SCALE)


def add_shared_colorbar(fig: plt.Figure, axes: list[plt.Axes], scatter) -> None:
    cbar = fig.colorbar(
        scatter,
        ax=axes,
        location="right",
        fraction=0.025,
        pad=0.018,
        shrink=0.58,
        aspect=24,
    )
    cbar.set_label("Separation Coefficient", fontsize=12 * FONT_SCALE, fontweight="bold")
    cbar.ax.tick_params(labelsize=11 * FONT_SCALE)


def add_top_colorbar(fig: plt.Figure, ax: plt.Axes, scatter) -> None:
    cbar = fig.colorbar(
        scatter,
        ax=ax,
        location="top",
        fraction=0.075,
        pad=0.035,
        aspect=28,
    )
    cbar.set_label("Separation Coefficient", fontsize=12 * FONT_SCALE, fontweight="bold")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=11 * FONT_SCALE, pad=2)


def save_standalone_colorbar(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(1.15, 4.1), constrained_layout=True)
    norm = hd_color_norm(data)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=HD_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax)
    cbar.set_label("Separation Coefficient", fontsize=12 * FONT_SCALE, fontweight="bold")
    cbar.ax.tick_params(labelsize=11 * FONT_SCALE)
    fig.savefig(OUT_FIG / "figure3A_drt_2d_map_colorbar_initial.png", dpi=300)
    fig.savefig(OUT_FIG / "figure3A_drt_2d_map_colorbar_initial.svg")
    plt.close(fig)


def save_sample_lists(top: list[str], bottom: list[str], data: pd.DataFrame) -> None:
    rows = []
    for group, samples in [("high_HD_diverse_EIS", top), ("low_HD_diverse_EIS", bottom)]:
        for sample in samples:
            hd = float(data.loc[data["sample"].eq(sample), "separation_coefficient"].iloc[0])
            fit_error = float(data.loc[data["sample"].eq(sample), "imag_loss_rel_rmse"].iloc[0])
            rows.append(
                {
                    "group": group,
                    "sample": sample,
                    "separation_coefficient": hd,
                    "imag_loss_rel_rmse": fit_error,
                }
            )
    pd.DataFrame(rows).to_csv(DRT_DATA_DIR / "figure3AB_top_bottom_samples.csv", index=False)


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    data = load_panel_a_data()

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), constrained_layout=True)
    top, bottom = plot_panel_b(axes[0], data)
    scatter = plot_panel_a(axes[1], data)
    add_top_colorbar(fig, axes[1], scatter)
    fig.savefig(OUT_FIG / "figure3AB_eis_drt_top_row_initial.png", dpi=300)
    fig.savefig(OUT_FIG / "figure3AB_eis_drt_top_row_initial.svg")
    plt.close(fig)

    fig_a, ax_a = plt.subplots(figsize=(5.9, 5.0), constrained_layout=True)
    scatter_a = plot_panel_a(ax_a, data)
    add_top_colorbar(fig_a, ax_a, scatter_a)
    fig_a.savefig(OUT_FIG / "figure3A_drt_2d_map_initial.png", dpi=300)
    fig_a.savefig(OUT_FIG / "figure3A_drt_2d_map_initial.svg")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(5.9, 5.0), constrained_layout=True)
    plot_panel_b(ax_b, data, show_curve_labels=True)
    fig_b.savefig(OUT_FIG / "figure3B_eis_top_bottom_initial.png", dpi=300)
    fig_b.savefig(OUT_FIG / "figure3B_eis_top_bottom_initial.svg")
    plt.close(fig_b)

    save_sample_lists(top, bottom, data)
    print(f"Figure 3A/B draft outputs written to: {OUT_FIG}")
    print(f"High-HD diverse samples: {top}")
    print(f"Low-HD diverse samples: {bottom}")


if __name__ == "__main__":
    main()
