#!/usr/bin/env python3
"""pyDRTtools EIS validation for HD separation.

This script uses the official ``pyDRTtools`` implementation from the Ciucci lab
to recover a distribution of relaxation times (DRT) from each raw EIS spectrum,
then tests whether integrated DRT descriptors correlate with the HD separation
coefficient.

The default batch settings use ``simple_run`` with a fixed regularization
parameter and the piecewise-linear RBF option. Figure 3 analysis is restricted
to the 561-sample initial-screening no-BO dataset so Bayesian-optimization
follow-up points do not bias the EIS/DRT-HD relationship.
"""

from __future__ import annotations

import math
import re
import time
import contextlib
import io
from pathlib import Path
from pandas.errors import EmptyDataError

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    with contextlib.redirect_stdout(io.StringIO()):
        import cvxopt
        from pyDRTtools.runs import EIS_object, simple_run
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script requires pyDRTtools. Run it with the conda env created as "
        "`conda activate DRT` or with "
        "`C:\\Users\\yibinjiang\\anaconda3\\envs\\DRT\\python.exe`."
    ) from exc

cvxopt.solvers.options["show_progress"] = False


FIG3_DIR = Path(__file__).resolve().parent
ROOT = FIG3_DIR.parents[1]
DATA_ROOT = ROOT / "hd_separation_ml" / "data_extraction"
EIS_DIR = DATA_ROOT / "all_data" / "EIS"
LABEL_FILE = DATA_ROOT / "separation_coefficient.xlsx"
ML_FEATURE_FILE = (
    ROOT
    / "hd_separation_ml"
    / "outputs"
    / "final_model_8features_561_seed17_rs_from_eis"
    / "feature_matrix_8features.csv"
)

OUT_DATA = FIG3_DIR / "data" / "eis_pydrttools_validation"
OUT_FIG = FIG3_DIR / "outputs" / "eis_pydrttools_validation"


PYDRT_RBF_TYPE = "PWL"
PYDRT_DATA_USED = "Combined Re-Im Data"
PYDRT_INDUCT_USED = 1
PYDRT_DER_USED = "1st order"
PYDRT_CV_TYPE = "custom"
PYDRT_REG_PARAM = 1e-3


COLORS = {
    "initial": "#256d85",
    "all": "#455a64",
    "accent": "#d9822b",
    "top": "#0b7285",
    "bottom": "#c2410c",
}


# Frequency bands expressed as physical relaxation-time bands using
# tau = 1/(2*pi*f). pyDRTtools itself uses tau=1/f for its default collocation
# points, and the recovered gamma(tau) is interpolated to a common physical tau
# grid before integration.
TAU_BANDS = [
    ("short_tau_1k_100kHz", 1.0 / (2 * math.pi * 100_000), 1.0 / (2 * math.pi * 1_000), "1-100 kHz"),
    ("mid_tau_10_1kHz", 1.0 / (2 * math.pi * 1_000), 1.0 / (2 * math.pi * 10), "10 Hz-1 kHz"),
    ("long_tau_0p1_10Hz", 1.0 / (2 * math.pi * 10), 1.0 / (2 * math.pi * 0.1), "0.1-10 Hz"),
    ("very_long_tau_0p1_1Hz", 1.0 / (2 * math.pi * 1), 1.0 / (2 * math.pi * 0.1), "0.1-1 Hz"),
]


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
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["freq_hz", "z_real_ohm", "z_imag_ohm"])
    df = df[df["freq_hz"] > 0].sort_values("freq_hz", ascending=False)
    df = df.groupby("freq_hz", as_index=False).median(numeric_only=True)
    return df


def load_labels() -> pd.DataFrame:
    labels = pd.read_excel(LABEL_FILE)
    labels["sample"] = labels["sample"].astype(str).str.replace(".json", "", regex=False)
    return labels[["sample", "separation_coefficient"]].dropna()


def load_initial_samples() -> set[str]:
    if not ML_FEATURE_FILE.exists():
        return set()
    return set(pd.read_csv(ML_FEATURE_FILE, usecols=["sample"])["sample"].astype(str))


def spearman(x: pd.Series, y: pd.Series) -> float:
    rx = pd.to_numeric(x, errors="coerce").rank(method="average").to_numpy(float)
    ry = pd.to_numeric(y, errors="coerce").rank(method="average").to_numpy(float)
    xx, yy = finite_xy(rx, ry)
    if len(xx) < 3 or np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def pearson(x: pd.Series, y: pd.Series) -> float:
    xx, yy = finite_xy(pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce"))
    if len(xx) < 3 or np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def finite_xy(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    return xx[ok], yy[ok]


def scalar_float(value) -> float:
    arr = np.asarray(value, dtype=float).ravel()
    return float(arr[0]) if arr.size else np.nan


def make_global_log_tau(n_tau: int = 90) -> np.ndarray:
    # Cover the full pyDRTtools default collocation range for spectra measured
    # from 100 kHz to 0.1 Hz. Band integrations below still use physical
    # tau=1/(2*pi*f) windows.
    tau_min = 1.0 / 100_000.0 / math.sqrt(10)
    tau_max = 1.0 / 0.1 * math.sqrt(10)
    return np.linspace(np.log(tau_min), np.log(tau_max), n_tau)


def recover_drt(
    df: pd.DataFrame,
    log_tau: np.ndarray,
    lam: float = PYDRT_REG_PARAM,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Recover a nonnegative DRT using the official pyDRTtools solver."""
    if len(df) < 10:
        raise ValueError(f"Too few EIS points: {len(df)}")

    freq = df["freq_hz"].to_numpy(float)
    z_re = df["z_real_ohm"].to_numpy(float)
    z_im = df["z_imag_ohm"].to_numpy(float)
    order = np.argsort(freq)[::-1]
    freq = freq[order]
    z_re = z_re[order]
    z_im = z_im[order]

    entry = EIS_object(freq, z_re, z_im)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        entry = simple_run(
            entry,
            rbf_type=PYDRT_RBF_TYPE,
            data_used=PYDRT_DATA_USED,
            induct_used=PYDRT_INDUCT_USED,
            der_used=PYDRT_DER_USED,
            cv_type=PYDRT_CV_TYPE,
            reg_param=lam,
        )
    wall_time_s = time.perf_counter() - t0

    out_tau = np.asarray(entry.out_tau_vec, dtype=float).ravel()
    out_gamma = np.asarray(entry.gamma, dtype=float).ravel()
    ok = np.isfinite(out_tau) & np.isfinite(out_gamma) & (out_tau > 0)
    if ok.sum() < 3:
        raise ValueError("pyDRTtools returned too few finite DRT points")

    out_tau = out_tau[ok]
    out_gamma = np.maximum(out_gamma[ok], 0.0)
    sort_idx = np.argsort(out_tau)
    out_tau = out_tau[sort_idx]
    out_gamma = out_gamma[sort_idx]
    out_log_tau = np.log(out_tau)

    tau = np.exp(log_tau)
    gamma = np.interp(log_tau, out_log_tau, out_gamma, left=0.0, right=0.0)
    weights = np.zeros_like(log_tau)
    weights[0] = 0.5 * (log_tau[1] - log_tau[0])
    weights[-1] = 0.5 * (log_tau[-1] - log_tau[-2])
    weights[1:-1] = 0.5 * (log_tau[2:] - log_tau[:-2])
    area = gamma * weights

    def r2_score(obs: np.ndarray, pred: np.ndarray) -> float:
        ss_res = float(np.sum((obs - pred) ** 2))
        ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    imag_rmse = float(np.sqrt(np.mean((entry.b_im - entry.mu_Z_im) ** 2)))
    imag_scale = float(max(np.mean(np.abs(entry.b_im)), 1e-12))
    combined_residual = np.concatenate(
        [(entry.b_re - entry.mu_Z_re), (entry.b_im - entry.mu_Z_im)]
    )
    combined_obs = np.concatenate([entry.b_re, entry.b_im])
    diagnostics = {
        "n_eis_points": float(len(freq)),
        "freq_min_used_hz": float(freq.min()),
        "freq_max_used_hz": float(freq.max()),
        "tau_min_s": float(tau.min()),
        "tau_max_s": float(tau.max()),
        "lambda": scalar_float(entry.lambda_value),
        "pydrttools_wall_time_s": float(wall_time_s),
        "pydrttools_n_rl": float(entry.N_RL),
        "pydrttools_R_ohm": scalar_float(entry.R),
        "pydrttools_L_h": scalar_float(entry.L),
        "real_loss_r2": float(r2_score(entry.b_re, entry.mu_Z_re)),
        "imag_loss_r2": float(r2_score(entry.b_im, entry.mu_Z_im)),
        "combined_loss_r2": float(r2_score(combined_obs, combined_obs - combined_residual)),
        "imag_loss_rel_rmse": imag_rmse / imag_scale,
    }

    dist = pd.DataFrame(
        {
            "tau_s": tau,
            "log10_tau_s": np.log10(tau),
            "gamma_ohm": gamma,
            "area_ohm": area,
        }
    )
    return dist, diagnostics


def summarize_distribution(dist: pd.DataFrame) -> dict[str, float]:
    total_area = float(dist["area_ohm"].sum())
    peak_idx = int(dist["gamma_ohm"].idxmax())
    row: dict[str, float] = {
        "drt_total_area_ohm": total_area,
        "drt_peak_gamma_ohm": float(dist.loc[peak_idx, "gamma_ohm"]),
        "drt_peak_tau_s": float(dist.loc[peak_idx, "tau_s"]),
        "drt_peak_log10_tau_s": float(dist.loc[peak_idx, "log10_tau_s"]),
    }
    for name, lo_tau, hi_tau, _ in TAU_BANDS:
        mask = (dist["tau_s"] >= lo_tau) & (dist["tau_s"] <= hi_tau)
        area = float(dist.loc[mask, "area_ohm"].sum())
        row[f"{name}__area_ohm"] = area
        row[f"{name}__fraction"] = area / total_area if total_area > 0 else np.nan
    row["long_over_short_tau_area"] = (
        row["long_tau_0p1_10Hz__area_ohm"]
        / max(row["short_tau_1k_100kHz__area_ohm"], 1e-12)
    )
    return row


def run_drt_batch(labels: pd.DataFrame, initial_samples: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_map = labels.set_index("sample")["separation_coefficient"].to_dict()
    feature_rows = []
    dist_rows = []
    errors = []
    global_log_tau = make_global_log_tau()

    paths = sorted(
        path
        for path in EIS_DIR.glob("*.txt")
        if path.stem in label_map and path.stem in initial_samples
    )
    for idx, path in enumerate(paths, 1):
        sample = path.stem
        try:
            df = read_eis(path)
            dist, diagnostics = recover_drt(df, global_log_tau)
            summary = summarize_distribution(dist)
            feature_rows.append(
                {
                    "sample": sample,
                    "separation_coefficient": float(label_map[sample]),
                    "dataset_subset": "initial_screening_no_BO",
                    **diagnostics,
                    **summary,
                }
            )
            dist["sample"] = sample
            dist_rows.append(dist)
        except Exception as exc:
            errors.append({"sample": sample, "error": str(exc)})

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(paths)} EIS files")

    features = pd.DataFrame(feature_rows)
    distributions = pd.concat(dist_rows, ignore_index=True) if dist_rows else pd.DataFrame()
    error_df = pd.DataFrame(errors)
    error_df.to_csv(OUT_DATA / "drt_batch_errors.csv", index=False)
    return features, distributions


def make_correlations(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        col
        for col in features.columns
        if col
        not in {
            "sample",
            "separation_coefficient",
            "dataset_subset",
        }
        and pd.api.types.is_numeric_dtype(features[col])
    ]
    subsets = {
        "initial_screening_no_BO": features[
            features["dataset_subset"].eq("initial_screening_no_BO")
        ],
    }
    rows = []
    for subset_name, subset in subsets.items():
        y = subset["separation_coefficient"]
        for feature in feature_cols:
            x = subset[feature]
            rows.append(
                {
                    "subset": subset_name,
                    "feature": feature,
                    "n": int(pd.to_numeric(x, errors="coerce").notna().sum()),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                    "abs_spearman": abs(spearman(x, y)),
                }
            )
    return pd.DataFrame(rows).sort_values(["subset", "abs_spearman"], ascending=[True, False])


def add_fit_line(ax: plt.Axes, x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, color: str) -> None:
    xx, yy = finite_xy(x, y)
    if len(xx) < 3 or np.nanstd(xx) == 0:
        return
    coef = np.polyfit(xx, yy, 1)
    xs = np.linspace(np.nanmin(xx), np.nanmax(xx), 100)
    ax.plot(xs, np.polyval(coef, xs), color=color, linewidth=2.0)


def corr_label(df: pd.DataFrame, x_col: str) -> str:
    return (
        f"Spearman rho = {spearman(df[x_col], df['separation_coefficient']):.2f}, "
        f"n = {int(pd.to_numeric(df[x_col], errors='coerce').notna().sum())}"
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e5e7eb", linewidth=0.8, zorder=0)


def plot_drt_panel(features: pd.DataFrame, distributions: pd.DataFrame, subset: str, stem: str) -> None:
    data = features[features["dataset_subset"].eq("initial_screening_no_BO")].copy()
    sample_order = data.sort_values("separation_coefficient")
    bottom_samples = sample_order.head(10)["sample"].tolist()
    top_samples = sample_order.tail(10)["sample"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), constrained_layout=True)
    ax = axes[0, 0]
    for samples, label, color in [
        (top_samples, "Top 10 HD separation", COLORS["top"]),
        (bottom_samples, "Bottom 10 HD separation", COLORS["bottom"]),
    ]:
        sub = distributions[distributions["sample"].isin(samples)]
        pivot = sub.pivot(index="log10_tau_s", columns="sample", values="gamma_ohm")
        x = 10 ** pivot.index.to_numpy(float)
        med = np.nanmedian(pivot.to_numpy(float), axis=1)
        q25 = np.nanpercentile(pivot.to_numpy(float), 25, axis=1)
        q75 = np.nanpercentile(pivot.to_numpy(float), 75, axis=1)
        ax.plot(x, med, color=color, linewidth=2.2, label=label)
        ax.fill_between(x, q25, q75, color=color, alpha=0.18, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Relaxation time tau (s)")
    ax.set_ylabel("gamma(tau) (ohm)")
    ax.set_title("A. pyDRTtools DRT distributions", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)

    ax = axes[0, 1]
    x_total = np.log10(data["drt_total_area_ohm"].clip(lower=1e-12))
    ax.scatter(
        x_total,
        data["separation_coefficient"],
        s=24,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        color=COLORS["initial"],
    )
    add_fit_line(ax, x_total, data["separation_coefficient"], COLORS["accent"])
    ax.set_xlabel("log10 total DRT area (ohm)")
    ax.set_ylabel("HD separation coefficient")
    ax.set_title("B. Total relaxation resistance", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.04,
        corr_label(data.assign(_x=x_total), "_x"),
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )
    style_axis(ax)

    ax = axes[1, 0]
    x_long_area = np.log10(data["long_tau_0p1_10Hz__area_ohm"].clip(lower=1e-12))
    ax.scatter(
        x_long_area,
        data["separation_coefficient"],
        s=24,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        color=COLORS["initial"],
    )
    add_fit_line(ax, x_long_area, data["separation_coefficient"], COLORS["accent"])
    ax.set_xlabel("log10 long-tau DRT area, equivalent 0.1-10 Hz (ohm)")
    ax.set_ylabel("HD separation coefficient")
    ax.set_title("C. Low-frequency relaxation resistance", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.04,
        corr_label(data.assign(_x=x_long_area), "_x"),
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )
    style_axis(ax)

    ax = axes[1, 1]
    x_frac = 100 * data["long_tau_0p1_10Hz__fraction"]
    sc = ax.scatter(
        x_total,
        x_frac,
        c=data["separation_coefficient"],
        cmap="viridis",
        s=34,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel("log10 total DRT area (ohm)")
    ax.set_ylabel("Long-tau DRT fraction (%)")
    ax.set_title("D. DRT kinetic map", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("HD separation coefficient")
    style_axis(ax)

    fig.suptitle(
        "pyDRTtools DRT validation on initial no-BO raw EIS data",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUT_FIG / f"{stem}.png", dpi=300)
    fig.savefig(OUT_FIG / f"{stem}.svg")
    plt.close(fig)


def plot_tau_correlations(features: pd.DataFrame, distributions: pd.DataFrame) -> None:
    pivot = distributions.pivot(index="sample", columns="log10_tau_s", values="gamma_ohm")
    meta = features.set_index("sample")[["separation_coefficient", "dataset_subset"]]
    joined = meta.join(pivot, how="inner")
    rows = []
    for subset_name, subset in {
        "initial_screening_no_BO": joined[joined["dataset_subset"].eq("initial_screening_no_BO")],
    }.items():
        y = subset["separation_coefficient"]
        for col in pivot.columns:
            rows.append(
                {
                    "subset": subset_name,
                    "log10_tau_s": float(col),
                    "tau_s": 10 ** float(col),
                    "spearman": spearman(subset[col], y),
                }
            )
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT_DATA / "drt_tau_resolved_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for subset_name, label, color in [
        ("initial_screening_no_BO", "Initial no-BO", COLORS["initial"]),
    ]:
        sub = corr[corr["subset"].eq(subset_name)]
        ax.plot(sub["tau_s"], sub["spearman"], label=label, color=color, linewidth=2.0)
    ax.axhline(0, color="#6b7280", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Relaxation time tau (s)")
    ax.set_ylabel("Spearman rho vs HD separation")
    ax.set_title("Tau-resolved DRT correlation", fontweight="bold")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.savefig(OUT_FIG / "figure3_drt_tau_resolved_correlation.png", dpi=300)
    fig.savefig(OUT_FIG / "figure3_drt_tau_resolved_correlation.svg")
    plt.close(fig)


def write_summary(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    correlations: pd.DataFrame,
    missing_eis: list[str],
) -> None:
    focus = [
        "drt_total_area_ohm",
        "long_tau_0p1_10Hz__fraction",
        "very_long_tau_0p1_1Hz__fraction",
        "mid_tau_10_1kHz__fraction",
        "short_tau_1k_100kHz__fraction",
        "long_over_short_tau_area",
        "imag_loss_r2",
        "imag_loss_rel_rmse",
    ]
    focus_corr = correlations[correlations["feature"].isin(focus)].sort_values(
        ["subset", "feature"]
    )
    errors_path = OUT_DATA / "drt_batch_errors.csv"
    try:
        errors = (
            pd.read_csv(errors_path)
            if errors_path.exists() and errors_path.stat().st_size
            else pd.DataFrame()
        )
    except EmptyDataError:
        errors = pd.DataFrame()
    lines = [
        "Figure 3 pyDRTtools DRT-HD validation",
        "=====================================",
        "",
        "Method: official pyDRTtools simple_run with CVXOPT quadratic programming.",
        f"Settings: rbf_type={PYDRT_RBF_TYPE}, data_used={PYDRT_DATA_USED}, "
        f"induct_used={PYDRT_INDUCT_USED}, cv_type={PYDRT_CV_TYPE}, lambda={PYDRT_REG_PARAM}.",
        "Recovered gamma(tau) is interpolated to a common tau grid before integration.",
        "Long-tau fraction maps to the low-frequency 0.1-10 Hz process window by tau = 1/(2*pi*f).",
        "",
        f"EIS files in source folder: {len(list(EIS_DIR.glob('*.txt')))}",
        f"Initial-screening separation labels: {len(labels)}",
        f"DRT-fitted spectra: {len(features)}",
        f"Initial-screening no-BO fitted spectra: {features['dataset_subset'].eq('initial_screening_no_BO').sum()}",
        f"Missing EIS labels: {missing_eis}",
        f"DRT skipped/error spectra: {len(errors)}",
        "",
        "Focused DRT correlations:",
        focus_corr[["subset", "feature", "n", "pearson", "spearman"]].round(4).to_string(index=False),
        "",
        "Top initial-screening DRT correlations:",
        correlations[correlations["subset"].eq("initial_screening_no_BO")]
        .head(20)[["feature", "n", "pearson", "spearman"]]
        .round(4)
        .to_string(index=False),
        "",
        "Interpretation:",
        "- Total DRT relaxation area is negatively correlated with HD separation.",
        "- Absolute long-tau / low-frequency DRT area is also negatively correlated with",
        "  HD separation, but the normalized long-tau fraction is weak and not a robust",
        "  negative trend in this batch inversion.",
        "- This is the official pyDRTtools validation; the separate raw-frequency proxy is kept in",
        "  data/eis_hd_validation/ for comparison.",
    ]
    (OUT_DATA / "drt_hd_validation_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    initial_samples = load_initial_samples()
    labels = labels[labels["sample"].isin(initial_samples)].copy()
    label_samples = set(labels["sample"].astype(str))
    eis_samples = {path.stem for path in EIS_DIR.glob("*.txt")}
    missing_eis = sorted(label_samples - eis_samples)

    features, distributions = run_drt_batch(labels, initial_samples)
    correlations = make_correlations(features)

    features.to_csv(OUT_DATA / "drt_features.csv", index=False)
    distributions.to_csv(OUT_DATA / "drt_distributions_long.csv", index=False)
    correlations.to_csv(OUT_DATA / "drt_feature_correlations.csv", index=False)

    plot_drt_panel(
        features,
        distributions,
        subset="initial_screening_no_BO",
        stem="figure3_drt_validation_initial",
    )
    plot_tau_correlations(features, distributions)
    write_summary(labels, features, correlations, missing_eis)

    print(f"DRT-fitted spectra: {len(features)}")
    print(f"Initial no-BO DRT-fitted spectra: {features['dataset_subset'].eq('initial_screening_no_BO').sum()}")
    print(f"Outputs written to: {OUT_DATA}")
    print(f"Figures written to: {OUT_FIG}")


if __name__ == "__main__":
    main()
