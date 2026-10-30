#!/usr/bin/env python3
"""Validate initial-screening EIS descriptors against HD separation.

This Figure 3 helper uses the current raw EIS files in
``hd_separation_ml/data_extraction/all_data/EIS`` and the current
``separation_coefficient.xlsx`` labels. It intentionally uses raw
frequency-domain integrals rather than an external DRT/FRT solver so the
analysis is lightweight and reproducible from the checked data alone. Figure 3
uses only the initial-screening no-BO samples.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG3_DIR = Path(__file__).resolve().parent
ROOT = FIG3_DIR.parents[1]
DATA_ROOT = ROOT / "hd_separation_ml" / "data_extraction"
EIS_DIR = DATA_ROOT / "all_data" / "EIS"
LABEL_FILE = DATA_ROOT / "separation_coefficient.xlsx"
INITIAL_SAMPLE_FILE = FIG3_DIR / "data" / "eis_initial_samples.csv"

OUT_DATA = FIG3_DIR / "data" / "eis_hd_validation"
OUT_FIG = FIG3_DIR / "outputs" / "eis_hd_validation"


BANDS = [
    ("very_low_0p1_1Hz", 0.1, 1.0, "0.1-1 Hz"),
    ("low_1_10Hz", 1.0, 10.0, "1-10 Hz"),
    ("mid_10_1kHz", 10.0, 1_000.0, "10 Hz-1 kHz"),
    ("high_1k_10kHz", 1_000.0, 10_000.0, "1-10 kHz"),
    ("very_high_10k_100kHz", 10_000.0, 100_000.0, "10-100 kHz"),
]


COLORS = {
    "initial": "#256d85",
    "all": "#455a64",
    "followup": "#b0bec5",
    "accent": "#d9822b",
    "top": "#0b7285",
    "bottom": "#c2410c",
}


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
            # CorrStudio-like export:
            # freq, amp, bias, time, Zreal, Zimag, |Z|, phase, range
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
    df = df[df["freq_hz"] > 0].sort_values("freq_hz")
    df = df.groupby("freq_hz", as_index=False).median(numeric_only=True)
    if df["z_abs_ohm"].isna().any():
        df["z_abs_ohm"] = np.hypot(df["z_real_ohm"], df["z_imag_ohm"])
    if df["phase_deg"].isna().any():
        df["phase_deg"] = np.degrees(np.arctan2(df["z_imag_ohm"], df["z_real_ohm"]))
    return df


def load_labels() -> pd.DataFrame:
    labels = pd.read_excel(LABEL_FILE)
    labels["sample"] = labels["sample"].astype(str).str.replace(".json", "", regex=False)
    return labels[["sample", "separation_coefficient"]].dropna()


def load_initial_samples() -> set[str]:
    return set(pd.read_csv(INITIAL_SAMPLE_FILE, usecols=["sample"])["sample"].astype(str))


def choose_frequency_grid(parsed: dict[str, pd.DataFrame]) -> np.ndarray:
    counts = pd.Series({sample: len(df) for sample, df in parsed.items()})
    mode_count = int(counts.mode().iloc[0])
    ref_sample = next(sample for sample, count in counts.items() if count == mode_count)
    grid = parsed[ref_sample]["freq_hz"].to_numpy(float)
    return np.sort(grid)


def interpolate_spectrum(df: pd.DataFrame, target_logf: np.ndarray, column: str) -> np.ndarray:
    logf = np.log10(df["freq_hz"].to_numpy(float))
    vals = df[column].to_numpy(float)
    order = np.argsort(logf)
    logf = logf[order]
    vals = vals[order]
    out = np.full_like(target_logf, np.nan, dtype=float)
    in_range = (target_logf >= logf.min() - 1e-9) & (target_logf <= logf.max() + 1e-9)
    out[in_range] = np.interp(target_logf[in_range], logf, vals)
    return out


def pearson(x: pd.Series, y: pd.Series) -> float:
    return pd.to_numeric(x, errors="coerce").corr(pd.to_numeric(y, errors="coerce"))


def spearman(x: pd.Series, y: pd.Series) -> float:
    return pd.to_numeric(x, errors="coerce").rank(method="average").corr(
        pd.to_numeric(y, errors="coerce").rank(method="average")
    )


def finite_xy(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    return xx[ok], yy[ok]


def build_interpolated_matrix(
    labels: pd.DataFrame, initial_samples: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    label_map = labels.set_index("sample")["separation_coefficient"].to_dict()
    parsed: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []

    for path in sorted(EIS_DIR.glob("*.txt")):
        sample = path.stem
        if sample not in label_map or sample not in initial_samples:
            continue
        try:
            parsed[sample] = read_eis(path)
        except Exception as exc:
            errors.append({"sample": sample, "error": str(exc)})

    if not parsed:
        raise RuntimeError(f"No EIS files with labels were found in {EIS_DIR}")

    target_freq = choose_frequency_grid(parsed)
    target_logf = np.log10(target_freq)
    grid = pd.DataFrame(
        {
            "freq_index": range(len(target_freq)),
            "freq_hz": target_freq,
            "log10_freq_hz": target_logf,
        }
    )

    rows = []
    for sample, df in parsed.items():
        row = {
            "sample": sample,
            "separation_coefficient": float(label_map[sample]),
            "dataset_subset": "initial_screening_no_BO",
            "n_points_raw": len(df),
        }
        for col in ["z_real_ohm", "z_imag_ohm", "z_abs_ohm", "phase_deg"]:
            vals = interpolate_spectrum(df, target_logf, col)
            for idx, value in enumerate(vals):
                row[f"{col}__f{idx:02d}"] = value
        rows.append(row)

    missing_eis = sorted((set(label_map) & initial_samples) - set(parsed))
    return pd.DataFrame(rows), grid, pd.DataFrame(errors), missing_eis


def make_per_frequency_correlations(wide: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    response_specs = [
        ("z_abs_ohm", "Zabs", False, False),
        ("z_abs_ohm", "log10_Zabs", True, False),
        ("z_real_ohm", "Zreal", False, False),
        ("z_real_ohm", "log10_Zreal", True, False),
        ("z_imag_ohm", "abs_Zimag", False, True),
        ("z_imag_ohm", "log10_abs_Zimag", True, True),
        ("phase_deg", "abs_phase_deg", False, True),
    ]
    subsets = {
        "initial_screening_no_BO": wide[wide["dataset_subset"].eq("initial_screening_no_BO")],
    }

    rows = []
    for subset_name, subset in subsets.items():
        y = subset["separation_coefficient"]
        for source_col, response, log_scale, absolute in response_specs:
            for _, freq_row in grid.iterrows():
                idx = int(freq_row["freq_index"])
                x = subset[f"{source_col}__f{idx:02d}"]
                if absolute:
                    x = x.abs()
                if log_scale:
                    x = np.log10(pd.to_numeric(x, errors="coerce").clip(lower=1e-12))
                rows.append(
                    {
                        "subset": subset_name,
                        "response": response,
                        "freq_index": idx,
                        "freq_hz": float(freq_row["freq_hz"]),
                        "log10_freq_hz": float(freq_row["log10_freq_hz"]),
                        "n": int(pd.Series(x).notna().sum()),
                        "pearson": pearson(pd.Series(x), y),
                        "spearman": spearman(pd.Series(x), y),
                    }
                )

    corr = pd.DataFrame(rows)
    corr["abs_spearman"] = corr["spearman"].abs()
    corr["abs_pearson"] = corr["pearson"].abs()
    return corr


def area_over_log_frequency(values: np.ndarray, logf: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(logf)
    if ok.sum() < 2:
        return np.nan
    return float(np.trapezoid(values[ok], logf[ok]))


def mean_finite(values: np.ndarray) -> float:
    if not np.isfinite(values).any():
        return np.nan
    return float(np.nanmean(values))


def make_band_features(wide: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    logf = grid["log10_freq_hz"].to_numpy(float)
    freq = grid["freq_hz"].to_numpy(float)
    rows = []
    for _, src in wide.iterrows():
        zreal = src[[f"z_real_ohm__f{i:02d}" for i in grid["freq_index"]]].to_numpy(float)
        zimag = src[[f"z_imag_ohm__f{i:02d}" for i in grid["freq_index"]]].to_numpy(float)
        zabs = src[[f"z_abs_ohm__f{i:02d}" for i in grid["freq_index"]]].to_numpy(float)
        phase = src[[f"phase_deg__f{i:02d}" for i in grid["freq_index"]]].to_numpy(float)

        row = {
            "sample": src["sample"],
            "separation_coefficient": src["separation_coefficient"],
            "dataset_subset": src["dataset_subset"],
        }
        for band_name, lo, hi, _ in BANDS:
            mask = (freq >= lo) & (freq <= hi)
            row[f"{band_name}__mean_Zreal"] = mean_finite(zreal[mask])
            row[f"{band_name}__mean_Zabs"] = mean_finite(zabs[mask])
            row[f"{band_name}__mean_log10_Zabs"] = mean_finite(
                np.log10(np.clip(zabs[mask], 1e-12, None))
            )
            row[f"{band_name}__mean_abs_Zimag"] = mean_finite(np.abs(zimag[mask]))
            row[f"{band_name}__mean_abs_phase_deg"] = mean_finite(np.abs(phase[mask]))
            row[f"{band_name}__area_Zabs_vs_logf"] = area_over_log_frequency(
                zabs[mask], logf[mask]
            )
            row[f"{band_name}__area_abs_Zimag_vs_logf"] = area_over_log_frequency(
                np.abs(zimag[mask]), logf[mask]
            )

        zabs_area_cols = [f"{band[0]}__area_Zabs_vs_logf" for band in BANDS]
        zimag_area_cols = [f"{band[0]}__area_abs_Zimag_vs_logf" for band in BANDS]
        row["total_Zabs_area_0p1_100kHz"] = float(np.nansum([row[c] for c in zabs_area_cols]))
        row["total_abs_Zimag_area_0p1_100kHz"] = float(
            np.nansum([row[c] for c in zimag_area_cols])
        )
        low_zabs = row["very_low_0p1_1Hz__area_Zabs_vs_logf"] + row[
            "low_1_10Hz__area_Zabs_vs_logf"
        ]
        low_zimag = row["very_low_0p1_1Hz__area_abs_Zimag_vs_logf"] + row[
            "low_1_10Hz__area_abs_Zimag_vs_logf"
        ]
        high_zabs = row["high_1k_10kHz__area_Zabs_vs_logf"] + row[
            "very_high_10k_100kHz__area_Zabs_vs_logf"
        ]
        high_zimag = row["high_1k_10kHz__area_abs_Zimag_vs_logf"] + row[
            "very_high_10k_100kHz__area_abs_Zimag_vs_logf"
        ]
        row["low_0p1_10Hz_fraction_Zabs"] = low_zabs / row["total_Zabs_area_0p1_100kHz"]
        row["low_0p1_10Hz_fraction_abs_Zimag"] = (
            low_zimag / row["total_abs_Zimag_area_0p1_100kHz"]
        )
        row["high_1k_100kHz_fraction_Zabs"] = high_zabs / row["total_Zabs_area_0p1_100kHz"]
        row["high_1k_100kHz_fraction_abs_Zimag"] = (
            high_zimag / row["total_abs_Zimag_area_0p1_100kHz"]
        )
        row["mid_10_1kHz_fraction_Zabs"] = (
            row["mid_10_1kHz__area_Zabs_vs_logf"] / row["total_Zabs_area_0p1_100kHz"]
        )
        row["mid_10_1kHz_fraction_abs_Zimag"] = (
            row["mid_10_1kHz__area_abs_Zimag_vs_logf"]
            / row["total_abs_Zimag_area_0p1_100kHz"]
        )
        row["Zreal_low_minus_high_ohm"] = (
            row["very_low_0p1_1Hz__mean_Zreal"] - row["very_high_10k_100kHz__mean_Zreal"]
        )
        row["Zabs_low_over_high"] = (
            row["very_low_0p1_1Hz__mean_Zabs"] / row["very_high_10k_100kHz__mean_Zabs"]
        )
        row["max_abs_Zimag_ohm"] = float(np.nanmax(np.abs(zimag)))
        if np.isfinite(np.abs(zimag)).any():
            idx = int(np.nanargmax(np.abs(zimag)))
            row["freq_at_max_abs_Zimag_hz"] = float(freq[idx])
            row["log10_freq_at_max_abs_Zimag"] = float(logf[idx])
        else:
            row["freq_at_max_abs_Zimag_hz"] = np.nan
            row["log10_freq_at_max_abs_Zimag"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def make_band_correlations(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        col
        for col in features.columns
        if col
        not in {
            "sample",
            "separation_coefficient",
            "dataset_subset",
        }
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
                }
            )
    corr = pd.DataFrame(rows)
    corr["abs_spearman"] = corr["spearman"].abs()
    return corr.sort_values(["subset", "abs_spearman"], ascending=[True, False])


def add_fit_line(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    xx, yy = finite_xy(x, y)
    if len(xx) < 3 or np.nanstd(xx) == 0:
        return
    coef = np.polyfit(xx, yy, 1)
    xs = np.linspace(np.nanmin(xx), np.nanmax(xx), 100)
    ax.plot(xs, np.polyval(coef, xs), color=color, linewidth=2.0)


def corr_label(df: pd.DataFrame, x_col: str) -> str:
    x = df[x_col]
    y = df["separation_coefficient"]
    return f"Spearman rho = {spearman(x, y):.2f}, n = {int(pd.to_numeric(x, errors='coerce').notna().sum())}"


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e5e7eb", linewidth=0.8, zorder=0)


def plot_validation_panel(features: pd.DataFrame, freq_corr: pd.DataFrame, subset: str, stem: str) -> None:
    data = features[features["dataset_subset"].eq("initial_screening_no_BO")].copy()

    corr_subset = freq_corr[freq_corr["subset"].eq(subset)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), constrained_layout=True)
    ax = axes[0, 0]
    for response, label, color in [
        ("log10_Zabs", "log10 |Z|", "#256d85"),
        ("log10_abs_Zimag", "log10 |Z''|", "#d9822b"),
        ("abs_phase_deg", "|phase|", "#6d4c41"),
    ]:
        sub = corr_subset[corr_subset["response"].eq(response)]
        ax.plot(sub["freq_hz"], sub["spearman"], label=label, linewidth=2.0, color=color)
    for _, lo, hi, band_label in BANDS:
        ax.axvspan(lo, hi, color="#f3f4f6", alpha=0.28, zorder=-1)
    ax.axhline(0, color="#6b7280", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Spearman rho vs HD separation")
    ax.set_title("A. Frequency-resolved EIS correlation", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)

    y_col = "separation_coefficient"
    x_total = "total_abs_Zimag_area_0p1_100kHz"
    ax = axes[0, 1]
    ax.scatter(
        np.log10(data[x_total]),
        data[y_col],
        s=24,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        color=COLORS["initial"],
    )
    add_fit_line(ax, np.log10(data[x_total]), data[y_col], COLORS["accent"])
    ax.set_xlabel("log10 total |Z''| area, 0.1 Hz-100 kHz")
    ax.set_ylabel("HD separation coefficient")
    ax.set_title("B. Total EIS loss integral", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.04,
        corr_label(data.assign(_x=np.log10(data[x_total])), "_x"),
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )
    style_axis(ax)

    x_frac = "low_0p1_10Hz_fraction_Zabs"
    ax = axes[1, 0]
    ax.scatter(
        100 * data[x_frac],
        data[y_col],
        s=24,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        color=COLORS["initial"],
    )
    add_fit_line(ax, 100 * data[x_frac], data[y_col], COLORS["accent"])
    ax.set_xlabel("Low-frequency |Z| area fraction, 0.1-10 Hz (%)")
    ax.set_ylabel("HD separation coefficient")
    ax.set_title("C. Slow-process fraction", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.04,
        corr_label(data.assign(_x=data[x_frac]), "_x"),
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )
    style_axis(ax)

    ax = axes[1, 1]
    sc = ax.scatter(
        np.log10(data[x_total]),
        100 * data[x_frac],
        c=data[y_col],
        cmap="viridis",
        s=34,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel("log10 total |Z''| area")
    ax.set_ylabel("Low-frequency |Z| fraction (%)")
    ax.set_title("D. EIS kinetic map", loc="left", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("HD separation coefficient")
    style_axis(ax)

    fig.suptitle(
        "EIS validation on initial no-BO raw data",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUT_FIG / f"{stem}.png", dpi=300)
    fig.savefig(OUT_FIG / f"{stem}.svg")
    plt.close(fig)


def plot_representative_bode(wide: pd.DataFrame, grid: pd.DataFrame) -> None:
    initial = wide[wide["dataset_subset"].eq("initial_screening_no_BO")].copy()
    top_samples = initial.nlargest(10, "separation_coefficient")["sample"].tolist()
    bottom_samples = initial.nsmallest(10, "separation_coefficient")["sample"].tolist()
    freq = grid["freq_hz"].to_numpy(float)
    idx_cols = [int(i) for i in grid["freq_index"]]

    def stack(samples: list[str], prefix: str, transform=None) -> np.ndarray:
        rows = []
        for sample in samples:
            row = wide[wide["sample"].eq(sample)].iloc[0]
            vals = row[[f"{prefix}__f{i:02d}" for i in idx_cols]].to_numpy(float)
            if transform is not None:
                vals = transform(vals)
            rows.append(vals)
        return np.vstack(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    panels = [
        ("z_abs_ohm", "|Z| (ohm)", None),
        ("z_imag_ohm", "|Z''| (ohm)", np.abs),
    ]
    for ax, (prefix, ylabel, transform) in zip(axes, panels):
        for samples, label, color in [
            (top_samples, "Top 10 HD separation", COLORS["top"]),
            (bottom_samples, "Bottom 10 HD separation", COLORS["bottom"]),
        ]:
            mat = stack(samples, prefix, transform)
            med = np.nanmedian(mat, axis=0)
            q25 = np.nanpercentile(mat, 25, axis=0)
            q75 = np.nanpercentile(mat, 75, axis=0)
            ax.plot(freq, med, color=color, linewidth=2.2, label=label)
            ax.fill_between(freq, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9)
        style_axis(ax)
    fig.suptitle("Representative EIS spectra for high vs low HD separation", fontweight="bold")
    fig.savefig(OUT_FIG / "figure3_eis_representative_bode_initial.png", dpi=300)
    fig.savefig(OUT_FIG / "figure3_eis_representative_bode_initial.svg")
    plt.close(fig)


def write_summary(
    labels: pd.DataFrame,
    wide: pd.DataFrame,
    parse_errors: pd.DataFrame,
    missing_eis: list[str],
    band_corr: pd.DataFrame,
) -> None:
    focus_features = [
        "total_Zabs_area_0p1_100kHz",
        "total_abs_Zimag_area_0p1_100kHz",
        "low_0p1_10Hz_fraction_Zabs",
        "low_0p1_10Hz_fraction_abs_Zimag",
        "high_1k_100kHz_fraction_Zabs",
        "high_1k_100kHz_fraction_abs_Zimag",
        "mid_10_1kHz_fraction_Zabs",
        "mid_10_1kHz_fraction_abs_Zimag",
    ]
    focus = band_corr[band_corr["feature"].isin(focus_features)].copy()
    focus = focus.sort_values(["subset", "feature"])

    lines = [
        "Figure 3 initial no-BO EIS-HD validation",
        "========================================",
        "",
        f"EIS files in source folder: {len(list(EIS_DIR.glob('*.txt')))}",
        f"Initial-screening separation labels: {len(labels)}",
        f"Initial-screening matched EIS spectra: {len(wide)}",
        f"Initial-screening no-BO matched spectra: {wide['dataset_subset'].eq('initial_screening_no_BO').sum()}",
        f"Missing EIS labels: {missing_eis}",
        f"Parse errors: {len(parse_errors)}",
        "",
        "Focused correlations for total integral and frequency fractions:",
        focus[["subset", "feature", "n", "pearson", "spearman"]].round(4).to_string(index=False),
        "",
        "Interpretation:",
        "- The total EIS area/integral features are negatively correlated with HD separation.",
        "- The low-frequency |Z| area fraction is also negatively correlated with HD separation.",
        "- The low-frequency |Z''| fraction is weaker in the raw-frequency proxy, so the stronger",
        "  low-frequency-fraction statement should be tied to |Z| area or to a later DRT/FRT",
        "  relaxation-area decomposition.",
    ]
    (OUT_DATA / "eis_hd_validation_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    initial_samples = load_initial_samples()
    labels = labels[labels["sample"].isin(initial_samples)].copy()
    wide, grid, parse_errors, missing_eis = build_interpolated_matrix(labels, initial_samples)
    freq_corr = make_per_frequency_correlations(wide, grid)
    band_features = make_band_features(wide, grid)
    band_corr = make_band_correlations(band_features)

    grid.to_csv(OUT_DATA / "eis_frequency_grid.csv", index=False)
    parse_errors.to_csv(OUT_DATA / "eis_parse_errors.csv", index=False)
    wide.to_csv(OUT_DATA / "eis_interpolated_frequency_matrix.csv", index=False)
    freq_corr.to_csv(OUT_DATA / "eis_per_frequency_correlations.csv", index=False)
    band_features.to_csv(OUT_DATA / "eis_band_integral_features.csv", index=False)
    band_corr.to_csv(OUT_DATA / "eis_band_feature_correlations.csv", index=False)

    plot_validation_panel(
        band_features,
        freq_corr,
        subset="initial_screening_no_BO",
        stem="figure3_eis_validation_initial",
    )
    plot_representative_bode(wide, grid)
    write_summary(labels, wide, parse_errors, missing_eis, band_corr)

    print(f"Matched EIS spectra: {len(wide)}")
    print(f"Initial no-BO spectra: {wide['dataset_subset'].eq('initial_screening_no_BO').sum()}")
    print(f"Outputs written to: {OUT_DATA}")
    print(f"Figures written to: {OUT_FIG}")


if __name__ == "__main__":
    main()
