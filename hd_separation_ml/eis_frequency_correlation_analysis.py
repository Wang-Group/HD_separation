from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "hd_separation_ml" / "data_extraction"
EIS_DIR = DATA_DIR / "all_data" / "EIS"
LABEL_FILE = DATA_DIR / "separation_coefficient.xlsx"
OUTPUT_DIR = ROOT / "hd_separation_ml" / "outputs" / "eis_frequency_analysis"


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


def pearson(x: pd.Series, y: pd.Series) -> float:
    return pd.to_numeric(x, errors="coerce").corr(pd.to_numeric(y, errors="coerce"))


def spearman(x: pd.Series, y: pd.Series) -> float:
    x_rank = pd.to_numeric(x, errors="coerce").rank(method="average")
    y_rank = pd.to_numeric(y, errors="coerce").rank(method="average")
    return x_rank.corr(y_rank)


def choose_frequency_grid(parsed: dict[str, pd.DataFrame]) -> np.ndarray:
    counts = pd.Series({sample: len(df) for sample, df in parsed.items()})
    mode_count = int(counts.mode().iloc[0])
    ref_sample = next(sample for sample, count in counts.items() if count == mode_count)
    return parsed[ref_sample]["freq_hz"].to_numpy(float)


def interpolate_spectrum(df: pd.DataFrame, target_logf: np.ndarray, column: str) -> np.ndarray:
    logf = np.log10(df["freq_hz"].to_numpy(float))
    order = np.argsort(logf)
    logf = logf[order]
    vals = df[column].to_numpy(float)[order]
    out = np.full_like(target_logf, np.nan, dtype=float)
    in_range = (target_logf >= logf.min() - 1e-9) & (target_logf <= logf.max() + 1e-9)
    out[in_range] = np.interp(target_logf[in_range], logf, vals)
    return out


def build_feature_matrix(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_map = labels.set_index("sample")["separation_coefficient"].to_dict()
    parsed: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []

    for path in sorted(EIS_DIR.glob("*.txt")):
        sample = path.stem
        if sample not in label_map:
            continue
        try:
            parsed[sample] = read_eis(path)
        except Exception as exc:
            errors.append({"sample": sample, "error": str(exc)})

    target_freq = choose_frequency_grid(parsed)
    order = np.argsort(np.log10(target_freq))
    target_freq = target_freq[order]
    target_logf = np.log10(target_freq)

    rows = []
    for sample, df in parsed.items():
        row = {
            "sample": sample,
            "separation_coefficient": float(label_map[sample]),
            "dataset_subset": (
                "initial_screening_no_BO"
                if not re.search(r"\d$", sample)
                else "digit_suffixed_followup_or_BO"
            ),
            "n_points_raw": len(df),
        }
        for col in ["z_real_ohm", "z_imag_ohm", "z_abs_ohm", "phase_deg"]:
            interp_vals = interpolate_spectrum(df, target_logf, col)
            for idx, value in enumerate(interp_vals):
                row[f"{col}__f{idx:02d}"] = value
        rows.append(row)

    wide = pd.DataFrame(rows)
    grid = pd.DataFrame(
        {
            "freq_index": range(len(target_freq)),
            "freq_hz": target_freq,
            "log10_freq_hz": target_logf,
        }
    )
    return wide, grid, pd.DataFrame(errors)


def make_per_frequency_correlations(wide: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    response_columns = [
        ("z_real_ohm", "Zreal"),
        ("z_imag_ohm", "Zimag_signed"),
        ("z_abs_ohm", "Zabs"),
        ("phase_deg", "phase_deg"),
    ]
    subsets = {
        "all_matched": wide,
        "initial_screening_no_BO": wide[wide["dataset_subset"].eq("initial_screening_no_BO")],
        "digit_suffixed_followup_or_BO": wide[
            wide["dataset_subset"].eq("digit_suffixed_followup_or_BO")
        ],
    }
    rows = []
    for subset_name, subset in subsets.items():
        y = subset["separation_coefficient"]
        for base_col, response in response_columns:
            for _, freq_row in grid.iterrows():
                idx = int(freq_row["freq_index"])
                col = f"{base_col}__f{idx:02d}"
                x = subset[col]
                rows.append(
                    {
                        "subset": subset_name,
                        "response": response,
                        "freq_index": idx,
                        "freq_hz": freq_row["freq_hz"],
                        "log10_freq_hz": freq_row["log10_freq_hz"],
                        "n": int(x.notna().sum()),
                        "pearson": pearson(x, y),
                        "spearman": spearman(x, y),
                    }
                )
                if base_col in {"z_imag_ohm", "phase_deg"}:
                    x_abs = x.abs()
                    rows.append(
                        {
                            "subset": subset_name,
                            "response": "abs_" + response,
                            "freq_index": idx,
                            "freq_hz": freq_row["freq_hz"],
                            "log10_freq_hz": freq_row["log10_freq_hz"],
                            "n": int(x_abs.notna().sum()),
                            "pearson": pearson(x_abs, y),
                            "spearman": spearman(x_abs, y),
                        }
                    )

        for base_col, response in [
            ("z_real_ohm", "log10_Zreal"),
            ("z_abs_ohm", "log10_Zabs"),
            ("z_imag_ohm", "log10_abs_Zimag"),
        ]:
            for _, freq_row in grid.iterrows():
                idx = int(freq_row["freq_index"])
                col = f"{base_col}__f{idx:02d}"
                vals = subset[col].abs() if base_col == "z_imag_ohm" else subset[col]
                x = np.log10(vals.clip(lower=1e-12))
                rows.append(
                    {
                        "subset": subset_name,
                        "response": response,
                        "freq_index": idx,
                        "freq_hz": freq_row["freq_hz"],
                        "log10_freq_hz": freq_row["log10_freq_hz"],
                        "n": int(pd.Series(x).notna().sum()),
                        "pearson": pearson(pd.Series(x), y),
                        "spearman": spearman(pd.Series(x), y),
                    }
                )

    corr = pd.DataFrame(rows)
    corr["abs_spearman"] = corr["spearman"].abs()
    corr["abs_pearson"] = corr["pearson"].abs()
    return corr


def band_indices(grid: pd.DataFrame, lo: float, hi: float) -> list[int]:
    return [
        int(row["freq_index"])
        for _, row in grid.iterrows()
        if lo <= float(row["freq_hz"]) < hi
    ]


def make_band_features(wide: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    bands = [
        ("very_low_0p1_1Hz", 0.1, 1.0),
        ("low_1_10Hz", 1.0, 10.0),
        ("mid_10_1kHz", 10.0, 1000.0),
        ("high_1k_10kHz", 1000.0, 10000.0),
        ("very_high_10k_100kHz", 10000.0, 100000.1),
    ]
    rows = []
    for _, row in wide.iterrows():
        out = {
            "sample": row["sample"],
            "separation_coefficient": row["separation_coefficient"],
            "dataset_subset": row["dataset_subset"],
        }
        for band_name, lo, hi in bands:
            idxs = band_indices(grid, lo, hi)
            logf = np.array([grid.loc[i, "log10_freq_hz"] for i in idxs], dtype=float)
            for base, label in [
                ("z_real_ohm", "Zreal"),
                ("z_imag_ohm", "Zimag_signed"),
                ("z_abs_ohm", "Zabs"),
                ("phase_deg", "phase_deg"),
            ]:
                vals = np.array([row[f"{base}__f{i:02d}"] for i in idxs], dtype=float)
                out[f"{band_name}__mean_{label}"] = float(np.nanmean(vals))
                if label in {"Zreal", "Zabs"}:
                    out[f"{band_name}__mean_log10_{label}"] = float(
                        np.nanmean(np.log10(np.clip(vals, 1e-12, None)))
                    )
                if label in {"Zimag_signed", "phase_deg"}:
                    out[f"{band_name}__mean_abs_{label}"] = float(np.nanmean(np.abs(vals)))
            if len(idxs) >= 2:
                z_abs = np.array([row[f"z_abs_ohm__f{i:02d}"] for i in idxs], dtype=float)
                z_imag_abs = np.abs(
                    np.array([row[f"z_imag_ohm__f{i:02d}"] for i in idxs], dtype=float)
                )
                valid_abs = np.isfinite(logf) & np.isfinite(z_abs)
                valid_imag = np.isfinite(logf) & np.isfinite(z_imag_abs)
                out[f"{band_name}__area_Zabs_vs_logf"] = (
                    float(np.trapezoid(z_abs[valid_abs], logf[valid_abs]))
                    if valid_abs.sum() >= 2
                    else np.nan
                )
                out[f"{band_name}__area_abs_Zimag_vs_logf"] = (
                    float(np.trapezoid(z_imag_abs[valid_imag], logf[valid_imag]))
                    if valid_imag.sum() >= 2
                    else np.nan
                )

        first_idx = int(grid["freq_index"].min())
        last_idx = int(grid["freq_index"].max())
        z_real_low = row[f"z_real_ohm__f{first_idx:02d}"]
        z_real_high = row[f"z_real_ohm__f{last_idx:02d}"]
        z_abs_low = row[f"z_abs_ohm__f{first_idx:02d}"]
        z_abs_high = row[f"z_abs_ohm__f{last_idx:02d}"]
        abs_zimag = np.array(
            [abs(row[f"z_imag_ohm__f{i:02d}"]) for i in grid["freq_index"]],
            dtype=float,
        )
        out["Zreal_low_minus_high_ohm"] = z_real_low - z_real_high
        out["Zabs_low_over_high"] = z_abs_low / z_abs_high if z_abs_high else np.nan
        out["max_abs_Zimag_ohm"] = float(np.nanmax(abs_zimag))
        max_idx = int(np.nanargmax(abs_zimag))
        out["freq_at_max_abs_Zimag_hz"] = float(grid.loc[max_idx, "freq_hz"])
        out["log10_freq_at_max_abs_Zimag"] = float(grid.loc[max_idx, "log10_freq_hz"])
        rows.append(out)
    return pd.DataFrame(rows)


def make_band_correlations(band_df: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all_matched": band_df,
        "initial_screening_no_BO": band_df[
            band_df["dataset_subset"].eq("initial_screening_no_BO")
        ],
        "digit_suffixed_followup_or_BO": band_df[
            band_df["dataset_subset"].eq("digit_suffixed_followup_or_BO")
        ],
    }
    rows = []
    for subset_name, subset in subsets.items():
        y = subset["separation_coefficient"]
        for col in subset.columns:
            if col in {"sample", "separation_coefficient", "dataset_subset"}:
                continue
            x = pd.to_numeric(subset[col], errors="coerce")
            s = spearman(x, y)
            rows.append(
                {
                    "subset": subset_name,
                    "feature": col,
                    "n": int(x.notna().sum()),
                    "pearson": pearson(x, y),
                    "spearman": s,
                    "abs_spearman": abs(s),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["subset", "abs_spearman"], ascending=[True, False]
    )


def write_summary(
    labels: pd.DataFrame,
    wide: pd.DataFrame,
    errors: pd.DataFrame,
    corr: pd.DataFrame,
    band_corr: pd.DataFrame,
) -> None:
    label_samples = set(labels["sample"])
    file_samples = {p.stem for p in EIS_DIR.glob("*.txt")}
    lines = [
        f"EIS files: {len(file_samples)}",
        f"Matched labeled spectra: {len(wide)}",
        f"Initial-screening no-BO spectra: {(wide.dataset_subset == 'initial_screening_no_BO').sum()}",
        f"Digit-suffixed BO/follow-up spectra: {(wide.dataset_subset == 'digit_suffixed_followup_or_BO').sum()}",
        f"Missing EIS labels: {sorted(label_samples - file_samples)}",
        f"Parse errors: {len(errors)}",
        "",
    ]
    for subset in [
        "initial_screening_no_BO",
        "all_matched",
        "digit_suffixed_followup_or_BO",
    ]:
        lines.append(f"== Top per-frequency Spearman correlations: {subset} ==")
        top = corr[corr["subset"].eq(subset)].sort_values(
            "abs_spearman", ascending=False
        ).head(15)
        lines.append(top[["response", "freq_hz", "spearman", "pearson", "n"]].to_string(index=False))
        lines.append("")
        lines.append(f"== Top band/shape Spearman correlations: {subset} ==")
        top_band = band_corr[band_corr["subset"].eq(subset)].head(15)
        lines.append(top_band[["feature", "spearman", "pearson", "n"]].to_string(index=False))
        lines.append("")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    wide, grid, errors = build_feature_matrix(labels)
    corr = make_per_frequency_correlations(wide, grid)
    band_df = make_band_features(wide, grid)
    band_corr = make_band_correlations(band_df)

    wide.to_csv(OUTPUT_DIR / "eis_frequency_feature_matrix.csv", index=False)
    grid.to_csv(OUTPUT_DIR / "frequency_grid.csv", index=False)
    errors.to_csv(OUTPUT_DIR / "eis_frequency_parse_errors.csv", index=False)
    corr.to_csv(OUTPUT_DIR / "per_frequency_correlations.csv", index=False)
    band_df.to_csv(OUTPUT_DIR / "band_frequency_features.csv", index=False)
    band_corr.to_csv(OUTPUT_DIR / "band_feature_correlations.csv", index=False)
    write_summary(labels, wide, errors, corr, band_corr)
    print((OUTPUT_DIR / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
