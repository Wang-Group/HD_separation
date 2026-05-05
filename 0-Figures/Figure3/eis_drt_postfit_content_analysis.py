#!/usr/bin/env python3
"""Post-fit pyDRTtools content analysis for initial no-BO HD separation.

This script asks whether DRT-derived quantities beyond total DRT area carry
information about HD separation. It computes direct Spearman correlations,
partial Spearman correlations after controlling log total DRT area, top/bottom
HD quantile contrasts, tau-resolved residual correlations, and a compact
cross-validated predictive check. All reported Figure 3 analyses are restricted
to the initial-screening no-BO dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FIG3_DIR = Path(__file__).resolve().parent
IN_DATA = FIG3_DIR / "data" / "eis_pydrttools_validation"
OUT_FIG = FIG3_DIR / "outputs" / "eis_pydrttools_validation"

FEATURES_FILE = IN_DATA / "drt_features.csv"
DISTRIBUTIONS_FILE = IN_DATA / "drt_distributions_long.csv"


CANDIDATE_FEATURES = [
    "drt_peak_gamma_ohm",
    "drt_peak_tau_s",
    "short_tau_1k_100kHz__area_ohm",
    "short_tau_1k_100kHz__fraction",
    "mid_tau_10_1kHz__area_ohm",
    "mid_tau_10_1kHz__fraction",
    "long_tau_0p1_10Hz__area_ohm",
    "long_tau_0p1_10Hz__fraction",
    "very_long_tau_0p1_1Hz__area_ohm",
    "very_long_tau_0p1_1Hz__fraction",
    "long_over_short_tau_area",
    "pydrttools_R_ohm",
    "pydrttools_L_h",
    "real_loss_r2",
    "imag_loss_r2",
    "combined_loss_r2",
    "imag_loss_rel_rmse",
]

COMPACT_MODEL_FEATURES = [
    "drt_total_area_ohm",
    "drt_peak_gamma_ohm",
    "drt_peak_tau_s",
    "short_tau_1k_100kHz__area_ohm",
    "mid_tau_10_1kHz__area_ohm",
    "long_tau_0p1_10Hz__area_ohm",
    "very_long_tau_0p1_1Hz__area_ohm",
    "short_tau_1k_100kHz__fraction",
    "mid_tau_10_1kHz__fraction",
    "long_tau_0p1_10Hz__fraction",
    "very_long_tau_0p1_1Hz__fraction",
    "long_over_short_tau_area",
]


def rank(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    return xx[ok], yy[ok]


def corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    xx, yy = finite_pair(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if len(xx) < 3 or np.nanstd(xx) == 0 or np.nanstd(yy) == 0:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    return corr(rank(x), rank(y))


def residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    zz = np.asarray(z, dtype=float).reshape(-1, 1)
    ok = np.isfinite(xx) & np.isfinite(zz[:, 0])
    out = np.full_like(xx, np.nan, dtype=float)
    if ok.sum() < 3:
        return out
    model = LinearRegression().fit(zz[ok], xx[ok])
    out[ok] = xx[ok] - model.predict(zz[ok])
    return out


def partial_spearman(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    z: pd.Series | np.ndarray,
) -> float:
    rx = rank(x)
    ry = rank(y)
    rz = rank(z)
    return corr(residualize(rx, rz), residualize(ry, rz))


def transformed_feature(frame: pd.DataFrame, col: str) -> np.ndarray:
    values = frame[col].to_numpy(float)
    if (
        "area_ohm" in col
        or col
        in {
            "drt_total_area_ohm",
            "drt_peak_gamma_ohm",
            "long_over_short_tau_area",
            "pydrttools_R_ohm",
            "pydrttools_L_h",
        }
    ):
        return np.log10(np.clip(np.abs(values), 1e-12, None))
    if col == "drt_peak_tau_s":
        return np.log10(np.clip(values, 1e-12, None))
    return values


def feature_partial_correlations(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {
        "initial_screening_no_BO": features[
            features["dataset_subset"].eq("initial_screening_no_BO")
        ],
    }
    for subset_name, subset in subsets.items():
        y = subset["separation_coefficient"].to_numpy(float)
        total = transformed_feature(subset, "drt_total_area_ohm")
        for col in CANDIDATE_FEATURES:
            x = transformed_feature(subset, col)
            rows.append(
                {
                    "subset": subset_name,
                    "feature": col,
                    "n": int(np.isfinite(x).sum()),
                    "spearman": spearman(x, y),
                    "partial_spearman_control_log_total_area": partial_spearman(
                        x, y, total
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_partial"] = out["partial_spearman_control_log_total_area"].abs()
    return out.sort_values(["subset", "abs_partial"], ascending=[True, False])


def top_bottom_medians(features: pd.DataFrame) -> pd.DataFrame:
    subset = features[features["dataset_subset"].eq("initial_screening_no_BO")].copy()
    bottom = subset[
        subset["separation_coefficient"]
        <= subset["separation_coefficient"].quantile(0.2)
    ]
    top = subset[
        subset["separation_coefficient"]
        >= subset["separation_coefficient"].quantile(0.8)
    ]
    cols = [
        "separation_coefficient",
        "drt_total_area_ohm",
        "drt_peak_gamma_ohm",
        "mid_tau_10_1kHz__area_ohm",
        "long_tau_0p1_10Hz__area_ohm",
        "short_tau_1k_100kHz__fraction",
        "long_tau_0p1_10Hz__fraction",
        "drt_peak_tau_s",
    ]
    rows = []
    for col in cols:
        bottom_med = float(bottom[col].median())
        top_med = float(top[col].median())
        rows.append(
            {
                "feature": col,
                "bottom20_median": bottom_med,
                "top20_median": top_med,
                "top_over_bottom": top_med / bottom_med if bottom_med else np.nan,
            }
        )
    return pd.DataFrame(rows)


def tau_partial_correlations(features: pd.DataFrame, distributions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {
        "initial_screening_no_BO": features[
            features["dataset_subset"].eq("initial_screening_no_BO")
        ],
    }
    for subset_name, subset in subsets.items():
        samples = set(subset["sample"])
        pivot = distributions[distributions["sample"].isin(samples)].pivot(
            index="sample", columns="tau_s", values="gamma_ohm"
        )
        meta = subset.set_index("sample").loc[pivot.index]
        y = meta["separation_coefficient"].to_numpy(float)
        total = np.log10(meta["drt_total_area_ohm"].clip(lower=1e-12).to_numpy(float))
        for tau in pivot.columns:
            gamma = pivot[tau].to_numpy(float)
            rows.append(
                {
                    "subset": subset_name,
                    "tau_s": float(tau),
                    "equivalent_freq_hz": 1.0 / (2.0 * np.pi * float(tau)),
                    "spearman": spearman(gamma, y),
                    "partial_spearman_control_log_total_area": partial_spearman(
                        gamma, y, total
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_partial"] = out["partial_spearman_control_log_total_area"].abs()
    return out


def predictive_checks(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {
        "initial_screening_no_BO": features[
            features["dataset_subset"].eq("initial_screening_no_BO")
        ],
    }
    for subset_name, subset in subsets.items():
        y = subset["separation_coefficient"].to_numpy(float)
        x_total = transformed_feature(subset, "drt_total_area_ohm").reshape(-1, 1)
        x_compact = subset[COMPACT_MODEL_FEATURES].copy()
        for col in x_compact.columns:
            x_compact[col] = transformed_feature(subset, col)
        x_compact = (
            x_compact.replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(float)
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=17)
        models = [
            (
                "ridge_total_area_only",
                make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
                x_total,
            ),
            (
                "ridge_compact_drt",
                make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
                x_compact,
            ),
            (
                "random_forest_compact_drt",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=8,
                    random_state=17,
                    n_jobs=-1,
                ),
                x_compact,
            ),
        ]
        for model_name, model, x_values in models:
            r2 = cross_val_score(model, x_values, y, cv=cv, scoring="r2")
            mae = -cross_val_score(
                model, x_values, y, cv=cv, scoring="neg_mean_absolute_error"
            )
            rows.append(
                {
                    "subset": subset_name,
                    "model": model_name,
                    "mean_cv_r2": float(r2.mean()),
                    "sd_cv_r2": float(r2.std()),
                    "mean_cv_mae": float(mae.mean()),
                }
            )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)


def plot_partial_features(partial: pd.DataFrame) -> None:
    subset = partial[partial["subset"].eq("initial_screening_no_BO")].head(10)
    labels = subset["feature"].str.replace("__", "\n", regex=False)
    values = subset["partial_spearman_control_log_total_area"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    colors = np.where(values >= 0, "#256d85", "#d9822b")
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#6b7280", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Partial Spearman rho after controlling log total DRT area")
    ax.set_title("DRT shape descriptors beyond total area", fontweight="bold")
    style_axis(ax)
    fig.savefig(OUT_FIG / "figure3_drt_postfit_partial_feature_correlations.png", dpi=300)
    fig.savefig(OUT_FIG / "figure3_drt_postfit_partial_feature_correlations.svg")
    plt.close(fig)


def plot_tau_partial(tau_corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for subset_name, label, color in [
        ("initial_screening_no_BO", "Initial no-BO", "#256d85"),
    ]:
        sub = tau_corr[tau_corr["subset"].eq(subset_name)]
        ax.plot(
            sub["tau_s"],
            sub["partial_spearman_control_log_total_area"],
            label=label,
            color=color,
            linewidth=2.0,
        )
    ax.axhline(0, color="#6b7280", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Relaxation time tau (s)")
    ax.set_ylabel("Partial Spearman rho")
    ax.set_title("Tau-resolved DRT signal beyond total area", fontweight="bold")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.savefig(OUT_FIG / "figure3_drt_postfit_tau_partial_correlation.png", dpi=300)
    fig.savefig(OUT_FIG / "figure3_drt_postfit_tau_partial_correlation.svg")
    plt.close(fig)


def write_summary(
    partial: pd.DataFrame,
    medians: pd.DataFrame,
    tau_corr: pd.DataFrame,
    predictive: pd.DataFrame,
) -> None:
    initial_partial = partial[partial["subset"].eq("initial_screening_no_BO")]
    initial_tau = tau_corr[tau_corr["subset"].eq("initial_screening_no_BO")]
    lines = [
        "Post-fit DRT content analysis, initial no-BO",
        "============================================",
        "",
        "Question: which DRT-derived quantities besides total DRT area relate to HD separation?",
        "",
        "Top direct and partial feature associations, initial 561:",
        initial_partial[
            [
                "feature",
                "n",
                "spearman",
                "partial_spearman_control_log_total_area",
            ]
        ]
        .head(12)
        .round(4)
        .to_string(index=False),
        "",
        "Initial top/bottom 20% HD medians:",
        medians.round(4).to_string(index=False),
        "",
        "Strongest initial tau-resolved gamma associations:",
        initial_tau.assign(abs_s=initial_tau["spearman"].abs())
        .sort_values("abs_s", ascending=False)
        .head(8)[
            [
                "tau_s",
                "equivalent_freq_hz",
                "spearman",
                "partial_spearman_control_log_total_area",
            ]
        ]
        .round(5)
        .to_string(index=False),
        "",
        "Exploratory 5-fold predictive checks:",
        predictive.round(4).to_string(index=False),
        "",
        "Interpretation:",
        "- Total DRT area remains the main single descriptor.",
        "- DRT peak height and mid/long-tau absolute areas track HD mostly because they",
        "  co-vary with the total relaxation resistance.",
        "- The strongest tau-resolved signal is near tau = 1-2 ms, equivalent to roughly",
        "  80-150 Hz; this signal partly remains after controlling total DRT area.",
        "- Short-tau fraction is positively associated with HD separation, but this is a",
        "  normalized shape effect and should be framed as relative redistribution rather",
        "  than an independent absolute resistance increase.",
    ]
    (IN_DATA / "drt_postfit_content_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURES_FILE)
    distributions = pd.read_csv(DISTRIBUTIONS_FILE)
    features = features[features["dataset_subset"].eq("initial_screening_no_BO")].copy()
    distributions = distributions[distributions["sample"].isin(features["sample"])].copy()

    partial = feature_partial_correlations(features)
    medians = top_bottom_medians(features)
    tau_corr = tau_partial_correlations(features, distributions)
    predictive = predictive_checks(features)

    partial.drop(columns=["abs_partial"]).to_csv(
        IN_DATA / "drt_postfit_feature_partial_correlations.csv", index=False
    )
    medians.to_csv(IN_DATA / "drt_postfit_top_bottom_medians.csv", index=False)
    tau_corr.drop(columns=["abs_partial"]).to_csv(
        IN_DATA / "drt_postfit_tau_partial_correlations.csv", index=False
    )
    predictive.to_csv(IN_DATA / "drt_postfit_predictive_checks.csv", index=False)

    plot_partial_features(partial)
    plot_tau_partial(tau_corr)
    write_summary(partial, medians, tau_corr, predictive)

    print(f"Post-fit DRT content outputs written to: {IN_DATA}")
    print(f"Post-fit DRT content figures written to: {OUT_FIG}")


if __name__ == "__main__":
    main()
