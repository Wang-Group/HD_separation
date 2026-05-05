#!/usr/bin/env python3
"""Initial-screening no-BO DRT fitting heatmap for Figure 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable


FIG3_DIR = Path(__file__).resolve().parent
DRT_DATA_DIR = FIG3_DIR / "data" / "eis_pydrttools_validation"
OUT_FIG = FIG3_DIR / "outputs" / "eis_pydrttools_validation"

FEATURES_FILE = DRT_DATA_DIR / "drt_features.csv"
DISTRIBUTIONS_FILE = DRT_DATA_DIR / "drt_distributions_long.csv"

HD_CMAP = "coolwarm"
DRT_CMAP = "magma"
GAMMA_LOG_OFFSET = 1e-4

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.weight": "bold",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "svg.fonttype": "none",
    }
)


INITIAL_SUBSET = "initial_screening_no_BO"


def load_heatmap_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(FEATURES_FILE)
    distributions = pd.read_csv(DISTRIBUTIONS_FILE)
    features = features[features["dataset_subset"].eq(INITIAL_SUBSET)].copy()

    features = features.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["sample", "separation_coefficient"]
    )
    available = set(distributions["sample"].unique())
    features = features[features["sample"].isin(available)].copy()
    features = features.sort_values(
        ["separation_coefficient", "drt_total_area_ohm", "sample"],
        ascending=[True, False, True],
    )
    return features, distributions


def make_matrix(features: pd.DataFrame, distributions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    samples = features["sample"].tolist()
    matrix = (
        distributions[distributions["sample"].isin(samples)]
        .pivot(index="sample", columns="log10_tau_s", values="gamma_ohm")
        .reindex(samples)
        .sort_index(axis=1)
    )
    gamma = matrix.to_numpy(float)
    log_tau = matrix.columns.to_numpy(float)
    return gamma, log_tau


def draw_heatmap(output_stem: str) -> None:
    features, distributions = load_heatmap_data()
    gamma, log_tau = make_matrix(features, distributions)
    gamma_plot = np.clip(gamma, 0, None) + GAMMA_LOG_OFFSET
    sep = features["separation_coefficient"].to_numpy(float)

    vmin = float(np.nanpercentile(gamma_plot, 5))
    vmax = float(np.nanpercentile(gamma_plot, 99))
    sep_norm = Normalize(vmin=float(sep.min()), vmax=float(sep.max()))

    fig = plt.figure(figsize=(6.8, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[34, 0.8, 1.0, 1.0],
        wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[0, 1], sharey=ax)
    cax_gamma = fig.add_subplot(gs[0, 2])
    cax_sep = fig.add_subplot(gs[0, 3])

    im = ax.imshow(
        gamma_plot,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=DRT_CMAP,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=(log_tau.min(), log_tau.max(), 0, len(features)),
    )
    ax_strip.imshow(
        sep[:, None],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=HD_CMAP,
        norm=sep_norm,
        extent=(0, 1, 0, len(features)),
    )

    tau_ticks = np.arange(np.ceil(log_tau.min()), np.floor(log_tau.max()) + 1)
    ax.set_xticks(tau_ticks)
    ax.set_xticklabels([rf"$10^{{{int(t)}}}$" for t in tau_ticks])
    ax.set_xlabel("Relaxation Time (s)")
    ax.set_ylabel("Samples")
    ax.set_yticks([0, len(features)])
    ax.set_yticklabels(["Low", "High"])
    ax.tick_params(width=1.6, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    ax_strip.set_xticks([])
    ax_strip.tick_params(left=False, labelleft=False)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    cbar_gamma = fig.colorbar(im, cax=cax_gamma)
    cbar_gamma.set_label("DRT Intensity (Ohm)", fontweight="bold")
    cbar_gamma.ax.tick_params(labelsize=10, width=1.2, length=4)
    cbar_gamma.outline.set_linewidth(1.2)

    sep_sm = ScalarMappable(norm=sep_norm, cmap=HD_CMAP)
    sep_sm.set_array([])
    cbar_sep = fig.colorbar(sep_sm, cax=cax_sep)
    cbar_sep.set_label("Separation Coefficient", fontweight="bold")
    cbar_sep.ax.tick_params(labelsize=10, width=1.2, length=4)
    cbar_sep.outline.set_linewidth(1.2)

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / f"{output_stem}.png", dpi=300)
    fig.savefig(OUT_FIG / f"{output_stem}.svg")
    plt.close(fig)

    order = features[
        [
            "sample",
            "separation_coefficient",
            "dataset_subset",
            "imag_loss_rel_rmse",
            "drt_total_area_ohm",
            "drt_peak_tau_s",
        ]
    ].copy()
    order.insert(0, "heatmap_row", np.arange(len(order)))
    order.to_csv(DRT_DATA_DIR / f"{output_stem}_sample_order.csv", index=False)


def main() -> None:
    draw_heatmap(
        output_stem="figure3_drt_fitting_heatmap_initial",
    )
    print(f"Initial no-BO DRT fitting heatmap written to: {OUT_FIG}")


if __name__ == "__main__":
    main()
