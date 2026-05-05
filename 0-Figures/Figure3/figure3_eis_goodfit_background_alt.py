#!/usr/bin/env python3
"""Alternate Figure 3B/AB with good-fit EIS spectra as a background layer.

This script intentionally writes timestamped files so existing Figure 3 outputs
are not overwritten while comparing the denser EIS visualization.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure3_eis_drt_ab_panels as base


BACKGROUND_FITS_FILE = base.DRT_DATA_DIR / "figure3B_background_good_fit_eis_fits.csv"
BACKGROUND_LINE_ALPHA = 0.80
BACKGROUND_LINEWIDTH = 0.52


def output_paths(stem: str, timestamp: str) -> tuple[Path, Path]:
    png = base.OUT_FIG / f"{stem}_{timestamp}.png"
    svg = base.OUT_FIG / f"{stem}_{timestamp}.svg"
    if png.exists() or svg.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {png} / {svg}")
    return png, svg


def load_good_fit_background_fits(
    data: pd.DataFrame, foreground_samples: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not BACKGROUND_FITS_FILE.exists():
        raise FileNotFoundError(
            f"Expected fitted background curves at {BACKGROUND_FITS_FILE}"
        )
    excluded = foreground_samples
    fits = pd.read_csv(BACKGROUND_FITS_FILE)
    fits = fits[
        (fits["freq_hz"] <= base.NYQUIST_MAX_FREQ_HZ)
        & (~fits["sample"].isin(excluded))
    ].copy()
    fits = fits[
        (fits["fit_z_real_rs_ohm"] >= 0)
        & (fits["fit_neg_z_imag_ohm"] >= 0)
        & (fits["fit_z_real_rs_ohm"] <= base.NYQUIST_XLIM_OHM)
        & (fits["fit_neg_z_imag_ohm"] <= base.NYQUIST_YLIM_OHM)
    ].copy()

    meta_cols = [
        "sample",
        "imag_loss_rel_rmse",
        "imag_loss_r2",
        "drt_total_area_ohm",
    ]
    meta = data[meta_cols].drop_duplicates("sample")
    summary = (
        fits.groupby(["sample", "separation_coefficient"], as_index=False)
        .agg(
            background_fit_max_x_ohm=("fit_z_real_rs_ohm", "max"),
            background_fit_max_y_ohm=("fit_neg_z_imag_ohm", "max"),
            n_background_points=("freq_hz", "size"),
        )
        .merge(meta, on="sample", how="left")
        .sort_values("separation_coefficient")
    )
    return fits, summary


def plot_good_fit_background(ax: plt.Axes, data: pd.DataFrame) -> pd.DataFrame:
    top, bottom = base.select_diverse_extreme_samples(data)
    foreground = set(top + bottom)
    fits, background = load_good_fit_background_fits(data, foreground)
    norm = base.hd_color_norm(data)
    cmap = plt.get_cmap(base.HD_CMAP)

    for row in background.itertuples(index=False):
        curve = fits[fits["sample"].eq(row.sample)].sort_values(
            "freq_hz", ascending=False
        )
        color = cmap(norm(float(row.separation_coefficient)))
        ax.plot(
            curve["fit_z_real_rs_ohm"],
            curve["fit_neg_z_imag_ohm"],
            color=color,
            alpha=BACKGROUND_LINE_ALPHA,
            linewidth=BACKGROUND_LINEWIDTH,
            zorder=0.2,
        )
    return background


def save_background_csv(background: pd.DataFrame, timestamp: str) -> Path:
    path = base.DRT_DATA_DIR / f"figure3B_goodfit_background_samples_{timestamp}.csv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    cols = [
        "sample",
        "separation_coefficient",
        "imag_loss_rel_rmse",
        "imag_loss_r2",
        "drt_total_area_ohm",
        "background_fit_max_x_ohm",
        "background_fit_max_y_ohm",
        "n_background_points",
    ]
    background[cols].to_csv(path, index=False)
    return path


def remove_gridlines(*axes: plt.Axes) -> None:
    for ax in axes:
        ax.grid(False)


def main() -> None:
    base.OUT_FIG.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = base.load_panel_a_data()

    fig_b, ax_b = plt.subplots(figsize=(5.9, 5.0), constrained_layout=True)
    background = plot_good_fit_background(ax_b, data)
    base.plot_panel_b(ax_b, data, show_curve_labels=True)
    remove_gridlines(ax_b)
    png_b, svg_b = output_paths(
        "figure3B_eis_top_bottom_goodfit_background_initial", timestamp
    )
    fig_b.savefig(png_b, dpi=300)
    fig_b.savefig(svg_b)
    plt.close(fig_b)

    fig_ab, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), constrained_layout=True)
    plot_good_fit_background(axes[0], data)
    base.plot_panel_b(axes[0], data)
    scatter = base.plot_panel_a(axes[1], data)
    base.add_top_colorbar(fig_ab, axes[1], scatter)
    remove_gridlines(*axes)
    png_ab, svg_ab = output_paths(
        "figure3AB_eis_drt_top_row_goodfit_background_initial", timestamp
    )
    fig_ab.savefig(png_ab, dpi=300)
    fig_ab.savefig(svg_ab)
    plt.close(fig_ab)

    csv_path = save_background_csv(background, timestamp)
    print(f"Good-fit background spectra: {len(background)}")
    print(f"Standalone B: {png_b}")
    print(f"Combined AB: {png_ab}")
    print(f"Background sample list: {csv_path}")


if __name__ == "__main__":
    main()
