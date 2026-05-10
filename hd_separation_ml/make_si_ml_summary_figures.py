from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ML_ROOT = Path(__file__).resolve().parent

HD_DIR = ML_ROOT / "outputs" / "final_model_10features_561_seed42_rf_et_only"
ETA_DIR = ML_ROOT / "outputs" / "03_eta10_activity_model_raw_descriptors"


def add_image_panel(ax, image_path, label=None):
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.set_axis_off()
    if label:
        ax.text(
            0.01,
            0.99,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
            fontname="Arial",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
        )


def save_figure(fig, out_base):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def make_s8():
    fig = plt.figure(figsize=(13.2, 9.4), dpi=300)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1],
        height_ratios=[0.92, 1.08],
        wspace=0.10,
        hspace=0.16,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    add_image_panel(ax_a, HD_DIR / "test_prediction_residual.png", "A")
    add_image_panel(ax_b, HD_DIR / "whole_dataset_shap_swarm.png", "B")
    add_image_panel(ax_c, HD_DIR / "whole_dataset_shap_bar.png", "C")

    save_figure(fig, HD_DIR / "figure_S8_hd_separation_ml_summary")


def make_s9():
    # The eta10 notebook already exports a clean A-D summary. Re-export it under
    # the SI figure name so manuscript replacement is stable and obvious.
    source = ETA_DIR / "eta10_activity_model_summary_ABCD.png"
    fig = plt.figure(figsize=(13.0, 8.1), dpi=300)
    ax = fig.add_subplot(111)
    add_image_panel(ax, source)
    save_figure(fig, ETA_DIR / "figure_S9_eta10_activity_ml_summary")


if __name__ == "__main__":
    make_s8()
    make_s9()
    print("Wrote:")
    print(HD_DIR / "figure_S8_hd_separation_ml_summary.png")
    print(HD_DIR / "figure_S8_hd_separation_ml_summary.pdf")
    print(ETA_DIR / "figure_S9_eta10_activity_ml_summary.png")
    print(ETA_DIR / "figure_S9_eta10_activity_ml_summary.pdf")
