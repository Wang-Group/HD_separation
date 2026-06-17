from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "figure1_bottom_this_work_from_all_data.csv"
OUTPUT_DIR = HERE / "outputs"
FRONT_CSV = OUTPUT_DIR / "this_work_pareto_front_points.csv"
FRONT_PNG = OUTPUT_DIR / "this_work_pareto_front_only.png"


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


def load_this_work_points() -> pd.DataFrame:
    source = pd.read_csv(DATA_PATH)
    points = source[
        ["sample", "source_type", "plot_x_overpotential_mV", "separation_coefficient"]
    ].copy()
    points = points.rename(
        columns={
            "plot_x_overpotential_mV": "x",
            "separation_coefficient": "y",
        }
    )
    points["x"] = pd.to_numeric(points["x"], errors="coerce").clip(upper=550)
    points["y"] = pd.to_numeric(points["y"], errors="coerce")
    points = points.dropna(subset=["x", "y"]).reset_index(drop=True)
    return points[np.isfinite(points["x"]) & np.isfinite(points["y"])].copy()


def save_front_csv(front: pd.DataFrame) -> None:
    export = front.rename(
        columns={
            "x": "plot_x_overpotential_mV",
            "y": "separation_coefficient",
        }
    )
    export[
        ["sample", "source_type", "plot_x_overpotential_mV", "separation_coefficient"]
    ].to_csv(FRONT_CSV, index=False)


def plot_front(front: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    group_style = {
        "initial_screening_561": {"color": "#4C78A8", "marker": "o", "label": "Initial screening"},
        "bo_extra_co_fe_36": {"color": "#F58518", "marker": "s", "label": "Co-Fe BO"},
        "bo_figure4_four_system_112": {"color": "#E45756", "marker": "^", "label": "Four-system BO"},
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        front["x"],
        front["y"],
        color="#D62728",
        linewidth=2.3,
        zorder=2,
    )

    for group, grouped in front.groupby("source_type", sort=False):
        style = group_style.get(
            group,
            {"color": "#666666", "marker": "o", "label": group.replace("_", " ")},
        )
        ax.scatter(
            grouped["x"],
            grouped["y"],
            s=62,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.55,
            label=style["label"],
            zorder=3,
        )

    label_offsets = {
        "Ag-Pb": (4, -9),
        "Co-Fe29": (4, 6),
        "Co-V24": (4, -10),
        "Co-V12": (4, 4),
        "Co-V10": (4, 10),
        "Co-Mo": (4, -9),
        "Co-Mo16": (4, 5),
        "Co-Mo12": (4, -10),
        "Co-Mo7": (4, 6),
        "Co-Mo26": (4, 6),
        "Fe-Ni21": (4, -10),
        "Fe-Ni18": (4, 6),
        "Fe-Ni11": (4, -10),
        "Fe-Ni7": (4, 5),
    }

    for _, row in front.iterrows():
        dx, dy = label_offsets.get(row["sample"], (4, 5))
        ax.annotate(
            row["sample"],
            (row["x"], row["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color="#222222",
        )

    ax.set_xlabel("Overpotential @ -10 mA cm$^{-2}$ (mV)", fontweight="bold")
    ax.set_ylabel("Separation coefficient @ -50 mA cm$^{-2}$", fontweight="bold")
    ax.set_title("This-Work Pareto Front", fontweight="bold", pad=10)

    x_pad = 12
    y_pad = 0.25
    ax.set_xlim(front["x"].min() - x_pad, front["x"].max() + 34)
    ax.set_ylim(front["y"].min() - y_pad, front["y"].max() + 0.45)
    ax.grid(True, color="#E6E6E6", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    for spine in ax.spines.values():
        spine.set_color("black")

    fig.tight_layout()
    fig.savefig(FRONT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = load_this_work_points()
    front = pareto_frontier_xy(points)
    save_front_csv(front)
    plot_front(front)
    print(f"This-work points: {len(points)}")
    print(f"Pareto-front points: {len(front)}")
    print(f"Saved CSV: {FRONT_CSV}")
    print(f"Saved PNG: {FRONT_PNG}")


if __name__ == "__main__":
    main()
