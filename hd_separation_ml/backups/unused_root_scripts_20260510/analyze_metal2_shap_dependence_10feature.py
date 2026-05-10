from pathlib import Path

"""Metal-2 SHAP dependence analysis for the final 10-feature model.

This script focuses on whether metal-2 descriptors behave conditionally with
respect to the standard-potential descriptor. The working hypothesis is that
metal 2 may act differently when it is more reducible/less reducible during
electrodeposition, which could modulate whether metal-2 properties behave like
alloy-forming descriptors or like surface/electronic modifiers.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
FEATURE_MATRIX = (
    ROOT
    / "outputs"
    / "feature_selection_561_rs_from_eis_no_eis_fitting"
    / "feature_matrix_561_no_eis_fitting.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "final_model_10features_561_seed42_rf_et_only" / "metal2_shap_dependence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FEATURES = [
    "Frequency_H_element1",
    "Frequency_D_element1",
    "delta_ZPE_H_element1",
    "electronegativity_element1",
    "work_function_element2",
    "delta_EOH_element2",
    "FE_ads_avg_element1",
    "work_function_element1",
    "E_std_element2",
    "MM_bond_element2",
]

DISPLAY_NAMES = {
    "Frequency_H_element1": "H Vibrational Frequency of Metal 1",
    "Frequency_D_element1": "D Vibrational Frequency of Metal 1",
    "delta_ZPE_H_element1": "H/D ZPE Difference of Metal 1",
    "electronegativity_element1": "Electronegativity of Metal 1",
    "work_function_element2": "Work Function of Metal 2",
    "delta_EOH_element2": "OH Adsorption Energy of Metal 2",
    "FE_ads_avg_element1": "H Adsorption Energy of Metal 1",
    "work_function_element1": "Work Function of Metal 1",
    "E_std_element2": "Standard Potential of Metal 2",
    "MM_bond_element2": "M-M Bond Energy of Metal 2",
    "separation_coefficient": "Separation Coefficient",
}

METAL2_FEATURES = [
    "work_function_element2",
    "delta_EOH_element2",
    "E_std_element2",
    "MM_bond_element2",
]


def configure_matplotlib():
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.linewidth"] = 1.8
    mpl.rcParams["xtick.major.width"] = 1.1
    mpl.rcParams["ytick.major.width"] = 1.1


def train_model():
    df = pd.read_csv(FEATURE_MATRIX).dropna(subset=["separation_coefficient"]).reset_index(drop=True)
    X = df[FEATURES].copy()
    y = df["separation_coefficient"].to_numpy(float)
    strat = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=strat,
    )
    model = RandomForestRegressor(
        n_estimators=600,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    pipe = make_pipeline(SimpleImputer(strategy="median"), model)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "test_r2": r2_score(y_test, pred),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    imputer = pipe.named_steps["simpleimputer"]
    X_all = pd.DataFrame(imputer.transform(X), columns=FEATURES, index=X.index)
    estimator = pipe.named_steps["randomforestregressor"]
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_all)
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{f}" for f in FEATURES], index=X.index)
    out = pd.concat([df[[c for c in ["sample", "separation_coefficient"] if c in df.columns]], X_all, shap_df], axis=1)
    out.to_csv(OUTPUT_DIR / "metal2_shap_table.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    return out, metrics


def style_axis(ax):
    ax.tick_params(axis="both", labelsize=10, colors="black")
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontname("Arial")
        item.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.title.set_fontweight("bold")
    ax.grid(True, color="#c9c9c9", alpha=0.28, linewidth=0.45)


def scatter_dependence(df, x, shap_feature, color_feature, stem):
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=300)
    sc = ax.scatter(
        df[x],
        df[f"SHAP_{shap_feature}"],
        c=df[color_feature],
        s=34,
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.35,
        alpha=0.82,
    )
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xlabel(DISPLAY_NAMES.get(x, x))
    ax.set_ylabel(f"SHAP: {DISPLAY_NAMES.get(shap_feature, shap_feature)}")
    ax.set_title(f"Colored by {DISPLAY_NAMES.get(color_feature, color_feature)}", fontsize=11, fontweight="bold")
    style_axis(ax)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    for item in cbar.ax.get_yticklabels():
        item.set_fontname("Arial")
        item.set_fontweight("bold")
    cbar.set_label(DISPLAY_NAMES.get(color_feature, color_feature), fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def binned_summary(df):
    q = pd.qcut(df["E_std_element2"], q=3, labels=["Low", "Mid", "High"], duplicates="drop")
    rows = []
    for group_name, group in df.groupby(q, observed=True):
        row = {
            "E_std_element2_bin": str(group_name),
            "n": len(group),
            "mean_separation": group["separation_coefficient"].mean(),
            "median_separation": group["separation_coefficient"].median(),
            "high_S_ge_8_fraction": (group["separation_coefficient"] >= 8).mean(),
            "E_std_element2_min": group["E_std_element2"].min(),
            "E_std_element2_max": group["E_std_element2"].max(),
        }
        for feature in METAL2_FEATURES:
            row[f"mean_abs_SHAP_{feature}"] = group[f"SHAP_{feature}"].abs().mean()
            row[f"mean_SHAP_{feature}"] = group[f"SHAP_{feature}"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "Estd2_binned_metal2_shap_summary.csv", index=False)
    return summary


def plot_binned_abs_shap(summary):
    long_rows = []
    for _, row in summary.iterrows():
        for feature in METAL2_FEATURES:
            long_rows.append(
                {
                    "E_std_element2_bin": row["E_std_element2_bin"],
                    "feature": DISPLAY_NAMES[feature],
                    "mean_abs_shap": row[f"mean_abs_SHAP_{feature}"],
                }
            )
    long_df = pd.DataFrame(long_rows)
    pivot = long_df.pivot(index="feature", columns="E_std_element2_bin", values="mean_abs_shap")
    pivot = pivot.loc[[DISPLAY_NAMES[f] for f in METAL2_FEATURES]]
    pivot = pivot[[col for col in ["Low", "Mid", "High"] if col in pivot.columns]]
    ax = pivot.plot(kind="barh", figsize=(6.7, 4.3), width=0.75, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_ylabel("")
    ax.grid(True, axis="x", color="#c9c9c9", alpha=0.32, linewidth=0.45)
    ax.legend(
        title="Standard Potential of Metal 2 Bin",
        frameon=True,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        prop={"family": "Arial", "weight": "bold", "size": 8.5},
        title_fontproperties={"family": "Arial", "weight": "bold", "size": 8.5},
    )
    style_axis(ax)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "Estd2_binned_metal2_mean_abs_shap.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "Estd2_binned_metal2_mean_abs_shap.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def correlation_tables(df):
    cols = METAL2_FEATURES + [f"SHAP_{f}" for f in METAL2_FEATURES] + ["separation_coefficient"]
    pearson = df[cols].corr(method="pearson")
    spearman = df[cols].corr(method="spearman")
    pearson.to_csv(OUTPUT_DIR / "metal2_feature_shap_pearson_corr.csv")
    spearman.to_csv(OUTPUT_DIR / "metal2_feature_shap_spearman_corr.csv")


def main():
    configure_matplotlib()
    df, metrics = train_model()
    correlation_tables(df)
    summary = binned_summary(df)
    plot_binned_abs_shap(summary)

    for color_feature in ["work_function_element2", "delta_EOH_element2", "MM_bond_element2", "separation_coefficient"]:
        scatter_dependence(
            df,
            "E_std_element2",
            "E_std_element2",
            color_feature,
            f"dependence_Estd2_SHAP_Estd2_colored_by_{color_feature}",
        )
    for shap_feature in ["work_function_element2", "delta_EOH_element2", "MM_bond_element2"]:
        scatter_dependence(
            df,
            "E_std_element2",
            shap_feature,
            shap_feature,
            f"dependence_Estd2_SHAP_{shap_feature}_colored_by_{shap_feature}",
        )

    print("Metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("\nE_std_element2-binned metal-2 SHAP summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
