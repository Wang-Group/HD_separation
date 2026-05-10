from pathlib import Path

"""Train and interpret the seed-42 9-feature H/D separation model.

The 9-feature set is the plateau point from the two-stage feature-family
selection workflow in
``bruteforce_classification_then_forward_regression_seed42_16pool.py``. This
script retrains the XGBoost regressor with the same stratified train/test split
and saves test-set and whole-dataset SHAP plots.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
FEATURE_MATRIX = (
    ROOT
    / "outputs"
    / "feature_selection_561_rs_from_eis_no_eis_fitting"
    / "feature_matrix_561_no_eis_fitting.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "06_bruteforce_classification_forward_regression_seed42_16pool" / "final_9feature_seed42_shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FEATURES = [
    "delta_EOH_element1",
    "Frequency_H_element1",
    "delta_ZPE_H_element1",
    "Frequency_D_element1",
    "work_function_element2",
    "delta_EOH_element2",
    "FE_ads_avg_element1",
    "electronegativity_element1",
    "E_std_element1",
]

DISPLAY_NAMES = {
    "delta_EOH_element1": "OH-affinity descriptor of metal 1",
    "Frequency_H_element1": "H vibrational frequency of metal 1",
    "delta_ZPE_H_element1": "H/D ZPE difference of metal 1",
    "Frequency_D_element1": "D vibrational frequency of metal 1",
    "work_function_element2": "Work function of metal 2",
    "delta_EOH_element2": "OH-affinity descriptor of metal 2",
    "FE_ads_avg_element1": "H adsorption energy of metal 1",
    "electronegativity_element1": "Electronegativity of metal 1",
    "E_std_element1": "Standard potential of metal 1",
}


def configure_matplotlib():
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.linewidth"] = 1.8
    mpl.rcParams["xtick.major.width"] = 1.1
    mpl.rcParams["ytick.major.width"] = 1.1


def style_current_figure(xlabel):
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(axis="both", colors="black", labelcolor="black", labelsize=10)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        ax.title.set_color("black")
        ax.title.set_fontweight("bold")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("black")
            label.set_fontweight("bold")
            label.set_fontname("Arial")
        for text in ax.texts:
            text.set_color("black")
            text.set_fontweight("bold")
        if ax.get_xlabel():
            ax.set_xlabel(xlabel, fontweight="bold", color="black")
    fig.tight_layout()
    return fig


def save_shap_plots(shap_values, X_values, stem):
    display_features = [DISPLAY_NAMES[f] for f in FEATURES]

    shap.summary_plot(
        shap_values,
        X_values,
        feature_names=display_features,
        show=False,
        plot_size=(7.0, 4.9),
        max_display=len(FEATURES),
    )
    fig = style_current_figure("SHAP Value")
    fig.savefig(OUTPUT_DIR / f"{stem}_shap_swarm.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / f"{stem}_shap_swarm.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    shap.summary_plot(
        shap_values,
        X_values,
        feature_names=display_features,
        show=False,
        plot_type="bar",
        plot_size=(7.0, 4.9),
        max_display=len(FEATURES),
    )
    fig = style_current_figure("Averaged |SHAP Value|")
    fig.savefig(OUTPUT_DIR / f"{stem}_shap_bar.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / f"{stem}_shap_bar.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    imp = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "display_name": display_features,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(OUTPUT_DIR / f"{stem}_mean_abs_shap.csv", index=False)
    return imp


def main():
    configure_matplotlib()
    df = pd.read_csv(FEATURE_MATRIX)
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

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES, index=X_test.index)
    X_all_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES, index=X.index)

    model = XGBRegressor(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_train_imp, y_train)
    pred = model.predict(X_test_imp)
    metrics = {
        "seed": SEED,
        "model": "XGBoost",
        "test_r2": r2_score(y_test, pred),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "n_train": len(X_train_imp),
        "n_test": len(X_test_imp),
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    explainer = shap.TreeExplainer(model)
    shap_test = explainer.shap_values(X_test_imp)
    shap_all = explainer.shap_values(X_all_imp)

    test_imp = save_shap_plots(shap_test, X_test_imp, "test")
    all_imp = save_shap_plots(shap_all, X_all_imp, "whole_dataset")

    print("Metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("\nTest SHAP importance:")
    print(test_imp.to_string(index=False))
    print("\nWhole dataset SHAP importance:")
    print(all_imp.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
