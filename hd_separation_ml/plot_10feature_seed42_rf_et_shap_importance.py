from pathlib import Path

"""Final 10-feature RF/ExtraTrees model interpretation for H/D separation.

The 10-feature set is taken from the RF/ExtraTrees-only forward-addition path in
``outputs/06c_forward_regression_seed42_all28pool_rf_et_only``. Random Forest
and Extra Trees are trained on the same seed-42 stratified split; the better
held-out model is used for SHAP plots, and built-in feature importance is saved
for both models.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
OUTPUT_DIR = ROOT / "outputs" / "final_model_10features_561_seed42_rf_et_only"
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
        ax.tick_params(axis="both", colors="black", labelcolor="black", labelsize=11)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
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
            ax.set_xlabel(xlabel, fontweight="bold", color="black", fontsize=12)
    fig.tight_layout()
    return fig


def models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=600,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=600,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def save_prediction_plot(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    lo = min(float(np.min(y_true)), float(np.min(y_pred))) - 0.25
    hi = max(float(np.max(y_true)), float(np.max(y_pred))) + 0.25

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)
    ax.scatter(
        y_true,
        y_pred,
        s=42,
        color="#FFC44D",
        edgecolor="#7A4A00",
        linewidth=0.55,
        alpha=0.82,
    )
    ax.plot([lo, hi], [lo, hi], "--", color="red", linewidth=2.0, label="Perfect Prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True", fontsize=13, fontweight="bold")
    ax.set_ylabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_title(f"{model_name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}", fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, color="#c9c9c9", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=True, prop={"family": "Arial", "weight": "bold", "size": 9})
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontname("Arial")
        item.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_prediction.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "test_prediction.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    residual = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300)

    axes[0].scatter(
        y_true,
        y_pred,
        s=42,
        color="#FFC44D",
        edgecolor="#7A4A00",
        linewidth=0.55,
        alpha=0.82,
    )
    axes[0].plot([lo, hi], [lo, hi], "--", color="red", linewidth=2.0, label="Perfect Prediction")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel("True Separation Coefficient", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Predicted Separation Coefficient", fontsize=12, fontweight="bold")
    axes[0].set_title(f"Test Set\nR² = {r2:.4f}, RMSE = {rmse:.4f}", fontsize=11, fontweight="bold", pad=7)
    axes[0].grid(True, color="#c9c9c9", linewidth=0.5, alpha=0.45)
    axes[0].legend(frameon=True, prop={"family": "Arial", "weight": "bold", "size": 8})

    axes[1].scatter(
        y_pred,
        residual,
        s=42,
        color="#FFC44D",
        edgecolor="#7A4A00",
        linewidth=0.55,
        alpha=0.82,
    )
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2.0)
    axes[1].set_xlabel("Predicted Separation Coefficient", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Residuals", fontsize=12, fontweight="bold")
    axes[1].set_title("Residual Plot", fontsize=11, fontweight="bold", pad=7)
    axes[1].grid(True, color="#c9c9c9", linewidth=0.5, alpha=0.45)

    for ax in axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontname("Arial")
            item.set_fontweight("bold")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_prediction_residual.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "test_prediction_residual.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_builtin_importance(model_rows):
    rows = []
    for model_name, model in model_rows:
        estimator = model.named_steps[list(model.named_steps)[-1]]
        for feature, display_name, importance in zip(
            FEATURES,
            [DISPLAY_NAMES.get(feature, feature) for feature in FEATURES],
            estimator.feature_importances_,
        ):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "display_name": display_name,
                    "feature_importance": importance,
                }
            )
    importance = pd.DataFrame(rows)
    importance.to_csv(OUTPUT_DIR / "feature_importance_by_model.csv", index=False)
    avg = (
        importance.groupby(["feature", "display_name"], as_index=False)
        .agg(mean_feature_importance=("feature_importance", "mean"))
        .sort_values("mean_feature_importance", ascending=False)
        .reset_index(drop=True)
    )
    avg.to_csv(OUTPUT_DIR / "feature_importance_mean_rf_et.csv", index=False)

    plot_df = avg.sort_values("mean_feature_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    ax.barh(
        plot_df["display_name"],
        plot_df["mean_feature_importance"],
        color="#7DB7FF",
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xlabel("Mean Feature Importance")
    ax.grid(True, axis="x", color="#c9c9c9", alpha=0.35, linewidth=0.45)
    ax.tick_params(axis="both", labelsize=10, colors="black")
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontname("Arial")
        item.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance_mean_rf_et.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "feature_importance_mean_rf_et.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return importance, avg


def save_shap_plots(shap_values, X_values, stem):
    display_features = [DISPLAY_NAMES.get(feature, feature) for feature in FEATURES]

    plt.figure(figsize=(7.2, 5.2), dpi=300)
    shap.summary_plot(shap_values, X_values, feature_names=display_features, show=False)
    fig = style_current_figure("SHAP Value")
    fig.savefig(OUTPUT_DIR / f"{stem}_shap_swarm.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / f"{stem}_shap_swarm.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    plt.figure(figsize=(6.2, 4.9), dpi=300)
    shap.summary_plot(
        shap_values,
        X_values,
        feature_names=display_features,
        plot_type="bar",
        show=False,
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

    trained = []
    metric_rows = []
    transformed = {}
    for model_name, estimator in models().items():
        pipeline = make_pipeline(SimpleImputer(strategy="median"), estimator)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        metric_rows.append(
            {
                "seed": SEED,
                "n_features": len(FEATURES),
                "model": model_name,
                "test_r2": r2_score(y_test, pred),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "features": ", ".join(FEATURES),
            }
        )
        trained.append((model_name, pipeline))
        transformed[model_name] = pred

    metrics = pd.DataFrame(metric_rows).sort_values("test_r2", ascending=False).reset_index(drop=True)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    save_builtin_importance(trained)

    best_model_name = metrics.iloc[0]["model"]
    best_pipeline = dict(trained)[best_model_name]
    imputer = best_pipeline.named_steps["simpleimputer"]
    estimator = best_pipeline.named_steps[list(best_pipeline.named_steps)[-1]]

    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES, index=X_test.index)
    X_all_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES, index=X.index)
    save_prediction_plot(y_test, transformed[best_model_name], best_model_name)

    explainer = shap.TreeExplainer(estimator)
    shap_test = explainer.shap_values(X_test_imp)
    shap_all = explainer.shap_values(X_all_imp)

    test_imp = save_shap_plots(shap_test, X_test_imp, "test")
    all_imp = save_shap_plots(shap_all, X_all_imp, "whole_dataset")

    print("Metrics:")
    print(metrics[["model", "test_r2", "test_rmse"]].to_string(index=False))
    print(f"\nBest model for SHAP: {best_model_name}")
    print("\nWhole dataset SHAP importance:")
    print(all_imp.to_string(index=False))
    print("\nTest-set SHAP importance:")
    print(test_imp.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
