from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "outputs" / "feature_selection_561_rs_from_eis_no_eis_fitting"
FEATURE_MATRIX = FEATURE_DIR / "feature_matrix_561_no_eis_fitting.csv"
CANDIDATE_FEATURES = FEATURE_DIR / "candidate_features.csv"
OUTPUT_DIR = ROOT / "outputs" / "05_initial_descriptor_family_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 5, 11, 17, 42, 88]
HIGH_SEPARATION_THRESHOLD = 8.0

FAMILIES = {
    "H adsorption": ["FE_ads_avg_element1", "FE_ads_avg_element2"],
    "OH affinity": ["delta_EOH_element1", "delta_EOH_element2"],
    "Electronegativity": ["electronegativity_element1", "electronegativity_element2"],
    "M-D bond": ["MD_bond_kJ_element1", "MD_bond_kJ_element2"],
    "M-H bond": ["MH_bond_kJ_element1", "MH_bond_kJ_element2"],
    "H/D ZPE": ["delta_ZPE_H_element1", "delta_ZPE_H_element2"],
    "Work function": ["work_function_element1", "work_function_element2"],
    "M-M bond": ["MM_bond_element1", "MM_bond_element2"],
    "Standard potential": ["E_std_element1", "E_std_element2"],
    "PZC": ["pzc_element1", "pzc_element2"],
    "OH frequency": ["Frequency_OH_element1", "Frequency_OH_element2"],
    "I_raw": ["I_raw"],
    "Rs": ["Rs_from_EIS"],
}

SUBSETS = {
    "All candidate features": None,
    "H + OH + electronegativity": [
        "FE_ads_avg_element1",
        "FE_ads_avg_element2",
        "delta_EOH_element1",
        "delta_EOH_element2",
        "electronegativity_element1",
        "electronegativity_element2",
    ],
    "Weighted H + OH + electronegativity": [
        "weighted_FE_ads_avg",
        "weighted_delta_EOH",
        "weighted_electronegativity",
    ],
    "H + OH": [
        "FE_ads_avg_element1",
        "FE_ads_avg_element2",
        "delta_EOH_element1",
        "delta_EOH_element2",
    ],
    "H + electronegativity": [
        "FE_ads_avg_element1",
        "FE_ads_avg_element2",
        "electronegativity_element1",
        "electronegativity_element2",
    ],
    "OH + electronegativity": [
        "delta_EOH_element1",
        "delta_EOH_element2",
        "electronegativity_element1",
        "electronegativity_element2",
    ],
    "H adsorption only": ["FE_ads_avg_element1", "FE_ads_avg_element2"],
    "OH affinity only": ["delta_EOH_element1", "delta_EOH_element2"],
    "Electronegativity only": ["electronegativity_element1", "electronegativity_element2"],
    "H + OH + electronegativity + M-D": [
        "FE_ads_avg_element1",
        "FE_ads_avg_element2",
        "delta_EOH_element1",
        "delta_EOH_element2",
        "electronegativity_element1",
        "electronegativity_element2",
        "MD_bond_kJ_element1",
        "MD_bond_kJ_element2",
    ],
}


def configure_matplotlib():
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.linewidth"] = 1.8
    mpl.rcParams["xtick.major.width"] = 1.1
    mpl.rcParams["ytick.major.width"] = 1.1


def stratification_bins(y, n_bins=5):
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")


def add_weighted_descriptors(df):
    frac1 = np.where(df["element1"].eq(df["element2"]), 1.0, 6.0 / 7.0)
    frac2 = np.where(df["element1"].eq(df["element2"]), 0.0, 1.0 / 7.0)
    df = df.copy()
    for base in ["FE_ads_avg", "delta_EOH", "electronegativity"]:
        df[f"weighted_{base}"] = (
            frac1 * df[f"{base}_element1"] + frac2 * df[f"{base}_element2"]
        )
    return df


def load_data():
    df = pd.read_csv(FEATURE_MATRIX)
    df = add_weighted_descriptors(df)
    candidate_features = pd.read_csv(CANDIDATE_FEATURES)["feature"].tolist()
    missing = [col for col in candidate_features if col not in df.columns]
    if missing:
        raise KeyError(f"Missing candidate features: {missing}")
    return df, candidate_features


def make_regressor(seed):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
    )


def make_classifier(seed):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
            class_weight="balanced",
        ),
    )


def evaluate_subsets(df, candidate_features):
    y = df["separation_coefficient"].to_numpy(float)
    y_class = y >= HIGH_SEPARATION_THRESHOLD
    rows = []

    for subset_name, subset_features in SUBSETS.items():
        features = candidate_features if subset_features is None else subset_features
        X = df[features]

        for seed in SEEDS:
            strat = stratification_bins(y)
            X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
                X,
                y,
                y_class,
                test_size=0.20,
                random_state=seed,
                stratify=strat,
            )

            reg = make_regressor(seed)
            reg.fit(X_train, y_train)
            pred = reg.predict(X_test)
            rows.append(
                {
                    "task": "regression",
                    "subset": subset_name,
                    "seed": seed,
                    "n_features": len(features),
                    "r2": r2_score(y_test, pred),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                }
            )

            cls = make_classifier(seed)
            cls.fit(X_train, yc_train)
            prob = cls.predict_proba(X_test)[:, 1]
            rows.append(
                {
                    "task": "classification_SHD_ge_8",
                    "subset": subset_name,
                    "seed": seed,
                    "n_features": len(features),
                    "r2": np.nan,
                    "rmse": np.nan,
                    "roc_auc": roc_auc_score(yc_test, prob),
                    "average_precision": average_precision_score(yc_test, prob),
                }
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "subset_model_results_by_seed.csv", index=False)

    summary = (
        raw.groupby(["task", "subset", "n_features"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
        )
        .sort_values(["task", "r2_mean", "roc_auc_mean"], ascending=[True, False, False])
    )
    summary.to_csv(OUTPUT_DIR / "subset_model_results_summary.csv", index=False)
    return raw, summary


def family_permutation_importance(df, candidate_features):
    y = df["separation_coefficient"].to_numpy(float)
    X = df[candidate_features]
    rows = []

    for seed in SEEDS:
        strat = stratification_bins(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=strat,
        )
        model = make_regressor(seed)
        model.fit(X_train, y_train)
        baseline = r2_score(y_test, model.predict(X_test))

        rng = np.random.default_rng(seed + 1000)
        for family, cols in FAMILIES.items():
            valid_cols = [col for col in cols if col in X_test.columns]
            drops = []
            for _ in range(10):
                X_perm = X_test.copy()
                order = rng.permutation(len(X_perm))
                X_perm.loc[:, valid_cols] = X_perm.loc[:, valid_cols].to_numpy()[order]
                drops.append(baseline - r2_score(y_test, model.predict(X_perm)))
            rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "n_features": len(valid_cols),
                    "baseline_r2": baseline,
                    "r2_drop_mean": float(np.mean(drops)),
                    "r2_drop_std_within_seed": float(np.std(drops)),
                }
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "family_permutation_importance_by_seed.csv", index=False)
    summary = (
        raw.groupby(["family", "n_features"], as_index=False)
        .agg(
            r2_drop_mean=("r2_drop_mean", "mean"),
            r2_drop_std_across_seed=("r2_drop_mean", "std"),
            baseline_r2_mean=("baseline_r2", "mean"),
        )
        .sort_values("r2_drop_mean", ascending=False)
    )
    summary.to_csv(OUTPUT_DIR / "family_permutation_importance_summary.csv", index=False)
    return raw, summary


def plot_family_importance(summary):
    configure_matplotlib()
    top = summary.sort_values("r2_drop_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=300)
    colors = [
        "#F35656" if name in {"H adsorption", "OH affinity", "Electronegativity"} else "#9FB4D9"
        for name in top["family"]
    ]
    ax.barh(top["family"], top["r2_drop_mean"], xerr=top["r2_drop_std_across_seed"], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Test R² Drop After Family Permutation")
    ax.grid(True, axis="x", color="#c9c9c9", alpha=0.35, linewidth=0.45)
    ax.tick_params(axis="both", labelsize=9, colors="black")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontname("Arial")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "family_permutation_importance.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "family_permutation_importance.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_subset_performance(summary):
    configure_matplotlib()
    reg = summary[summary["task"].eq("regression")].copy()
    reg = reg.sort_values("r2_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=300)
    colors = [
        "#F19FA0" if "H + OH + electronegativity" in name else "#B7C8E8"
        for name in reg["subset"]
    ]
    ax.barh(reg["subset"], reg["r2_mean"], xerr=reg["r2_std"], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Mean Test R² Across Seeds")
    ax.grid(True, axis="x", color="#c9c9c9", alpha=0.35, linewidth=0.45)
    ax.tick_params(axis="both", labelsize=8.5, colors="black")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontname("Arial")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "subset_regression_performance.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "subset_regression_performance.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    cls = summary[summary["task"].eq("classification_SHD_ge_8")].copy()
    cls = cls.sort_values("roc_auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=300)
    colors = [
        "#F19FA0" if "H + OH + electronegativity" in name else "#B7C8E8"
        for name in cls["subset"]
    ]
    ax.barh(cls["subset"], cls["roc_auc_mean"], xerr=cls["roc_auc_std"], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xlabel(f"ROC-AUC for SH/D ≥ {HIGH_SEPARATION_THRESHOLD:g}")
    ax.grid(True, axis="x", color="#c9c9c9", alpha=0.35, linewidth=0.45)
    ax.tick_params(axis="both", labelsize=8.5, colors="black")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontname("Arial")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "subset_high_separation_classification_auc.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "subset_high_separation_classification_auc.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    df, candidate_features = load_data()
    raw_subset, subset_summary = evaluate_subsets(df, candidate_features)
    raw_perm, perm_summary = family_permutation_importance(df, candidate_features)
    plot_family_importance(perm_summary)
    plot_subset_performance(subset_summary)
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("\nFamily permutation importance:")
    print(perm_summary.to_string(index=False))
    print("\nSubset model summary:")
    print(subset_summary.to_string(index=False))


if __name__ == "__main__":
    main()
