from __future__ import annotations

"""Two-stage feature-family selection for H/D separation.

This reproducible analysis starts from the seed-42 16-feature pool, uses
high-separation classification to identify robust descriptor families, and then
performs forward regression to test whether adding those features improves
cross-validated prediction of the continuous H/D separation coefficient.

Main outputs are written to
``outputs/06_bruteforce_classification_forward_regression_seed42_16pool/``.
"""

from itertools import combinations
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "outputs" / "feature_selection_561_rs_from_eis_no_eis_fitting"
SEED42_DIR = ROOT / "outputs" / "feature_selection_561_rs_from_eis_no_eis_fitting_seed42"
FEATURE_MATRIX = FEATURE_DIR / "feature_matrix_561_no_eis_fitting.csv"
HISTORY = SEED42_DIR / "backward_elimination_history.csv"

OUTPUT_DIR = ROOT / "outputs" / "06_bruteforce_classification_forward_regression_seed42_16pool"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
HIGH_SEPARATION_THRESHOLD = 8.0
MAX_CLASSIFICATION_FEATURES = 4
FORWARD_REGRESSION_MAX_FEATURES = 12
PARALLEL_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))

FEATURE_FAMILIES = {
    "I_raw": "I_raw",
    "Rs_from_EIS": "Rs from EIS",
    "work_function_element1": "Work function",
    "work_function_element2": "Work function",
    "pzc_element1": "PZC",
    "pzc_element2": "PZC",
    "MM_bond_element1": "M-M bond",
    "MM_bond_element2": "M-M bond",
    "E_std_element1": "Standard potential",
    "E_std_element2": "Standard potential",
    "FE_ads_avg_element1": "H adsorption",
    "FE_ads_avg_element2": "H adsorption",
    "delta_EOH_element1": "OH affinity",
    "delta_EOH_element2": "OH affinity",
    "electronegativity_element1": "Electronegativity",
    "electronegativity_element2": "Electronegativity",
    "MH_bond_kJ_element1": "M-H/M-D isotope bond",
    "MH_bond_kJ_element2": "M-H/M-D isotope bond",
    "MD_bond_kJ_element1": "M-H/M-D isotope bond",
    "MD_bond_kJ_element2": "M-H/M-D isotope bond",
    "Frequency_OH_element1": "OH vibration",
    "Frequency_OH_element2": "OH vibration",
    "Frequency_H_element1": "H/D vibration-ZPE",
    "Frequency_H_element2": "H/D vibration-ZPE",
    "Frequency_D_element1": "H/D vibration-ZPE",
    "Frequency_D_element2": "H/D vibration-ZPE",
    "delta_ZPE_H_element1": "H/D vibration-ZPE",
    "delta_ZPE_H_element2": "H/D vibration-ZPE",
}


def configure_matplotlib():
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.linewidth"] = 1.8
    mpl.rcParams["xtick.major.width"] = 1.1
    mpl.rcParams["ytick.major.width"] = 1.1


def load_seed42_16_feature_pool() -> list[str]:
    history = pd.read_csv(HISTORY)
    row = history.loc[history["n_features"].eq(16)]
    if row.empty:
        raise ValueError(f"Could not find n_features == 16 in {HISTORY}")
    return [item.strip() for item in row.iloc[0]["features"].split(",")]


def load_dataset(feature_pool: list[str]):
    df = pd.read_csv(FEATURE_MATRIX)
    missing = [feature for feature in feature_pool if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing features from matrix: {missing}")
    df = df.dropna(subset=["separation_coefficient"]).reset_index(drop=True)
    X = df[feature_pool].copy()
    y_reg = df["separation_coefficient"].to_numpy(float)
    y_cls = y_reg >= HIGH_SEPARATION_THRESHOLD
    return df, X, y_reg, y_cls


def classifier_models(y_cls):
    pos = int(np.sum(y_cls))
    neg = int(len(y_cls) - pos)
    scale_pos_weight = neg / max(pos, 1)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=40,
            max_depth=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=40,
            max_depth=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=35,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    return models


def regressor_models():
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=80,
            max_depth=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    return models


def all_combinations(feature_pool: list[str], max_size: int):
    for size in range(1, max_size + 1):
        yield from combinations(feature_pool, size)


def evaluate_classifier_combo(X, y_cls, features, model_name, model, folds):
    aucs = []
    aps = []
    X_subset = X.loc[:, list(features)]
    for train_idx, test_idx in folds:
        pipeline = make_pipeline(SimpleImputer(strategy="median"), clone(model))
        pipeline.fit(X_subset.iloc[train_idx], y_cls[train_idx])
        if hasattr(pipeline[-1], "predict_proba"):
            score = pipeline.predict_proba(X_subset.iloc[test_idx])[:, 1]
        else:
            score = pipeline.decision_function(X_subset.iloc[test_idx])
        aucs.append(roc_auc_score(y_cls[test_idx], score))
        aps.append(average_precision_score(y_cls[test_idx], score))
    return {
        "model": model_name,
        "n_features": len(features),
        "features": ", ".join(features),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs, ddof=1)),
        "average_precision_mean": float(np.mean(aps)),
        "average_precision_std": float(np.std(aps, ddof=1)),
    }


def brute_force_classification(X, y_cls, feature_pool):
    cached = OUTPUT_DIR / "classification_combo_results.csv"
    if cached.exists():
        return pd.read_csv(cached)

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X, y_cls)
    )
    models = classifier_models(y_cls)
    combos = list(all_combinations(feature_pool, MAX_CLASSIFICATION_FEATURES))
    def evaluate_all_models(features):
        return [
            evaluate_classifier_combo(X, y_cls, features, model_name, model, folds)
            for model_name, model in models.items()
        ]

    print(
        f"Classification search: {len(combos)} combinations x {len(models)} models; "
        f"parallel jobs={PARALLEL_JOBS}",
        flush=True,
    )
    nested_rows = Parallel(n_jobs=PARALLEL_JOBS, batch_size=8, verbose=10)(
        delayed(evaluate_all_models)(features) for features in combos
    )
    rows = [row for group in nested_rows for row in group]

    results = pd.DataFrame(rows).sort_values(
        ["roc_auc_mean", "average_precision_mean"], ascending=False
    )
    results.to_csv(OUTPUT_DIR / "classification_combo_results.csv", index=False)
    results.groupby("model").head(50).to_csv(
        OUTPUT_DIR / "classification_top50_combos_per_model.csv", index=False
    )
    return results


def calculate_feature_weights(classification_results, feature_pool):
    rows = []
    for model_name, group in classification_results.groupby("model"):
        max_auc = group["roc_auc_mean"].max()
        selected = group.loc[group["roc_auc_mean"].ge(max_auc - 0.01)].copy()
        if selected.empty:
            selected = group.head(100).copy()
        for _, row in selected.iterrows():
            features = [item.strip() for item in row["features"].split(",")]
            combo_weight = max(row["roc_auc_mean"] - 0.5, 0) * max(row["average_precision_mean"], 1e-9)
            for feature in features:
                rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "combo_features": row["features"],
                        "combo_size": row["n_features"],
                        "combo_roc_auc": row["roc_auc_mean"],
                        "combo_average_precision": row["average_precision_mean"],
                        "distributed_weight": combo_weight / len(features),
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "classification_feature_weight_contributions.csv", index=False)
    summary = (
        raw.groupby("feature", as_index=False)
        .agg(
            weighted_score=("distributed_weight", "sum"),
            occurrence=("feature", "count"),
            mean_combo_auc=("combo_roc_auc", "mean"),
            mean_combo_ap=("combo_average_precision", "mean"),
        )
        .sort_values("weighted_score", ascending=False)
    )
    all_features = pd.DataFrame({"feature": feature_pool})
    summary = all_features.merge(summary, on="feature", how="left").fillna(
        {
            "weighted_score": 0.0,
            "occurrence": 0,
            "mean_combo_auc": 0.0,
            "mean_combo_ap": 0.0,
        }
    )
    total = summary["weighted_score"].sum()
    summary["normalized_weight"] = summary["weighted_score"] / total if total else 0.0
    summary = summary.sort_values("weighted_score", ascending=False)
    summary.to_csv(OUTPUT_DIR / "classification_feature_weight_summary.csv", index=False)
    return summary


def calculate_family_weights(feature_weights):
    family_rows = []
    for _, row in feature_weights.iterrows():
        family_rows.append(
            {
                "feature": row["feature"],
                "family": FEATURE_FAMILIES.get(row["feature"], "Other"),
                "weighted_score": row["weighted_score"],
                "occurrence": row["occurrence"],
            }
        )

    family_df = pd.DataFrame(family_rows)
    family_summary = (
        family_df.groupby("family", as_index=False)
        .agg(weighted_score=("weighted_score", "sum"), occurrence=("occurrence", "sum"))
        .sort_values("weighted_score", ascending=False)
        .reset_index(drop=True)
    )
    total = family_summary["weighted_score"].sum()
    family_summary["normalized_weight"] = family_summary["weighted_score"] / total if total else 0.0
    family_summary.to_csv(OUTPUT_DIR / "classification_family_weight_summary.csv", index=False)

    representatives = (
        family_df.sort_values("weighted_score", ascending=False)
        .drop_duplicates("family")
        .merge(family_summary[["family", "normalized_weight"]], on="family", how="left")
        .sort_values("normalized_weight", ascending=False)
        .reset_index(drop=True)
    )
    representatives.to_csv(OUTPUT_DIR / "classification_family_representative_features.csv", index=False)
    representatives.head(6).to_csv(
        OUTPUT_DIR / "classification_core_family_representatives_top6.csv",
        index=False,
    )
    return family_summary, representatives


def regression_folds(y_reg):
    bins = pd.qcut(y_reg, q=5, labels=False, duplicates="drop")
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(np.zeros_like(y_reg), bins))


def evaluate_regression_features(X, y_reg, features, folds):
    rows = []
    X_subset = X.loc[:, list(features)]
    for model_name, model in regressor_models().items():
        r2s = []
        rmses = []
        for train_idx, test_idx in folds:
            pipeline = make_pipeline(SimpleImputer(strategy="median"), clone(model))
            pipeline.fit(X_subset.iloc[train_idx], y_reg[train_idx])
            pred = pipeline.predict(X_subset.iloc[test_idx])
            r2s.append(r2_score(y_reg[test_idx], pred))
            rmses.append(float(np.sqrt(mean_squared_error(y_reg[test_idx], pred))))
        rows.append(
            {
                "model": model_name,
                "n_features": len(features),
                "features": ", ".join(features),
                "cv_r2_mean": float(np.mean(r2s)),
                "cv_r2_std": float(np.std(r2s, ddof=1)),
                "cv_rmse_mean": float(np.mean(rmses)),
                "cv_rmse_std": float(np.std(rmses, ddof=1)),
            }
        )
    result = pd.DataFrame(rows)
    return result


def forward_regression(X, y_reg, feature_pool, core_features):
    path_path = OUTPUT_DIR / "forward_regression_selected_path_by_model.csv"
    summary_path = OUTPUT_DIR / "forward_regression_selected_path_summary.csv"
    trials_path = OUTPUT_DIR / "forward_regression_candidate_trials_by_model.csv"
    if path_path.exists() and summary_path.exists() and trials_path.exists():
        return pd.read_csv(path_path), pd.read_csv(summary_path)

    folds = regression_folds(y_reg)
    selected = list(core_features)
    remaining = [feature for feature in feature_pool if feature not in selected]
    path_rows = []
    candidate_rows = []

    current_eval = evaluate_regression_features(X, y_reg, selected, folds)
    current_eval["step_added_feature"] = "BASE"
    current_eval["mean_r2_across_models"] = current_eval["cv_r2_mean"].mean()
    path_rows.append(current_eval)

    while len(selected) < min(FORWARD_REGRESSION_MAX_FEATURES, len(feature_pool)):
        trial_summaries = []
        for candidate in remaining:
            features = selected + [candidate]
            eval_df = evaluate_regression_features(X, y_reg, features, folds)
            mean_r2 = eval_df["cv_r2_mean"].mean()
            eval_df["candidate_added_feature"] = candidate
            eval_df["mean_r2_across_models"] = mean_r2
            candidate_rows.append(eval_df)
            trial_summaries.append((mean_r2, candidate, eval_df))

        trial_summaries.sort(key=lambda item: item[0], reverse=True)
        best_mean_r2, best_feature, best_eval = trial_summaries[0]
        selected.append(best_feature)
        remaining.remove(best_feature)
        best_eval = best_eval.copy()
        best_eval["step_added_feature"] = best_feature
        best_eval["mean_r2_across_models"] = best_mean_r2
        path_rows.append(best_eval)
        print(
            f"Forward regression selected {len(selected)} features: "
            f"{best_feature}; mean model CV R2={best_mean_r2:.4f}"
        )

    path = pd.concat(path_rows, ignore_index=True)
    candidates = pd.concat(candidate_rows, ignore_index=True)
    path.to_csv(OUTPUT_DIR / "forward_regression_selected_path_by_model.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "forward_regression_candidate_trials_by_model.csv", index=False)

    path_summary = (
        path.groupby(["n_features", "features", "step_added_feature"], as_index=False)
        .agg(
            mean_r2_across_models=("cv_r2_mean", "mean"),
            mean_rmse_across_models=("cv_rmse_mean", "mean"),
            best_model_r2=("cv_r2_mean", "max"),
        )
        .sort_values("n_features")
    )
    path_summary.to_csv(OUTPUT_DIR / "forward_regression_selected_path_summary.csv", index=False)
    return path, path_summary


def save_plateau_feature_set(path_summary, n_features=9):
    row = path_summary.loc[path_summary["n_features"].eq(n_features)]
    if row.empty:
        return []
    features = [item.strip() for item in row.iloc[0]["features"].split(",")]
    pd.DataFrame(
        {
            "feature": features,
            "selection_origin": [
                "classification_core" if idx < 4 else "forward_regression_addition"
                for idx, _ in enumerate(features)
            ],
            "n_feature_plateau": n_features,
        }
    ).to_csv(OUTPUT_DIR / f"plateau_{n_features}feature_set.csv", index=False)
    return features


def plot_classification_weights(summary):
    configure_matplotlib()
    plot_df = summary.sort_values("weighted_score", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=300)
    colors = []
    for feature in plot_df["feature"]:
        if feature.startswith("FE_ads"):
            colors.append("#F35656")
        elif feature.startswith("delta_EOH"):
            colors.append("#F19FA0")
        elif feature.startswith("electronegativity"):
            colors.append("#F1E2AD")
        else:
            colors.append("#AAC0FC")
    ax.barh(plot_df["feature"], plot_df["normalized_weight"], color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xlabel("Normalized Weighted Occurrence in Top AUC Combinations")
    ax.grid(True, axis="x", color="#c9c9c9", alpha=0.35, linewidth=0.45)
    ax.tick_params(axis="both", labelsize=8.5, colors="black")
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontname("Arial")
        item.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "classification_feature_weight_summary.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "classification_feature_weight_summary.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_forward_regression(path_summary):
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.3, 4.8), dpi=300)
    ax.plot(
        path_summary["n_features"],
        path_summary["mean_r2_across_models"],
        color="#FF42AD",
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="Mean across models",
    )
    ax.plot(
        path_summary["n_features"],
        path_summary["best_model_r2"],
        color="#3B82F6",
        marker="s",
        markersize=5,
        linewidth=1.6,
        label="Best model",
    )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("5-Fold CV R$^2$")
    ax.grid(True, color="#c9c9c9", alpha=0.35, linewidth=0.45)
    ax.legend(frameon=True, prop={"family": "Arial", "weight": "bold", "size": 9})
    ax.tick_params(axis="both", labelsize=10, colors="black")
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontname("Arial")
        item.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forward_regression_cv_curve.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "forward_regression_cv_curve.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    configure_matplotlib()
    feature_pool = load_seed42_16_feature_pool()
    pd.DataFrame({"feature": feature_pool}).to_csv(OUTPUT_DIR / "seed42_16_feature_pool.csv", index=False)
    df, X, y_reg, y_cls = load_dataset(feature_pool)

    print(f"Feature pool size: {len(feature_pool)}")
    print(f"Dataset size: {len(df)}; high-S class count: {int(y_cls.sum())}")
    print(f"XGBoost available: {HAS_XGB}")
    print(f"Parallel jobs: {PARALLEL_JOBS}", flush=True)

    classification_results = brute_force_classification(X, y_cls, feature_pool)
    feature_weights = calculate_feature_weights(classification_results, feature_pool)
    family_weights, family_representatives = calculate_family_weights(feature_weights)
    core4 = feature_weights.head(4)["feature"].tolist()
    pd.DataFrame({"feature": core4}).to_csv(OUTPUT_DIR / "classification_core4_features.csv", index=False)

    path, path_summary = forward_regression(X, y_reg, feature_pool, core4)
    plateau9 = save_plateau_feature_set(path_summary, n_features=9)

    plot_classification_weights(feature_weights)
    plot_forward_regression(path_summary)

    print("\nCore four features from classification weighted combinations:")
    print("\n".join(core4))
    print("\nTop classification combinations:")
    print(classification_results.head(20).to_string(index=False))
    print("\nClassification family weights:")
    print(family_weights.to_string(index=False))
    print("\nRepresentative feature per family:")
    print(family_representatives.to_string(index=False))
    print("\nForward regression selected path:")
    print(path_summary.to_string(index=False))
    if plateau9:
        print("\nPractical 9-feature plateau set:")
        print("\n".join(plateau9))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
