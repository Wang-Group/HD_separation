from __future__ import annotations

"""Forward-addition regression using only Random Forest and Extra Trees.

This script starts from the all-28 high-separation classification core and then
adds features greedily using the mean 5-fold CV R2 of Random Forest and Extra
Trees only. XGBoost is deliberately excluded from the feature-addition decision
because it showed less stable held-out behavior in this small, correlated
descriptor set.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "outputs" / "feature_selection_561_rs_from_eis_no_eis_fitting"
FEATURE_MATRIX = FEATURE_DIR / "feature_matrix_561_no_eis_fitting.csv"
CANDIDATE_FEATURES = FEATURE_DIR / "candidate_features.csv"
CORE4_FEATURES = (
    ROOT
    / "outputs"
    / "06b_bruteforce_classification_forward_regression_seed42_all28pool"
    / "classification_core4_features.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "06c_forward_regression_seed42_all28pool_rf_et_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_FEATURES = 16


def configure_matplotlib():
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.linewidth"] = 1.8
    mpl.rcParams["xtick.major.width"] = 1.1
    mpl.rcParams["ytick.major.width"] = 1.1


def selection_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=120,
            max_depth=8,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def final_test_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=600,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=600,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def load_inputs():
    df = pd.read_csv(FEATURE_MATRIX).dropna(subset=["separation_coefficient"]).reset_index(drop=True)
    feature_pool = pd.read_csv(CANDIDATE_FEATURES)["feature"].dropna().astype(str).tolist()
    core4 = pd.read_csv(CORE4_FEATURES)["feature"].dropna().astype(str).tolist()
    missing = [feature for feature in feature_pool + core4 if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing features in feature matrix: {missing}")
    X = df[feature_pool].copy()
    y = df["separation_coefficient"].to_numpy(float)
    return df, X, y, feature_pool, core4


def cv_folds(y):
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    return list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(
            np.zeros_like(y), bins
        )
    )


def evaluate_cv(X, y, features, folds):
    rows = []
    X_subset = X.loc[:, list(features)]
    for model_name, model in selection_models().items():
        r2s = []
        rmses = []
        for train_idx, test_idx in folds:
            pipeline = make_pipeline(SimpleImputer(strategy="median"), clone(model))
            pipeline.fit(X_subset.iloc[train_idx], y[train_idx])
            pred = pipeline.predict(X_subset.iloc[test_idx])
            r2s.append(r2_score(y[test_idx], pred))
            rmses.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
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
    return pd.DataFrame(rows)


def greedy_forward_addition(X, y, feature_pool, core4):
    folds = cv_folds(y)
    selected = list(core4)
    remaining = [feature for feature in feature_pool if feature not in selected]
    path_rows = []
    candidate_rows = []

    current_eval = evaluate_cv(X, y, selected, folds)
    current_eval["step_added_feature"] = "BASE"
    path_rows.append(current_eval)

    while len(selected) < min(MAX_FEATURES, len(feature_pool)):
        trials = []
        for candidate in remaining:
            trial_features = selected + [candidate]
            eval_df = evaluate_cv(X, y, trial_features, folds)
            mean_r2 = eval_df["cv_r2_mean"].mean()
            eval_df["candidate_added_feature"] = candidate
            eval_df["mean_r2_across_models"] = mean_r2
            candidate_rows.append(eval_df)
            trials.append((mean_r2, candidate, eval_df))

        trials.sort(key=lambda item: item[0], reverse=True)
        best_mean_r2, best_feature, best_eval = trials[0]
        selected.append(best_feature)
        remaining.remove(best_feature)
        best_eval = best_eval.copy()
        best_eval["step_added_feature"] = best_feature
        path_rows.append(best_eval)
        print(
            f"Selected {len(selected)} features: {best_feature}; "
            f"mean RF/ET CV R2={best_mean_r2:.4f}",
            flush=True,
        )

    path = pd.concat(path_rows, ignore_index=True)
    candidates = pd.concat(candidate_rows, ignore_index=True)
    path.to_csv(OUTPUT_DIR / "forward_regression_selected_path_by_model.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "forward_regression_candidate_trials_by_model.csv", index=False)

    summary = (
        path.groupby(["n_features", "features", "step_added_feature"], as_index=False)
        .agg(
            mean_r2_across_models=("cv_r2_mean", "mean"),
            mean_rmse_across_models=("cv_rmse_mean", "mean"),
            best_model_r2=("cv_r2_mean", "max"),
        )
        .sort_values("n_features")
    )
    summary.to_csv(OUTPUT_DIR / "forward_regression_selected_path_summary.csv", index=False)
    return summary


def evaluate_heldout_test(X, y, path_summary):
    strat = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=strat,
    )
    rows = []
    for _, path_row in path_summary.iterrows():
        features = [item.strip() for item in path_row["features"].split(",")]
        for model_name, model in final_test_models().items():
            pipeline = make_pipeline(SimpleImputer(strategy="median"), clone(model))
            pipeline.fit(X_train[features], y_train)
            pred = pipeline.predict(X_test[features])
            rows.append(
                {
                    "n_features": len(features),
                    "model": model_name,
                    "test_r2": r2_score(y_test, pred),
                    "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "features": ", ".join(features),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "heldout_test_results_by_path.csv", index=False)
    return result


def plot_path(path_summary, heldout):
    configure_matplotlib()
    test_summary = (
        heldout.groupby("n_features", as_index=False)
        .agg(best_test_r2=("test_r2", "max"), mean_test_r2=("test_r2", "mean"))
        .sort_values("n_features")
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.7), dpi=300)
    ax.plot(
        path_summary["n_features"],
        path_summary["mean_r2_across_models"],
        color="#FF42AD",
        marker="o",
        linewidth=2.0,
        label="Mean 5-Fold CV R²",
    )
    ax.plot(
        path_summary["n_features"],
        path_summary["best_model_r2"],
        color="#3B82F6",
        marker="s",
        linewidth=1.7,
        label="Best CV R²",
    )
    ax.plot(
        test_summary["n_features"],
        test_summary["best_test_r2"],
        color="#111111",
        marker="^",
        linewidth=1.7,
        label="Best Held-Out Test R²",
    )
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("R²")
    ax.grid(True, color="#c9c9c9", alpha=0.35, linewidth=0.45)
    ax.legend(frameon=True, prop={"family": "Arial", "weight": "bold", "size": 9})
    ax.tick_params(axis="both", labelsize=10, colors="black")
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontname("Arial")
        item.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forward_regression_rf_et_only_cv_test_curve.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "forward_regression_rf_et_only_cv_test_curve.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    _, X, y, feature_pool, core4 = load_inputs()
    pd.DataFrame({"feature": core4}).to_csv(OUTPUT_DIR / "starting_core4_features.csv", index=False)
    path_summary = greedy_forward_addition(X, y, feature_pool, core4)
    heldout = evaluate_heldout_test(X, y, path_summary)
    plot_path(path_summary, heldout)

    print("\nForward path summary:")
    print(path_summary[["n_features", "step_added_feature", "mean_r2_across_models", "best_model_r2"]].to_string(index=False))
    print("\nHeld-out test summary:")
    print(
        heldout.sort_values(["n_features", "test_r2"], ascending=[True, False])[
            ["n_features", "model", "test_r2", "test_rmse"]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
