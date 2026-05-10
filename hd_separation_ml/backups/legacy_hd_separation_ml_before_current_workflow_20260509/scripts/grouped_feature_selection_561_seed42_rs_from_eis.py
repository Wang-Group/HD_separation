"""Chemistry-constrained grouped feature selection for H/D separation.

This workflow is intended to replace direct all-feature backward elimination
when strongly correlated descriptors have different chemical meanings.

Protocol:
1. Read the corrected 561-sample no-BO feature matrix produced by
   notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb.
2. Compute the Spearman correlation matrix for all candidate descriptors.
3. Group only chemically redundant descriptors when they are also strongly
   correlated. For example, M-H and M-D bond energies are grouped, but OH
   affinity and M-H/M-D bond energies are not grouped even if correlated.
4. Choose one representative descriptor per group using a chemistry-first
   priority list.
5. Perform backward elimination on the representative descriptors using
   5-fold cross-validated R2 on the training set.
6. Evaluate compact subsets from the elimination path and train final models.

Run from the repository root or from hd_separation_ml:

    python hd_separation_ml/grouped_feature_selection_561_seed42_rs_from_eis.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split


RANDOM_SEED = 42
TEST_SIZE = 0.20
SPEARMAN_ABS_THRESHOLD = 0.85

ML_DIR = Path(__file__).resolve().parent
FEATURE_MATRIX = (
    ML_DIR
    / "outputs"
    / "feature_selection_561_rs_from_eis_no_eis_fitting"
    / "feature_matrix_561_no_eis_fitting.csv"
)
OUTPUT_DIR = ML_DIR / "outputs" / "04_grouped_feature_selection_561_seed42_rs_from_eis"

METAL_DESCRIPTOR_BASE_COLS = [
    "work_function",
    "pzc",
    "MM_bond",
    "E_std",
    "FE_ads_avg",
    "delta_EOH",
    "electronegativity",
    "MH_bond_kJ",
    "MD_bond_kJ",
    "Frequency_OH",
    "Frequency_H",
    "Frequency_D",
    "delta_ZPE_H",
]

CANDIDATE_FEATURES = ["I_raw", "Rs_from_EIS"] + [
    f"{base}_element{idx}"
    for base in METAL_DESCRIPTOR_BASE_COLS
    for idx in (1, 2)
]


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def load_feature_matrix() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not FEATURE_MATRIX.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {FEATURE_MATRIX}\n"
            "Run notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb first."
        )
    df = pd.read_csv(FEATURE_MATRIX)
    missing = [col for col in CANDIDATE_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(f"Missing candidate feature columns: {missing}")
    X = df[CANDIDATE_FEATURES].copy().apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = df["separation_coefficient"].astype(float)
    metadata_cols = [
        col for col in ["sample", "element1", "element2", "separation_coefficient"]
        if col in df.columns
    ]
    return X, y, df[metadata_cols].copy()


def chemistry_equivalence_class(feature: str) -> str:
    """Return a class for descriptors that may be grouped if correlated.

    Correlation alone is not enough: OH affinity, H adsorption, and M-H/M-D
    bond energies describe different chemistry and are kept separate.
    """
    if feature in ["I_raw", "Rs_from_EIS"]:
        return feature
    if feature in ["MH_bond_kJ_element1", "MD_bond_kJ_element1"]:
        return "MHD_bond_element1"
    if feature in ["MH_bond_kJ_element2", "MD_bond_kJ_element2"]:
        return "MHD_bond_element2"
    if feature in [
        "Frequency_H_element1",
        "Frequency_D_element1",
        "delta_ZPE_H_element1",
    ]:
        return "HD_vibration_ZPE_element1"
    if feature in [
        "Frequency_H_element2",
        "Frequency_D_element2",
        "delta_ZPE_H_element2",
    ]:
        return "HD_vibration_ZPE_element2"
    return feature


def physical_family(feature: str) -> str:
    if feature in ["I_raw", "Rs_from_EIS"]:
        return "electrochemical_response"
    if feature.startswith(("work_function", "electronegativity", "pzc", "E_std")):
        return "electronic_redox"
    if feature.startswith(("FE_ads_avg", "delta_EOH")):
        return "adsorption_affinity"
    if feature.startswith(("MH_bond_kJ", "MD_bond_kJ", "Frequency_", "delta_ZPE_H")):
        return "isotope_bond_vibration"
    if feature.startswith("MM_bond"):
        return "metal_metal_cohesion"
    return "other"


def build_chemistry_constrained_groups(
    spearman: pd.DataFrame,
    target_corr: pd.Series,
    threshold: float = SPEARMAN_ABS_THRESHOLD,
) -> pd.DataFrame:
    features = list(spearman.columns)
    parent = {feature: feature for feature in features}

    def find(feature: str) -> str:
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]
        return feature

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(features):
        for b in features[i + 1:]:
            same_chemistry = chemistry_equivalence_class(a) == chemistry_equivalence_class(b)
            strongly_correlated = abs(spearman.loc[a, b]) >= threshold
            if same_chemistry and strongly_correlated:
                union(a, b)

    components: dict[str, list[str]] = {}
    for feature in features:
        components.setdefault(find(feature), []).append(feature)

    groups = list(components.values())
    groups.sort(key=lambda group: (-len(group), group[0]))

    rows = []
    for idx, group in enumerate(groups, start=1):
        representative = choose_representative(group, target_corr)
        rows.append(
            {
                "group_id": f"G{idx:02d}",
                "n_features": len(group),
                "representative_feature": representative,
                "representative_target_spearman": float(target_corr[representative]),
                "equivalence_classes": ";".join(
                    sorted({chemistry_equivalence_class(feature) for feature in group})
                ),
                "physical_families": ";".join(
                    sorted({physical_family(feature) for feature in group})
                ),
                "features": ", ".join(group),
            }
        )
    return pd.DataFrame(rows)


def choose_representative(group: list[str], target_corr: pd.Series) -> str:
    """Choose one representative per chemistry-constrained group.

    The priority list is chemistry-first, not purely statistical. Singletons
    are returned unchanged. If no priority feature is present, the feature with
    the largest absolute target Spearman correlation is used.
    """
    priority = [
        "MD_bond_kJ_element1",
        "MD_bond_kJ_element2",
        "delta_ZPE_H_element1",
        "delta_ZPE_H_element2",
        "Frequency_OH_element1",
        "Frequency_OH_element2",
    ]
    for feature in priority:
        if feature in group:
            return feature
    return max(group, key=lambda feature: abs(target_corr[feature]))


def parse_feature_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_split(X: pd.DataFrame, y: pd.Series) -> SplitData:
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_bins,
    )
    return SplitData(X_train, X_test, y_train, y_test)


def backward_elimination(
    split: SplitData,
    representative_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimator = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    current = list(dict.fromkeys(representative_features))
    history_rows = []
    removal_rows = []
    step = 0

    while len(current) >= 1:
        current_score = cross_val_score(
            estimator,
            split.X_train[current],
            split.y_train,
            cv=5,
            scoring="r2",
            n_jobs=-1,
        ).mean()
        history_rows.append(
            {
                "step": step,
                "n_features": len(current),
                "cv_r2": current_score,
                "features": ", ".join(current),
            }
        )
        if len(current) == 1:
            break

        best_score = -np.inf
        best_remaining = None
        for feature in current:
            remaining = [item for item in current if item != feature]
            score_after_removal = cross_val_score(
                estimator,
                split.X_train[remaining],
                split.y_train,
                cv=5,
                scoring="r2",
                n_jobs=-1,
            ).mean()
            removal_rows.append(
                {
                    "step": step,
                    "remove_feature": feature,
                    "cv_r2_after_removal": score_after_removal,
                }
            )
            if score_after_removal > best_score:
                best_score = score_after_removal
                best_remaining = remaining

        current = best_remaining
        step += 1

    return pd.DataFrame(history_rows), pd.DataFrame(removal_rows)


def train_final_models(
    split: SplitData,
    selected_features: list[str],
    feature_set_name: str,
) -> pd.DataFrame:
    model_configs = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 15, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [10, 15, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
        },
    }

    rows = []
    for model_name, config in model_configs.items():
        grid_search = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        grid_search.fit(split.X_train[selected_features], split.y_train)
        model = grid_search.best_estimator_
        y_train_pred = model.predict(split.X_train[selected_features])
        y_test_pred = model.predict(split.X_test[selected_features])
        rows.append(
            {
                "feature_set": feature_set_name,
                "model_name": model_name,
                "best_params": json.dumps(grid_search.best_params_),
                "best_cv_score": grid_search.best_score_,
                "train_r2": r2_score(split.y_train, y_train_pred),
                "test_r2": r2_score(split.y_test, y_test_pred),
                "train_rmse": rmse(split.y_train, y_train_pred),
                "test_rmse": rmse(split.y_test, y_test_pred),
                "n_features": len(selected_features),
                "selected_features": ", ".join(selected_features),
            }
        )
    return pd.DataFrame(rows).sort_values("test_r2", ascending=False)


def group_permutation_importance(
    model: ExtraTreesRegressor,
    split: SplitData,
    groups_df: pd.DataFrame,
    n_repeats: int = 30,
) -> pd.DataFrame:
    baseline_r2 = r2_score(split.y_test, model.predict(split.X_test))
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for _, group in groups_df.iterrows():
        features = parse_feature_list(group["features"])
        drops = []
        for _ in range(n_repeats):
            X_permuted = split.X_test.copy()
            for feature in features:
                X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())
            permuted_r2 = r2_score(split.y_test, model.predict(X_permuted))
            drops.append(baseline_r2 - permuted_r2)
        rows.append(
            {
                "group_id": group["group_id"],
                "representative_feature": group["representative_feature"],
                "n_features": group["n_features"],
                "physical_families": group["physical_families"],
                "importance_r2_drop_mean": float(np.mean(drops)),
                "importance_r2_drop_std": float(np.std(drops, ddof=1)),
                "features": group["features"],
            }
        )
    return pd.DataFrame(rows).sort_values("importance_r2_drop_mean", ascending=False)


def plot_elimination(history_df: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(
        history_df["n_features"],
        history_df["cv_r2"],
        marker="o",
        linewidth=2,
        color="#f28e2b",
    )
    ax.invert_xaxis()
    ax.set_xlabel("Number of Representative Features", fontweight="bold")
    ax.set_ylabel("5-Fold CV R2", fontweight="bold")
    ax.set_title("Chemistry-Constrained Grouped Feature Selection", fontweight="bold")
    ax.grid(alpha=0.25)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "backward_elimination_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_group_importance(importance_df: pd.DataFrame) -> None:
    top = importance_df.head(14).iloc[::-1].copy()
    labels = [
        f"{row.group_id}: {row.representative_feature}"
        for _, row in top.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.38 * len(top) + 1.2)))
    ax.barh(
        labels,
        top["importance_r2_drop_mean"],
        xerr=top["importance_r2_drop_std"],
        color="#f28e2b",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xlabel("Grouped Permutation Importance (R2 Drop)", fontweight="bold")
    ax.set_title("Descriptor Group Importance", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "group_permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, metadata = load_feature_matrix()
    split = make_split(X, y)

    spearman = X.corr(method="spearman")
    target_corr = X.apply(lambda column: column.corr(y, method="spearman"))
    spearman.to_csv(OUTPUT_DIR / "spearman_feature_correlation_matrix.csv")
    target_corr.rename("spearman_to_target").to_csv(
        OUTPUT_DIR / "spearman_feature_target_correlation.csv"
    )

    groups_df = build_chemistry_constrained_groups(spearman, target_corr)
    groups_df.to_csv(OUTPUT_DIR / "chemistry_constrained_groups.csv", index=False)

    representative_features = groups_df["representative_feature"].tolist()
    pd.DataFrame({"representative_feature": representative_features}).to_csv(
        OUTPUT_DIR / "representative_features.csv",
        index=False,
    )

    history_df, removal_df = backward_elimination(split, representative_features)
    history_df.to_csv(OUTPUT_DIR / "backward_elimination_history.csv", index=False)
    removal_df.to_csv(OUTPUT_DIR / "backward_elimination_removal_scores.csv", index=False)
    plot_elimination(history_df)

    best_row = history_df.loc[history_df["cv_r2"].idxmax()]
    feature_sets = {
        "best_cv": parse_feature_list(best_row["features"]),
    }
    for n_features in [6, 10]:
        rows = history_df[history_df["n_features"].eq(n_features)]
        if not rows.empty:
            feature_sets[f"compact_{n_features}"] = parse_feature_list(
                rows.iloc[0]["features"]
            )

    selected_summary = pd.DataFrame(
        [
            {
                "feature_set": name,
                "n_features": len(features),
                "features": ", ".join(features),
            }
            for name, features in feature_sets.items()
        ]
    )
    selected_summary.to_csv(OUTPUT_DIR / "selected_feature_sets.csv", index=False)

    final_results = []
    for set_name, selected in feature_sets.items():
        result = train_final_models(split, selected, set_name)
        final_results.append(result)
    final_results_df = pd.concat(final_results, ignore_index=True)
    final_results_df.to_csv(OUTPUT_DIR / "final_model_results.csv", index=False)
    best_by_feature_set = final_results_df.loc[
        final_results_df.groupby("feature_set")["test_r2"].idxmax()
    ].sort_values("test_r2", ascending=False)
    best_by_feature_set.to_csv(OUTPUT_DIR / "best_model_by_feature_set.csv", index=False)

    full_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    full_model.fit(split.X_train, split.y_train)
    group_importance = group_permutation_importance(full_model, split, groups_df)
    group_importance.to_csv(OUTPUT_DIR / "group_permutation_importance.csv", index=False)
    plot_group_importance(group_importance)

    summary = {
        "n_samples": int(len(X)),
        "n_candidate_features": int(len(CANDIDATE_FEATURES)),
        "n_chemistry_constrained_groups": int(len(groups_df)),
        "spearman_abs_threshold": SPEARMAN_ABS_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "note": (
            "Only chemically redundant descriptors are grouped. OH affinity, "
            "H adsorption, and M-H/M-D bond strength are retained as distinct "
            "descriptor families even when correlated."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSelected feature sets:")
    print(selected_summary.to_string(index=False))
    print("\nBest model by feature set:")
    print(
        best_by_feature_set[
            ["feature_set", "model_name", "best_cv_score", "test_r2", "test_rmse"]
        ].to_string(index=False)
    )
    print("\nTop grouped permutation importance:")
    print(group_importance.head(12).to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
