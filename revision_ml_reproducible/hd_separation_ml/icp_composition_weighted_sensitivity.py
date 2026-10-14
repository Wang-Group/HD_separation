"""Compare current and ICP-composition-weighted H/D separation models.

This is a sensitivity analysis, not a replacement for the manuscript-facing
feature-selection workflow. All scenarios use the same ICP-eligible samples,
the same seed-42 stratified split identities, and the same RF/ExtraTrees settings.
"""

from __future__ import annotations

import json
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

from icp_master import MASTER, load_initial_composition


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
FEATURE_DIR = ROOT / "outputs" / "feature_selection_561_rs_from_eis_no_eis_fitting"
FEATURE_MATRIX = FEATURE_DIR / "feature_matrix_561_no_eis_fitting.csv"
CANDIDATE_FEATURES = FEATURE_DIR / "candidate_features.csv"
COMPOSITION_TABLE = MASTER
OUTPUT_DIR = ROOT / "outputs" / "icp_composition_weighted_sensitivity_20260927"

SEED = 42
CURRENT_10 = [
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

SCENARIO_LABELS = {
    "current10_raw": "Current 10",
    "current10_plus_icp_fraction": "Current 10 + ICP fraction",
    "current10_icp_scaled": "Current 10 ICP-scaled",
    "raw28": "Raw 28",
    "raw28_plus_icp_fraction": "Raw 28 + ICP fraction",
    "raw28_icp_scaled": "Raw 28 ICP-scaled",
    "nominal_weighted15": "6:1 weighted 15",
    "icp_weighted15": "ICP weighted 15",
    "raw28_plus_nominal13": "Raw 28 + 6:1",
    "raw28_plus_icp13": "Raw 28 + ICP",
}


def configure_matplotlib() -> None:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["axes.linewidth"] = 1.35
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def models() -> dict[str, object]:
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


def build_weighted_matrix() -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    features = pd.read_csv(FEATURE_MATRIX).dropna(subset=["separation_coefficient"]).copy()
    composition = load_initial_composition(COMPOSITION_TABLE)[
        [
            "sample",
            "separation_coefficient",
            "actual_frac_element1",
            "actual_frac_element2",
            "nominal_frac_element1",
            "nominal_frac_element2",
            "merge_status",
            "matched_zero_component_accepted",
        ]
    ]
    data = features.drop(columns="separation_coefficient").merge(
        composition, on="sample", how="inner", validate="one_to_one")

    raw28 = pd.read_csv(CANDIDATE_FEATURES)["feature"].dropna().astype(str).tolist()
    pair_bases = [
        feature.removesuffix("_element1")
        for feature in raw28
        if feature.endswith("_element1") and feature.replace("_element1", "_element2") in raw28
    ]
    nominal_weighted: list[str] = []
    icp_weighted: list[str] = []
    nominal_scaled: list[str] = []
    icp_scaled: list[str] = []
    for base in pair_bases:
        element1 = f"{base}_element1"
        element2 = f"{base}_element2"
        nominal_name = f"nominal_weighted_{base}"
        icp_name = f"icp_weighted_{base}"
        data[nominal_name] = (
            data["nominal_frac_element1"].astype(float) * data[element1].astype(float)
            + data["nominal_frac_element2"].astype(float) * data[element2].astype(float)
        )
        data[icp_name] = (
            data["actual_frac_element1"].astype(float) * data[element1].astype(float)
            + data["actual_frac_element2"].astype(float) * data[element2].astype(float)
        )
        nominal_weighted.append(nominal_name)
        icp_weighted.append(icp_name)

        for suffix, fraction_column in [
            ("element1", "nominal_frac_element1"),
            ("element2", "nominal_frac_element2"),
        ]:
            source = f"{base}_{suffix}"
            output = f"nominal_scaled_{base}_{suffix}"
            data[output] = data[fraction_column].astype(float) * data[source].astype(float)
            nominal_scaled.append(output)
        for suffix, fraction_column in [
            ("element1", "actual_frac_element1"),
            ("element2", "actual_frac_element2"),
        ]:
            source = f"{base}_{suffix}"
            output = f"icp_scaled_{base}_{suffix}"
            data[output] = data[fraction_column].astype(float) * data[source].astype(float)
            icp_scaled.append(output)

    unchanged = [feature for feature in raw28 if not feature.endswith(("_element1", "_element2"))]
    return (
        data,
        raw28,
        unchanged + nominal_weighted,
        unchanged + icp_weighted,
        unchanged + nominal_scaled,
        unchanged + icp_scaled,
    )


def evaluate_scenarios(
    data: pd.DataFrame,
    scenarios: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = data["separation_coefficient"].to_numpy(float)
    indices = np.arange(len(data))
    strat = pd.qcut(y, q=5, labels=False, duplicates="drop")

    # Preserve the formal 561-sample seed-42 split identities, then restrict
    # both partitions to the ICP-eligible samples. Re-splitting after applying
    # the composition policy obscures the composition-only comparison.
    full_data = pd.read_csv(FEATURE_MATRIX).dropna(subset=["separation_coefficient"]).reset_index(drop=True)
    full_indices = np.arange(len(full_data))
    full_strat = pd.qcut(
        full_data["separation_coefficient"], q=5, labels=False, duplicates="drop"
    )
    full_train_idx, full_test_idx = train_test_split(
        full_indices,
        test_size=0.20,
        random_state=SEED,
        stratify=full_strat,
    )
    train_samples = set(full_data.iloc[full_train_idx]["sample"])
    test_samples = set(full_data.iloc[full_test_idx]["sample"])
    train_idx = np.flatnonzero(data["sample"].isin(train_samples).to_numpy())
    test_idx = np.flatnonzero(data["sample"].isin(test_samples).to_numpy())
    if len(train_idx) + len(test_idx) != len(data):
        raise ValueError("Some ICP-eligible samples are absent from the formal 561-sample split.")

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(indices, strat)
    )

    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for scenario, feature_names in scenarios.items():
        X = data[feature_names].copy()
        for model_name, estimator in models().items():
            cv_r2: list[float] = []
            cv_rmse: list[float] = []
            for fold_train, fold_test in folds:
                pipeline = make_pipeline(SimpleImputer(strategy="median"), clone(estimator))
                pipeline.fit(X.iloc[fold_train], y[fold_train])
                pred = pipeline.predict(X.iloc[fold_test])
                cv_r2.append(float(r2_score(y[fold_test], pred)))
                cv_rmse.append(float(np.sqrt(mean_squared_error(y[fold_test], pred))))

            pipeline = make_pipeline(SimpleImputer(strategy="median"), clone(estimator))
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            test_pred = pipeline.predict(X.iloc[test_idx])
            test_r2 = float(r2_score(y[test_idx], test_pred))
            test_rmse = float(np.sqrt(mean_squared_error(y[test_idx], test_pred)))
            metric_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "model": model_name,
                    "n_samples": len(data),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_features": len(feature_names),
                    "test_r2": test_r2,
                    "test_rmse": test_rmse,
                    "cv_r2_mean": float(np.mean(cv_r2)),
                    "cv_r2_std": float(np.std(cv_r2, ddof=1)),
                    "cv_rmse_mean": float(np.mean(cv_rmse)),
                    "cv_rmse_std": float(np.std(cv_rmse, ddof=1)),
                    "features": ", ".join(feature_names),
                }
            )

            fitted_estimator = pipeline.named_steps[list(pipeline.named_steps)[-1]]
            for feature, importance in zip(feature_names, fitted_estimator.feature_importances_):
                importance_rows.append(
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

            for row_idx, pred in zip(test_idx, test_pred):
                prediction_rows.append(
                    {
                        "scenario": scenario,
                        "model": model_name,
                        "sample": data.iloc[row_idx]["sample"],
                        "y_true": y[row_idx],
                        "y_pred": float(pred),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(importance_rows), pd.DataFrame(prediction_rows)


def draw_summary(metrics: pd.DataFrame, importance: pd.DataFrame, predictions: pd.DataFrame) -> None:
    configure_matplotlib()
    order = list(SCENARIO_LABELS)
    labels = [SCENARIO_LABELS[key] for key in order]
    model_order = ["Random Forest", "Extra Trees"]
    colors = {"Random Forest": "#377eb8", "Extra Trees": "#e76f51"}
    x = np.arange(len(order))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.8), dpi=300)
    for model_idx, model_name in enumerate(model_order):
        sub = metrics.loc[metrics["model"].eq(model_name)].set_index("scenario").reindex(order)
        offset = (model_idx - 0.5) * width
        axes[0, 0].bar(x + offset, sub["test_r2"], width, color=colors[model_name], label=model_name)
        axes[0, 1].bar(
            x + offset,
            sub["cv_r2_mean"],
            width,
            yerr=sub["cv_r2_std"],
            color=colors[model_name],
            capsize=2.5,
            label=model_name,
        )

    for ax, ylabel, title in [
        (axes[0, 0], "Held-out test R2", "Seed-42 held-out performance"),
        (axes[0, 1], "Mean 5-fold CV R2", "Cross-validated performance"),
    ]:
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", color="#dfe3e8", linewidth=0.65)
        ax.legend(frameon=False, fontsize=8.5)

    icp_importance = (
        importance.loc[importance["scenario"].eq("icp_weighted15")]
        .groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"))
        .sort_values("mean_importance", ascending=False)
        .head(12)
        .sort_values("mean_importance")
    )
    axes[1, 0].barh(
        icp_importance["feature"].str.replace("icp_weighted_", "", regex=False),
        icp_importance["mean_importance"],
        color="#55a868",
        edgecolor="#222222",
        linewidth=0.45,
    )
    axes[1, 0].set_xlabel("Mean RF/ET feature importance")
    axes[1, 0].set_title("ICP-weighted 15-feature model", fontweight="bold")
    axes[1, 0].grid(True, axis="x", color="#dfe3e8", linewidth=0.65)

    icp_metrics = metrics.loc[metrics["scenario"].eq("icp_weighted15")].sort_values("test_r2", ascending=False)
    best_model = str(icp_metrics.iloc[0]["model"])
    best_metrics = icp_metrics.iloc[0]
    pred = predictions.loc[
        predictions["scenario"].eq("icp_weighted15") & predictions["model"].eq(best_model)
    ]
    lo = min(float(pred["y_true"].min()), float(pred["y_pred"].min())) - 0.2
    hi = max(float(pred["y_true"].max()), float(pred["y_pred"].max())) + 0.2
    axes[1, 1].scatter(
        pred["y_true"],
        pred["y_pred"],
        s=34,
        color="#f2c14e",
        edgecolor="#6b4f00",
        linewidth=0.45,
        alpha=0.82,
    )
    axes[1, 1].plot([lo, hi], [lo, hi], "--", color="#d62728", linewidth=1.5)
    axes[1, 1].set_xlim(lo, hi)
    axes[1, 1].set_ylim(lo, hi)
    axes[1, 1].set_xlabel("Measured H/D separation coefficient")
    axes[1, 1].set_ylabel("Predicted H/D separation coefficient")
    axes[1, 1].set_title(
        f"ICP-weighted test set, {best_model}\nR2={best_metrics['test_r2']:.3f}, RMSE={best_metrics['test_rmse']:.3f}",
        fontweight="bold",
    )
    axes[1, 1].grid(True, color="#dfe3e8", linewidth=0.65)

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=8.3)
    fig.suptitle("H/D separation ML sensitivity to ICP composition weighting", fontsize=13.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for suffix in ("png", "pdf"):
        path = OUTPUT_DIR / f"icp_composition_weighted_ml_summary.{suffix}"
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, raw28, nominal15, icp15, _nominal_scaled28, icp_scaled28 = build_weighted_matrix()
    nominal13 = [feature for feature in nominal15 if feature.startswith("nominal_weighted_")]
    icp13 = [feature for feature in icp15 if feature.startswith("icp_weighted_")]

    current10_icp_scaled = []
    for feature in CURRENT_10:
        base, suffix = feature.rsplit("_", maxsplit=1)
        current10_icp_scaled.append(f"icp_scaled_{base}_{suffix}")

    scenarios = {
        "current10_raw": CURRENT_10,
        "current10_plus_icp_fraction": CURRENT_10 + ["actual_frac_element1"],
        "current10_icp_scaled": current10_icp_scaled,
        "raw28": raw28,
        "raw28_plus_icp_fraction": raw28 + ["actual_frac_element1"],
        "raw28_icp_scaled": icp_scaled28,
        "nominal_weighted15": nominal15,
        "icp_weighted15": icp15,
        "raw28_plus_nominal13": raw28 + nominal13,
        "raw28_plus_icp13": raw28 + icp13,
    }

    metrics, importance, predictions = evaluate_scenarios(data, scenarios)
    data.to_csv(
        OUTPUT_DIR / "feature_matrix_icp_eligible_with_weighted_descriptors.csv",
        index=False,
    )
    metrics.to_csv(OUTPUT_DIR / "scenario_model_metrics.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "scenario_feature_importance.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "scenario_test_predictions.csv", index=False)
    with (OUTPUT_DIR / "analysis_policy.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "purpose": "Sensitivity analysis; not a replacement for the manuscript feature-selection workflow.",
                "n_samples": len(data),
                "seed": SEED,
                "formal_split_source_n": 561,
                "n_train": int(metrics["n_train"].iloc[0]),
                "n_test": int(metrics["n_test"].iloc[0]),
                "split": "Formal 561-sample seed-42 stratified split identities, restricted to the ICP-eligible samples; identical rows across scenarios.",
                "models": ["Random Forest", "Extra Trees"],
                "imputation": "Training-fold median",
                "composition_policy": "Same as the latest Figure 4 ICP analysis; updated W measurements retained except W-Nb (both metals undetected), Au bimetals without ICP excluded, pure metals retained, single zero fractions accepted.",
                "feature_selection": "None; fixed scenario feature sets are compared to isolate composition-weighting effects.",
            },
            handle,
            indent=2,
        )
    draw_summary(metrics, importance, predictions)

    print(f"Samples: {len(data)}")
    print(f"ICP zero-component rows retained: {int(data['matched_zero_component_accepted'].astype(bool).sum())}")
    print("\nScenario metrics:")
    print(
        metrics[
            ["scenario", "model", "n_features", "test_r2", "test_rmse", "cv_r2_mean", "cv_r2_std"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
