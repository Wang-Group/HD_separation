from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs" / "03_eta10_activity_model_raw_descriptors"
OUT_DIR = OUTPUT_ROOT / "sensitivity_eta10_artifacts"

DATA_CSV = OUTPUT_ROOT / "eta10_ge20_model_dataset_raw_descriptors.csv"
SELECTED_FEATURES_CSV = OUTPUT_ROOT / "selected_features_eta10_raw_descriptors.csv"

RANDOM_SEED = 17
TARGET = "eta10_mV"


def stratify_bins(y: pd.Series):
    try:
        bins = pd.qcut(y, q=8, labels=False, duplicates="drop")
        if len(pd.Series(bins).dropna().unique()) < 2:
            return None
        return bins
    except Exception:
        return None


def make_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=None,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_split=5,
            min_samples_leaf=1,
            max_depth=None,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def evaluate_variant(name: str, data: pd.DataFrame, features: list[str]) -> tuple[list[dict], list[dict], pd.DataFrame]:
    work = data.dropna(subset=[TARGET]).copy()
    X = work[features]
    y = pd.to_numeric(work[TARGET], errors="coerce")
    keep = y.notna()
    work = work.loc[keep].copy()
    X = X.loc[keep]
    y = y.loc[keep]

    split = train_test_split(
        X,
        y,
        work["sample"].to_numpy(),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=stratify_bins(y),
    )
    X_train, X_test, y_train, y_test, sample_train, sample_test = split
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    metric_rows = []
    importance_rows = []
    prediction_rows = []

    for model_name, model in make_models().items():
        pipeline = make_pipeline(SimpleImputer(strategy="median"), model)
        cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
        cv_mse = -cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")

        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        estimator = pipeline.steps[-1][1]

        metric_rows.append(
            {
                "variant": name,
                "model_name": model_name,
                "n_samples": len(work),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_crossing_1": int((work["n_crossings"].astype(int) == 1).sum()),
                "n_crossing_gt1": int((work["n_crossings"].astype(int) > 1).sum()),
                "eta10_min_mV": float(y.min()),
                "eta10_median_mV": float(y.median()),
                "eta10_max_mV": float(y.max()),
                "cv_r2_mean": float(cv_r2.mean()),
                "cv_r2_std": float(cv_r2.std()),
                "cv_rmse_mV_mean": float(np.sqrt(cv_mse).mean()),
                "cv_rmse_mV_std": float(np.sqrt(cv_mse).std()),
                "train_r2": float(r2_score(y_train, y_train_pred)),
                "test_r2": float(r2_score(y_test, y_test_pred)),
                "train_rmse_mV": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                "test_rmse_mV": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            }
        )

        for feature, importance in zip(features, estimator.feature_importances_):
            importance_rows.append(
                {
                    "variant": name,
                    "model_name": model_name,
                    "feature": feature,
                    "feature_importance": float(importance),
                }
            )

        for sample, observed, predicted in zip(sample_test, y_test, y_test_pred):
            prediction_rows.append(
                {
                    "variant": name,
                    "model_name": model_name,
                    "sample": sample,
                    "observed_eta10_mV": float(observed),
                    "predicted_eta10_mV": float(predicted),
                    "residual_mV": float(observed - predicted),
                }
            )

    return metric_rows, importance_rows, pd.DataFrame(prediction_rows)


def plot_summary(metrics: pd.DataFrame, importance: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    rf_metrics = metrics[metrics["model_name"].eq("Random Forest")].copy()
    rf_importance = importance[importance["model_name"].eq("Random Forest")].copy()
    pivot = rf_importance.pivot(index="feature", columns="variant", values="feature_importance").fillna(0)
    pivot = pivot.loc[pivot["baseline_current"].sort_values(ascending=True).index]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    axes[0].bar(rf_metrics["variant"], rf_metrics["test_r2"], color="#4C78A8")
    axes[0].axhline(rf_metrics.loc[rf_metrics["variant"].eq("baseline_current"), "test_r2"].iloc[0], color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Test R2")
    axes[0].set_title("Random Forest test R2")
    axes[0].tick_params(axis="x", labelrotation=35)

    axes[1].bar(rf_metrics["variant"], rf_metrics["test_rmse_mV"], color="#F58518")
    axes[1].axhline(rf_metrics.loc[rf_metrics["variant"].eq("baseline_current"), "test_rmse_mV"].iloc[0], color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Test RMSE (mV)")
    axes[1].set_title("Random Forest test RMSE")
    axes[1].tick_params(axis="x", labelrotation=35)

    im = axes[2].imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    axes[2].set_yticks(np.arange(len(pivot.index)))
    axes[2].set_yticklabels(pivot.index)
    axes[2].set_xticks(np.arange(len(pivot.columns)))
    axes[2].set_xticklabels(pivot.columns, rotation=35, ha="right")
    axes[2].set_title("RF feature importance")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "eta10_artifact_sensitivity_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "eta10_artifact_sensitivity_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_CSV)
    selected_features = pd.read_csv(SELECTED_FEATURES_CSV)["selected_feature"].tolist()

    variants = {
        "baseline_current": data,
        "single_crossing_only": data[data["n_crossings"].astype(int).eq(1)],
        "eta10_ge50_all_crossings": data[pd.to_numeric(data[TARGET], errors="coerce").ge(50.0)],
        "eta10_ge50_single_crossing": data[
            data["n_crossings"].astype(int).eq(1)
            & pd.to_numeric(data[TARGET], errors="coerce").ge(50.0)
        ],
    }

    all_metrics = []
    all_importance = []
    all_predictions = []
    sample_rows = []

    baseline_samples = set(variants["baseline_current"]["sample"])
    for variant, frame in variants.items():
        excluded = sorted(baseline_samples - set(frame["sample"]))
        for sample in excluded:
            sample_rows.append({"variant": variant, "excluded_sample": sample})

        metric_rows, importance_rows, predictions = evaluate_variant(variant, frame, selected_features)
        all_metrics.extend(metric_rows)
        all_importance.extend(importance_rows)
        all_predictions.append(predictions)

    metrics = pd.DataFrame(all_metrics)
    importance = pd.DataFrame(all_importance)
    predictions = pd.concat(all_predictions, ignore_index=True)
    excluded_samples = pd.DataFrame(sample_rows)

    baseline_rf = importance[
        importance["variant"].eq("baseline_current")
        & importance["model_name"].eq("Random Forest")
    ].set_index("feature")["feature_importance"]
    stability_rows = []
    for (variant, model_name), group in importance.groupby(["variant", "model_name"]):
        values = group.set_index("feature")["feature_importance"]
        stability_rows.append(
            {
                "variant": variant,
                "model_name": model_name,
                "spearman_vs_baseline_rf": float(values.corr(baseline_rf, method="spearman")),
                "pearson_vs_baseline_rf": float(values.corr(baseline_rf, method="pearson")),
                "top3_features": ";".join(
                    group.sort_values("feature_importance", ascending=False)["feature"].head(3)
                ),
            }
        )
    stability = pd.DataFrame(stability_rows)

    metrics.to_csv(OUT_DIR / "eta10_artifact_sensitivity_metrics.csv", index=False)
    importance.to_csv(OUT_DIR / "eta10_artifact_sensitivity_feature_importance.csv", index=False)
    predictions.to_csv(OUT_DIR / "eta10_artifact_sensitivity_test_predictions.csv", index=False)
    excluded_samples.to_csv(OUT_DIR / "eta10_artifact_sensitivity_excluded_samples.csv", index=False)
    stability.to_csv(OUT_DIR / "eta10_artifact_sensitivity_feature_stability.csv", index=False)
    plot_summary(metrics, importance)

    summary = {
        "selected_features": selected_features,
        "variants": {
            name: {
                "n_samples": int(len(frame)),
                "n_single_crossing": int((frame["n_crossings"].astype(int) == 1).sum()),
                "n_multi_crossing": int((frame["n_crossings"].astype(int) > 1).sum()),
            }
            for name, frame in variants.items()
        },
        "metric_file": str(OUT_DIR / "eta10_artifact_sensitivity_metrics.csv"),
        "feature_stability_file": str(OUT_DIR / "eta10_artifact_sensitivity_feature_stability.csv"),
    }
    (OUT_DIR / "eta10_artifact_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(metrics.sort_values(["variant", "model_name"]).to_string(index=False))
    print(f"Saved sensitivity outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
