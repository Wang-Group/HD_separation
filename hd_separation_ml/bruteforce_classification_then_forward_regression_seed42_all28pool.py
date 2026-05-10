from __future__ import annotations

"""All-candidate feature-family selection for H/D separation.

This is the direct all-feature counterpart to
``bruteforce_classification_then_forward_regression_seed42_16pool.py``. It uses
the same high-separation classification plus forward-regression workflow, but
starts from every candidate descriptor in ``candidate_features.csv`` rather than
the seed-42 16-feature prefiltered pool.
"""

import pandas as pd

import feature_selection_workflow_core as workflow


workflow.OUTPUT_DIR = (
    workflow.ROOT
    / "outputs"
    / "06b_bruteforce_classification_forward_regression_seed42_all28pool"
)
workflow.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_candidate_feature_pool() -> list[str]:
    feature_file = workflow.FEATURE_DIR / "candidate_features.csv"
    features = pd.read_csv(feature_file)["feature"].dropna().astype(str).tolist()
    return [feature.strip() for feature in features]


def main():
    workflow.configure_matplotlib()
    feature_pool = load_all_candidate_feature_pool()
    pd.DataFrame({"feature": feature_pool}).to_csv(
        workflow.OUTPUT_DIR / "all28_feature_pool.csv",
        index=False,
    )
    df, X, y_reg, y_cls = workflow.load_dataset(feature_pool)

    print(f"Feature pool size: {len(feature_pool)}")
    print(f"Dataset size: {len(df)}; high-S class count: {int(y_cls.sum())}")
    print(f"XGBoost available: {workflow.HAS_XGB}")
    print(f"Parallel jobs: {workflow.PARALLEL_JOBS}", flush=True)

    classification_results = workflow.brute_force_classification(X, y_cls, feature_pool)
    feature_weights = workflow.calculate_feature_weights(classification_results, feature_pool)
    family_weights, family_representatives = workflow.calculate_family_weights(feature_weights)
    core4 = feature_weights.head(4)["feature"].tolist()
    pd.DataFrame({"feature": core4}).to_csv(
        workflow.OUTPUT_DIR / "classification_core4_features.csv",
        index=False,
    )

    path, path_summary = workflow.forward_regression(X, y_reg, feature_pool, core4)
    plateau9 = workflow.save_plateau_feature_set(path_summary, n_features=9)

    workflow.plot_classification_weights(feature_weights)
    workflow.plot_forward_regression(path_summary)

    print("\nCore four features from all-28 classification weighted combinations:")
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
    print(f"\nSaved outputs to: {workflow.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
