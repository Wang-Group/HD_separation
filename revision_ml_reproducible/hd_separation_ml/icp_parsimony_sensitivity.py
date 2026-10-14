"""Re-evaluate sparse selection rules using cached nested inner-CV candidates.

This is a retrospective sensitivity analysis on the previous outer partitions,
not a new independent test. Selection rules read inner scores only. One-SE is
the primary prespecified simplicity rule; fixed penalties are diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from icp_intrinsic_descriptor_analysis import ROOT, DescriptorRepresentation, load_data, scores
from icp_nested_group_cv import (
    OUTPUT as SOURCE, SCENARIOS, FRACTION, choose_best, make_model,
    interpret_outer, feature_label, markdown_table,
)

OUTPUT = ROOT / "outputs" / "icp_parsimony_sensitivity_20260928"
RULES = {"max_score": None, "one_se": None,
         "penalty_0.001": .001, "penalty_0.003": .003, "penalty_0.005": .005}


def select_candidate(candidates, rule, n_inner):
    best = choose_best(candidates)
    if rule == "max_score":
        return best, best["inner_r2_mean"]
    if rule == "one_se":
        threshold = best["inner_r2_mean"] - best["inner_r2_std"] / np.sqrt(n_inner)
        eligible = [c for c in candidates if c["inner_r2_mean"] >= threshold - 1e-12]
        min_k = min(c["n_features"] for c in eligible)
        return choose_best([c for c in eligible if c["n_features"] == min_k]), float(threshold)
    penalty = RULES[rule]
    chosen = max(candidates, key=lambda c: (c["inner_r2_mean"] - penalty * c["n_features"],
        -c["n_features"], -c["config_id"]))
    return chosen, None


def evaluate_task(path, data, assignments, manifest):
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = {k: payload[k] for k in ["repeat", "fold", "scenario", "model"]}
    test_names = set(assignments.loc[assignments["repeat"].eq(key["repeat"]) &
        assignments.fold.eq(key["fold"]), "sample"])
    training = data[~data["sample"].isin(test_names)].copy()
    testing = data[data["sample"].isin(test_names)].copy()
    assert not set(training.metal_pair) & set(testing.metal_pair)
    transform = DescriptorRepresentation(key["scenario"]).fit(training)
    tr, te = transform.transform(training), transform.transform(testing)
    columns = list(tr.columns)
    rank = np.array([payload["outer_train_rfe_ranks"][c] for c in columns])
    order = np.argsort(rank, kind="stable")
    best = choose_best(payload["candidates"])
    cached_metric = next(row for row in payload["metrics"] if row["strategy"] == "nested_selection")
    cached_predictions = pd.DataFrame([row for row in payload["predictions"]
        if row["strategy"] == "nested_selection"]).set_index("sample").reindex(testing["sample"])
    np.testing.assert_allclose(cached_predictions.y_true, testing.separation_coefficient)
    cached_columns = [columns[j] for j in np.sort(order[:best["n_features"]])]
    assert cached_columns == cached_metric["selected_features"]
    cache = {(best["n_features"], best["config_id"]): (
        cached_predictions.y_pred.to_numpy(), payload["feature_importance"])}
    metrics, predictions, features, importance = [], [], [], []
    seed = 2000 + key["repeat"] * 100 + key["fold"]
    y = testing.separation_coefficient.to_numpy(float)
    for rule in RULES:
        chosen, threshold = select_candidate(payload["candidates"], rule, manifest["inner_folds"])
        cache_key = (chosen["n_features"], chosen["config_id"])
        mask = np.sort(order[:chosen["n_features"]])
        selected_columns = [columns[j] for j in mask]
        if cache_key in cache:
            pred, shap_rows = cache[cache_key]
        else:
            model = make_model(key["model"], chosen["config"], manifest["final_trees"], seed)
            model.fit(tr.to_numpy()[:, mask], training.separation_coefficient.to_numpy(float))
            pred = model.predict(te.to_numpy()[:, mask])
            shap_rows = []
            if rule == "one_se":
                shap_rows, _ = interpret_outer(model, te.to_numpy()[:, mask], y,
                    columns, selected_columns, seed)
            cache[cache_key] = (pred, shap_rows)
        selected_icp = FRACTION in selected_columns
        metrics.append({**key, "rule": rule, "n_features": len(mask),
            "selected_features": selected_columns, "icp_selected": selected_icp,
            "inner_r2_mean": chosen["inner_r2_mean"], "best_inner_r2_mean": best["inner_r2_mean"],
            "inner_r2_cost": best["inner_r2_mean"] - chosen["inner_r2_mean"],
            "threshold": threshold, "penalty": RULES[rule], "config_id": chosen["config_id"],
            "config": chosen["config"], **scores(y, pred)})
        predictions += [{**key, "rule": rule, "sample": name, "y_true": float(obs), "y_pred": float(p)}
            for name, obs, p in zip(testing["sample"], y, pred)]
        features += [{**key, "rule": rule, "feature": col, "selected": col in selected_columns} for col in columns]
        if rule == "one_se":
            assert len(shap_rows) == len(columns)
            importance += [{**key, **row} for row in shap_rows]
    return key, metrics, predictions, features, importance


def summarize(results, output):
    metrics = pd.DataFrame([row for result in results for row in result[1]])
    predictions = pd.DataFrame([row for result in results for row in result[2]])
    features = pd.DataFrame([row for result in results for row in result[3]])
    importance = pd.DataFrame([row for result in results for row in result[4]])
    keys = ["scenario", "model", "rule"]
    baseline = metrics[metrics.rule.eq("max_score")][["repeat", "fold", "scenario", "model", "r2", "n_features"]]
    baseline = baseline.rename(columns={"r2": "baseline_r2", "n_features": "baseline_n_features"})
    metrics = metrics.merge(baseline, on=["repeat", "fold", "scenario", "model"], validate="many_to_one")
    metrics["delta_r2"] = metrics.r2 - metrics.baseline_r2
    metrics["features_removed"] = metrics.baseline_n_features - metrics.n_features
    metrics.to_csv(output / "outer_fold_metrics.csv", index=False)
    predictions.to_csv(output / "outer_predictions.csv", index=False)
    features.to_csv(output / "feature_selection_per_fold.csv", index=False)
    importance.to_csv(output / "one_se_shap_per_fold.csv", index=False)
    summary = metrics.groupby(keys, as_index=False).agg(r2_mean=("r2", "mean"),
        r2_std=("r2", "std"), rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
        n_features_mean=("n_features", "mean"), n_features_min=("n_features", "min"),
        n_features_max=("n_features", "max"), mean_delta_r2=("delta_r2", "mean"),
        mean_features_removed=("features_removed", "mean"), icp_selected_count=("icp_selected", "sum"),
        n_outer_folds=("r2", "size"))
    summary.to_csv(output / "model_summary.csv", index=False)
    stability = features.groupby(keys + ["feature"], as_index=False).agg(
        selection_frequency=("selected", "mean"), selected_count=("selected", "sum"))
    stability.to_csv(output / "selection_stability.csv", index=False)
    imp = importance.groupby(["scenario", "model", "feature"], as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"), selection_frequency=("selected", "mean"))
    imp.to_csv(output / "one_se_shap_summary.csv", index=False)
    pooled = []
    for (scenario, model, rule, repeat), rows in predictions.groupby(keys + ["repeat"]):
        assert len(rows) == 554 and rows["sample"].nunique() == 554
        pooled.append(dict(scenario=scenario, model=model, rule=rule, repeat=repeat,
            **scores(rows.y_true, rows.y_pred)))
    pd.DataFrame(pooled).to_csv(output / "pooled_oof_metrics.csv", index=False)
    return summary, imp


def draw(summary, importance, output):
    plt.rcParams.update({"font.family": "Arial", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), layout="constrained")
    colors = {"RF": "#237b95", "ET": "#d47c27"}
    markers = {"raw26": "o", "raw26_fraction": "s", "icp13": "^"}
    for ax, rule, title in zip(axes[0], ["max_score", "one_se"], ["Maximum inner-CV score", "One-standard-error selection"]):
        for model in colors:
            for scenario in SCENARIOS:
                row = summary[(summary.model == model) & (summary.scenario == scenario) & (summary.rule == rule)].iloc[0]
                ax.scatter(row.n_features_mean, row.r2_mean, color=colors[model], marker=markers[scenario],
                    s=75, label=f"{model}: {SCENARIOS[scenario]}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(2, 27)
        ax.set_ylim(.35, .58)
        ax.set_xlabel("Mean number of selected features")
        ax.set_ylabel(r"Mean outer validation $R^2$")
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    sub = summary[summary.scenario.eq("raw26_fraction")]
    for model in colors:
        line = sub[sub.model.eq(model) & sub.rule.ne("one_se")].sort_values("n_features_mean")
        axes[1, 0].plot(line.n_features_mean, line.r2_mean, "o-", color=colors[model], label=model)
        row = sub[sub.model.eq(model) & sub.rule.eq("one_se")].iloc[0]
        axes[1, 0].scatter(row.n_features_mean, row.r2_mean, marker="*", s=160,
            color=colors[model], edgecolor="black", zorder=3)
    axes[1, 0].set_xlabel("Mean number of selected features")
    axes[1, 0].set_ylabel(r"Mean outer validation $R^2$")
    axes[1, 0].set_title("26 properties + ICP: fixed penalty sensitivity", fontweight="bold")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=.2)
    axes[1, 0].text(.03, .04, "Stars: one-SE; circles: fixed penalties and no penalty",
        transform=axes[1, 0].transAxes, fontsize=9)
    sub = importance[importance.scenario.eq("raw26_fraction")]
    order = sub.groupby("feature").mean_abs_shap.mean().nlargest(9).index.tolist()
    if FRACTION not in order:
        order.append(FRACTION)
    order = list(reversed(order))
    for i, model in enumerate(colors):
        rows = sub[sub.model.eq(model)].set_index("feature").reindex(order)
        axes[1, 1].barh(np.arange(len(order)) + (i - .5) * .35, rows.mean_abs_shap,
            height=.34, color=colors[model])
    axes[1, 1].set_yticks(np.arange(len(order)), [feature_label(f) for f in order])
    axes[1, 1].set_xlabel("Mean absolute outer validation SHAP")
    axes[1, 1].set_title("Sparse model: 26 properties + ICP", fontweight="bold")
    fig.suptitle("Feature-count regularization | same 15 grouped outer folds", fontsize=15)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"parsimony_comparison.{extension}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    manifest = json.loads((SOURCE / "analysis_manifest.json").read_text())
    for relative, expected in manifest["input_hashes"].items():
        if hashlib.sha256((ROOT.parent / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed since original nested CV: {relative}")
    paths = sorted((SOURCE / "checkpoints").glob("*.json"))
    assert len(paths) == 90
    data, _ = load_data()
    assignments = pd.read_csv(SOURCE / "outer_cv_assignments.csv")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    policy = dict(source_manifest_fingerprint=manifest["fingerprint"], jobs=args.jobs,
        primary_rule="one_se", rules=RULES,
        one_se_definition="Choose smallest k with inner mean R2 >= best mean - best inner fold SD / sqrt(3); then highest mean at that k.",
        penalty_definition="Maximize mean inner R2 - lambda * number of features; lambda fixed, not chosen on outer scores.",
        scope="Post-hoc simplicity sensitivity using existing inner candidates and outer partitions; not independent confirmation.",
        uncertainty="One-SE is a simplicity heuristic. Correlated CV folds mean SD/sqrt(3) is not a calibrated confidence bound.",
        counts=manifest["counts"], source_checkpoints={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUTPUT / "analysis_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    results = []
    start = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        pending = Parallel(n_jobs=args.jobs, return_as="generator_unordered", pre_dispatch=args.jobs)(
            delayed(evaluate_task)(p, data, assignments, manifest) for p in paths)
        for result in pending:
            results.append(result)
            if len(results) % 10 == 0:
                print(f"Completed {len(results)}/90 model-fold tasks in {(time.perf_counter()-start):.1f}s", flush=True)
    summary, importance = summarize(results, OUTPUT)
    draw(summary, importance, OUTPUT)
    report = f"""# Feature-count regularization sensitivity

The earlier manuscript-facing model used 10 selected features, starting from a
four-feature core. The recent nested model instead maximized mean inner-CV R2
and preferred smaller models only for exact score ties. Tree depth and leaf
regularization were present, but there was no predictor-count penalty.

## Rules

The primary simplicity rule is one-SE: among candidates within one approximate
standard error of the highest inner R2, choose the smallest feature count and
then the highest-scoring configuration at that count. The standard error is
the best candidate's inner fold SD divided by sqrt(3), not the SD itself. This
is a conventional parsimony heuristic, not a calibrated confidence interval
or a statistical equivalence test for these dependent CV folds.

Explicit fixed penalties are also evaluated: maximize inner R2 - lambda*k for
lambda = 0.001, 0.003, and 0.005. Lambda is not selected to maximize outer R2.
The penalty is on feature count, rather than an L1 coefficient penalty on the
forest. Candidate feature counts and all tree settings retain the original
search grid. In particular, 6 and 10 features are not candidates in this grid.
Correlated isotope features are still in the pool; selecting a feature is not
proof that it represents independent physical information.

## Evaluation

The cached inner candidate scores and training-only RFE rankings are reused.
Models are refit with 600 trees on each outer training set, using the same
seeds and medians as the original nested run. The original maximum-score
baseline predictions are reused exactly. All 12 CPU workers are allowed,
with one native/model thread per worker. No outer response chooses k, lambda,
or hyperparameters. However, this rule comparison was motivated by inspecting
the previous results and uses the same outer partitions: it is retrospective
sensitivity analysis, not independent validation of a newly chosen procedure.

## Results

{markdown_table(summary[["scenario", "model", "rule", "n_features_mean", "r2_mean", "r2_std", "mean_delta_r2", "icp_selected_count"]])}

Fold SD is descriptive. Positive or negative mean R2 changes are not presented
as statistically significant. Elemental-reference missingness, including
Eu/Nb and missing adsorption energies/PZC, remains unchanged.

## Files and reproduction

`outer_fold_metrics.csv` includes rule choices, inner thresholds, selected
features, and paired outer-score differences. `selection_stability.csv` and
`one_se_shap_summary.csv` describe selection and predictive interpretation.
`analysis_policy.json` records source hashes and the selection definitions.

```powershell
& C:\\Users\\yibinjiang\\anaconda3\\python.exe .\\hd_separation_ml\\icp_parsimony_sensitivity.py --jobs 12
```

Method reference:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html
"""
    (OUTPUT / "README.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Results: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
