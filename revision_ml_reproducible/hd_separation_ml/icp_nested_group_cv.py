"""Nested metal-pair CV with fold-local RFE and forest hyperparameter search.

The complete analysis is prespecified in a manifest. Outer validation responses
never enter imputation, feature selection, model selection, or parameter tuning.
Independent outer tasks are checkpointed to make long runs resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedGroupKFold

from icp_intrinsic_descriptor_analysis import (
    BASES, LABELS, ROOT, WORKBOOK, COMPOSITION_TABLE, FEATURE_MATRIX,
    DescriptorRepresentation, load_data, scores,
)

OUTPUT = ROOT / "outputs" / "icp_nested_group_cv_20260928"
SCENARIOS = {
    "raw26": "Separate elemental 26",
    "raw26_fraction": "Separate 26 + ICP fraction",
    "icp13": "ICP-weighted 13",
}
MODELS = {"RF": RandomForestRegressor, "ET": ExtraTreesRegressor}
FRACTION = "actual_frac_element1"
CONFIGS = [dict(max_depth=depth, min_samples_leaf=leaf, max_features=features)
    for depth, leaf, features in itertools.product([8, None], [1, 3, 6], ["sqrt", 1.0])]


def make_model(name, config, trees, seed):
    return MODELS[name](n_estimators=trees, random_state=seed, n_jobs=1, **config)


def feature_counts(p):
    return sorted(set(k for k in [4, 8, 12, 18, p] if k <= p))


def group_folds(data, splits, seed):
    strata = pd.qcut(data.separation_coefficient, 5, labels=False, duplicates="drop")
    groups = data.metal_pair.to_numpy()
    cv = StratifiedGroupKFold(splits, shuffle=True, random_state=seed)
    folds = list(cv.split(data, strata, groups))
    for train, valid in folds:
        if set(groups[train]) & set(groups[valid]):
            raise ValueError("Metal-pair overlap in grouped split.")
        if set(train) & set(valid) or len(train) + len(valid) != len(data):
            raise ValueError("Invalid partition.")
    return folds


def rfe_order(X, y, model_name, trees, seed):
    estimator = make_model(model_name,
        dict(max_depth=8, min_samples_leaf=2, max_features="sqrt"), trees, seed)
    selector = RFE(estimator, n_features_to_select=4, step=1).fit(X, y)
    return np.argsort(selector.ranking_, kind="stable"), selector.ranking_


def choose_best(candidates):
    # Only exact score ties prefer fewer predictors, then a deterministic config ID.
    return max(candidates, key=lambda r: (r["inner_r2_mean"], -r["n_features"], -r["config_id"]))


def search_training_fold(train_data, scenario, model_name, inner, args, seed):
    y = train_data.separation_coefficient.to_numpy(float)
    p = 13 if scenario == "icp13" else 26 + int(scenario == "raw26_fraction")
    counts = feature_counts(p)
    score_rows, selections = [], []
    for inner_fold, (train, valid) in enumerate(inner):
        transform = DescriptorRepresentation(scenario).fit(train_data.iloc[train])
        tr = transform.transform(train_data.iloc[train])
        va = transform.transform(train_data.iloc[valid])
        X, V = tr.to_numpy(), va.to_numpy()
        order, ranks = rfe_order(X, y[train], model_name, args.selector_trees, seed + inner_fold)
        for feature, rank in zip(tr.columns, ranks):
            selections.append(dict(inner_fold=inner_fold, feature=feature, rfe_rank=int(rank)))
        for k in counts:
            mask = np.sort(order[:k])
            for config_id, config in enumerate(CONFIGS):
                model = make_model(model_name, config, args.search_trees, seed + inner_fold)
                model.fit(X[:, mask], y[train])
                pred = model.predict(V[:, mask])
                score_rows.append(dict(inner_fold=inner_fold, n_features=k, config_id=config_id,
                    **scores(y[valid], pred)))
    candidates = []
    table = pd.DataFrame(score_rows)
    for (k, config_id), rows in table.groupby(["n_features", "config_id"]):
        candidates.append(dict(n_features=int(k), config_id=int(config_id),
            inner_r2_mean=float(rows.r2.mean()), inner_r2_std=float(rows.r2.std()),
            inner_rmse_mean=float(rows.rmse.mean()), config=CONFIGS[int(config_id)]))
    return choose_best(candidates), choose_best([r for r in candidates if r["n_features"] == p]), candidates, selections


def interpret_outer(model, X, y, all_columns, selected, seed):
    values = np.asarray(shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        .shap_values(X, check_additivity=True))
    mean_abs = np.abs(values).mean(axis=0)
    mapping = dict(zip(selected, mean_abs))
    rows = [dict(feature=feature, selected=feature in mapping,
        mean_abs_shap=float(mapping.get(feature, 0))) for feature in all_columns]
    perm = None
    if FRACTION in selected:
        base_r2 = scores(y, model.predict(X))["r2"]
        column = selected.index(FRACTION)
        rng = np.random.default_rng(seed)
        arrays = []
        for _ in range(10):
            arr = X.copy()
            arr[:, column] = arr[rng.permutation(len(arr)), column]
            arrays.append(arr)
        predicted = model.predict(np.concatenate(arrays)).reshape(len(arrays), len(X))
        drops = [base_r2 - scores(y, pred)["r2"] for pred in predicted]
        perm = dict(icp_permutation_r2_drop=float(np.mean(drops)),
            icp_permutation_std=float(np.std(drops, ddof=1)))
    return rows, perm


def run_task(data, train, test, repeat, fold, scenario, model_name, args, fingerprint):
    start = time.perf_counter()
    task_id = f"repeat{repeat}_fold{fold}_{scenario}_{model_name}"
    checkpoint = args.output / "checkpoints" / f"{task_id}.json"
    if checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload["fingerprint"] != fingerprint:
            raise ValueError(f"Checkpoint does not match run manifest: {checkpoint}")
        return task_id, payload, True
    training = data.iloc[train].reset_index(drop=True)
    testing = data.iloc[test].reset_index(drop=True)
    seed = 2000 + repeat * 100 + fold
    inner = group_folds(training, args.inner_folds, seed)
    best, best_all, candidates, inner_ranks = search_training_fold(
        training, scenario, model_name, inner, args, seed)
    transform = DescriptorRepresentation(scenario).fit(training)
    train_frame, test_frame = transform.transform(training), transform.transform(testing)
    X, V = train_frame.to_numpy(), test_frame.to_numpy()
    y, y_test = training.separation_coefficient.to_numpy(float), testing.separation_coefficient.to_numpy(float)
    order, ranks = rfe_order(X, y, model_name, args.selector_trees, seed)
    all_columns = list(train_frame.columns)
    settings = {
        "nested_selection": (best, np.sort(order[:best["n_features"]])),
        "tuned_all_features": (best_all, np.arange(len(all_columns))),
        "fixed_all_features": (dict(config=dict(max_depth=None, min_samples_leaf=1,
            max_features="sqrt"), n_features=len(all_columns), inner_r2_mean=None,
            config_id=-1), np.arange(len(all_columns))),
    }
    metric_rows, predictions = [], []
    importance, permutation = [], None
    selected_icp = False
    selected_no_icp_pred = None
    for strategy, (choice, mask) in settings.items():
        model = make_model(model_name, choice["config"], args.final_trees, seed)
        model.fit(X[:, mask], y)
        pred = model.predict(V[:, mask])
        chosen_columns = [all_columns[j] for j in mask]
        metric_rows.append(dict(strategy=strategy, n_features=len(mask),
            inner_r2_mean=choice["inner_r2_mean"], config=choice["config"],
            selected_features=chosen_columns, n_train=len(train), n_test=len(test),
            **scores(y_test, pred)))
        predictions += [dict(strategy=strategy, sample=sample, y_true=float(obs), y_pred=float(estimate))
            for sample, obs, estimate in zip(testing["sample"], y_test, pred)]
        if strategy == "nested_selection":
            importance, permutation = interpret_outer(model, V[:, mask], y_test,
                all_columns, chosen_columns, seed)
            selected_icp = FRACTION in chosen_columns
            if scenario == "raw26_fraction":
                if selected_icp:
                    reduced_mask = [j for j in mask if all_columns[j] != FRACTION]
                    reduced = make_model(model_name, choice["config"], args.final_trees, seed)
                    reduced.fit(X[:, reduced_mask], y)
                    selected_no_icp_pred = reduced.predict(V[:, reduced_mask])
                else:
                    selected_no_icp_pred = pred.copy()
                metric_rows.append(dict(strategy="selected_without_icp", n_features=len(mask) - int(selected_icp),
                    inner_r2_mean=None, config=choice["config"], selected_features=[c for c in chosen_columns if c != FRACTION],
                    n_train=len(train), n_test=len(test), **scores(y_test, selected_no_icp_pred)))
                predictions += [dict(strategy="selected_without_icp", sample=sample,
                    y_true=float(obs), y_pred=float(estimate)) for sample, obs, estimate in
                    zip(testing["sample"], y_test, selected_no_icp_pred)]
    inner_assignments = [dict(inner_fold=i, sample=training.iloc[j]["sample"],
        metal_pair=training.iloc[j]["metal_pair"]) for i, (_, va) in enumerate(inner) for j in va]
    payload = dict(fingerprint=fingerprint, repeat=repeat, fold=fold, scenario=scenario,
        model=model_name, metrics=metric_rows, predictions=predictions,
        candidates=candidates, inner_feature_ranks=inner_ranks,
        inner_assignments=inner_assignments, feature_importance=importance,
        outer_train_rfe_ranks=dict(zip(all_columns, map(int, ranks))),
        selected_icp=selected_icp, icp_permutation=permutation,
        elapsed_seconds=time.perf_counter() - start)
    checkpoint.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return task_id, payload, False


def feature_label(name):
    if name == FRACTION:
        return "ICP M1 molar fraction"
    for suffix, label in [("_element1", "M1"), ("_element2", "M2")]:
        if name.endswith(suffix):
            return f"{LABELS[name.removesuffix(suffix)]} ({label})"
    return LABELS[name]


def aggregate(payloads, data, output):
    metrics, predictions, importance, searches, inner_assignments, ranks, perm = [], [], [], [], [], [], []
    keys = ["repeat", "fold", "scenario", "model"]
    for payload in payloads:
        key = {k: payload[k] for k in keys}
        metrics.extend({**key, **row} for row in payload["metrics"])
        predictions.extend({**key, **row} for row in payload["predictions"])
        importance.extend({**key, **row} for row in payload["feature_importance"])
        searches.extend({**key, **row} for row in payload["candidates"])
        inner_assignments.extend({**key, **row} for row in payload["inner_assignments"])
        ranks.extend({**key, **row} for row in payload["inner_feature_ranks"])
        if payload["scenario"] == "raw26_fraction":
            perm.append({**key, "selected_icp": payload["selected_icp"],
                **(payload["icp_permutation"] or dict(icp_permutation_r2_drop=0.0, icp_permutation_std=0.0))})
    metrics, predictions, importance = map(pd.DataFrame, [metrics, predictions, importance])
    # Choose the model family from inner validation only, separately in each outer fold.
    selected = metrics[metrics.strategy.eq("nested_selection")]
    winners = selected.sort_values(["inner_r2_mean", "n_features", "model"],
        ascending=[False, True, True]).drop_duplicates(["repeat", "fold", "scenario"])
    winner_keys = winners[keys].copy()
    auto_metrics = winners.copy()
    auto_metrics["selected_model"] = auto_metrics.model
    auto_metrics["model"] = "Inner-selected RF/ET"
    auto_predictions = predictions[predictions.strategy.eq("nested_selection")].merge(winner_keys, on=keys)
    auto_predictions["model"] = "Inner-selected RF/ET"
    auto_importance = importance.merge(winner_keys, on=keys)
    auto_importance["model"] = "Inner-selected RF/ET"
    metrics = pd.concat([metrics, auto_metrics], ignore_index=True)
    predictions = pd.concat([predictions, auto_predictions], ignore_index=True)
    importance = pd.concat([importance, auto_importance], ignore_index=True)
    metrics.to_csv(output / "outer_fold_metrics.csv", index=False)
    predictions.to_csv(output / "outer_predictions.csv", index=False)
    importance.to_csv(output / "outer_feature_importance.csv", index=False)
    pd.DataFrame(searches).to_csv(output / "inner_search_results.csv", index=False)
    pd.DataFrame(inner_assignments).to_csv(output / "inner_cv_assignments.csv", index=False)
    pd.DataFrame(ranks).to_csv(output / "inner_rfe_rankings.csv", index=False)
    pd.DataFrame(perm).to_csv(output / "icp_permutation_outer_folds.csv", index=False)
    winner_keys.to_csv(output / "inner_selected_model_per_outer_fold.csv", index=False)
    group = ["scenario", "model", "strategy"]
    summary = metrics.groupby(group, as_index=False).agg(r2_mean=("r2", "mean"),
        r2_std=("r2", "std"), rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
        n_features_mean=("n_features", "mean"), n_features_min=("n_features", "min"),
        n_features_max=("n_features", "max"), n_outer_folds=("r2", "size"))
    summary.to_csv(output / "model_summary.csv", index=False)
    feature_summary = importance.groupby(["scenario", "model", "feature"], as_index=False).agg(
        selected_count=("selected", "sum"), selection_frequency=("selected", "mean"),
        mean_abs_shap=("mean_abs_shap", "mean"), std_abs_shap=("mean_abs_shap", "std"),
        n_outer_folds=("selected", "size"))
    feature_summary.to_csv(output / "feature_selection_stability.csv", index=False)
    repeat_scores = []
    for (scenario, model, strategy, repeat), rows in predictions.groupby(group + ["repeat"]):
        assert len(rows) == len(data) and rows["sample"].nunique() == len(data)
        repeat_scores.append(dict(scenario=scenario, model=model, strategy=strategy,
            repeat=repeat, **scores(rows.y_true, rows.y_pred)))
    pd.DataFrame(repeat_scores).to_csv(output / "pooled_oof_metrics_per_repeat.csv", index=False)
    pairs = []
    for model in [*MODELS, "Inner-selected RF/ET"]:
        subset = metrics[metrics.model.eq(model) & metrics.strategy.eq("nested_selection")]
        pivot = subset.pivot(index=["repeat", "fold"], columns="scenario", values="r2")
        for _, row in pivot.reset_index().iterrows():
            pairs.append(dict(model=model, repeat=int(row["repeat"]), fold=int(row["fold"]),
                comparison="ICP fraction added to raw descriptors",
                delta_r2=float(row.raw26_fraction - row.raw26)))
        if model in MODELS:
            subset = metrics[metrics.model.eq(model) & metrics.scenario.eq("raw26_fraction")]
            pivot = subset.pivot(index=["repeat", "fold"], columns="strategy", values="r2")
            for _, row in pivot.reset_index().iterrows():
                pairs.append(dict(model=model, repeat=int(row["repeat"]), fold=int(row["fold"]),
                    comparison="ICP fraction retained vs removed after selection",
                    delta_r2=float(row.nested_selection - row.selected_without_icp)))
    paired = pd.DataFrame(pairs)
    paired.to_csv(output / "paired_outer_fold_differences.csv", index=False)
    paired.groupby(["model", "comparison"]).agg(mean_delta=("delta_r2", "mean"),
        std_delta=("delta_r2", "std"), positive_folds=("delta_r2", lambda x: int((x > 0).sum())),
        n_folds=("delta_r2", "size")).to_csv(output / "paired_comparison_summary.csv")
    return summary, feature_summary, paired


def plot_results(summary, stability, paired, output, n_samples):
    plt.rcParams.update({"font.family": "Arial", "font.size": 10,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), layout="constrained")
    colors = {"RF": "#237b95", "ET": "#d47c27"}
    positions = np.arange(len(SCENARIOS))
    for ax, strategy, title in [(axes[0, 0], "fixed_all_features", "Fixed-model baseline"),
            (axes[0, 1], "nested_selection", "Nested feature selection and tuning")]:
        for i, model in enumerate(colors):
            sub = summary[summary.model.eq(model) & summary.strategy.eq(strategy)].set_index(
                "scenario").reindex(SCENARIOS)
            ax.errorbar(sub.r2_mean, positions + (i - .5) * .18, xerr=sub.r2_std,
                fmt="o", color=colors[model], capsize=3, label=model)
        ax.set_yticks(positions, list(SCENARIOS.values()))
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(r"Outer validation $R^2$ (mean and fold SD)")
        ax.grid(axis="x", alpha=.2)
        ax.legend(frameon=False)
    relevant = summary[summary.strategy.isin(["fixed_all_features", "nested_selection"])]
    bounds = (max(-1, float((relevant.r2_mean - relevant.r2_std).min()) - .03),
              float((relevant.r2_mean + relevant.r2_std).max()) + .03)
    for ax in axes[0]:
        ax.set_xlim(*bounds)
    sub = stability[stability.scenario.eq("raw26_fraction") & stability.model.isin(MODELS)]
    order = sub.groupby("feature").selection_frequency.mean().nlargest(11).index.tolist()
    if FRACTION not in order:
        order.append(FRACTION)
    order = list(reversed(order))
    for i, model in enumerate(colors):
        rows = sub[sub.model.eq(model)].set_index("feature").reindex(order)
        axes[1, 0].barh(np.arange(len(order)) + (i - .5) * .36,
            rows.selection_frequency, height=.35, color=colors[model])
    axes[1, 0].set_yticks(np.arange(len(order)), [feature_label(f) for f in order])
    axes[1, 0].set_xlim(0, 1.08)
    axes[1, 0].set_xlabel("Fraction of outer training folds selecting the feature")
    axes[1, 0].set_title("Separate properties + ICP: selection stability", fontweight="bold")
    sub = paired[paired.comparison.eq("ICP fraction added to raw descriptors") & paired.model.isin(MODELS)]
    for i, model in enumerate(colors):
        values = sub[sub.model.eq(model)].delta_r2.to_numpy()
        offsets = np.linspace(-.16, .16, len(values))
        axes[1, 1].scatter(np.full(len(values), i) + offsets, values, s=30,
            color=colors[model], alpha=.75)
        axes[1, 1].plot([i - .28, i + .28], [values.mean()] * 2, color="black", lw=2)
    axes[1, 1].axhline(0, color="0.5", linestyle="--", lw=1)
    axes[1, 1].set_xticks([0, 1], list(colors))
    axes[1, 1].set_ylabel(r"Paired outer-fold $\Delta R^2$: with ICP minus without ICP")
    axes[1, 1].set_title("Does measured composition improve prediction?", fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=.2)
    fig.suptitle(f"Nested metal-pair cross-validation | {n_samples} catalysts", fontsize=16)
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"nested_icp_comparison.{extension}", dpi=220)
    plt.close(fig)


def markdown_table(frame):
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in values) + " |")
    return "\n".join(rows)


def write_report(summary, stability, output, args):
    chosen = summary[summary.strategy.eq("nested_selection")][[
        "scenario", "model", "r2_mean", "r2_std", "rmse_mean", "mae_mean", "n_features_mean"]]
    icp = stability[stability.feature.eq(FRACTION)][[
        "model", "selected_count", "n_outer_folds", "selection_frequency", "mean_abs_shap"]]
    comparisons = pd.read_csv(output / "paired_comparison_summary.csv")
    report = f"""# Nested ICP descriptor analysis

All results are for the same 554 initial-screening samples and 389 unordered
metal pairs as the previous fixed-model comparison. BO samples are excluded.
ICP policy, elemental-property definitions, and missing source values are
unchanged. In particular, missing Eu/Nb properties and incomplete adsorption
and PZC reference data still limit physical interpretation.

## Evaluation design

- Outer evaluation: {args.repeats} repetitions of five-fold stratified
  unordered-metal-pair CV, using seeds 42 onward. This reuses the previous
  grouped outer partitions. Both orderings of a metal pair stay together.
- Inner evaluation: {args.inner_folds}-fold stratified metal-pair CV on the
  current outer training set. Binning and splitting use only that set.
- Imputation: training-fold median of the two elemental endpoints, computed
  separately inside every inner split and refitted on each outer training set.
- Selection: sklearn RFE, removing one feature at a time to four remaining
  features, fitted separately on each inner training set. RF and ET use their
  respective selectors ({args.selector_trees} trees, max_depth=8,
  min_samples_leaf=2, max_features=sqrt). The full elimination path supplies
  candidate subsets of 4, 8, 12, 18, and all predictors, where available.
- Hyperparameter search: all 12 combinations of max_depth={{8, None}},
  min_samples_leaf={{1, 3, 6}}, and max_features={{sqrt, 1.0}}. Every feature
  count is crossed with every configuration. The highest mean inner R2 wins;
  only exact ties prefer fewer features and then a fixed configuration order.
- Predictive forests use {args.search_trees} trees during the inner search and
  {args.final_trees} trees for all outer refits. These budgets were fixed in
  advance and are the same for all representations. This is an approximate
  two-stage search, not exhaustive optimization of forest size or all subsets.
- RFE rankings and medians are refitted on the complete outer training set
  after choosing feature count and hyperparameters. Outer test responses
  never select features, parameters, or model family.
- Controls: fixed all-feature forests and tuned all-feature forests are
  evaluated on exactly the same outer test rows. In the augmented model,
  deleting ICP after selection and refitting with the remaining selected
  features gives a conditional ablation without rescreening.
- RF and ET results are reported separately. The additional inner-selected
  RF/ET result chooses the model family from inner CV, never from outer scores.
- SHAP uses outer validation observations. Nonselected predictors contribute
  zero to cross-fold mean SHAP; selection frequency is reported separately.
  ICP permutation importance uses ten shuffles per outer validation fold.
- Fold standard deviations are descriptive, not confidence intervals. CV
  folds share training observations. Paired differences are not independent
  experimental replicates, and no naive t-test is used.

## Nested results

{markdown_table(chosen)}

## ICP fraction selection

{markdown_table(icp)}

Selection frequency measures whether this search retains the measured fraction;
it is not a causal effect or a proof of unique information. Correlated elemental
properties can substitute for one another. Interpretation should consider the
paired performance comparison and the conditional removal check below.

{markdown_table(comparisons)}

## Reproduction and audit

Run from the repository root:

```powershell
& C:\\Users\\yibinjiang\\anaconda3\\python.exe .\\hd_separation_ml\\icp_nested_group_cv.py
```

Completed tasks are checkpointed with a manifest fingerprint. The manifest
includes data and source hashes, versions, search space, and evaluation policy.
`outer_cv_assignments.csv` and `inner_cv_assignments.csv` preserve sample-level
fold assignments. `inner_search_results.csv` contains every candidate's inner
validation score; `inner_rfe_rankings.csv` records training-only feature ranks.
`outer_fold_metrics.csv` and `outer_predictions.csv` provide all validation
results. `feature_selection_stability.csv` includes per-model feature selection
rates and validation SHAP. `pooled_oof_metrics_per_repeat.csv` reports R2 from
each full repetition's out-of-fold predictions, distinct from averaging fold R2.

This comparison addresses selection/tuning robustness within the tested
chemical library. It does not test entirely unseen elements or establish a
surface mechanism. ICP-normalized weighted descriptors remain empirical
bulk-composition proxies.

Method references:
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--selector-trees", type=int, default=128)
    parser.add_argument("--search-trees", type=int, default=160)
    parser.add_argument("--final-trees", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    data, _ = load_data()
    assignments, tasks = [], []
    for repeat in range(args.repeats):
        for fold, (train, test) in enumerate(group_folds(data, 5, 42 + repeat)):
            training = data.iloc[train].reset_index(drop=True)
            for itr, iva in group_folds(training, args.inner_folds, 2000 + repeat * 100 + fold):
                if set(training.iloc[itr]["sample"]) & set(data.iloc[test]["sample"]):
                    raise ValueError("Outer validation sample used in inner training.")
                if set(training.iloc[iva]["sample"]) & set(data.iloc[test]["sample"]):
                    raise ValueError("Outer validation sample used in inner validation.")
            assignments.extend(dict(repeat=repeat, fold=fold, sample=data.iloc[j]["sample"],
                metal_pair=data.iloc[j]["metal_pair"]) for j in test)
            for scenario in SCENARIOS:
                for model_name in MODELS:
                    tasks.append((train, test, repeat, fold, scenario, model_name))
    print(f"Audited {len(data)} rows, {data.metal_pair.nunique()} pairs, "
        f"{args.repeats * 5} outer folds, {len(tasks)} model searches.", flush=True)
    if args.audit_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    sources = [WORKBOOK, COMPOSITION_TABLE, FEATURE_MATRIX, Path(__file__),
        ROOT / "icp_intrinsic_descriptor_analysis.py", ROOT / "icp_composition_weighted_sensitivity.py"]
    manifest = dict(n_samples=len(data), n_pairs=data.metal_pair.nunique(),
        outer_repeats=args.repeats, outer_folds=5, inner_folds=args.inner_folds,
        outer_seeds=list(range(42, 42 + args.repeats)), scenarios=SCENARIOS,
        models=list(MODELS), selector_trees=args.selector_trees,
        search_trees=args.search_trees, final_trees=args.final_trees,
        selection="Fold-local sklearn RFE, step=1, minimum=4; feature count selected by inner CV",
        counts={s: feature_counts(13 if s == "icp13" else 26 + int(s == "raw26_fraction")) for s in SCENARIOS},
        predictor_configs=CONFIGS, selector_config=dict(max_depth=8, min_samples_leaf=2, max_features="sqrt"),
        scorer="Mean inner fold R2; exact ties prefer fewer features", versions=dict(sklearn=sklearn.__version__,
        numpy=np.__version__, pandas=pd.__version__, shap=shap.__version__),
        input_hashes={str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = args.output / "analysis_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Existing results have a different manifest; choose a separate output directory.")
    manifest_path.write_text(json.dumps(dict(fingerprint=fingerprint, **manifest), indent=2), encoding="utf-8")
    data.to_csv(args.output / "analysis_dataset_unimputed.csv", index=False)
    pd.DataFrame(assignments).to_csv(args.output / "outer_cv_assignments.csv", index=False)
    payloads = []
    start = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs, return_as="generator_unordered", pre_dispatch=args.jobs)(
            delayed(run_task)(data, *task, args, fingerprint) for task in tasks)
        for task_id, payload, cached in results:
            payloads.append(payload)
            selected = next(row for row in payload["metrics"] if row["strategy"] == "nested_selection")
            print(f"[{len(payloads)}/{len(tasks)}] {task_id}: k={selected['n_features']}, "
                f"inner R2={selected['inner_r2_mean']:.3f}, outer R2={selected['r2']:.3f}, "
                f"ICP={payload['selected_icp']}, cached={cached}, "
                f"elapsed={(time.perf_counter()-start)/60:.1f} min", flush=True)
    summary, stability, paired = aggregate(payloads, data, args.output)
    plot_results(summary, stability, paired, args.output, len(data))
    write_report(summary, stability, args.output, args)
    print(summary.to_string(index=False), flush=True)
    print(f"Completed; outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
