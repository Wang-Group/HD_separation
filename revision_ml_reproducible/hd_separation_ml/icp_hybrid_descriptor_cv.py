"""Nested grouped CV for separate elemental plus explicit ICP-weighted features."""

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

from icp_intrinsic_descriptor_analysis import BASES, ROOT, DescriptorRepresentation, load_data, scores
from icp_nested_group_cv import (
    OUTPUT as SOURCE, CONFIGS, FRACTION, choose_best, feature_label, group_folds,
    interpret_outer, make_model, markdown_table, rfe_order,
)
from icp_parsimony_sensitivity import OUTPUT as PARSIMONY, select_candidate

OUTPUT = ROOT / "outputs" / "icp_hybrid_descriptor_cv_20260928"
WEIGHTED = "icp_weighted_"
NEW_SCENARIOS = {"hybrid39": "Separate 26 + weighted 13", "hybrid40": "Separate 26 + weighted 13 + fraction"}
SCENARIOS = {"icp13": "ICP-weighted 13", "raw26": "Separate elemental 26",
             "raw26_fraction": "Separate 26 + ICP fraction", **NEW_SCENARIOS}
STRATEGIES = ["fixed_all_features", "tuned_all_features", "nested_selection", "one_se"]
FIXED = dict(max_depth=None, min_samples_leaf=1, max_features="sqrt")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HybridRepresentation:
    def __init__(self, scenario):
        if scenario not in NEW_SCENARIOS:
            raise ValueError(scenario)
        self.scenario = scenario

    def fit(self, data):
        self.endpoints_ = DescriptorRepresentation("raw26").fit(data)
        return self

    def transform(self, data):
        out = self.endpoints_.transform(data)
        f = data[FRACTION].to_numpy(float)
        if not np.isfinite(f).all() or ((f < 0) | (f > 1)).any():
            raise ValueError("Invalid ICP fraction.")
        for base in BASES:
            out[WEIGHTED + base] = f * out[f"{base}_element1"] + (1 - f) * out[f"{base}_element2"]
        if self.scenario == "hybrid40":
            out[FRACTION] = f
        return out


def candidate_counts(p):
    return sorted({k for k in (4, 8, 12, 18, 26, p) if k <= p})


def search(training, name, model_name, seed):
    p = 39 if name == "hybrid39" else 40
    y = training.separation_coefficient.to_numpy(float)
    rows, ranks = [], []
    for inner_fold, (tr, va) in enumerate(group_folds(training, 3, seed)):
        transformer = HybridRepresentation(name).fit(training.iloc[tr])
        frame = transformer.transform(training.iloc[tr])
        X, V = frame.to_numpy(), transformer.transform(training.iloc[va]).to_numpy()
        order, ranking = rfe_order(X, y[tr], model_name, 128, seed + inner_fold)
        ranks += [dict(inner_fold=inner_fold, feature=c, rfe_rank=int(r)) for c, r in zip(frame.columns, ranking)]
        for k in candidate_counts(p):
            mask = np.sort(order[:k])
            for config_id, config in enumerate(CONFIGS):
                model = make_model(model_name, config, 160, seed + inner_fold).fit(X[:, mask], y[tr])
                rows.append(dict(inner_fold=inner_fold, n_features=k, config_id=config_id,
                                 **scores(y[va], model.predict(V[:, mask]))))
    rows = pd.DataFrame(rows)
    candidates = [dict(n_features=int(k), config_id=int(c), config=CONFIGS[int(c)],
        inner_r2_mean=float(values.r2.mean()), inner_r2_std=float(values.r2.std()),
        inner_rmse_mean=float(values.rmse.mean()))
        for (k, c), values in rows.groupby(["n_features", "config_id"])]
    return candidates, ranks


def run_task(data, train, test, repeat, fold, name, model_name, output, fingerprint):
    checkpoint = output / "checkpoints" / f"repeat{repeat}_fold{fold}_{name}_{model_name}.json"
    if checkpoint.exists():
        result = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert result["fingerprint"] == fingerprint
        return result, True
    start = time.perf_counter()
    training, testing = data.iloc[train].reset_index(drop=True), data.iloc[test].reset_index(drop=True)
    assert not set(training.metal_pair) & set(testing.metal_pair)
    seed = 2000 + repeat * 100 + fold
    candidates, inner_ranks = search(training, name, model_name, seed)
    transform = HybridRepresentation(name).fit(training)
    tr = transform.transform(training)
    X, V = tr.to_numpy(), transform.transform(testing).to_numpy()
    columns = list(tr.columns)
    y, yt = training.separation_coefficient.to_numpy(), testing.separation_coefficient.to_numpy()
    order, ranks = rfe_order(X, y, model_name, 128, seed)
    best = choose_best(candidates)
    simple, threshold = select_candidate(candidates, "one_se", 3)
    all_best = choose_best([c for c in candidates if c["n_features"] == X.shape[1]])
    choices = {"fixed_all_features": dict(config=FIXED, n_features=X.shape[1], inner_r2_mean=None,
                                          config_id=-1),
               "tuned_all_features": all_best, "nested_selection": best, "one_se": simple}
    metrics, predictions, importance, ablations = [], [], [], []
    cache = {}
    for strategy, choice in choices.items():
        mask = np.sort(order[:choice["n_features"]])
        selected = [columns[j] for j in mask]
        config_key = (len(mask), json.dumps(choice["config"], sort_keys=True))
        if config_key not in cache:
            model = make_model(model_name, choice["config"], 600, seed).fit(X[:, mask], y)
            cache[config_key] = model, model.predict(V[:, mask])
        model, pred = cache[config_key]
        n_weighted = sum(c.startswith(WEIGHTED) for c in selected)
        metrics.append(dict(strategy=strategy, n_features=len(mask), selected_features=selected,
            n_weighted=n_weighted, n_endpoint=len(mask) - n_weighted - int(FRACTION in selected),
            fraction_selected=FRACTION in selected, n_train=len(training), n_test=len(testing),
            config=choice["config"], config_id=choice["config_id"], inner_r2_mean=choice["inner_r2_mean"],
            one_se_threshold=threshold if strategy == "one_se" else None, **scores(yt, pred)))
        predictions += [dict(strategy=strategy, sample=s, y_true=float(obs), y_pred=float(p))
                        for s, obs, p in zip(testing["sample"], yt, pred)]
        if strategy in ("one_se", "nested_selection"):
            shap_rows, _ = interpret_outer(model, V[:, mask], yt, columns, selected, seed)
            importance += [dict(strategy=strategy, **row) for row in shap_rows]
            reduced_mask = [j for j in mask if not columns[j].startswith(WEIGHTED)]
            if not n_weighted:
                reduced_pred = pred
                status = "no_weighted_feature_selected"
            elif reduced_mask:
                reduced = make_model(model_name, choice["config"], 600, seed).fit(X[:, reduced_mask], y)
                reduced_pred = reduced.predict(V[:, reduced_mask])
                status = "refit_remaining_selected_features_without_rescreening"
            else:
                reduced_pred = np.full(len(yt), y.mean())
                status = "all_selected_features_weighted_training_mean_fallback"
            reduced_score = scores(yt, reduced_pred)
            ablations.append(dict(strategy=strategy, status=status, n_weighted_removed=n_weighted,
                original_r2=scores(yt, pred)["r2"], reduced_r2=reduced_score["r2"],
                delta_r2=scores(yt, pred)["r2"] - reduced_score["r2"],
                original_rmse=scores(yt, pred)["rmse"], reduced_rmse=reduced_score["rmse"]))
    result = dict(fingerprint=fingerprint, repeat=repeat, fold=fold, scenario=name, model=model_name,
        metrics=metrics, predictions=predictions, importance=importance, ablations=ablations,
        candidates=candidates, inner_ranks=inner_ranks,
        outer_ranks=dict(zip(columns, map(int, ranks))), seconds=time.perf_counter() - start)
    checkpoint.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, False


def cached_controls():
    metrics = pd.read_csv(SOURCE / "outer_fold_metrics.csv")
    metrics = metrics[metrics.model.isin(["RF", "ET"]) & metrics.strategy.isin(STRATEGIES)].copy()
    predictions = pd.read_csv(SOURCE / "outer_predictions.csv")
    predictions = predictions[predictions.model.isin(["RF", "ET"]) & predictions.strategy.isin(STRATEGIES)].copy()
    simple = pd.read_csv(PARSIMONY / "outer_fold_metrics.csv")
    simple = simple[simple.rule.eq("one_se")].copy()
    simple["strategy"] = "one_se"
    simple_preds = pd.read_csv(PARSIMONY / "outer_predictions.csv")
    simple_preds = simple_preds[simple_preds.rule.eq("one_se")].copy()
    simple_preds["strategy"] = "one_se"
    columns = ["repeat", "fold", "scenario", "model", "strategy", "n_features", "r2", "rmse", "mae"]
    pred_cols = ["repeat", "fold", "scenario", "model", "strategy", "sample", "y_true", "y_pred"]
    return pd.concat([metrics[columns], simple[columns]], ignore_index=True), pd.concat(
        [predictions[pred_cols], simple_preds[pred_cols]], ignore_index=True)


def summarize(payloads, output):
    key_cols = ["repeat", "fold", "scenario", "model"]
    frames = {}
    for category in ("metrics", "predictions", "importance", "ablations", "candidates", "inner_ranks"):
        frames[category] = pd.DataFrame([{**{k: p[k] for k in key_cols}, **r}
                                         for p in payloads for r in p[category]])
    old_metrics, old_predictions = cached_controls()
    metrics = pd.concat([old_metrics, frames["metrics"]], ignore_index=True)
    predictions = pd.concat([old_predictions, frames["predictions"]], ignore_index=True)
    keys = ["scenario", "model", "strategy"]
    assert len(metrics) == 600
    assert len(predictions) == 66480
    assert not predictions.duplicated(keys + ["repeat", "sample"]).any()
    assert predictions.groupby(keys + ["repeat"])["sample"].nunique().eq(554).all()
    metrics.to_csv(output / "outer_fold_metrics.csv", index=False)
    predictions.to_csv(output / "outer_predictions.csv", index=False)
    for name in ("importance", "ablations", "candidates", "inner_ranks"):
        frames[name].to_csv(output / ("hybrid_" + name + ".csv"), index=False)
    summary = metrics.groupby(keys, as_index=False).agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), n_features_mean=("n_features", "mean"),
        n_features_min=("n_features", "min"), n_features_max=("n_features", "max"), n_folds=("r2", "size"))
    summary.to_csv(output / "model_summary.csv", index=False)
    blocks = frames["metrics"].groupby(keys, as_index=False).agg(n_endpoint_mean=("n_endpoint", "mean"),
        n_weighted_mean=("n_weighted", "mean"), fraction_selected_count=("fraction_selected", "sum"))
    blocks.to_csv(output / "selected_feature_blocks.csv", index=False)
    imp = frames["importance"]
    stability = imp.groupby(keys + ["feature"], as_index=False).agg(selection_frequency=("selected", "mean"),
        selected_count=("selected", "sum"), mean_abs_shap=("mean_abs_shap", "mean"))
    stability.to_csv(output / "hybrid_selection_stability.csv", index=False)
    paired = []
    for added, base in [("hybrid39", "raw26"), ("hybrid39", "raw26_fraction"),
                        ("hybrid40", "raw26_fraction"), ("hybrid40", "hybrid39"), ("hybrid39", "icp13")]:
        a, b = metrics[metrics.scenario.eq(added)], metrics[metrics.scenario.eq(base)]
        join = ["repeat", "fold", "model", "strategy"]
        rows = a.merge(b, on=join, validate="one_to_one", suffixes=("_added", "_base"))
        paired += [{**{k: r[k] for k in join}, "comparison": f"{added} - {base}",
                    "delta_r2": r.r2_added - r.r2_base, "delta_rmse": r.rmse_added - r.rmse_base}
                   for _, r in rows.iterrows()]
    paired = pd.DataFrame(paired)
    paired.to_csv(output / "paired_outer_folds.csv", index=False)
    comparisons = paired.groupby(["comparison", "model", "strategy"], as_index=False).agg(
        delta_r2_mean=("delta_r2", "mean"), delta_r2_std=("delta_r2", "std"),
        positive_folds=("delta_r2", lambda s: int((s > 0).sum())), delta_rmse_mean=("delta_rmse", "mean"))
    comparisons.to_csv(output / "paired_comparison_summary.csv", index=False)
    ablations = frames["ablations"].groupby(keys, as_index=False).agg(
        mean_delta_r2=("delta_r2", "mean"), std_delta_r2=("delta_r2", "std"),
        positive_folds=("delta_r2", lambda s: int((s > 0).sum())))
    ablations.to_csv(output / "weighted_block_removal_summary.csv", index=False)
    pooled = [dict(zip(keys + ["repeat"], values), **scores(rows.y_true, rows.y_pred))
              for values, rows in predictions.groupby(keys + ["repeat"])]
    pd.DataFrame(pooled).to_csv(output / "pooled_oof_metrics.csv", index=False)
    return summary, stability, comparisons, blocks


def label(feature):
    return "Weighted " + feature_label(feature.removeprefix(WEIGHTED)) if feature.startswith(WEIGHTED) else feature_label(feature)


def plot(summary, stability, output):
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True, sharex=True, layout="constrained")
    for ax, strategy, title in zip(axes, ("tuned_all_features", "one_se"),
                                  ("All features, inner tuning", "Nested RFE + one-SE selection")):
        for offset, model, color in [(-.12, "RF", "#187e92"), (.12, "ET", "#d47c27")]:
            rows = summary[(summary.model == model) & (summary.strategy == strategy)].set_index("scenario").loc[list(SCENARIOS)]
            ax.errorbar(rows.r2_mean, np.arange(len(rows)) + offset, xerr=rows.r2_std,
                        fmt="o", capsize=3, color=color, label=model)
        ax.set_yticks(np.arange(len(SCENARIOS)), list(SCENARIOS.values()))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(r"Mean grouped outer-validation $R^2$ (fold SD)")
        ax.set_xlim(.29, .66)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Separate elemental + ICP-weighted descriptors | same 554 catalysts", fontsize=15)
    for ext in ("png", "pdf"):
        fig.savefig(output / ("hybrid_descriptor_comparison." + ext), dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.5), layout="constrained")
    for ax, scenario_name in zip(axes, NEW_SCENARIOS):
        sub = stability[(stability.scenario == scenario_name) & (stability.strategy == "one_se")]
        order = sub.groupby("feature").mean_abs_shap.mean().nlargest(12).index[::-1]
        for offset, model, color in [(-.18, "RF", "#187e92"), (.18, "ET", "#d47c27")]:
            rows = sub[sub.model == model].set_index("feature").loc[order]
            ax.barh(np.arange(len(order)) + offset, rows.mean_abs_shap, height=.35, color=color, label=model)
        ax.set_yticks(np.arange(len(order)), [label(f) for f in order], fontsize=9)
        ax.set_xlabel("Mean absolute validation SHAP (nonselected = 0)")
        ax.set_title(NEW_SCENARIOS[scenario_name], fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Sparse hybrid models: predictive attribution, not independent physical effects", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output / ("hybrid_sparse_attribution." + ext), dpi=220)
    plt.close(fig)


def write_report(summary, comparisons, blocks, output):
    report = """# Separate elemental plus ICP-weighted descriptor comparison

## Information sets

- raw26: M1 and M2 properties kept separately (13 each).
- raw26_fraction: the 26 endpoint properties plus measured M1 mole fraction.
- icp13: each descriptor explicitly averaged using measured ICP mole fractions.
- hybrid39: all 26 endpoint properties plus all 13 explicit weighted means.
- hybrid40: hybrid39 plus the measured M1 fraction as an explicit column.

The main question is whether explicit means give RF/ET a more useful input
representation than endpoints plus a separate fraction. Weighted means are
deterministic transformations of endpoints and fractions, not additional
independent physical measurements. If the endpoints differ, a weighted mean
can also encode the fraction. Thus hybrid39-vs-raw26 adds composition information
AND an engineered representation; hybrid40-vs-raw26_fraction more directly
checks explicit feature engineering with the same underlying information.

All properties are intrinsic reference values. NO LSV current, Rs, eta10,
sample potential, or potential correction is included. ICP values describe
bulk metal composition, not measured active-site or surface properties.

## Evaluation

Same 554 initial-screening samples and 389 unordered metal pairs as the earlier
analysis; no BO. Exact saved 3 x 5 grouped outer and 3 grouped inner partitions.
Fold-local pooled endpoint median imputation is performed before weighting;
the endpoints and derived weighted means share the SAME imputed values.
Eu/Nb and incomplete adsorption/PZC references remain a limitation. This is
the 554-row ML policy, not the 496-row reference-complete upper-envelope plot.

RFE is refitted on each inner training set and again on the outer training set.
Selectors: model-matched RF/ET, 128 trees, max depth 8, minimum leaf 2, sqrt
features, step 1. Hybrid candidate counts are 4/8/12/18/26/all (39 or 40).
For a matched link to previous results, baseline counts retain their original
4/8/12/18/all grid. No exhaustive subset search is claimed; 6 and 10 are not
candidates. This controls runtimes but limits conclusions about the optimal
feature count. Six counts x the same 12 forest configurations are assessed
with 160 trees per inner model and 600 trees per outer refit.

Four strategies are retained, not chosen by their outer performance:
1. All features, fixed settings (unlimited depth, leaf 1, sqrt features).
2. All features, inner-CV tuning.
3. Maximum inner mean R2 selects feature count and settings.
4. One-SE: smallest count within best inner mean minus its fold SD/sqrt(3),
   then best-scoring settings at that count. This is a parsimony heuristic,
   not a statistical equivalence test for dependent folds.

Historical raw26/raw26_fraction/icp13 metrics and predictions are reused after
source-hash and split verification. Their one-SE controls are reused from the
previous parsimony analysis. RF/ET results are shown separately; no outer-score
selection of model family. Sparse validation SHAP and feature selection rates
describe model predictions, not causality or independent physical relevance.
Correlated raw and weighted properties can substitute for each other.

After sparse selection, a diagnostic refits the same settings without selected
weighted columns. It does not rescreen features. If all selected columns are
weighted, its fallback is the training-response mean. This is a conditional
ablation, not proof that weighting adds fundamentally new information.
Independent permutation of deterministic raw/weighted columns is not used
as the main importance claim because it breaks their algebraic relationship.

The idea was motivated by earlier results on these same data/outer partitions.
Despite fold-local fitting, this is a retrospective sensitivity comparison,
not an independent validation of a representation chosen after seeing these
results. Fold SD is descriptive, not a confidence interval; no naive t-tests
on repeated folds are performed. Dataset composition and source-property gaps
limit generalization to new elements, surfaces and measurement conditions.

## Results

"""
    for strategy in STRATEGIES:
        report += "\n### " + strategy + "\n\n" + markdown_table(summary[summary.strategy.eq(strategy)][
            ["scenario", "model", "r2_mean", "r2_std", "rmse_mean", "n_features_mean"]]) + "\n"
    report += "\n## Selected feature blocks (one-SE)\n\n" + markdown_table(blocks[blocks.strategy.eq("one_se")])
    report += "\n\n## Paired outer differences (one-SE)\n\n" + markdown_table(comparisons[comparisons.strategy.eq("one_se")])
    report += """

## Reproduction

```powershell
& C:\\Users\\yibinjiang\\anaconda3\\python.exe .\\hd_separation_ml\\icp_hybrid_descriptor_cv.py --jobs 12
```

Source hashes, versions, candidate grids and all policies are recorded in
`analysis_manifest.json`. Checkpoints validate the fingerprint before reuse.
The saved inner and outer assignments allow group-leakage checks. Predictions,
inner search scores, RFE ranks, selection rates, SHAP, paired differences and
weighted-block deletion diagnostics are retained as CSV files. All weighted
columns have an explicit `icp_weighted_` prefix in selection and importance tables.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    source_manifest = json.loads((SOURCE / "analysis_manifest.json").read_text(encoding="utf-8"))
    for path, expected in source_manifest["input_hashes"].items():
        assert sha(ROOT.parent / path) == expected, f"Source changed: {path}"
    data, _ = load_data()
    assignments = pd.read_csv(SOURCE / "outer_cv_assignments.csv")
    source_inner = pd.read_csv(SOURCE / "inner_cv_assignments.csv")
    tasks, inner_rows = [], []
    for repeat in range(3):
        for fold, (tr, te) in enumerate(group_folds(data, 5, 42 + repeat)):
            expected = assignments.loc[assignments["repeat"].eq(repeat) & assignments.fold.eq(fold), "sample"]
            assert set(data.iloc[te]["sample"]) == set(expected)
            training = data.iloc[tr].reset_index(drop=True)
            for inner_fold, (itr, iva) in enumerate(group_folds(training, 3, 2000 + repeat * 100 + fold)):
                assert not set(training.iloc[itr]["sample"]) & set(data.iloc[te]["sample"])
                inner_rows += [dict(repeat=repeat, fold=fold, inner_fold=inner_fold,
                    sample=training.iloc[j]["sample"], metal_pair=training.iloc[j].metal_pair) for j in iva]
            tasks += [(tr, te, repeat, fold, name, model) for name in NEW_SCENARIOS for model in ("RF", "ET")]
    inner = pd.DataFrame(inner_rows)
    cols = list(inner.columns)
    pd.testing.assert_frame_equal(inner.sort_values(cols).reset_index(drop=True),
        source_inner[cols].drop_duplicates().sort_values(cols).reset_index(drop=True))
    hashes = dict(source_manifest["input_hashes"])
    sources = [Path(__file__), ROOT / "icp_parsimony_sensitivity.py", SOURCE / "analysis_manifest.json",
        SOURCE / "outer_cv_assignments.csv", SOURCE / "inner_cv_assignments.csv",
        SOURCE / "outer_fold_metrics.csv", SOURCE / "outer_predictions.csv",
        PARSIMONY / "outer_fold_metrics.csv", PARSIMONY / "outer_predictions.csv"]
    hashes.update({str(p.relative_to(ROOT.parent)): sha(p) for p in sources})
    policy = dict(scenarios=SCENARIOS, n_samples=len(data), n_pairs=data.metal_pair.nunique(),
        n_tasks=len(tasks), strategies=STRATEGIES, configs=CONFIGS, source_hashes=hashes,
        counts={s: candidate_counts(39 if s == "hybrid39" else 40) for s in NEW_SCENARIOS},
        selector_trees=128, search_trees=160, final_trees=600, versions=source_manifest["versions"],
        weighting="ICP fraction applied after shared fold-local endpoint imputation; no electrochemical inputs",
        source_fingerprint=source_manifest["fingerprint"],
        caveat="Retrospective representation comparison using previous outer partitions; not independent confirmation")
    fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    policy.update(fingerprint=fingerprint, jobs=args.jobs)
    manifest_path = output / "analysis_manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] == fingerprint
    manifest_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    assignments.to_csv(output / "outer_cv_assignments.csv", index=False)
    inner.to_csv(output / "inner_cv_assignments.csv", index=False)
    data.to_csv(output / "analysis_dataset_unimputed.csv", index=False)
    print(f"Audited {len(data)} samples; {len(tasks)} hybrid tasks; {args.jobs} workers.", flush=True)
    start, payloads = time.perf_counter(), []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(run_task)(data, *task, output, fingerprint) for task in tasks)
        for payload, reused in results:
            payloads.append(payload)
            print(f"Completed {len(payloads)}/{len(tasks)}: {payload['scenario']} {payload['model']} "
                  f"repeat{payload['repeat']} fold{payload['fold']}; {time.perf_counter() - start:.1f}s", flush=True)
    summary, stability, comparisons, blocks = summarize(payloads, output)
    plot(summary, stability, output)
    write_report(summary, comparisons, blocks, output)
    print(summary[summary.strategy.isin(["tuned_all_features", "one_se"])].to_string(index=False), flush=True)
    print(f"Finished in {time.perf_counter() - start:.1f}s. Output: {output}", flush=True)


if __name__ == "__main__":
    main()
