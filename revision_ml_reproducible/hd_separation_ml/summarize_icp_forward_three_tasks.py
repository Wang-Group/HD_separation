"""Report repeated-CV performance and selection frequencies for three tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icp_bo_augmentation_cv import representation
from icp_forward_three_tasks import PROTOCOLS, TABLES
from icp_intrinsic_descriptor_analysis import BASES, FAMILIES, LABELS, scores
from icp_revision_inputs import sha

ROOT = Path(__file__).resolve().parent
STRATEGIES = ["all39", "forward10", "forward_best", "forward_one_se"]
NAMES = dict(stratified_random="Stratified random entities", unseen_pair="Unseen metal pairs",
             known_pair_new_composition="Known pairs: new compositions")
METHODS = dict(all39="All 39", forward10="Forward 10", forward_best="Inner-best subset", forward_one_se="One-SE subset")
SHORT_TASKS = ["Random\nentities", "Unseen\npairs", "Known-pair\ncompositions"]


def load_results(output):
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    for file, expected in {**manifest["source_hashes"], **manifest["code_hashes"]}.items():
        if sha(Path(file)) != expected:
            raise ValueError(f"Changed analysis dependency: {file}")
    provenance = pd.read_csv(output / "checkpoint_provenance.csv")
    if len(provenance) != 45 or provenance.duplicated(["protocol", "repeat", "fold"]).any():
        raise ValueError("Expected 45 distinct fold checkpoints")
    records = []
    for row in provenance.itertuples(index=False):
        file = Path(row.checkpoint)
        if sha(file) != row.sha256:
            raise ValueError(f"Changed checkpoint: {file}")
        record = json.loads(file.read_text(encoding="utf-8"))
        if record["fingerprint"] != row.fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        records.append(record)
    tables = {category:pd.DataFrame([dict(**{k:r[k] for k in ("protocol", "repeat", "fold")}, **entry)
              for r in records for entry in r[category]]) for category in TABLES}
    source = Path(manifest["source"])
    data = pd.read_csv(source / "combined_dataset.csv").set_index("sample", drop=False)
    assignments = pd.read_csv(source / "outer_assignments.csv")
    predictions, settings = tables["predictions"], tables["settings"]
    np.testing.assert_allclose(predictions.y_true, data.loc[predictions["sample"], "separation_coefficient"], atol=1e-12, rtol=0)
    for (protocol, repeat, fold), rows in predictions.groupby(["protocol", "repeat", "fold"]):
        members = assignments[(assignments.protocol==protocol) & (assignments.repeat==repeat) &
                              (assignments.fold==fold) & assignments.role.eq("testing")]["sample"]
        if protocol != "known_pair_new_composition":
            members = members[data.loc[members, "cohort"].to_numpy()=="initial"]
        if set(rows.strategy) != set(STRATEGIES):
            raise ValueError("Missing prediction strategy")
        for _, part in rows.groupby("strategy"):
            if part["sample"].duplicated().any() or set(part["sample"]) != set(members):
                raise ValueError("Test membership changed")
    if len(settings) != 45*len(STRATEGIES) or settings.duplicated(["protocol", "repeat", "fold", "strategy"]).any():
        raise ValueError("Missing or duplicated selected models")
    if not settings.groupby(["protocol", "strategy"]).size().eq(15).all():
        raise ValueError("Every strategy needs 15 fits per task")
    for row in settings.itertuples(index=False):
        if len(row.features) != row.n_features or len(set(row.features)) != row.n_features:
            raise ValueError("Incorrect feature count")
    return data, tables, manifest


def performance_tables(predictions, settings, data):
    repeats, folds = [], []
    for (protocol, repeat, strategy), part in predictions.groupby(["protocol", "repeat", "strategy"]):
        cohort = "BO" if protocol=="known_pair_new_composition" else "initial"
        if part["sample"].duplicated().any() or set(part["sample"]) != set(data.loc[data.cohort.eq(cohort), "sample"]):
            raise ValueError("Incomplete repeat test population")
        repeats.append(dict(protocol=protocol, repeat=repeat, strategy=strategy, n=len(part), **scores(part.y_true, part.y_pred)))
    for (protocol, repeat, fold, strategy), part in predictions.groupby(["protocol", "repeat", "fold", "strategy"]):
        folds.append(dict(protocol=protocol, repeat=repeat, fold=fold, strategy=strategy,
                          n=len(part), **scores(part.y_true, part.y_pred)))
    repeats = pd.DataFrame(repeats)
    baseline = repeats[repeats.strategy.eq("all39")][["protocol", "repeat", "r2", "rmse", "mae"]]
    baseline = baseline.rename(columns={c:"all39_"+c for c in ["r2", "rmse", "mae"]})
    repeats = repeats.merge(baseline, on=["protocol", "repeat"], validate="many_to_one")
    for metric in ["r2", "rmse", "mae"]:
        repeats["delta_"+metric] = repeats[metric] - repeats["all39_"+metric]
    summary = repeats.groupby(["protocol", "strategy"], as_index=False).agg(
        n_test=("n", "first"), n_repeats=("repeat", "nunique"), r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        delta_r2_mean=("delta_r2", "mean"), delta_r2_sd=("delta_r2", "std"),
        delta_rmse_mean=("delta_rmse", "mean"), delta_rmse_sd=("delta_rmse", "std"))
    counts = settings.groupby(["protocol", "strategy"], as_index=False).agg(n_fits=("fold", "size"),
        features_mean=("n_features", "mean"), features_sd=("n_features", "std"), features_median=("n_features", "median"),
        features_min=("n_features", "min"), features_max=("n_features", "max"))
    return summary.merge(counts, on=["protocol", "strategy"], validate="one_to_one"), repeats, pd.DataFrame(folds)


def count_selection(rows, field):
    keys = ["protocol", "strategy", field]
    summary = rows.groupby(keys, as_index=False).agg(selected_fits=("selected", "sum"), total_fits=("selected", "size"),
        selection_frequency=("selected", "mean"), weighted_selected_fits=("weighted", "sum"),
        weighted_selection_frequency=("weighted", "mean"))
    repeat = rows.groupby(keys+["repeat"], as_index=False).agg(selected_fits=("selected", "sum"), total_fits=("selected", "size"),
        selection_frequency=("selected", "mean"))
    spread = repeat.groupby(keys, as_index=False).agg(repeat_frequency_sd=("selection_frequency", "std"))
    overall = rows.groupby(["strategy", field], as_index=False).agg(selected_fits=("selected", "sum"),
        total_fits=("selected", "size"), selection_frequency=("selected", "mean"))
    return summary.merge(spread, on=keys, validate="one_to_one"), repeat, overall


def selection_tables(settings, columns):
    individual, properties, bundles = [], [], []
    for row in settings.to_dict("records"):
        key = {k:row[k] for k in ("protocol", "repeat", "fold", "strategy")}
        features = set(row["features"])
        for feature in columns:
            selected = feature in features
            individual.append(dict(**key, feature=feature, selected=selected,
                weighted=selected and feature.startswith("icp_weighted_"),
                forward_rank=(row["features"].index(feature)+1) if selected and row["strategy"]!="all39" else np.nan))
        for base in BASES:
            versions = {base+"_element1", base+"_element2", "icp_weighted_"+base}
            properties.append(dict(**key, property=base, selected=bool(versions & features),
                                   weighted="icp_weighted_"+base in features))
        for name, bases in FAMILIES.items():
            versions = {c for base in bases for c in (base+"_element1", base+"_element2", "icp_weighted_"+base)}
            bundles.append(dict(**key, family=name, selected=bool(versions & features),
                                weighted=any("icp_weighted_"+base in features for base in bases)))
    individual = pd.DataFrame(individual)
    feature, feature_repeat, overall = count_selection(individual, "feature")
    ranks = individual.groupby(["protocol", "strategy", "feature"], as_index=False).forward_rank.mean()
    feature = feature.merge(ranks, on=["protocol", "strategy", "feature"], validate="one_to_one")
    prop, prop_repeat, _ = count_selection(pd.DataFrame(properties), "property")
    family, family_repeat, _ = count_selection(pd.DataFrame(bundles), "family")
    return dict(feature_selection_frequency=feature, feature_frequency_by_repeat=feature_repeat,
        feature_frequency_all_tasks=overall, property_selection_frequency=prop,
        property_frequency_by_repeat=prop_repeat, descriptor_family_selection_frequency=family,
        descriptor_family_frequency_by_repeat=family_repeat, selection_indicator=individual)


def importance_table(permutation, data):
    repeat = permutation.groupby(["protocol", "repeat", "strategy", "property", "permutation"], as_index=False).agg(
        sse_increase=("sse_increase", "sum"), n=("n", "sum"))
    sst = {}
    for protocol in PROTOCOLS:
        y = data.loc[data.cohort.eq("BO" if protocol=="known_pair_new_composition" else "initial"), "separation_coefficient"]
        sst[protocol] = float(np.square(y-y.mean()).sum())
    repeat["r2_drop"] = repeat.sse_increase / repeat.protocol.map(sst)
    repeat = repeat.groupby(["protocol", "repeat", "strategy", "property"], as_index=False).r2_drop.mean()
    return repeat.groupby(["protocol", "strategy", "property"], as_index=False).agg(
        r2_drop_mean=("r2_drop", "mean"), r2_drop_repeat_sd=("r2_drop", "std"))


def save_figure(fig, output, stem):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def draw(summary, frequencies, path, output):
    plt.rcParams.update({"font.family":"Arial", "font.size":10, "pdf.fonttype":42,
                        "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(15,4.9), layout="constrained")
    colors = ["#777777", "#087F80", "#526CA7", "#AC7834"]
    limits = (min(0, float((summary.r2_mean-summary.r2_sd).min())-.04),
              max(.75, float((summary.r2_mean+summary.r2_sd).max())+.07))
    for ax, protocol in zip(axes, PROTOCOLS):
        rows = summary[summary.protocol.eq(protocol)].set_index("strategy").loc[STRATEGIES]
        x = np.arange(len(rows))
        ax.bar(x, rows.r2_mean, color=colors, width=.65)
        ax.errorbar(x, rows.r2_mean, yerr=rows.r2_sd, fmt="none", color="#222222", capsize=3)
        for i, row in enumerate(rows.itertuples()):
            ax.text(i, row.r2_mean+row.r2_sd+.014, f"{row.r2_mean:.3f}", ha="center", fontsize=10)
            count = str(int(row.features_min)) if row.features_min==row.features_max else f"{int(row.features_min)}-{int(row.features_max)}"
            ax.text(i,.025,count+"\nfeatures",ha="center",va="bottom",color="white",fontsize=9)
        ax.set_xticks(x, [METHODS[s] for s in STRATEGIES], rotation=18)
        ax.set(title=NAMES[protocol], ylabel="Pooled out-of-fold R-squared", ylim=limits)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.supxlabel("Mean and SD across three repeats of five-fold CV; selection and tuning use inner training folds.",fontsize=10)
    save_figure(fig,output,"three_task_performance")
    for strategy in ["forward10", "forward_one_se", "forward_best"]:
        fig, axes = plt.subplots(1,3,figsize=(13.8,7.2),layout="constrained",sharey=True)
        for ax, version, title in zip(axes, ["element1","element2","weighted"],["M1 properties","M2 properties","ICP-weighted properties"]):
            features = [("icp_weighted_"+b if version=="weighted" else b+"_"+version) for b in BASES]
            matrix = frequencies[frequencies.strategy.eq(strategy)].pivot(index="feature",columns="protocol",values="selected_fits").loc[features,PROTOCOLS].to_numpy()
            heat = ax.imshow(matrix,aspect="auto",vmin=0,vmax=15,cmap="YlGnBu")
            for i in range(len(BASES)):
                for j in range(3):
                    ax.text(j,i,str(int(matrix[i,j])),ha="center",va="center",fontsize=10,
                            color="white" if matrix[i,j]>=10 else "#111111")
            ax.set_xticks(range(3),SHORT_TASKS)
            ax.set_yticks(range(len(BASES)),[LABELS[b] for b in BASES])
            ax.set_title(title)
            ax.tick_params(length=0)
        fig.suptitle(f"{METHODS[strategy]}: feature selection count out of 15 fits per task",fontsize=13)
        bar = fig.colorbar(heat,ax=axes,shrink=.7,pad=.02,ticks=[0,5,10,15])
        bar.set_label("Selected fits (out of 15)")
        save_figure(fig,output,f"{strategy}_feature_frequency")
    fig, axes = plt.subplots(1,3,figsize=(15,4.6),layout="constrained")
    for ax, protocol in zip(axes, PROTOCOLS):
        rows = path[path.protocol.eq(protocol)]
        for _, fold in rows.groupby(["repeat","fold"]):
            ax.plot(fold.n_features,fold.mean_r2,color="#087F80",alpha=.16,lw=1)
        mean=rows.groupby("n_features").mean_r2.mean()
        ax.plot(mean.index,mean,lw=2.5,color="#087F80")
        ax.set(title=NAMES[protocol],xlabel="Number of selected features",ylabel="Inner-CV mean RF/ET R-squared")
    fig.supxlabel("Thin: outer-training folds. Thick: their mean. The sqrt split-sampling rule changes at 4, 9 and 16 features.",fontsize=10)
    save_figure(fig,output,"three_task_inner_paths")


def report(summary, frequencies, prop, family, manifest, output):
    audit = pd.read_csv(output/"split_audit.csv")
    random = audit[audit.level.eq("outer") & audit.protocol.eq("stratified_random")]
    seen_fraction = np.average(random.fraction_test_pair_seen, weights=random.n_test)
    lines=["# Forward Feature Selection Across Three Prediction Tasks", "",
        "## Design", "",
        "Population: 554 initial-screening samples and 111 independent BO samples, including 27 Co-Fe BO samples.",
        "The descriptor pool is 26 intrinsic elemental properties plus 13 ICP-weighted properties (39 inputs).",
        "All tasks use three repeats of five outer folds, with three inner folds for selection/tuning.",
        "The random task adds 15 new fits; the 30 completed grouped-task fits are reused after hash verification.", "",
        "1. **Stratified random entities:** hold out initial samples by response-quantile-stratified random folds. Test n = 554 per repeat. Metal pairs may overlap. BO augments training, excluding held-out composition bins.",
        "2. **Unseen metal pairs:** hold out unordered metal pairs. Test n = 554 initial samples per repeat. BO enters training only for training pairs.",
        "3. **Known pairs, new compositions:** hold out 5 mol% composition bins within the five BO pairs. Test n = 111 BO samples per repeat. Matching initial rows are purged.", "",
        "Shared ICP measurements and sample IDs never cross train/test boundaries. All imputers, feature selection and RF/ET tuning use training data only.",
        f"In the random task, {seen_fraction:.1%} of held-out initial samples have a metal pair present in training (sample-weighted over all repeats).",
        "It mixes seen- and unseen-pair cases and does not establish dedicated unseen-pair or BO-composition generalization.",
        "Averaging uses pooled out-of-fold predictions within each repeat, then mean/SD across three repeats. It is not the average of fold-wise R2.",
        "SD is split variability, not an independent experimental confidence interval. The three task scores are not collapsed into one grand average.", "",
        "## Mean Performance", "",
        "| Task | Strategy | R2 mean +/- SD | RMSE mean +/- SD | MAE mean +/- SD | Features mean (range) |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for protocol in PROTOCOLS:
        rows=summary[summary.protocol.eq(protocol)].set_index("strategy")
        for strategy in STRATEGIES:
            r=rows.loc[strategy]
            lines.append(f"| {NAMES[protocol]} | {METHODS[strategy]} | {r.r2_mean:.4f} +/- {r.r2_sd:.4f} | {r.rmse_mean:.4f} +/- {r.rmse_sd:.4f} | {r.mae_mean:.4f} +/- {r.mae_sd:.4f} | {r.features_mean:.1f} ({int(r.features_min)}-{int(r.features_max)}) |")
    lines += ["", "All39, forward-10, inner-best (1-16 features) and one-SE use the same test folds and six RF/ET tuning configurations.",
        "Forward-10 fixes only the subset size; feature identities are reselected within each outer training set.",
        "Forward addition uses 120-tree RF/ET inner scores, tuning uses 80 trees, and final fits use 400 trees.",
        "The one-SE rule chooses the smallest path subset within one standard error of the best inner score.",
        "The path is greedy and capped at 16. Its curves also reflect sqrt feature-sampling jumps, not a physical number-of-descriptors threshold.", "",
        "## Selection Frequencies", "",
        "A feature's frequency is selected fits / 15 within one task and one strategy, not a tree importance value.",
        "The same property can appear as M1, M2 or ICP-weighted inputs; these are counted individually in feature_selection_frequency.csv.",
        "Property frequency counts a fit once when any of its three versions is present. Descriptor-family counts additionally combine correlated H/D vibrations/ZPE and H/D bond energies.",
        "Per-repeat counts use five fits; their SD is available. The optional all-task frequency uses 45 fits and is descriptive only.",
        "All39 frequencies are mechanically 100% and are a control, not evidence of importance.", "",
        "### Forward-10 Property Frequency", "",
        "| Property (any version) | Random entities | Unseen pairs | Known-pair compositions |",
        "| --- | ---: | ---: | ---: |"]
    for base in BASES:
        rows=prop[prop.strategy.eq("forward10") & prop.property.eq(base)].set_index("protocol")
        values=[f"{int(rows.loc[p,'selected_fits'])}/15 ({rows.loc[p,'selection_frequency']:.0%})" for p in PROTOCOLS]
        lines.append("| "+LABELS[base]+" | "+" | ".join(values)+" |")
    lines += ["", "### Most Frequently Selected Individual Inputs", ""]
    for protocol in PROTOCOLS:
        rows=frequencies[frequencies.protocol.eq(protocol) & frequencies.strategy.eq("forward10")]
        rows=rows.sort_values(["selected_fits","forward_rank","feature"],ascending=[False,True,True]).head(10)
        lines.append(f"**{NAMES[protocol]}**")
        lines.extend(f"- `{r.feature}`: {int(r.selected_fits)}/15 ({r.selection_frequency:.0%})." for r in rows.itertuples())
        lines.append("")
    lines += ["## Interpretation", "",
        "Selection frequency quantifies stability of predictive feature use, not unique physical causality.",
        "H vibration, D vibration and ZPE are perfectly correlated in the current descriptor table; H/D bond energies are also nearly collinear.",
        "Selecting one representative but not another does not make the latter physically irrelevant. Search tie order may choose among equivalent inputs.",
        "Within a fixed metal pair, every weighted property is an affine function of the same ICP fraction, so its unique mechanism cannot be identified from within-pair ranking alone.",
        "Held-out property permutation is retained for inner-best and one-SE models only; its results do not feed back into selection. Partial shuffles of correlated properties may form inconsistent descriptors.",
        "The earlier historical RF R2 = 0.606054 was reproduced separately. That historical selector used the full response vector before its reported holdout and is not the nested random task reported here.", "",
        "## Reproduction", "", "```powershell",
        "python -X utf8 hd_separation_ml/icp_forward_three_tasks.py --jobs 10",
        "python -X utf8 hd_separation_ml/summarize_icp_forward_three_tasks.py", "```", "",
        "For revised y, rerun the corrected-ICP baseline, Co-Fe comparison and two-task forward analysis first.",
        "Then pass their outputs through --source and --grouped and use a fresh --output. Source/code hashes reject stale reuse.",
        "The primary summary is summary.csv. Raw repeat/fold metrics, per-fit selections, frequencies, overlap audits and checkpoint provenance are retained."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/forward_three_tasks_20261013")
    args=parser.parse_args()
    output=args.output.resolve()
    data,tables,manifest=load_results(output)
    summary,repeats,folds=performance_tables(tables["predictions"],tables["settings"],data)
    columns=list(representation("hybrid39").fit(data).transform(data).columns)
    selection=selection_tables(tables["settings"],columns)
    for name,table in {"summary":summary,"repeat_metrics":repeats,"fold_metrics":folds,**selection}.items():
        table.to_csv(output/f"{name}.csv",index=False)
    choices=tables["settings"].copy()
    choices["features"]=choices.features.map(json.dumps)
    choices.to_csv(output/"selected_feature_sets.csv",index=False)
    importance_table(tables["permutation"],data).to_csv(output/"heldout_property_importance.csv",index=False)
    draw(summary,selection["feature_selection_frequency"],tables["path"],output)
    report(summary,selection["feature_selection_frequency"],selection["property_selection_frequency"],
           selection["descriptor_family_selection_frequency"],manifest,output)
    old=pd.read_csv(Path(manifest["grouped_source"])/"summary.csv")
    old=old[old.population.isin(["initial","all_BO"])].set_index(["protocol","strategy"])
    reused=summary[summary.protocol.ne("stratified_random")].set_index(["protocol","strategy"])
    np.testing.assert_allclose(reused.r2_mean,old.loc[reused.index,"r2"],atol=1e-12,rtol=0)
    np.testing.assert_allclose(reused.rmse_mean,old.loc[reused.index,"rmse"],atol=1e-12,rtol=0)
    (output/"report_verification.json").write_text(json.dumps(dict(targets_and_membership_verified=True,
        n_outer_fits=45, feature_frequency_denominator=15, grouped_metrics_reproduced=True,
        reporter_sha256=sha(Path(__file__)),analysis_fingerprint=manifest["fingerprint"]),indent=2),encoding="utf-8")
    print(summary.to_string(index=False))


if __name__=="__main__":
    main()
