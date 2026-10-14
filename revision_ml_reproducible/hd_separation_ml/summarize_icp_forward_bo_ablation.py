"""Report paired BO augmentation effects and feature-selection stability."""

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
from icp_forward_bo_ablation import ARMS, PROTOCOLS, TABLES
from icp_intrinsic_descriptor_analysis import BASES, LABELS, scores
from icp_revision_inputs import sha
from summarize_icp_forward_three_tasks import (
    METHODS, NAMES, STRATEGIES, importance_table, performance_tables, save_figure, selection_tables,
)

ROOT = Path(__file__).resolve().parent
ARM_NAMES = {"without_bo":"Without BO training", "with_bo":"With BO training"}
COLORS = {"without_bo":"#777777", "with_bo":"#087F80"}


def load_ablation(output):
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    for file, expected in {**manifest["source_hashes"], **manifest["code_hashes"]}.items():
        if sha(Path(file)) != expected:
            raise ValueError(f"Changed dependency: {file}")
    provenance = pd.read_csv(output / "checkpoint_provenance.csv")
    if len(provenance) != 90 or provenance.duplicated(["arm","protocol","repeat","fold"]).any():
        raise ValueError("Expected 90 distinct arm/fold checkpoints")
    records = []
    for row in provenance.itertuples(index=False):
        file = Path(row.checkpoint)
        if sha(file) != row.sha256:
            raise ValueError(f"Changed checkpoint: {file}")
        record = json.loads(file.read_text(encoding="utf-8"))
        if record["fingerprint"] != row.fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        records.append(dict(arm=row.arm, **record))
    tables = {category:pd.DataFrame([dict(**{k:r[k] for k in ("arm","protocol","repeat","fold")}, **entry)
        for r in records for entry in r[category]]) for category in TABLES}
    data = pd.read_csv(Path(manifest["source"]) / "combined_dataset.csv").set_index("sample", drop=False)
    prediction = tables["predictions"]
    np.testing.assert_allclose(prediction.y_true, data.loc[prediction["sample"], "separation_coefficient"], atol=1e-12, rtol=0)
    keys = ["protocol","repeat","fold","strategy","sample"]
    if prediction.duplicated(["arm"]+keys).any():
        raise ValueError("Duplicate prediction")
    if set(prediction.arm) != set(ARMS):
        raise ValueError("Missing training arm")
    a = prediction[prediction.arm.eq("without_bo")].set_index(keys).sort_index()
    b = prediction[prediction.arm.eq("with_bo")].set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index,b.index)
    np.testing.assert_allclose(a.y_true,b.y_true,atol=1e-12,rtol=0)
    memberships = pd.read_csv(output / "outer_assignments.csv")
    for (arm,protocol,repeat,fold,strategy), part in prediction.groupby(["arm","protocol","repeat","fold","strategy"]):
        expected = memberships[(memberships.arm==arm) & (memberships.protocol==protocol) &
            (memberships.repeat==repeat) & (memberships.fold==fold) & memberships.role.eq("test")]["sample"]
        if set(part["sample"]) != set(expected):
            raise ValueError("Test samples differ from frozen split")
    settings = tables["settings"]
    if len(settings)!=360 or settings.duplicated(["arm","protocol","repeat","fold","strategy"]).any():
        raise ValueError("Missing selected models")
    for r in settings.itertuples():
        if r.n_features != len(set(r.features)):
            raise ValueError("Feature-count mismatch")
    inner = tables["inner_assignments"].query("arm == 'without_bo'")
    if not data.loc[inner["sample"],"cohort"].eq("initial").all():
        raise ValueError("BO outcome used by no-BO tuning")
    if not memberships.query("arm == 'without_bo' and role == 'train'").cohort.eq("initial").all():
        raise ValueError("BO data used by no-BO training")
    fixed = pd.DataFrame([dict(**{k:r[k] for k in ("protocol","repeat","fold")}, **entry)
        for r in records if r["arm"]=="without_bo" for entry in r["fixed_control"]])
    return data,tables,fixed,manifest


def paired_deltas(repeats):
    keys = ["protocol","repeat","strategy"]
    a = repeats[repeats.arm.eq("without_bo")][keys+["r2","rmse","mae","n"]]
    b = repeats[repeats.arm.eq("with_bo")][keys+["r2","rmse","mae","n"]]
    pair = a.merge(b,on=keys,suffixes=("_without","_with"),validate="one_to_one")
    if len(pair)!=len(a) or len(pair)!=len(b) or not pair.n_without.eq(pair.n_with).all():
        raise ValueError("Unmatched repeat metrics")
    for metric in ["r2","rmse","mae"]:
        pair["bo_delta_"+metric] = pair[metric+"_with"] - pair[metric+"_without"]
    aggregations = {f"bo_delta_{m}_{s}":(f"bo_delta_{m}","mean" if s=="mean" else "std")
                    for m in ["r2","rmse","mae"] for s in ["mean","sd"]}
    summary = pair.groupby(["protocol","strategy"],as_index=False).agg(**aggregations)
    return summary,pair


def composition_pair_tables(predictions):
    subset = predictions[predictions.protocol.eq("known_pair_new_composition")]
    rows = []
    for (arm, strategy, repeat, pair), part in subset.groupby(["arm", "strategy", "repeat", "metal_pair"]):
        rows.append(dict(arm=arm, strategy=strategy, repeat=repeat, metal_pair=pair,
                         n=len(part), **scores(part.y_true, part.y_pred)))
    repeats = pd.DataFrame(rows)
    summary = repeats.groupby(["arm", "strategy", "metal_pair"], as_index=False).agg(
        n=("n", "first"), r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"))
    return summary, repeats


def fixed_control_tables(fixed, predictions):
    base = predictions[predictions.arm.eq("without_bo") & predictions.strategy.eq("all39")].copy()
    base["arm"] = "without_bo"
    added = fixed.assign(arm="with_bo",strategy="all39")
    key = ["protocol","repeat","fold","sample"]
    a,b = base.set_index(key).sort_index(), added.set_index(key).sort_index()
    pd.testing.assert_index_equal(a.index,b.index)
    np.testing.assert_allclose(a.y_true,b.y_true,atol=1e-12,rtol=0)
    rows = []
    for (arm,protocol,repeat), p in pd.concat([base,added],ignore_index=True).groupby(["arm","protocol","repeat"]):
        if p["sample"].duplicated().any():
            raise ValueError("Duplicated locked-control OOF sample")
        rows.append(dict(arm=arm,protocol=protocol,repeat=repeat,strategy="all39",n=len(p),**scores(p.y_true,p.y_pred)))
    repeats = pd.DataFrame(rows)
    summary = repeats.groupby(["arm","protocol"],as_index=False).agg(
        r2_mean=("r2","mean"),r2_sd=("r2","std"),rmse_mean=("rmse","mean"),rmse_sd=("rmse","std"))
    delta,_ = paired_deltas(repeats)
    return summary,repeats,delta


def draw(summary, frequency, prop, fixed, output):
    plt.rcParams.update({"font.family":"Arial","font.size":10,"pdf.fonttype":42,
                        "axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,3,figsize=(15.2,5.0),layout="constrained",sharey=True)
    lower = min(0,float((summary.r2_mean-summary.r2_sd).min())-.035)
    upper = max(.75,float((summary.r2_mean+summary.r2_sd).max())+.08)
    for ax,protocol in zip(axes,PROTOCOLS):
        for j,arm in enumerate(ARMS):
            p = summary[(summary.protocol==protocol)&(summary.arm==arm)].set_index("strategy").loc[STRATEGIES]
            x = np.arange(4)+(j-.5)*.38
            ax.bar(x,p.r2_mean,width=.35,color=COLORS[arm],label=ARM_NAMES[arm])
            ax.errorbar(x,p.r2_mean,yerr=p.r2_sd,fmt="none",color="#222222",capsize=2)
            for xx,row in zip(x,p.itertuples()):
                ax.text(xx,row.r2_mean+row.r2_sd+.012,f"{row.r2_mean:.3f}",ha="center",va="bottom",fontsize=8.3,rotation=65)
        ax.axhline(0,color="#333333",lw=.7)
        ax.set_xticks(range(4),["All 39","Forward 10","Inner-best","One-SE"],rotation=20)
        ax.set(title=NAMES[protocol],ylim=(lower,upper))
        ax.grid(axis="y",alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Pooled out-of-fold R-squared")
    axes[0].legend(loc="upper left",frameon=False,fontsize=9)
    fig.supxlabel("Identical test samples in each arm; mean and SD over 3 repeats of 5-fold CV",fontsize=10)
    save_figure(fig,output,"bo_ablation_performance")
    column_keys = [(p,a) for p in PROTOCOLS for a in ARMS]
    labels = [f"{task}\n{arm}" for task in ["Random","Unseen pairs","BO compositions"] for arm in ["No BO","+ BO"]]
    for strategy in ["forward10","forward_best","forward_one_se"]:
        fig,axes = plt.subplots(1,3,figsize=(16,7.8),layout="constrained",sharey=True)
        part = frequency[frequency.strategy.eq(strategy)]
        for ax,version,title in zip(axes,["element1","element2","weighted"],["M1 properties","M2 properties","ICP-weighted properties"]):
            names = ["icp_weighted_"+b if version=="weighted" else b+"_"+version for b in BASES]
            matrix = part.pivot(index="feature",columns=["protocol","arm"],values="selected_fits").loc[names,column_keys].to_numpy()
            heat = ax.imshow(matrix,aspect="auto",vmin=0,vmax=15,cmap="YlGnBu")
            for i in range(len(names)):
                for j in range(6):
                    ax.text(j,i,str(int(matrix[i,j])),ha="center",va="center",fontsize=9,color="white" if matrix[i,j]>=10 else "#111111")
            ax.set_xticks(range(6),labels,rotation=40,ha="right")
            ax.set_yticks(range(len(BASES)),[LABELS[b] for b in BASES])
            ax.set_title(title)
            ax.tick_params(length=0)
        fig.suptitle(f"{METHODS[strategy]}: selection count out of 15 fits per task and training arm",fontsize=13)
        fig.colorbar(heat,ax=axes,pad=.015,shrink=.6,ticks=[0,5,10,15]).set_label("Selected fits / 15")
        save_figure(fig,output,strategy+"_bo_feature_frequency")
    fig,axes = plt.subplots(1,2,figsize=(11.8,7.8),layout="constrained",sharey=True)
    part = prop[prop.strategy.eq("forward10")]
    for ax,field,title in zip(axes,["selected_fits","weighted_selected_fits"],["Property selected in any form","ICP-weighted form selected"]):
        matrix = part.pivot(index="property",columns=["protocol","arm"],values=field).loc[BASES,column_keys].to_numpy()
        heat = ax.imshow(matrix,aspect="auto",vmin=0,vmax=15,cmap="YlGnBu")
        for i in range(len(BASES)):
            for j in range(6):
                ax.text(j,i,str(int(matrix[i,j])),ha="center",va="center",color="white" if matrix[i,j]>=10 else "#111111")
        ax.set_xticks(range(6),labels,rotation=35,ha="right")
        ax.set_yticks(range(len(BASES)),[LABELS[b] for b in BASES])
        ax.set_title(title)
        ax.tick_params(length=0)
    fig.suptitle("Forward-10 property selection stability",fontsize=13)
    fig.colorbar(heat,ax=axes,pad=.025,shrink=.6,ticks=[0,5,10,15]).set_label("Selected fits / 15")
    save_figure(fig,output,"forward10_bo_property_frequency")
    fig,axes = plt.subplots(1,3,figsize=(11.5,4.3),layout="constrained",sharey=True)
    for ax,protocol in zip(axes,PROTOCOLS):
        p = fixed[fixed.protocol.eq(protocol)].set_index("arm").loc[ARMS]
        ax.bar(range(2),p.r2_mean,color=[COLORS[a] for a in ARMS],width=.55)
        ax.errorbar(range(2),p.r2_mean,yerr=p.r2_sd,fmt="none",color="#222222",capsize=3)
        for i,row in enumerate(p.itertuples()):
            ax.text(i,row.r2_mean+row.r2_sd+.012,f"{row.r2_mean:.3f}",ha="center",fontsize=11)
        ax.set_xticks(range(2),["No BO","+ BO"])
        ax.set(title=NAMES[protocol],ylim=(min(0,float((fixed.r2_mean-fixed.r2_sd).min())-.03),
            max(.75,float((fixed.r2_mean+fixed.r2_sd).max())+.07)))
        ax.grid(axis="y",alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Pooled out-of-fold R-squared")
    fig.supxlabel("All 39 inputs; model and hyperparameters chosen using initial data only, then held fixed",fontsize=10)
    save_figure(fig,output,"bo_locked_model_control")


def report(summary, delta, fixed, prop, audit, output):
    lines = ["# Matched BO Augmentation and Feature Selection", "", "## Design", "",
        "554 initial-screening and 111 independent BO samples (including 27 Co-Fe BO samples).",
        "Three tasks x two training arms x four selection strategies. Each uses three repeats of five-fold outer CV, with three inner folds.",
        "The same frozen test samples, target values, initial training rows, seeds, descriptor definitions, model grid and tree budgets are used across arms.",
        "The 45 with-BO selection runs are verified and reused. The 45 without-BO runs independently rebuild the forward path and retune the model.", "",
        "- Random entities: 554 initial test samples per repeat; response-quantile stratification. Initial related pairs can overlap.",
        "- Unseen pairs: 554 initial test samples per repeat; unordered metal pairs are isolated. Eligible training BO must belong to training pairs.",
        "- New compositions: 111 BO test samples per repeat; held-out 5 mol% composition bins are excluded from both arms' training. No-BO means no BO TRAINING data, not no BO test data.",
        "- Shared ICP groups/sample IDs cannot cross training/test boundaries. BO training rows in held-out composition bins are excluded in every task.", "",
        "For new compositions, initial-only data cannot provide a BO composition-series validation set. The no-BO arm therefore selects/tunes by initial-only grouped-pair CV, then predicts the held-out BO samples. The with-BO arm uses composition-grouped inner CV. This is a comparison of complete training procedures, not solely sample-count changes. The locked-model control below separates the data-addition effect.",
        "All held-out BO compositions retain their metal pair in initial-only training (verified in the split audit).", "",
        "39 candidate inputs = 26 intrinsic M1/M2 properties + 13 ICP-weighted properties; no operating covariates.",
        "Forward selection tests paths from one to sixteen features. Forward-10 fixes the count, not the feature identities. Inner-best and one-SE choose count/configuration within training only. RF and Extra Trees are candidates; XGBoost is not used.", "",
        "## Performance", "",
        "Each repeat pools every out-of-fold prediction before calculating R2/RMSE/MAE. Values below are mean +/- SD over three repeats; SD describes split variability, not experimental uncertainty or an independent confidence interval. Scores from different tasks are not grand-averaged.", "",
        "| Task | Strategy | R2 without BO | R2 with BO | Paired change | RMSE without / with |",
        "|---|---|---:|---:|---:|---:|"]
    index = summary.set_index(["protocol","strategy","arm"])
    differences = delta.set_index(["protocol","strategy"])
    for protocol in PROTOCOLS:
        for strategy in STRATEGIES:
            a,b = (index.loc[(protocol,strategy,arm)] for arm in ARMS)
            d = differences.loc[(protocol,strategy)]
            lines.append(f"| {NAMES[protocol]} | {METHODS[strategy]} | {a.r2_mean:.3f} +/- {a.r2_sd:.3f} | {b.r2_mean:.3f} +/- {b.r2_sd:.3f} | {d.bo_delta_r2_mean:+.3f} +/- {d.bo_delta_r2_sd:.3f} | {a.rmse_mean:.3f} / {b.rmse_mean:.3f} |")
    lines.extend(["", "Complete R2, RMSE, MAE, feature-count distributions and paired repeat differences are in the CSV tables.",
        "The paired changes compare the same test observations. Repeated folds are correlated; no significance claim is inferred from the three split repeats alone.", "",
        "## Locked-Model Control", "",
        "All39: choose RF/ET and hyperparameters using initial-only inner CV, then fit on initial-only versus initial+eligible BO with that choice fixed. Training-only preprocessing is refitted for each arm. This control never uses BO responses to select hyperparameters and uses the same matched tests.", "",
        "| Task | R2 without BO | R2 with BO, locked settings | Change |", "|---|---:|---:|---:|"])
    for protocol in PROTOCOLS:
        f = fixed[fixed.protocol.eq(protocol)].set_index("arm")
        a,b = f.loc["without_bo"],f.loc["with_bo"]
        lines.append(f"| {NAMES[protocol]} | {a.r2_mean:.3f} +/- {a.r2_sd:.3f} | {b.r2_mean:.3f} +/- {b.r2_sd:.3f} | {b.r2_mean-a.r2_mean:+.3f} |")
    lines.extend(["", "## Feature Stability", "",
        "Individual feature frequencies use a denominator of 15 outer fits PER task, training arm and strategy. Counts of zero are retained. Property-level counts count M1/M2/weighted variants once per fit, not three times. Descriptor-family tables combine correlated H/D terms.",
        "All39 inputs are included by definition, so their 100% frequency is not evidence of importance.", "",
        "| Property (forward-10, any form) | Random no BO / +BO | Unseen no BO / +BO | Compositions no BO / +BO |",
        "|---|---:|---:|---:|"])
    freq = prop[prop.strategy.eq("forward10")].set_index(["property","protocol","arm"])
    for base in BASES:
        values = [f"{int(freq.loc[(base,p,'without_bo'),'selected_fits'])}/15 / {int(freq.loc[(base,p,'with_bo'),'selected_fits'])}/15" for p in PROTOCOLS]
        lines.append("| "+LABELS[base]+" | "+" | ".join(values)+" |")
    lines.extend(["", "Feature selection can yield a compact model and identify stable predictive descriptors without increasing R2. Compare the paired within-arm changes against All39 in summary.csv before asserting an accuracy gain.",
        "H-frequency, D-frequency and ZPE are perfectly correlated within each representation here; H/D bond energies are nearly collinear. Selection of one instead of another does not establish that the others are physically unimportant. Within a fixed metal pair, every weighted property is an affine function of ICP composition. Selection frequencies and permutation scores are predictive diagnostics, not causal assignments.",
        "Historical single-split R2 around 0.60 is not the same estimator as these repeated nested-CV scores. Earlier historical selection was not isolated from its final split; the archived reproduction is kept separately.", "",
        "## Reproduction", "", "```powershell",
        "python -X utf8 hd_separation_ml/icp_forward_bo_ablation.py --audit-only",
        "python -X utf8 hd_separation_ml/icp_forward_bo_ablation.py --jobs 10",
        "python -X utf8 hd_separation_ml/summarize_icp_forward_bo_ablation.py", "```", "",
        "Prerequisite: the verified three-task with-BO run. For changed targets, regenerate upstream datasets/folds and both training arms using new output directories. Never reuse checkpoints from different target/input/code hashes.",
        "Manifests, checkpoints, actual sample memberships, inner assignments, selected feature sets, every OOF prediction and report verification are retained.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT / "outputs/forward_bo_ablation_20261013")
    args = parser.parse_args()
    output = args.output.resolve()
    data,tables,fixed,manifest = load_ablation(output)
    columns = list(representation("hybrid39").fit(data).transform(data).columns)
    collected = {}
    for arm in ARMS:
        t = {k:v[v.arm.eq(arm)].drop(columns="arm") for k,v in tables.items()}
        summary,repeats,folds = performance_tables(t["predictions"],t["settings"],data)
        result = dict(summary=summary,repeat_metrics=repeats,fold_metrics=folds,
                      heldout_property_importance=importance_table(t["permutation"],data),
                      **selection_tables(t["settings"],columns))
        for key,frame in result.items():
            collected.setdefault(key,[]).append(frame.assign(arm=arm))
    collected = {k:pd.concat(v,ignore_index=True) for k,v in collected.items()}
    delta,repeat_delta = paired_deltas(collected["repeat_metrics"])
    fixed_summary,fixed_repeats,fixed_delta = fixed_control_tables(fixed,tables["predictions"])
    pair_summary,pair_repeats = composition_pair_tables(tables["predictions"])
    collected.update(bo_effect_summary=delta,bo_effect_by_repeat=repeat_delta,
                     composition_pair_summary=pair_summary,composition_pair_repeats=pair_repeats,
                     locked_control_summary=fixed_summary,locked_control_repeats=fixed_repeats,locked_control_bo_effect=fixed_delta)
    for key,frame in collected.items():
        frame.to_csv(output / f"{key}.csv",index=False)
    settings = tables["settings"].copy()
    settings["features"] = settings.features.map(json.dumps)
    settings["config"] = settings.config.map(json.dumps)
    settings.to_csv(output / "selected_feature_sets.csv",index=False)
    reference = pd.read_csv(Path(manifest["with_bo_source"]) / "summary.csv").set_index(["protocol","strategy"])
    actual = collected["summary"].query("arm == 'with_bo'").set_index(["protocol","strategy"]).loc[reference.index]
    for metric in ["r2_mean","r2_sd","rmse_mean","mae_mean"]:
        np.testing.assert_allclose(actual[metric],reference[metric],atol=1e-12,rtol=0)
    draw(collected["summary"],collected["feature_selection_frequency"],collected["property_selection_frequency"],fixed_summary,output)
    audit = pd.read_csv(output / "split_audit.csv")
    known = audit.query("arm == 'without_bo' and level == 'outer' and protocol == 'known_pair_new_composition'")
    if not known.fraction_test_pair_seen.eq(1).all():
        raise ValueError("Not all test compositions have their pair represented in initial-only training")
    report(collected["summary"],delta,fixed_summary,collected["property_selection_frequency"],audit,output)
    proof = dict(completed_outer_arm_fits=90,reported_models=360,matched_test_samples=True,zero_bo_tuning=True,
        unchanged_with_bo_results=True,frequency_denominator_per_task_arm=15,
        fingerprint=manifest["fingerprint"],reporter_sha256=sha(Path(__file__)))
    (output / "report_verification.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")
    print(collected["summary"].to_string(index=False))
    print("\nPAIRED BO EFFECT\n"+delta.to_string(index=False))


if __name__ == "__main__":
    main()
