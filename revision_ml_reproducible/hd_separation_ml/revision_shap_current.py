"""Stage 1: cross-fitted SHAP for current nested and retrospective frozen models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed, parallel_config

from revision_followup_common import (
    ROOT,FEATURES,BASES,PROTOCOLS,NAMES,label,load_current,start_output,matched_plan,
    seed_for,fit_model,save_json,
)


def job(key,data,assignments,choice,expected,output,fingerprint):
    protocol,repeat,fold,kind = key
    stem = f"{kind}_{protocol}_r{repeat}_f{fold}"
    file = output / "checkpoints" / (stem+".json")
    samples_file = output / "checkpoints" / (stem+".csv.gz")
    if file.exists() and samples_file.exists():
        result = json.loads(file.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Stale SHAP checkpoint")
        return result
    _,train,test = matched_plan((protocol,repeat,fold),data,assignments)
    selected = choice["features"]
    model,V,pred = fit_model(train,test,selected,choice["model"],choice["config"],400,seed_for((protocol,repeat,fold)))
    old = expected.set_index("sample").loc[test["sample"],"y_pred"].to_numpy()
    np.testing.assert_allclose(pred,old,atol=1e-10,rtol=0)
    explainer = shap.TreeExplainer(model,feature_perturbation="tree_path_dependent")
    values = np.asarray(explainer.shap_values(V[selected].to_numpy(),check_additivity=True))
    reconstructed = values.sum(axis=1)+float(np.asarray(explainer.expected_value).ravel()[0])
    np.testing.assert_allclose(reconstructed,pred,atol=1e-6,rtol=0)
    full = np.zeros((len(test),len(FEATURES)))
    full[:,[FEATURES.index(f) for f in selected]] = values
    common = dict(protocol=protocol,repeat=int(repeat),fold=int(fold),kind=kind)
    feature_rows = [dict(feature=c,n=len(test),selected=c in selected,
        abs_sum=float(np.abs(full[:,i]).sum()),signed_sum=float(full[:,i].sum())) for i,c in enumerate(FEATURES)]
    property_rows = []
    for base in BASES:
        indices = [i for i,c in enumerate(FEATURES) if c in {base+"_element1",base+"_element2","icp_weighted_"+base}]
        grouped = full[:,indices].sum(axis=1)
        property_rows.append(dict(property=base,n=len(test),abs_sum=float(np.abs(grouped).sum())))
    sample_rows = [dict(sample=row["sample"],metal_pair=row.metal_pair,feature=c,
        value=float(V.iloc[i][c]),shap_value=float(values[i,j]),y_true=float(row.separation_coefficient),
        y_pred=float(pred[i]),base_value=float(np.asarray(explainer.expected_value).ravel()[0]))
        for i,(_,row) in enumerate(test.iterrows()) for j,c in enumerate(selected)]
    pd.DataFrame(sample_rows).to_csv(samples_file,index=False,compression="gzip")
    result = dict(**common,fingerprint=fingerprint,feature_rows=feature_rows,property_rows=property_rows,
        max_prediction_difference=float(np.max(np.abs(pred-old))),
        max_additivity_error=float(np.max(np.abs(reconstructed-pred))),samples_file=str(samples_file))
    save_json(file,result)
    return result


def aggregate(rows,term):
    repeat = rows.groupby(["protocol","kind","repeat",term],as_index=False).agg(n=("n","sum"),abs_sum=("abs_sum","sum"))
    repeat["mean_abs_shap"] = repeat.abs_sum/repeat.n
    summary = repeat.groupby(["protocol","kind",term],as_index=False).agg(
        mean_abs_shap=("mean_abs_shap","mean"),repeat_sd=("mean_abs_shap","std"))
    return repeat,summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT / "outputs/reviewer_stage1_shap_20261013")
    parser.add_argument("--jobs",type=int,default=10)
    args = parser.parse_args()
    data,tables,assignments,spec,frozen,context = load_current()
    output = args.output.resolve()
    fingerprint = start_output(output,__file__,context,
        dict(stage=1,models=["nested_forward10","task1_frozen10"],arm="with_bo",shap=shap.__version__,
        explanation="tree_path_dependent; held-out inputs; training-tree path counts; no test background"))
    work = []
    for p in PROTOCOLS:
        for r in range(3):
            for f in range(5):
                s = tables["settings"]
                s = s[s.arm.eq("with_bo") & s.protocol.eq(p) & s.repeat.eq(r) & s.fold.eq(f) & s.strategy.eq("forward10")]
                if len(s)!=1:
                    raise ValueError("Missing nested model")
                for kind,choice,source in [("nested_forward10",s.iloc[0].to_dict(),tables["predictions"]),
                                           ("task1_frozen10",spec,frozen)]:
                    pred = source[source.arm.eq("with_bo") & source.protocol.eq(p) & source.repeat.eq(r) & source.fold.eq(f)]
                    if kind=="nested_forward10":
                        pred = pred[pred.strategy.eq("forward10")]
                    work.append(((p,r,f,kind),choice,pred))
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs)(delayed(job)(key,data,assignments,choice,pred,output,fingerprint)
            for key,choice,pred in work)
    features = pd.DataFrame([dict(**{k:r[k] for k in ("protocol","kind","repeat","fold")},**x)
        for r in results for x in r["feature_rows"]])
    properties = pd.DataFrame([dict(**{k:r[k] for k in ("protocol","kind","repeat","fold")},**x)
        for r in results for x in r["property_rows"]])
    frepeat,fsummary = aggregate(features,"feature")
    prepeat,psummary = aggregate(properties,"property")
    samples = pd.concat([pd.read_csv(r["samples_file"]).assign(**{k:r[k] for k in ("protocol","kind","repeat","fold")})
                         for r in results],ignore_index=True)
    samples.to_csv(output / "heldout_shap_values.csv.gz",index=False,compression="gzip")
    for name,frame in [("feature_fold_sums",features),("feature_repeat_summary",frepeat),
        ("feature_shap_summary",fsummary),("property_repeat_summary",prepeat),("property_shap_summary",psummary)]:
        frame.to_csv(output / (name+".csv"),index=False)
    plt.rcParams.update({"font.family":"Arial","font.size":10,"pdf.fonttype":42})
    order = fsummary.groupby("feature").mean_abs_shap.mean().nlargest(15).index[::-1].tolist()
    fig,axes = plt.subplots(1,3,figsize=(15,7.4),sharey=True)
    fig.subplots_adjust(left=.25,right=.98,bottom=.14,top=.83,wspace=.18)
    for ax,p in zip(axes,PROTOCOLS):
        for kind,offset,color,name in [("nested_forward10",-.18,"#248385","Nested forward-10"),
                                     ("task1_frozen10",.18,"#c65546","Frozen Task-1 sensitivity")]:
            part = fsummary[fsummary.protocol.eq(p)&fsummary.kind.eq(kind)].set_index("feature").loc[order]
            ax.barh(np.arange(len(order))+offset,part.mean_abs_shap,height=.34,color=color,label=name)
        ax.set_title(NAMES[p]); ax.set_xlabel("Mean absolute SHAP value")
        ax.set_yticks(range(len(order)),[label(c) for c in order]); ax.grid(axis="x",alpha=.15); ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.6,.96))
    fig.text(.5,.035,"With-BO outer-test predictions. Unselected features contribute zero; correlated-feature attributions are not causal effects.",ha="center",fontsize=10)
    for ext in ("png","pdf"):
        fig.savefig(output / ("current_feature_shap_comparison."+ext),dpi=210)
    plt.close(fig)
    # One repeat avoids displaying each sample three times in the dependence summary.
    order = fsummary[fsummary.kind.eq("task1_frozen10")].groupby("feature").mean_abs_shap.mean().nlargest(10).index[::-1].tolist()
    fig,axes = plt.subplots(1,3,figsize=(15,6.8),sharey=True)
    fig.subplots_adjust(left=.25,right=.93,bottom=.16,top=.86,wspace=.18)
    rng = np.random.default_rng(20261013)
    for ax,p in zip(axes,PROTOCOLS):
        part = samples[samples.kind.eq("task1_frozen10")&samples.protocol.eq(p)&samples.repeat.eq(0)]
        for i,c in enumerate(order):
            g = part[part.feature.eq(c)]
            color = g.value.rank(pct=True).to_numpy()
            dots = ax.scatter(g.shap_value,i+rng.uniform(-.27,.27,len(g)),c=color,cmap="coolwarm",vmin=0,vmax=1,s=10,alpha=.65,edgecolors="none")
        ax.axvline(0,color="#999999",lw=.7); ax.set_title(NAMES[p]); ax.set_xlabel("SHAP contribution to predicted H/D separation")
        ax.set_yticks(range(len(order)),[label(c) for c in order]); ax.grid(axis="x",alpha=.12)
    bar = fig.add_axes([.95,.22,.012,.54]); cb=fig.colorbar(dots,cax=bar,ticks=[0,1]); cb.ax.set_yticklabels(["Low","High"])
    cb.set_label("Feature value percentile within task")
    fig.suptitle("Frozen model: held-out SHAP distributions, repeat 1",fontsize=14)
    fig.text(.5,.035,"Retrospective sensitivity model. Colors include training-imputed descriptors; the three tasks have different test populations.",ha="center",fontsize=10)
    for ext in ("png","pdf"):
        fig.savefig(output / ("frozen_shap_distribution."+ext),dpi=210)
    plt.close(fig)
    lines = ["# Stage 1: Updated SHAP", "", "90 outer-model refits: two model specifications, three tasks, three repeats and five folds. "
        "Only the with-BO training arm is interpreted here. All predictions were reproduced to 1e-10 and SHAP additivity checked to 1e-6.", "",
        "Tree-path-dependent TreeSHAP uses the fitted trees' training path counts; no held-out background data are supplied. "
        "The frozen-model explanations remain retrospective, while the nested models retain their original fold-local selection.", "",
        "Feature summaries pool sums of absolute SHAP across test samples within each repeat, then average the three repeats. "
        "Unselected features are zero in the overall summary. Property-group attribution is abs(sum of signed SHAP for M1/M2/weighted versions), "
        "not a new conditional/group SHAP calculation. Individual values and fold sums are retained.", "",
        "H/D/ZPE redundancy and other correlations can redistribute attribution. SHAP explains model predictions, not causal effects. "
        "For Task 3, endpoint-property attributions may explain differences BETWEEN pairs, not variation of composition WITHIN a pair. "
        "The existing within-pair permutation results provide complementary evidence on composition dependence.", "",
        "The distribution plot uses repeat 1 to avoid triple plotting observations. Rank-based feature colors include training-imputed values. "
        "No fitted SHAP ranking was used to reselect or retune these models.", "",
        "Primary SHAP citation: Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions, NeurIPS 30.", "",
        "Reproduce: `python hd_separation_ml/revision_shap_current.py --jobs 10`."]
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    verification = dict(complete=True,refits=len(results),max_prediction_difference=max(r["max_prediction_difference"] for r in results),
        max_additivity_error=max(r["max_additivity_error"] for r in results),original_models_unchanged=True)
    save_json(output / "verification.json",verification)
    print(json.dumps(verification,indent=2),flush=True)
    print(fsummary.sort_values("mean_abs_shap",ascending=False).groupby(["protocol","kind"]).head(3).to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
