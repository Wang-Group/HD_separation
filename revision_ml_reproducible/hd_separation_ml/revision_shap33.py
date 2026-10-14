"""Held-out SHAP for reduced nested models and both frozen specifications."""
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

from revision_followup_common import (PROTOCOLS,NAMES,label,matched_plan,seed_for,
    fit_model,start_output,save_json)
from revision_remaining_common import load_final,OUTPUT,POOL,BASE11

KINDS=("nested33_forward10","original_frozen10","reselected_frozen10")
ARMS=("without_bo","with_bo")


def job(key,data,assignments,choice,expected,output,fingerprint):
    arm,protocol,repeat,fold,kind=key
    common=dict(zip(("arm","protocol","repeat","fold","kind"),key))
    stem=f"{arm}_{protocol}_r{repeat}_f{fold}_{kind}"
    path=output / "checkpoints" / (stem+".json")
    samples_path=path.with_suffix(".csv.gz")
    if path.exists() and samples_path.exists():
        result=json.loads(path.read_text(encoding="utf-8"))
        if result["fingerprint"]!=fingerprint:
            raise ValueError("Stale SHAP checkpoint")
        return result
    original,augmented,test=matched_plan((protocol,repeat,fold),data,assignments)
    train=augmented if arm=="with_bo" else original
    features=choice["features"]
    if not set(features)<=set(POOL):
        raise ValueError("Removed feature in reduced/frozen model")
    model,V,pred=fit_model(train,test,features,choice["model"],choice["config"],400,seed_for((protocol,repeat,fold)))
    old=expected.set_index("sample").loc[test["sample"],"y_pred"].to_numpy()
    np.testing.assert_allclose(pred,old,atol=1e-10,rtol=0)
    explainer=shap.TreeExplainer(model,feature_perturbation="tree_path_dependent")
    values=np.asarray(explainer.shap_values(V[features].to_numpy(),check_additivity=True))
    base=float(np.asarray(explainer.expected_value).ravel()[0])
    np.testing.assert_allclose(values.sum(axis=1)+base,pred,atol=1e-6,rtol=0)
    full=np.zeros((len(test),len(POOL)))
    full[:,[POOL.index(c) for c in features]]=values
    feature_rows=[dict(feature=c,n=len(test),selected=c in features,abs_sum=float(np.abs(full[:,i]).sum()))
                  for i,c in enumerate(POOL)]
    families=[]
    for b in BASE11:
        columns=[POOL.index(c) for c in (b+"_element1",b+"_element2","icp_weighted_"+b)]
        families.append(dict(property=b,n=len(test),abs_sum=float(np.abs(full[:,columns].sum(axis=1)).sum())))
    rows=[dict(sample=r["sample"],metal_pair=r.metal_pair,feature=c,value=float(V.iloc[i][c]),
        shap_value=float(values[i,j]),y_true=float(r.separation_coefficient),y_pred=float(pred[i]),base_value=base)
        for i,(_,r) in enumerate(test.iterrows()) for j,c in enumerate(features)]
    pd.DataFrame(rows).to_csv(samples_path,index=False,compression="gzip")
    result=dict(**common,fingerprint=fingerprint,feature_rows=feature_rows,property_rows=families,
        max_prediction_difference=float(np.max(np.abs(pred-old))),
        max_additivity_error=float(np.max(np.abs(values.sum(axis=1)+base-pred))),samples_file=str(samples_path))
    save_json(path,result)
    print(stem+" complete",flush=True)
    return result


def aggregate(rows,term):
    groups=["arm","protocol","kind"]
    repeat=rows.groupby(groups+["repeat",term],as_index=False).agg(n=("n","sum"),abs_sum=("abs_sum","sum"))
    repeat["mean_abs_shap"]=repeat.abs_sum/repeat.n
    summary=repeat.groupby(groups+[term],as_index=False).agg(mean_abs_shap=("mean_abs_shap","mean"),repeat_sd=("mean_abs_shap","std"))
    summary["rank"]=summary.groupby(groups).mean_abs_shap.rank(method="min",ascending=False).astype(int)
    return repeat,summary


def plots(feature,properties,output):
    plt.rcParams.update({"font.family":"Arial","font.size":10,"pdf.fonttype":42})
    nested=feature[feature.kind.eq("nested33_forward10")]
    order=nested.groupby("feature").mean_abs_shap.mean().nlargest(13).index[::-1]
    colors=["#238781" if c.startswith("icp_weighted_") else "#bd5d43" if c.endswith("element1") else "#577aac" for c in order]
    fig,axes=plt.subplots(2,3,figsize=(15,10),sharey=True)
    fig.subplots_adjust(left=.25,right=.98,bottom=.10,top=.90,wspace=.18,hspace=.25)
    for i,arm in enumerate(ARMS):
        for ax,p in zip(axes[i],PROTOCOLS):
            part=nested[nested.arm.eq(arm)&nested.protocol.eq(p)].set_index("feature").loc[order]
            ax.barh(range(len(order)),part.mean_abs_shap,color=colors)
            ax.set_yticks(range(len(order)),[label(c) for c in order])
            ax.set_title(NAMES[p]+(" | +BO" if arm=="with_bo" else " | no BO"))
            ax.set_xlabel("Mean absolute SHAP");ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("Nested forward-10 from 33 candidates",fontsize=16)
    fig.text(.5,.025,"Red: M1; blue: M2; teal: ICP-weighted. Held-out explanations; unselected features contribute zero.",ha="center")
    for ext in ("png","pdf"):
        fig.savefig(output / ("nested33_shap."+ext),dpi=200)
    plt.close(fig)
    names={"nested33_forward10":"Nested 33","original_frozen10":"Original frozen","reselected_frozen10":"Reselected frozen"}
    chosen=properties.groupby("property").mean_abs_shap.mean().nlargest(8).index[::-1]
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True)
    fig.subplots_adjust(left=.19,right=.98,bottom=.16,top=.82,wspace=.18)
    for ax,p in zip(axes,PROTOCOLS):
        for k,offset,color in zip(KINDS,(-.24,0,.24),("#238781","#777777","#bd5d43")):
            part=properties[properties.arm.eq("with_bo")&properties.protocol.eq(p)&properties.kind.eq(k)].set_index("property").loc[chosen]
            ax.barh(np.arange(len(chosen))+offset,part.mean_abs_shap,height=.23,color=color,label=names[k])
        ax.set_yticks(range(len(chosen)),[label(b+"_element1")[3:] for b in chosen])
        ax.set_title(NAMES[p]);ax.set_xlabel("Mean |summed signed family SHAP|")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False)
    fig.text(.5,.035,"With BO. Frozen specifications are retrospective sensitivities; signed-family sums are not conditional/group SHAP.",ha="center",fontsize=9)
    for ext in ("png","pdf"):
        fig.savefig(output / ("frozen_vs_nested_family_shap."+ext),dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=10)
    parser.add_argument("--output",type=Path,default=OUTPUT / "shap33")
    args=parser.parse_args()
    data,tables,assignments,specs,frozen,context=load_final()
    output=args.output.resolve()
    fingerprint=start_output(output,__file__,context,dict(kinds=KINDS,arms=ARMS,shap_version=shap.__version__,
        explanation="tree_path_dependent; fitted training path counts; held-out samples; no held-out background"))
    work=[]
    for arm in ARMS:
        for p in PROTOCOLS:
            for r in range(3):
                for f in range(5):
                    part=tables["settings"]
                    choice=part[part.arm.eq(arm)&part.protocol.eq(p)&part.repeat.eq(r)&part.fold.eq(f)&part.strategy.eq("forward10")].iloc[0].to_dict()
                    for kind in KINDS:
                        source=tables["predictions"] if kind==KINDS[0] else frozen[kind]
                        pred=source[source.arm.eq(arm)&source.protocol.eq(p)&source.repeat.eq(r)&source.fold.eq(f)]
                        if kind==KINDS[0]:
                            pred=pred[pred.strategy.eq("forward10")]
                        work.append(((arm,p,r,f,kind),choice if kind==KINDS[0] else specs[kind],pred))
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results=Parallel(n_jobs=args.jobs)(delayed(job)(k,data,assignments,c,p,output,fingerprint) for k,c,p in work)
    tables_out={}
    for term,category in (("feature","feature_rows"),("property","property_rows")):
        folds=pd.DataFrame([dict(**{k:r[k] for k in ("arm","protocol","repeat","fold","kind")},**v) for r in results for v in r[category]])
        repeats,summary=aggregate(folds,term)
        for suffix,frame in (("folds",folds),("repeats",repeats),("summary",summary)):
            frame.to_csv(output / f"{term}_{suffix}.csv",index=False)
        tables_out[term]=summary
    samples=pd.concat([pd.read_csv(r["samples_file"]).assign(**{k:r[k] for k in ("arm","protocol","repeat","fold","kind")}) for r in results],ignore_index=True)
    samples.to_csv(output / "heldout_shap_values.csv.gz",index=False,compression="gzip")
    plots(tables_out["feature"],tables_out["property"],output)
    save_json(output / "verification.json",dict(complete=True,refits=len(results),
        max_prediction_difference=max(r["max_prediction_difference"] for r in results),
        max_additivity_error=max(r["max_additivity_error"] for r in results),source_models_unchanged=True))
    print(tables_out["property"].query('rank <= 3').to_string(index=False),flush=True)


if __name__=="__main__":
    main()
