"""Matched composition-representation ablations on the final 665-row snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from revision_remaining_common import (load_final,add_recipes,reweight,OUTPUT,POOL,RAW22)
from revision_followup_common import (PROTOCOLS,NAMES,start_output,save_json,seed_for,
    clean_inputs,representation,make_model,aggregate_predictions)
from icp_forward_33_comparison import prepare,KEYS
from icp_forward_selection_revision import evaluate_subset,choose_candidate

SCENARIOS=("raw22","precursor33","icp33_plus_fraction")


def feature_frame(frame,data,scenario):
    if scenario=="raw22":
        return frame[RAW22].copy()
    if scenario=="precursor33":
        return reweight(frame,data,"precursor")[POOL]
    if scenario=="icp33_plus_fraction":
        result=frame[POOL].copy()
        result["actual_frac_element1"]=data.actual_frac_element1.to_numpy(float)
        return result
    if scenario=="icp33":
        return frame[POOL].copy()
    raise ValueError(scenario)


def matched_inner(inner,memberships,data,scenario):
    result=[]
    members=pd.DataFrame(memberships)
    for fold,d in enumerate(inner):
        a=data.loc[members.loc[members.inner_fold.eq(fold)&members.role.eq("train"),"sample"]]
        b=data.loc[members.loc[members.inner_fold.eq(fold)&members.role.eq("validation"),"sample"]]
        np.testing.assert_allclose(a.separation_coefficient,d["y"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(b.separation_coefficient,d["z"],atol=1e-12,rtol=0)
        X=feature_frame(pd.DataFrame(d["X"],columns=POOL),a,scenario)
        V=feature_frame(pd.DataFrame(d["V"],columns=POOL),b,scenario)
        result.append(dict(X=X.to_numpy(),V=V.to_numpy(),y=d["y"],z=d["z"],columns=list(X.columns)))
    return result


def job(key,data,assignments,settings,expected,options,output,fingerprint):
    path=output / "checkpoints" / ("_".join(map(str,key))+".json")
    if path.exists():
        result=json.loads(path.read_text(encoding="utf-8"))
        if result["fingerprint"]!=fingerprint:
            raise ValueError("Stale composition checkpoint")
        return result
    train,test,inner,memberships,audit=prepare(key,data,assignments)
    seed=seed_for(key[1:])
    transform=representation("hybrid39").fit(clean_inputs(train))
    X,V=transform.transform(clean_inputs(train)),transform.transform(clean_inputs(test))
    result=dict(zip(KEYS,key),fingerprint=fingerprint,settings=[],predictions=[],tuning=[],
        memberships=memberships,split_audit=audit,reproduction_errors=[])

    def final_fit(name,choice,train_frame,test_frame,compare_strategy=None):
        columns=choice["features"]
        model=make_model(choice["model"],choice["config"],options["final_trees"],seed)
        model.fit(train_frame[columns].to_numpy(),train.separation_coefficient)
        pred=model.predict(test_frame[columns].to_numpy())
        if compare_strategy is not None:
            ref=expected[expected.strategy.eq(compare_strategy)].set_index("sample").loc[test["sample"],"y_pred"]
            np.testing.assert_allclose(pred,ref,atol=1e-10,rtol=0)
            result["reproduction_errors"].append(float(np.max(np.abs(pred-ref))))
        result["settings"].append(dict(scenario=name,features=columns,model=choice["model"],config=choice["config"]))
        result["predictions"].extend(dict(scenario=name,sample=r["sample"],metal_pair=r.metal_pair,
            y_true=float(r.separation_coefficient),y_pred=float(p)) for (_,r),p in zip(test.iterrows(),pred))

    # Reproduce both saved baselines before interpreting representation changes.
    all_choice=settings[settings.strategy.eq("all33")].iloc[0].to_dict()
    ten_choice=settings[settings.strategy.eq("forward10")].iloc[0].to_dict()
    final_fit("icp33",all_choice,X,V,"all33")
    final_fit("selected10_icp",ten_choice,X,V,"forward10")
    final_fit("selected10_precursor",ten_choice,reweight(X,train,"precursor"),reweight(V,test,"precursor"))
    for scenario in SCENARIOS:
        split=matched_inner(inner,memberships,data,scenario)
        columns=split[0]["columns"]
        candidates=[]
        for name in ("RF","ET"):
            for ci,config in enumerate(options["configs"]):
                rows=evaluate_subset(split,list(range(len(columns))),name,config,options["search_trees"],seed)
                candidates.append(dict(model=name,config_id=ci,config=config,n_features=len(columns),features=columns,
                    inner_r2_mean=float(np.mean([r["r2"] for r in rows]))))
                result["tuning"].extend(dict(scenario=scenario,model=name,config_id=ci,**r) for r in rows)
        final_fit(scenario,choose_candidate(candidates),feature_frame(X,train,scenario),feature_frame(V,test,scenario))
    save_json(path,result)
    print(path.stem+" complete",flush=True)
    return result


def report(predictions,output):
    repeats,summary=aggregate_predictions(predictions,["arm","protocol","scenario"])
    comparisons=[]
    for first,second in (("precursor33","icp33"),("raw22","icp33"),("icp33","icp33_plus_fraction"),
                         ("selected10_precursor","selected10_icp")):
        cols=["arm","protocol","repeat","sample"]
        a=predictions[predictions.scenario.eq(first)].set_index(cols).sort_index()
        b=predictions[predictions.scenario.eq(second)].set_index(cols).sort_index()
        pd.testing.assert_index_equal(a.index,b.index)
        np.testing.assert_allclose(a.y_true,b.y_true,atol=1e-12,rtol=0)
        joined=repeats[repeats.scenario.eq(first)].merge(repeats[repeats.scenario.eq(second)],on=["arm","protocol","repeat"],suffixes=("_before","_after"),validate="one_to_one")
        joined["comparison"]=second+" minus "+first
        joined["delta_r2"]=joined.r2_after-joined.r2_before
        joined["delta_rmse"]=joined.rmse_after-joined.rmse_before
        comparisons.append(joined)
    paired=pd.concat(comparisons,ignore_index=True)
    paired_summary=paired.groupby(["arm","protocol","comparison"],as_index=False).agg(
        delta_r2_mean=("delta_r2","mean"),delta_r2_sd=("delta_r2","std"),delta_rmse_mean=("delta_rmse","mean"))
    for name,frame in (("repeat_metrics",repeats),("summary",summary),("paired_repeats",paired),("paired_summary",paired_summary)):
        frame.to_csv(output / (name+".csv"),index=False)
    plt.rcParams.update({"font.family":"Arial","font.size":10,"pdf.fonttype":42})
    order=("raw22","precursor33","icp33","icp33_plus_fraction")
    labels=("Raw 22","Precursor 33","ICP 33","ICP 33 + fraction")
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharey=True)
    fig.subplots_adjust(left=.08,right=.98,bottom=.15,top=.89,hspace=.48,wspace=.18)
    for i,arm in enumerate(("without_bo","with_bo")):
        for ax,p in zip(axes[i],PROTOCOLS):
            part=summary[summary.arm.eq(arm)&summary.protocol.eq(p)].set_index("scenario").loc[list(order)]
            ax.errorbar(range(4),part.r2_mean,yerr=part.r2_sd,fmt="o",color="#247e80",capsize=4)
            for j,r in enumerate(part.itertuples()):
                ax.annotate(f"{r.r2_mean:.3f}",(j,r.r2_mean+r.r2_sd),xytext=(0,6),textcoords="offset points",ha="center",fontsize=9)
            ax.axhline(0,color="#aaaaaa",lw=.7);ax.grid(axis="y",alpha=.16)
            ax.set_xticks(range(4),labels,rotation=25,ha="right")
            ax.set_title(NAMES[p]+(" | +BO" if arm=="with_bo" else " | no BO"))
            ax.set_ylim(min(-.3,float((summary.r2_mean-summary.r2_sd).min())-.1),max(.75,float((summary.r2_mean+summary.r2_sd).max())+.1))
        axes[i,0].set_ylabel("Out-of-fold R2")
    fig.suptitle("Matched precursor versus ICP composition",fontsize=15)
    fig.text(.5,.025,"Same samples, targets, outer/inner splits and tuning grid. Full declared pools; mean +/- SD across three repeats.",ha="center",fontsize=10)
    for ext in ("png","pdf"):
        fig.savefig(output / ("composition_comparison."+ext),dpi=200)
    plt.close(fig)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=10)
    parser.add_argument("--output",type=Path,default=OUTPUT / "composition_bridge")
    args=parser.parse_args()
    data,tables,assignments,_,_,context=load_final()
    data=add_recipes(data)
    output=args.output.resolve()
    fingerprint=start_output(output,__file__,context,dict(scenarios=SCENARIOS,full_pool_tuning=True,
        additional_check="fixed ICP-selected 10 inputs/config, substitute precursor weights and retrain",
        selection_policy="No new forward search in representation ablation",options=context["options"]))
    data[["sample","cohort","metal_pair","element1","element2","actual_frac_element1","nominal_frac_element1","recipe_source","volume1","volume2"]].to_csv(output / "composition_input_audit.csv",index=False)
    work=[]
    for arm in ("without_bo","with_bo"):
        for p in PROTOCOLS:
            for r in range(3):
                for f in range(5):
                    s=tables["settings"];s=s[s.arm.eq(arm)&s.protocol.eq(p)&s.repeat.eq(r)&s.fold.eq(f)]
                    expected=tables["predictions"];expected=expected[expected.arm.eq(arm)&expected.protocol.eq(p)&expected.repeat.eq(r)&expected.fold.eq(f)]
                    work.append(((arm,p,r,f),s,expected))
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results=Parallel(n_jobs=args.jobs)(delayed(job)(k,data,assignments,s,e,context["options"],output,fingerprint) for k,s,e in work)
    for category in ("predictions","settings","tuning","memberships","split_audit"):
        frame=pd.DataFrame([dict(**{k:r[k] for k in KEYS},**v) for r in results for v in r[category]])
        frame.to_csv(output / (category+".csv"),index=False)
        if category=="predictions":
            summary=report(frame,output)
    save_json(output / "verification.json",dict(complete=True,jobs=len(results),baseline_refits=2*len(results),
        max_reproduction_error=max(v for r in results for v in r["reproduction_errors"]),
        all_comparisons_use_matched_targets_tests=True,shared_icp_and_composition_splits_unchanged=True))
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
