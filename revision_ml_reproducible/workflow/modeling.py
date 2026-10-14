"""Same numerical methods as the archived revision, without historical-run loaders."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from inputs import KEYS, REFERENCE, read_json, write_json
from icp_electrochemical_exclusion import FEATURES, clean_inputs
from icp_bo_augmentation_cv import representation
from icp_forward_bo_ablation import matched_plan, no_bo_inner, seed_for
from icp_forward_three_tasks import check_split, plan_for, forward_path
from icp_two_prediction_tasks import inner_partitions
from icp_forward_selection_revision import evaluate_subset, select_models, choose_candidate
from icp_nested_group_cv import make_model
from revision_followup_common import aggregate_predictions
from revision_shap33 import aggregate as shap_aggregate, plots as shap_plots

REMOVED=("Frequency_D","delta_ZPE_H")
POOL33=[c for c in FEATURES if c not in {f for b in REMOVED for f in (b+"_element1",b+"_element2","icp_weighted_"+b)}]


def pool_columns(pool):
    return POOL33 if pool==33 else FEATURES


def prepare(key,data,assignments,pool):
    original,augmented,test=matched_plan(key[1:],data,assignments)
    train=original if key[0]=="without_bo" else augmented
    if key[0]=="without_bo":
        splits=no_bo_inner(original,key[1],seed_for(key[1:]))
    else:
        groups,_=plan_for(key[1:],data,assignments)
        splits=((a,b,key[1]) for a,b in inner_partitions(groups["original"],groups["augmented"],key[1],True,seed_for(key[1:])))
    inner=[]; members=[]; audits=[dict(level="outer",inner_fold=-1,**check_split(train,test,key[1]))]
    columns=pool_columns(pool)
    for fold,(a,b,protocol) in enumerate(splits):
        if (set(a["sample"])|set(b["sample"]))&set(test["sample"]): raise ValueError("Outer test entered selection")
        if (set(a["sample"])|set(b["sample"]))-set(train["sample"]): raise ValueError("Inner samples outside training")
        transform=representation("hybrid39").fit(clean_inputs(a))
        X=transform.transform(clean_inputs(a))[columns]; V=transform.transform(clean_inputs(b))[columns]
        inner.append(dict(X=X.to_numpy(),V=V.to_numpy(),y=a.separation_coefficient.to_numpy(),z=b.separation_coefficient.to_numpy(),columns=columns))
        audits.append(dict(level="inner",inner_fold=fold,**check_split(a,b,protocol)))
        for role,frame in (("train",a),("validation",b)):
            members.extend(dict(inner_fold=fold,role=role,sample=r["sample"],tuning_protocol=protocol) for _,r in frame.iterrows())
    return train,test,inner,members,audits


def selected_settings(source,key):
    table=pd.DataFrame(read_json(source / "settings.json"))
    for field,value in zip(KEYS,key): table=table[table[field].eq(value)]
    return table


def fit_models(key,train,test,choices,trees,explain):
    transform=representation("hybrid39").fit(clean_inputs(train))
    X=transform.transform(clean_inputs(train)); V=transform.transform(clean_inputs(test))
    result=dict(zip(KEYS,key),predictions=[],settings=[],shap=[])
    for name,choice in choices.items():
        selected=choice["features"]
        if len(set(selected))!=len(selected) or set(selected)-set(FEATURES): raise ValueError("Invalid feature list")
        model=make_model(choice["model"],choice["config"],trees,seed_for(key[1:]))
        model.fit(X[selected].to_numpy(),train.separation_coefficient)
        pred=model.predict(V[selected].to_numpy())
        result["settings"].append(dict(strategy=name,**choice))
        result["predictions"].extend(dict(strategy=name,sample=r["sample"],metal_pair=r.metal_pair,
            y_true=float(r.separation_coefficient),y_pred=float(p)) for (_,r),p in zip(test.iterrows(),pred))
        if explain and name in ("forward10","original_frozen10","reselected_frozen10"):
            explainer=shap.TreeExplainer(model,feature_perturbation="tree_path_dependent")
            values=np.asarray(explainer.shap_values(V[selected].to_numpy(),check_additivity=True))
            base=float(np.asarray(explainer.expected_value).ravel()[0])
            np.testing.assert_allclose(values.sum(axis=1)+base,pred,atol=1e-6,rtol=0)
            result["shap"].extend(dict(strategy=name,sample=r["sample"],feature=c,value=float(V.iloc[i][c]),
                shap_value=float(values[i,j]),base_value=base,y_pred=float(pred[i]))
                for i,(_,r) in enumerate(test.iterrows()) for j,c in enumerate(selected))
    return result


def refit_job(key,data,assignments,source,config,output,fingerprint,explain,include_frozen=True):
    file=output / "checkpoints" / ("_".join(map(str,key))+".json")
    if file.exists():
        record=read_json(file)
        if record["fingerprint"]!=fingerprint: raise ValueError("Stale checkpoint")
        return record
    original,augmented,test=matched_plan(key[1:],data,assignments)
    train=original if key[0]=="without_bo" else augmented
    choices={r["strategy"]:{k:r[k] for k in ("features","config","model")} for r in selected_settings(source,key).to_dict("records")
             if r["strategy"].startswith("all") or r["strategy"]=="forward10"}
    if include_frozen:
        for name in ("original_frozen10","reselected_frozen10"):
            spec=read_json(REFERENCE / (name+".json"))
            choices[name]={k:spec[k] for k in ("features","config","model")}
    result=fit_models(key,train,test,choices,config["search_options"]["final_trees"],explain)
    result["fingerprint"]=fingerprint
    result["split_audit"]=[check_split(train,test,key[1])]
    write_json(file,result); print(file.stem+" refitted",flush=True)
    return result


def train_job(key,data,assignments,config,pool,output,fingerprint,smoke=False):
    name="_".join(map(str,key)); file=output / "checkpoints" / (name+".json")
    if file.exists():
        result=read_json(file)
        if result["fingerprint"]!=fingerprint: raise ValueError("Stale training checkpoint")
        return result
    train,test,inner,members,audit=prepare(key,data,assignments,pool)
    seed=seed_for(key[1:]); columns=pool_columns(pool)
    path_options=dict(config["forward_options"]); options=dict(config["search_options"])
    if smoke:
        path_options.update(max_features=2,selection_trees=4)
        options.update(search_trees=4,final_trees=8,configs=options["configs"][:2])
    # No historical forward/tuning cache is consumed by this scratch workflow.
    state=forward_path(inner,path_options,seed,output / "paths" / (name+".json"),fingerprint)
    tune_file=output / "tuning" / (name+".json")
    tuned=read_json(tune_file) if tune_file.exists() else dict(fingerprint=fingerprint,candidates=[],scores=[],completed=[])
    if tuned["fingerprint"]!=fingerprint: raise ValueError("Stale tuning checkpoint")
    for step in state["steps"]+[dict(n_features=pool,mask=list(range(pool)))]:
        size=step["n_features"]
        if size in tuned["completed"]: continue
        for model in ("RF","ET"):
            for ci,parameters in enumerate(options["configs"]):
                rows=evaluate_subset(inner,step["mask"],model,parameters,options["search_trees"],seed)
                tuned["candidates"].append(dict(model=model,config_id=ci,config=parameters,n_features=size,
                    features=[columns[i] for i in step["mask"]],inner_r2_mean=float(np.mean([r["r2"] for r in rows])),
                    inner_r2_std=float(np.std([r["r2"] for r in rows],ddof=1))))
                tuned["scores"].extend(dict(n_features=size,model=model,config_id=ci,**r) for r in rows)
        tuned["completed"].append(size); write_json(tune_file,tuned)
    reduced=[c for c in tuned["candidates"] if c["n_features"]!=pool]
    full=[c for c in tuned["candidates"] if c["n_features"]==pool]
    if smoke:
        choices={"smoke_best":choose_candidate(reduced),"all"+str(pool):choose_candidate(full)}
    else:
        choices,_=select_models(reduced,full)
        choices["all"+str(pool)]=choices.pop("all39")
    result=fit_models(key,train,test,choices,options["final_trees"],not smoke)
    result.update(fingerprint=fingerprint,inner_assignments=members,split_audit=audit,tuning_scores=tuned["scores"],
        path=[{k:v for k,v in s.items() if k!="inner_scores"} for s in state["steps"]])
    write_json(file,result); print(name+" trained",flush=True)
    return result


def flatten(results,category):
    return pd.DataFrame([dict(**{k:r[k] for k in KEYS},**v) for r in results for v in r.get(category,[])])


def summarize(results,output,pool):
    for category in ("predictions","path","inner_assignments","tuning_scores","split_audit","shap"):
        frame=flatten(results,category)
        if len(frame): frame.to_csv(output / (category+(".csv.gz" if category=="shap" else ".csv")),index=False)
    settings=flatten(results,"settings")
    write_json(output / "settings.json",settings.to_dict("records"))
    predictions=flatten(results,"predictions")
    repeats,summary=aggregate_predictions(predictions,["arm","protocol","strategy"])
    repeats.to_csv(output / "repeat_metrics.csv",index=False); summary.to_csv(output / "summary.csv",index=False)
    frequencies=[]
    for (arm,protocol,strategy),part in settings.groupby(["arm","protocol","strategy"]):
        for feature in pool_columns(pool):
            count=sum(feature in f for f in part.features)
            frequencies.append(dict(arm=arm,protocol=protocol,strategy=strategy,feature=feature,selected_fits=count,total_fits=len(part)))
    pd.DataFrame(frequencies).to_csv(output / "feature_selection_frequency.csv",index=False)
    s=flatten(results,"shap")
    if len(s) and pool==33:
        folds=[]; families=[]
        for key,part in s.groupby(KEYS+["strategy"]):
            common=dict(zip(KEYS+['kind'],key))
            common["kind"]={"forward10":"nested33_forward10"}.get(common["kind"],common["kind"])
            pivot=part.pivot(index="sample",columns="feature",values="shap_value").reindex(columns=POOL33,fill_value=0.)
            for c in POOL33: folds.append(dict(**common,feature=c,n=len(pivot),abs_sum=float(pivot[c].abs().sum())))
            bases=[c[len("icp_weighted_"):] for c in POOL33 if c.startswith("icp_weighted_")]
            for b in bases:
                columns=[b+"_element1",b+"_element2","icp_weighted_"+b]
                families.append(dict(**common,property=b,n=len(pivot),abs_sum=float(pivot[columns].sum(axis=1).abs().sum())))
        _,f=shap_aggregate(pd.DataFrame(folds),"feature"); _,p=shap_aggregate(pd.DataFrame(families),"property")
        f.to_csv(output / "feature_shap_summary.csv",index=False); p.to_csv(output / "property_shap_summary.csv",index=False)
        if set(p.kind)=={"nested33_forward10","original_frozen10","reselected_frozen10"} and len(results)==90:
            shap_plots(f,p,output)
    return summary


def compare_reference(results,source):
    pred=flatten(results,"predictions")
    errors=[]
    index=KEYS+["sample"]
    for strategy,part in pred.groupby("strategy"):
        if strategy in ("original_frozen10","reselected_frozen10"):
            ref=pd.read_csv(REFERENCE / (strategy+"_predictions.csv"))
        else:
            ref=pd.read_csv(source / "predictions.csv")
            ref=ref[ref.strategy.eq(strategy)]
        a=part.set_index(index).sort_index(); b=ref.set_index(index).loc[a.index].sort_index()
        np.testing.assert_allclose(a.y_true,b.y_true,atol=1e-12,rtol=0)
        np.testing.assert_allclose(a.y_pred,b.y_pred,atol=1e-10,rtol=0)
        errors.append(float(np.max(np.abs(a.y_pred-b.y_pred))))
    return max(errors)
