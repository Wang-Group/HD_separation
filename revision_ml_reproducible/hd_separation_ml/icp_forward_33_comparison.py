"""Matched 39-versus-33 feature-pool comparison with fold-local forward selection.

Only D vibration and H/D ZPE are removed, in M1/M2/weighted forms. Exact
historical inner-CV evaluations are reusable only for identical ordered inputs,
fold membership, estimator settings and seeds. Changed subsets are refitted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from revision_followup_common import (ROOT,SOURCE,FEATURES,PROTOCOLS,load_current,start_output,
    clean_inputs,representation,matched_plan,seed_for,make_model,save_json,sha)
from icp_electrochemical_exclusion import check_path
from icp_forward_bo_ablation import no_bo_inner
from icp_forward_three_tasks import check_split,plan_for
from icp_forward_selection_revision import PATH_CONFIG,evaluate_subset,select_models
from icp_two_prediction_tasks import inner_partitions
from icp_identity_diagnostics import property_permutations

REMOVED_BASES = ("Frequency_D","delta_ZPE_H")
REMOVED = [f for f in FEATURES if f in {name for b in REMOVED_BASES
    for name in (b+"_element1",b+"_element2","icp_weighted_"+b)}]
POOL = [f for f in FEATURES if f not in REMOVED]
PATH_AUDIT = ROOT / "outputs/electrochemical_exclusion_verified_20261013/forward_path_audit.csv"
OUTPUT = ROOT / "outputs/forward33_comparison_20261013"
KEYS = ["arm","protocol","repeat","fold"]
CATEGORIES = ["predictions","settings","permutation","inner_assignments","tuning_scores","path"]


def source_records():
    provenance=pd.read_csv(SOURCE / "checkpoint_provenance.csv")
    paths=pd.read_csv(PATH_AUDIT)
    merged=provenance.merge(paths[KEYS+["path","sha256"]],on=KEYS,suffixes=("_checkpoint","_path"),validate="one_to_one")
    if len(merged)!=90 or merged[KEYS].duplicated().any():
        raise ValueError("Expected 90 source models and paths")
    records,files={},[]
    for row in merged.itertuples(index=False):
        checkpoint,path=Path(row.checkpoint),Path(row.path)
        if sha(checkpoint)!=row.sha256_checkpoint or sha(path)!=row.sha256_path:
            raise ValueError("Changed cached source")
        old=json.loads(checkpoint.read_text(encoding="utf-8"))
        state=json.loads(path.read_text(encoding="utf-8"))
        check_path(state,old["fingerprint"],16)
        if [[FEATURES[i] for i in s["mask"]] for s in state["steps"]] != [[FEATURES[i] for i in s["mask"]] for s in old["path"]]:
            raise ValueError("Source path does not match model checkpoint")
        records[(row.arm,row.protocol,int(row.repeat),int(row.fold))]=dict(checkpoint=str(checkpoint),path=str(path))
        files.extend([checkpoint,path])
    return records,files


def prepare(key,data,assignments):
    arm,protocol,repeat,fold=key
    original,augmented,test=matched_plan((protocol,repeat,fold),data,assignments)
    train=augmented if arm=="with_bo" else original
    seed=seed_for((protocol,repeat,fold))
    if arm=="without_bo":
        splits=no_bo_inner(original,protocol,seed)
    else:
        groups,_=plan_for((protocol,repeat,fold),data,assignments)
        splits=((a,b,protocol) for a,b in inner_partitions(groups["original"],groups["augmented"],protocol,True,seed))
    inner,membership,audit=[],[],[]
    audit.append(dict(level="outer",inner_fold=-1,**check_split(train,test,protocol)))
    for fold,(a,b,tuning_protocol) in enumerate(splits):
        if (set(a["sample"])|set(b["sample"])) & set(test["sample"]):
            raise ValueError("Test rows entered selection")
        if (set(a["sample"])|set(b["sample"]))-set(train["sample"]):
            raise ValueError("Inner rows outside outer training")
        audit.append(dict(level="inner",inner_fold=fold,**check_split(a,b,tuning_protocol)))
        transform=representation("hybrid39").fit(clean_inputs(a))
        X,V=transform.transform(clean_inputs(a))[POOL],transform.transform(clean_inputs(b))[POOL]
        if not np.isfinite(X.to_numpy()).all() or not np.isfinite(V.to_numpy()).all():
            raise ValueError("Nonfinite transformed features")
        inner.append(dict(X=X.to_numpy(),V=V.to_numpy(),y=a.separation_coefficient.to_numpy(),
            z=b.separation_coefficient.to_numpy(),columns=POOL))
        for role,rows in (("train",a),("validation",b)):
            membership.extend(dict(inner_fold=fold,role=role,sample=row["sample"],cohort=row.cohort,
                metal_pair=row.metal_pair,icp_group=row.icp_group,composition_group=row.composition_group,
                tuning_protocol=tuning_protocol) for _,row in rows.iterrows())
    return train,test,inner,membership,audit


def trial_key(trial,columns=FEATURES):
    return tuple(columns[i] for i in trial["mask"])


def validate_cached_inner(inner,membership,old,state,seed,options):
    fields=["inner_fold","role","sample"]
    new=pd.DataFrame(membership)[fields]
    previous=pd.DataFrame(old["inner_assignments"])[fields]
    pd.testing.assert_frame_equal(new.reset_index(drop=True),previous.reset_index(drop=True),check_dtype=False)
    cached=next(t for t in state["trials"] if t["n_features"]==1 and t["added_feature"] in POOL)
    mask=[POOL.index(cached["added_feature"])]
    rows=[dict(model=name,**r) for name in ("RF","ET")
          for r in evaluate_subset(inner,mask,name,PATH_CONFIG,options["selection_trees"],seed)]
    actual=pd.DataFrame(rows).set_index(["model","inner_fold"]).sort_index()
    expected=pd.DataFrame(cached["inner_scores"]).set_index(["model","inner_fold"]).sort_index()
    difference=float(np.max(np.abs(actual[["r2","rmse","mae"]].to_numpy()-expected[["r2","rmse","mae"]].to_numpy())))
    np.testing.assert_allclose(actual[["r2","rmse","mae"]],expected[["r2","rmse","mae"]],atol=1e-10,rtol=0)
    return difference


def reduced_path(inner,options,seed,old_state,file,fingerprint,progress):
    if file.exists():
        state=json.loads(file.read_text(encoding="utf-8"))
        if state["fingerprint"]!=fingerprint:
            raise ValueError("Stale reduced forward path")
    else:
        state=dict(fingerprint=fingerprint,selected=[],steps=[],trials=[])
    cache={trial_key(r):r for r in old_state["trials"]}
    for size in range(len(state["steps"])+1,options["max_features"]+1):
        trials=[]
        for index,feature in enumerate(POOL):
            if index in state["selected"]:
                continue
            mask=state["selected"]+[index]
            ordered=tuple(POOL[i] for i in mask)
            previous=cache.get(ordered)
            if previous is None:
                rows=[dict(model=name,**r) for name in ("RF","ET")
                    for r in evaluate_subset(inner,mask,name,PATH_CONFIG,options["selection_trees"],seed)]
            else:
                rows=previous["inner_scores"]
            trials.append(dict(n_features=size,added_feature=feature,added_index=index,mask=mask,
                mean_r2=float(np.mean([r["r2"] for r in rows])),inner_scores=rows,reused_identical_subset=previous is not None))
        winner=sorted(trials,key=lambda r:(-r["mean_r2"],r["added_index"]))[0]
        state["selected"]=winner["mask"]
        state["steps"].append(winner); state["trials"].extend(trials)
        save_json(file,state)
        save_json(progress,dict(stage="forward",step=size,new_trials=sum(not r["reused_identical_subset"] for r in state["trials"]),
            reused_trials=sum(r["reused_identical_subset"] for r in state["trials"]),updated=time.time()))
        if size%4==0:
            print(f"{file.stem}: reduced forward {size}/16",flush=True)
    return state


def job(key,data,assignments,source,forward_options,search_options,output,fingerprint):
    arm,protocol,repeat,fold=key
    name=f"{arm}_{protocol}_r{repeat}_f{fold}"
    destination=output / "checkpoints" / (name+".json")
    progress=output / "progress" / (name+".json")
    if destination.exists():
        result=json.loads(destination.read_text(encoding="utf-8"))
        if result["fingerprint"]!=fingerprint:
            raise ValueError("Stale reduced model")
        return result
    begin=time.perf_counter()
    train,test,inner,membership,audit=prepare(key,data,assignments)
    old=json.loads(Path(source["checkpoint"]).read_text(encoding="utf-8"))
    old_state=json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    seed=seed_for((protocol,repeat,fold))
    cache_error=validate_cached_inner(inner,membership,old,old_state,seed,forward_options)
    state=reduced_path(inner,forward_options,seed,old_state,output / "paths" / (name+".json"),fingerprint,progress)
    tuning_file=output / "tuning" / (name+".json")
    if tuning_file.exists():
        tuned_state=json.loads(tuning_file.read_text(encoding="utf-8"))
        if tuned_state["fingerprint"]!=fingerprint:
            raise ValueError("Stale tuning checkpoint")
    else:
        tuned_state=dict(fingerprint=fingerprint,candidates=[],tuning_scores=[],completed_sizes=[])
    old_tuning=pd.DataFrame(old["tuning_scores"])
    steps=state["steps"]+[dict(n_features=33,mask=list(range(33)))]
    for step in steps:
        size=step["n_features"]
        if size in tuned_state["completed_sizes"]:
            continue
        chosen=[POOL[i] for i in step["mask"]]
        previous_step=next((s for s in old_state["steps"] if s["n_features"]==size and list(trial_key(s))==chosen),None)
        for model in ("RF","ET"):
            for ci,config in enumerate(search_options["configs"]):
                previous=old_tuning[(old_tuning.model==model)&(old_tuning.config_id==ci)&(old_tuning.n_features==size)] if previous_step is not None else pd.DataFrame()
                if len(previous)==3:
                    rows=previous[["inner_fold","r2","rmse","mae"]].to_dict("records")
                    reused=True
                else:
                    rows=evaluate_subset(inner,step["mask"],model,config,search_options["search_trees"],seed)
                    reused=False
                tuned_state["candidates"].append(dict(model=model,n_features=size,config_id=ci,config=config,features=chosen,
                    inner_r2_mean=float(np.mean([r["r2"] for r in rows])),inner_r2_std=float(np.std([r["r2"] for r in rows],ddof=1)),
                    reused_identical_subset=reused))
                tuned_state["tuning_scores"].extend(dict(model=model,n_features=size,config_id=ci,reused_identical_subset=reused,**r) for r in rows)
        tuned_state["completed_sizes"].append(size)
        save_json(tuning_file,tuned_state)
        save_json(progress,dict(stage="tuning",step=size,updated=time.time()))
    path_candidates=[c for c in tuned_state["candidates"] if c["n_features"]!=33]
    full_candidates=[c for c in tuned_state["candidates"] if c["n_features"]==33]
    choices,threshold=select_models(path_candidates,full_candidates)
    choices["all33"]=choices.pop("all39")
    transform=representation("hybrid39").fit(clean_inputs(train))
    X,V=transform.transform(clean_inputs(train)),transform.transform(clean_inputs(test))
    result=dict(zip(KEYS,key),fingerprint=fingerprint,predictions=[],settings=[],permutation=[],
        inner_assignments=membership,split_audit=audit,tuning_scores=tuned_state["tuning_scores"],
        path=[{k:v for k,v in s.items() if k!="inner_scores"} for s in state["steps"]],
        cached_inner_replay_error=cache_error,n_new_forward_trials=sum(not t["reused_identical_subset"] for t in state["trials"]),
        n_reused_forward_trials=sum(t["reused_identical_subset"] for t in state["trials"]))
    cache={}
    for strategy,choice in choices.items():
        selected=choice["features"]
        if not set(selected)<=set(POOL):
            raise ValueError("Removed descriptor entered final model")
        cache_key=(choice["model"],choice["config_id"],tuple(selected))
        if cache_key not in cache:
            cache[cache_key]=make_model(choice["model"],choice["config"],search_options["final_trees"],seed).fit(
                X[selected].to_numpy(),train.separation_coefficient)
        estimator=cache[cache_key]; pred=estimator.predict(V[selected].to_numpy())
        result["settings"].append(dict(strategy=strategy,**{k:choice[k] for k in
            ("model","n_features","features","config","inner_r2_mean","inner_r2_std")},one_se_threshold=threshold,
            n_train_initial=int(train.cohort.eq("initial").sum()),n_train_bo=int(train.cohort.eq("BO").sum())))
        result["predictions"].extend(dict(strategy=strategy,sample=row["sample"],metal_pair=row.metal_pair,
            y_true=float(row.separation_coefficient),y_pred=float(p)) for (_,row),p in zip(test.iterrows(),pred))
        if strategy in ("forward_best","forward_one_se"):
            result["permutation"].extend(dict(strategy=strategy,**r) for r in property_permutations(estimator,V[selected],test,pred,seed))
    reference=pd.DataFrame(old["predictions"])
    for strategy,p in pd.DataFrame(result["predictions"]).groupby("strategy"):
        old_strategy="all39" if strategy=="all33" else strategy
        ref=reference[reference.strategy.eq(old_strategy)].set_index("sample")
        if set(ref.index)!=set(p["sample"]):
            raise ValueError("Mismatched outer test samples")
        np.testing.assert_allclose(p.y_true,ref.loc[p["sample"],"y_true"],atol=1e-12,rtol=0)
    result["elapsed_seconds"]=time.perf_counter()-begin
    save_json(destination,result)
    save_json(progress,dict(stage="complete",step=16,elapsed_seconds=result["elapsed_seconds"],updated=time.time()))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--jobs",type=int,default=10)
    parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args()
    if args.jobs<1:
        parser.error("jobs must be positive")
    if len(POOL)!=33 or len(REMOVED)!=6:
        raise ValueError("Incorrect reduced pool")
    data,tables,assignments,_,_,context=load_current()
    records,files=source_records()
    source_manifest=json.loads((SOURCE / "analysis_manifest.json").read_text(encoding="utf-8"))
    forward_options=source_manifest["options"]
    output=args.output.resolve()
    fingerprint=start_output(output,__file__,context,dict(candidate_features=POOL,removed_features=REMOVED,
        representative="Frequency_H, separately for M1, M2 and ICP-weighted",forward_options=forward_options,
        search_options=context["options"],cache="Exact ordered-subset scores with identical splits/seeds/settings only; first candidate replayed in each job",
        design="Matched nested 33-versus-39, three tasks, three repeats x five folds, with/without BO",
        selection="Forward path 1-16; tune RF/ET; report full pool, forward10, inner-best and one-SE",
        scope="No other correlation pruning; no operating covariates or potential shifts"),
        extra_files=[PATH_AUDIT,SOURCE / "analysis_manifest.json",*files])
    for folder in ("paths","tuning","progress"):
        (output / folder).mkdir(exist_ok=True)
    audit=[]
    for key in records:
        _,_,_,membership,rows=prepare(key,data,assignments)
        old=json.loads(Path(records[key]["checkpoint"]).read_text(encoding="utf-8"))
        fields=["inner_fold","role","sample"]
        pd.testing.assert_frame_equal(pd.DataFrame(membership)[fields],pd.DataFrame(old["inner_assignments"])[fields],check_dtype=False)
        audit.extend(dict(zip(KEYS,key),**row) for row in rows)
    pd.DataFrame(audit).to_csv(output / "split_audit.csv",index=False)
    pd.DataFrame(dict(feature=FEATURES,in_39=True,in_33=[f in POOL for f in FEATURES])).to_csv(output / "candidate_pool_comparison.csv",index=False)
    print("AUDIT: 90 outer training arms and 270 inner splits match the original runs; only six descriptors removed",flush=True)
    if args.audit_only:
        return
    start=time.perf_counter(); results=[]
    with parallel_config(backend="loky",inner_max_num_threads=1):
        for result in Parallel(n_jobs=args.jobs,return_as="generator_unordered")(
            delayed(job)(key,data,assignments,records[key],forward_options,context["options"],output,fingerprint) for key in records):
            results.append(result)
            print(f"33-feature comparison: {len(results)}/90 complete; elapsed {time.perf_counter()-start:.1f}s",flush=True)
    for category in CATEGORIES:
        pd.DataFrame([dict(**{k:r[k] for k in KEYS},**row) for r in results for row in r[category]]).to_csv(output / f"{category}.csv",index=False)
    save_json(output / "verification.json",dict(complete=True,jobs=len(results),final_strategies=4,
        removed_features_never_selected=True,matched_outer_tests=True,matched_inner_memberships=True,
        max_cached_inner_replay_error=max(r["cached_inner_replay_error"] for r in results),
        new_forward_trials=sum(r["n_new_forward_trials"] for r in results),reused_forward_trials=sum(r["n_reused_forward_trials"] for r in results),
        elapsed_seconds=time.perf_counter()-start,original_models_unchanged=True))
    pd.DataFrame([dict(**{k:r[k] for k in KEYS},checkpoint=str(output / "checkpoints" / (f"{r['arm']}_{r['protocol']}_r{r['repeat']}_f{r['fold']}.json")),
        fingerprint=fingerprint) for r in results]).to_csv(output / "checkpoint_provenance.csv",index=False)
    print("All 33-feature nested comparisons complete",flush=True)


if __name__=="__main__":
    main()
