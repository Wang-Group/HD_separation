"""Supported portable commands. Archived helper-script CLIs are not entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from inputs import ROOT, REFERENCE, KEYS, read_json, write_json, load_data, start_run, versions, regenerate_splits, resolve_source
from modeling import prepare, refit_job, train_job, summarize, compare_reference, fit_models
from icp_revision_inputs import sha
from icp_forward_bo_ablation import matched_plan
from icp_task1_frozen_sensitivity import consensus_features, tune_job, choose_global_config, task1_only


def package_files():
    for file in sorted(ROOT.rglob("*")):
        relative=file.relative_to(ROOT)
        if not file.is_file() or any(p in ("runs","__pycache__",".pytest_cache",".venv") for p in relative.parts): continue
        if file.name=="package_manifest.json": continue
        yield file


def inventory_check():
    inventory=read_json(ROOT / "source_inventory.json")
    for item in inventory["files"]:
        file=ROOT / item["path"]
        if not file.is_file() or sha(file)!=item["sha256"]: raise ValueError(f"Changed/missing snapshot: {item['path']}")
        if item["role"] in ("input","raw_provenance") and item["path"]!=item["repository_relative_path"]:
            raise ValueError("Raw relative path changed")
    return len(inventory["files"])


def seal():
    inventory_check()
    files={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in package_files()}
    write_json(ROOT / "package_manifest.json",dict(schema="hd-revision-package-v1",files=files,
        total_bytes=sum(p["bytes"] for p in files.values()),main_candidate_pool=33,contact="Cheng Wang, wangchengxmu@xmu.edu.cn"))
    return dict(files=len(files),size_mb=sum(p["bytes"] for p in files.values())/1e6)


def check_seal():
    manifest=ROOT / "package_manifest.json"
    if not manifest.exists(): return dict(sealed=False)
    expected=read_json(manifest)["files"]
    for relative,info in expected.items():
        file=ROOT / relative
        if not file.is_file() or sha(file)!=info["sha256"]: raise ValueError(f"Package integrity error: {relative}")
    return dict(sealed=True,files_checked=len(expected))


def settings(source):
    return pd.DataFrame(read_json(source / "settings.json"))


def keys_for(assignments,limit=0):
    order=["stratified_random","unseen_pair","known_pair_new_composition"]
    keys=[(arm,p,r,f) for arm in ("without_bo","with_bo") for p in order for r in range(3) for f in range(5)]
    wanted=set(assignments[["protocol","repeat","fold"]].itertuples(index=False,name=None))
    if any(k[1:] not in wanted for k in keys): raise ValueError("Incomplete outer assignment grid")
    return keys[:limit] if limit else keys


def split_audit(data,assignments,source,output,pool,compare=True):
    rows=[]; members=[]
    for key in keys_for(assignments):
        _,_,_,inner,audit=prepare(key,data,assignments,pool)
        rows.extend(dict(zip(KEYS,key),**r) for r in audit)
        members.extend(dict(zip(KEYS,key),**r) for r in inner)
    table=pd.DataFrame(members)
    if compare:
        fields=KEYS+["inner_fold","role","sample"]
        old=pd.read_csv(source / "inner_assignments.csv")[fields]
        # Ordering within each partition matters to reproducibility, not merely set equality.
        groups=fields[:-1]
        for frame in (old,table): frame["position"]=frame.groupby(groups,sort=False).cumcount()
        index=groups+["position"]
        pd.testing.assert_frame_equal(table[fields+["position"]].set_index(index).sort_index(),old.set_index(index).sort_index(),check_dtype=False)
    table.to_csv(output / "inner_assignments.csv",index=False)
    pd.DataFrame(rows).to_csv(output / "split_audit.csv",index=False)
    return dict(outer_arms=90,inner_partitions=270,matched_archived_inner_order=compare)


def frozen_fit(key,data,assignments,spec,config,output,fingerprint):
    file=output / "checkpoints" / ("_".join(map(str,key))+".json")
    if file.exists():
        record=read_json(file)
        if record["fingerprint"]!=fingerprint: raise ValueError("Stale frozen fit")
        return record
    original,augmented,test=matched_plan(key[1:],data,assignments)
    train=original if key[0]=="without_bo" else augmented
    result=fit_models(key,train,test,{"reselected_frozen10":{k:spec[k] for k in ("features","config","model")}},
        config["search_options"]["final_trees"],True)
    result["fingerprint"]=fingerprint; write_json(file,result)
    return result


def freeze(data,assignments,source,config,output,fingerprint,jobs,verify):
    features,ranking=consensus_features(settings(source),pd.read_csv(source / "path.csv"))
    members=task1_only(pd.read_csv(source / "inner_assignments.csv"))
    with parallel_config(backend="loky",inner_max_num_threads=1):
        tuned=Parallel(n_jobs=jobs)(delayed(tune_job)((r,f),data,assignments,members,features,
            config["search_options"],output,fingerprint) for r in range(3) for f in range(5))
    scores=pd.DataFrame([v for result in tuned for v in result["scores"]])
    choice,ranking_config=choose_global_config(scores,config["search_options"]["configs"])
    spec=dict(features=features,model=choice["model"],config=choice["config"],trees=config["search_options"]["final_trees"])
    write_json(output / "frozen_specification.json",spec)
    ranking.to_csv(output / "consensus_features.csv",index=False)
    ranking_config.to_csv(output / "configuration_ranking.csv",index=False)
    scores.to_csv(output / "tuning_scores.csv",index=False)
    if verify and spec!=read_json(REFERENCE / "reselected_frozen10.json"): raise ValueError("Frozen specification differs from reference")
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results=Parallel(n_jobs=jobs)(delayed(frozen_fit)(k,data,assignments,spec,config,output,fingerprint) for k in keys_for(assignments))
    summarize(results,output,33)
    return dict(jobs=len(results),retrospective_sensitivity=True,independent_nested_estimate=False,
        max_prediction_error=compare_reference(results,source) if verify else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("audit","refit","train","freeze","composition","figures","targets-template","test","seal"))
    parser.add_argument("--pool",type=int,choices=(33,39),default=33)
    parser.add_argument("--jobs",type=int,default=10)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source",type=Path,help="Package-relative reference/pool33 or a fresh training run")
    parser.add_argument("--targets",type=Path,help="Package-local replacement 665-row sample/y CSV")
    parser.add_argument("--regenerate-splits",action="store_true")
    parser.add_argument("--smoke",action="store_true",help="Two-step, low-tree training test; never scientific results")
    parser.add_argument("--limit",type=int,default=0,help="First N outer jobs only; partial output")
    parser.add_argument("--no-shap",action="store_true")
    parser.add_argument("--bootstrap",type=int,default=1000)
    args=parser.parse_args()
    if args.jobs<1 or args.limit<0 or args.bootstrap<20: parser.error("Invalid jobs/limit/bootstrap")
    for name in ("source","targets","output"):
        path=getattr(args,name)
        if path is not None:
            path=(ROOT / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_relative_to(ROOT): parser.error(f"--{name} must stay inside the self-contained package")
            setattr(args,name,path)
    if args.command=="seal": print(json.dumps(seal(),indent=2)); return
    seal_result=check_seal(); inventory_check()
    if args.command=="test":
        import pytest
        (ROOT / "runs").mkdir(exist_ok=True)
        selected=[ROOT / "workflow/test_portable.py"]+[ROOT / "hd_separation_ml" / (n+".py") for n in (
            "test_revision_remaining","test_revision_followups","test_icp_forward_33_comparison",
            "test_icp_forward_bo_ablation","test_icp_task1_frozen_sensitivity")]
        raise SystemExit(pytest.main([*(str(p) for p in selected),"-q","--junitxml="+str(ROOT / "runs/test_results.xml")]))
    if args.targets and args.command not in ("train","freeze","composition","figures","audit"):
        parser.error("Changed targets require fresh train/freeze/composition/figures, not archived refit")
    if args.targets and args.command in ("freeze","composition") and args.source is None:
        parser.error("Changed targets require --source pointing to their newly trained 33-candidate run")
    if args.command in ("freeze","composition") and args.pool!=33:
        parser.error("Use pool 33 for the primary frozen/composition follow-up")
    data,audit=load_data(args.targets)
    output,fingerprint=start_run(args,data)
    config=read_json(ROOT / "config.json")
    source=resolve_source(args.source,args.pool)
    assignments=pd.read_csv(REFERENCE / "outer_assignments.csv")
    if args.regenerate_splits or args.targets:
        assignments=regenerate_splits(data,output)
        if not args.targets:
            fields=["protocol","repeat","fold","role","sample"]
            old=pd.read_csv(REFERENCE / "outer_assignments.csv")
            pd.testing.assert_frame_equal(assignments[fields].reset_index(drop=True),old[fields].reset_index(drop=True),check_dtype=False)
    assignments.to_csv(output / "outer_assignments.csv",index=False)
    data.to_csv(output / "analysis_dataset.csv",index=False)
    audit.to_csv(output / "inclusion_audit.csv",index=False)
    keys=keys_for(assignments,args.limit or (1 if args.smoke else 0))
    status=dict(command=args.command,complete=False,smoke=args.smoke,partial=len(keys)!=90,
        initial=int(data.cohort.eq("initial").sum()),bo=int(data.cohort.eq("BO").sum()),**seal_result)
    if args.command=="audit":
        status.update(split_audit(data,assignments,source,output,args.pool,compare=not bool(args.targets)),versions=versions())
    elif args.command=="targets-template":
        data[["sample","separation_coefficient"]].to_csv(output / "targets.csv",index=False)
    elif args.command in ("refit","train"):
        if args.command=="refit":
            if versions()!=config["versions"]: raise ValueError("Exact replay requires the documented ML runtime")
            with parallel_config(backend="loky",inner_max_num_threads=1):
                results=Parallel(n_jobs=args.jobs)(delayed(refit_job)(k,data,assignments,source,config,output,fingerprint,not args.no_shap) for k in keys)
            status["max_prediction_error"]=compare_reference(results,source)
        else:
            with parallel_config(backend="loky",inner_max_num_threads=1):
                results=Parallel(n_jobs=args.jobs)(delayed(train_job)(k,data,assignments,config,args.pool,output,fingerprint,args.smoke) for k in keys)
            if not args.targets and not args.smoke: status["max_prediction_error"]=compare_reference(results,source)
        summarize(results,output,args.pool); status["jobs"]=len(results)
    elif args.command=="freeze":
        status.update(freeze(data,assignments,source,config,output,fingerprint,args.jobs,verify=not bool(args.targets)))
    elif args.command=="composition":
        from analysis import composition
        status.update(composition(keys,data,assignments,source,config,output,fingerprint,args.jobs))
    elif args.command=="figures":
        from analysis import descriptor_figures
        status.update(descriptor_figures(data,output,args.bootstrap))
    status["complete"]=True
    status["scientific_full_run"]=not args.smoke and not status["partial"]
    write_json(output / "verification.json",status)
    print(json.dumps(status,indent=2)); print(output)


if __name__=="__main__":
    main()
