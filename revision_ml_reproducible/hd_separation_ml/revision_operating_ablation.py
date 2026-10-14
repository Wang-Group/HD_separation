"""Matched nested tuning of intrinsic, operating, and potential-shifted inputs."""
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

from revision_followup_common import (ROOT, FEATURES, label, load_current, start_output, save_json,
    matched_plan, seed_for, representation, clean_inputs, make_model, aggregate_predictions)
from revision_correlations import COVARIATES, shifted_features
from icp_forward_bo_ablation import no_bo_inner
from icp_forward_selection_revision import evaluate_subset, choose_candidate
from icp_forward_three_tasks import check_split

CORRELATIONS = ROOT / "outputs/reviewer_stage2_correlations_20261013_v2"
CASES = {"intrinsic39": (False,False), "intrinsic39_operating": (False,True),
         "shifted39": (True,False), "shifted39_operating": (True,True)}
TITLES = {"intrinsic39":"Intrinsic 39", "intrinsic39_operating":"Intrinsic 39 + operating",
          "shifted39":"Potential-shifted 39", "shifted39_operating":"Shifted 39 + operating"}


def inputs(transform,data,case):
    shift, operating = CASES[case]
    frame = transform.transform(clean_inputs(data))
    if shift:
        frame = shifted_features(frame,data.potential_offset)
    if operating:
        for c in COVARIATES:
            frame[c] = data[c].to_numpy(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Missing transformed or operating input")
    return frame


def job(key,case,data,assignments,context,output,fingerprint,reference):
    path = output / "checkpoints" / f"{key[0]}_r{key[1]}_f{key[2]}_{case}.json"
    if path.exists():
        result=json.loads(path.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Stale operating ablation")
        return result
    train,_,test = matched_plan(key,data,assignments)
    seed = seed_for(key)
    inner,memberships=[],[]
    for fold,(tr,va,protocol) in enumerate(no_bo_inner(train,key[0],seed)):
        check_split(tr,va,protocol)
        if (set(tr["sample"]) | set(va["sample"])) & set(test["sample"]):
            raise ValueError("Outer test entered tuning")
        transform = representation("hybrid39").fit(clean_inputs(tr))
        X,V = inputs(transform,tr,case),inputs(transform,va,case)
        inner.append(dict(X=X.to_numpy(),V=V.to_numpy(),y=tr.separation_coefficient.to_numpy(),z=va.separation_coefficient.to_numpy()))
        for role,rows in (("train",tr),("validation",va)):
            memberships.extend(dict(inner_fold=fold,role=role,sample=s) for s in rows["sample"])
    candidates,tuning=[],[]
    for name in ("RF","ET"):
        for ci,config in enumerate(context["options"]["configs"]):
            metrics=evaluate_subset(inner,list(range(X.shape[1])),name,config,context["options"]["search_trees"],seed)
            candidates.append(dict(model=name,config_id=ci,config=config,n_features=X.shape[1],
                inner_r2_mean=float(np.mean([m["r2"] for m in metrics])),inner_r2_std=float(np.std([m["r2"] for m in metrics],ddof=1))))
            tuning.extend(dict(model=name,config_id=ci,**m) for m in metrics)
    selected=choose_candidate(candidates)
    transform=representation("hybrid39").fit(clean_inputs(train))
    X,V=inputs(transform,train,case),inputs(transform,test,case)
    model=make_model(selected["model"],selected["config"],context["options"]["final_trees"],seed)
    model.fit(X.to_numpy(),train.separation_coefficient)
    pred=model.predict(V.to_numpy())
    delta=None
    if case=="intrinsic39":
        ref=reference[(reference.protocol.eq(key[0])) & (reference.repeat.eq(key[1])) & (reference.fold.eq(key[2]))].set_index("sample").loc[test["sample"]]
        np.testing.assert_allclose(test.separation_coefficient,ref.y_true,atol=1e-12)
        delta=float(np.max(np.abs(pred-ref.y_pred)))
        np.testing.assert_allclose(pred,ref.y_pred,atol=1e-10,rtol=0)
    explainer=shap.TreeExplainer(model,feature_perturbation="tree_path_dependent")
    values=np.asarray(explainer.shap_values(V.to_numpy(),check_additivity=True))
    error=float(np.max(np.abs(values.sum(axis=1)+float(np.asarray(explainer.expected_value).ravel()[0])-pred)))
    if error>1e-6:
        raise ValueError("SHAP additivity failed")
    result=dict(protocol=key[0],repeat=int(key[1]),fold=int(key[2]),case=case,fingerprint=fingerprint,
        selected=selected,inner_assignments=memberships,tuning=tuning,baseline_reproduction_difference=delta,shap_additivity_error=error,
        predictions=[dict(sample=s,y_true=float(y),y_pred=float(p)) for s,y,p in zip(test["sample"],test.separation_coefficient,pred)],
        shap=[dict(feature=c,abs_sum=float(np.abs(values[:,i]).sum()),n=len(test)) for i,c in enumerate(V.columns)])
    save_json(path,result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT / "outputs/reviewer_stage3_operating_20261013")
    parser.add_argument("--jobs",type=int,default=10)
    args=parser.parse_args()
    data,tables,assignments,_,_,context=load_current()
    audited=pd.read_csv(CORRELATIONS / "audited_initial_dataset.csv").set_index("sample")
    for c in [*COVARIATES,"potential_offset"]:
        data[c]=data["sample"].map(audited[c])
    np.testing.assert_allclose(data.loc[audited.index,"separation_coefficient"],audited.separation_coefficient)
    ref=tables["predictions"].query("arm == 'without_bo' and strategy == 'all39'")
    output=args.output.resolve()
    fingerprint=start_output(output,__file__,context,dict(cohort="initial only, n=554",protocol="unseen_pair",
        cases=CASES,nested_tuning="3 repeats x 5 outer x 3 inner; 12 RF/ET candidates; 80 search and 400 final trees",
        selection="All declared features, no new forward selection; four matched diagnostic ablations",
        missing="Training-only intrinsic imputation then sample-specific shift; no missing operating values",
        operating=COVARIATES),extra_files=[CORRELATIONS / "audited_initial_dataset.csv",CORRELATIONS / "analysis_manifest.json",ROOT / "revision_correlations.py"])
    keys=[tuple(r) for r in assignments.query("protocol == 'unseen_pair'")[["protocol","repeat","fold"]].drop_duplicates().itertuples(index=False,name=None)]
    results=[]
    with parallel_config(backend="loky",inner_max_num_threads=1):
        for result in Parallel(n_jobs=args.jobs,return_as="generator_unordered")(
            delayed(job)(key,case,data,assignments,context,output,fingerprint,ref) for key in keys for case in CASES):
            results.append(result)
            print(f"Matched operating ablation {len(results)}/60",flush=True)
    for field in ("predictions","inner_assignments","tuning","shap"):
        frame=pd.DataFrame([dict(**{k:r[k] for k in ("protocol","repeat","fold","case")},**row) for r in results for row in r[field]])
        frame.to_csv(output / f"{field}.csv",index=False)
    pred=pd.read_csv(output / "predictions.csv")
    repeats,summary=aggregate_predictions(pred,["case"])
    repeats.to_csv(output / "repeat_metrics.csv",index=False)
    summary.to_csv(output / "summary.csv",index=False)
    pd.DataFrame([dict(**{k:r[k] for k in ("protocol","repeat","fold","case")},**r["selected"]) for r in results]).to_csv(output / "selected_models.csv",index=False)
    differences=[]
    for case in list(CASES)[1:]:
        for repeat in range(3):
            part=repeats[repeats.repeat.eq(repeat)].set_index("case")
            differences.append(dict(case=case,repeat=repeat,r2_delta=part.loc[case,"r2"]-part.loc["intrinsic39","r2"],
                rmse_delta=part.loc[case,"rmse"]-part.loc["intrinsic39","rmse"]))
    pd.DataFrame(differences).to_csv(output / "paired_repeat_differences.csv",index=False)
    s=pd.read_csv(output / "shap.csv").groupby(["case","repeat","feature"],as_index=False)[["abs_sum","n"]].sum()
    s["mean_abs_shap"]=s.abs_sum/s.n
    s.to_csv(output / "shap_repeat_summary.csv",index=False)
    importance=s.groupby(["case","feature"],as_index=False).agg(mean_abs_shap=("mean_abs_shap","mean"),repeat_sd=("mean_abs_shap","std"))
    importance.to_csv(output / "shap_summary.csv",index=False)
    plt.rcParams.update({"font.family":"Arial","font.size":11})
    fig,ax=plt.subplots(figsize=(10,5))
    fig.subplots_adjust(left=.32,right=.96,bottom=.18,top=.88)
    stats=summary.set_index("case").loc[list(CASES)]
    ax.barh(range(4),stats.r2_mean,xerr=stats.r2_sd,color=["#318c8e","#318c8e","#c66048","#c66048"],alpha=.9,capsize=4)
    ax.set_yticks(range(4),[TITLES[c] for c in CASES]); ax.invert_yaxis()
    ax.set_xlim(0,.75); ax.set_xlabel("Out-of-fold R2, mean and SD across three repeats")
    for i,r in enumerate(stats.itertuples()):
        ax.text(r.r2_mean+r.r2_sd+.01,i,f"{r.r2_mean:.3f}",va="center")
    ax.set_title("Matched unseen-pair evaluation: initial screening only")
    fig.text(.5,.02,"Same 554 test samples, same folds and inner model grid. Operating = current, EIS Rs, corrected potential.",ha="center",fontsize=9)
    for ext in ("png","pdf"):
        fig.savefig(output / f"operating_ablation.{ext}",dpi=200)
    plt.close(fig)
    lines=["# Matched operating-variable and potential-shift ablation","",
        "Initial-only evaluation: all 554 samples have all three operating covariates. BO covariates are unavailable in these archived inputs and were not fabricated. Unseen unordered metal pairs are isolated in both outer and inner CV.","",
        "Twelve model/config candidates were tuned separately in each outer fold using the same three inner splits. All 39 (or 42) declared features are used. This controls the information sources, not a new forward-selection search. Original selected-feature results remain unchanged.","",
        "| Inputs | R2 mean +/- SD | RMSE |","|---|---:|---:|"]
    for r in stats.itertuples():
        lines.append(f"| {TITLES[r.Index]} | {r.r2_mean:.3f} +/- {r.r2_sd:.3f} | {r.rmse_mean:.3f} |")
    lines += ["","The intrinsic baseline reproduces all archived all39/no-BO predictions. Potential-shifted descriptors explicitly contain a measured operating potential; they are not intrinsic elemental properties. SHAP tables are provided separately for each case, retaining M1/M2/weighted distinctions. Correlated attributions are not causal effects.","",
        "The corrected potential is E_actual + 0.9*0.05*Rs. Offset = corrected potential + 0.098 V. Add the offset to work function; subtract it from PZC and standard potential, including the weighted versions. Unknown intrinsic references are imputed from training data before applying this shift."]
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save_json(output / "verification.json",dict(complete=True,refits=len(results),max_baseline_difference=max(r["baseline_reproduction_difference"] for r in results if r["baseline_reproduction_difference"] is not None),
        max_shap_additivity_error=max(r["shap_additivity_error"] for r in results),original_models_unchanged=True))
    print(summary.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
