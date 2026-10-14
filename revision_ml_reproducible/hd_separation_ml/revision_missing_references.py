"""Matched-test refit sensitivities to incomplete elemental reference properties."""
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

from revision_followup_common import (ROOT,FEATURES,PROTOCOLS,NAMES,load_current,start_output,save_json,
    matched_plan,seed_for,fit_model,aggregate_predictions)
from revision_correlations import observed_features
from icp_forward_three_tasks import check_split

POLICIES = ["listed_elements_only", "selected_properties_complete"]
POLICY_LABELS = {"listed_elements_only":"Exclude unlisted Eu/Nb",
                 "selected_properties_complete":"Complete selected properties"}


def eligibility(data,features,policy,elements):
    if policy=="listed_elements_only":
        return data.element1.isin(elements) & data.element2.isin(elements)
    if policy=="selected_properties_complete":
        return observed_features(data)[features].notna().all(axis=1)
    raise ValueError(policy)


def job(key,arm,kind,choice,expected,data,assignments,elements,output,fingerprint):
    path=output / "checkpoints" / f"{kind}_{arm}_{key[0]}_r{key[1]}_f{key[2]}.json"
    if path.exists():
        result=json.loads(path.read_text(encoding="utf-8"))
        if result["fingerprint"]!=fingerprint:
            raise ValueError("Stale missing-reference checkpoint")
        return result
    original,augmented,test=matched_plan(key,data,assignments)
    train=augmented if arm=="with_bo" else original
    expected=expected.set_index("sample")
    np.testing.assert_allclose(test.separation_coefficient,expected.loc[test["sample"],"y_true"],atol=1e-12)
    result=dict(protocol=key[0],repeat=int(key[1]),fold=int(key[2]),arm=arm,kind=kind,
        fingerprint=fingerprint,predictions=[],coverage=[],exclusions=[])
    for policy in POLICIES:
        trmask=eligibility(train,choice["features"],policy,elements)
        temask=eligibility(test,choice["features"],policy,elements)
        tr,te=train.loc[trmask].copy(),test.loc[temask].copy()
        if len(tr)<20 or len(te)<2:
            raise ValueError("Insufficient retained data")
        check_split(tr,te,key[0])
        model,V,pred=fit_model(tr,te,choice["features"],choice["model"],choice["config"],400,seed_for(key))
        if policy=="selected_properties_complete":
            np.testing.assert_allclose(V[choice["features"]],observed_features(te)[choice["features"]],atol=1e-12,rtol=0)
        for source,values in (("baseline_same_test",expected.loc[te["sample"],"y_pred"].to_numpy()),("restricted_refit",pred)):
            result["predictions"].extend(dict(policy=policy,fit=source,sample=s,y_true=float(y),y_pred=float(p))
                for s,y,p in zip(te["sample"],te.separation_coefficient,values))
        result["coverage"].append(dict(policy=policy,n_train_original=len(train),n_train_retained=len(tr),
            n_train_bo_original=int(train.cohort.eq("BO").sum()),n_train_bo_retained=int(tr.cohort.eq("BO").sum()),
            n_test_original=len(test),n_test_retained=len(te),features=choice["features"],model=choice["model"],config=choice["config"]))
        for role,frame,mask in (("train",train,trmask),("test",test,temask)):
            missing=observed_features(frame)[choice["features"]].isna()
            result["exclusions"].extend(dict(policy=policy,role=role,sample=row["sample"],element1=row.element1,element2=row.element2,
                missing_selected_features=missing.columns[missing.iloc[i]].tolist())
                for i,(_,row) in enumerate(frame.iterrows()) if not mask.iloc[i])
    save_json(path,result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT / "outputs/reviewer_stage4_missing_20261013")
    parser.add_argument("--jobs",type=int,default=10)
    args=parser.parse_args()
    data,tables,assignments,spec,frozen,context=load_current()
    elements=set(pd.read_csv(ROOT / "revision_inputs/elemental_properties.csv").element)
    absent=sorted((set(data.element1)|set(data.element2))-elements)
    output=args.output.resolve()
    fingerprint=start_output(output,__file__,context,dict(policies=POLICIES,unlisted_elements=absent,
        design="Fixed saved features/configs; refit after training restriction; compare baseline on identical eligible test rows",
        feature_selection="Not rerun; nested features retain original outer-training selection. Frozen specification remains retrospective.",
        complete="Only the selected properties must be known; a zero-weight missing component is irrelevant to a weighted feature",
        arms=["without_bo","with_bo"],tasks=PROTOCOLS),extra_files=[ROOT / "revision_correlations.py"])
    missing=observed_features(data).isna()
    pd.concat([data[["sample","cohort","element1","element2"]],missing.add_prefix("missing_")],axis=1).to_csv(output / "reference_missingness_by_sample.csv",index=False)
    data.loc[~eligibility(data,[],"listed_elements_only",elements),["sample","cohort","element1","element2"]].to_csv(output / "unlisted_element_samples.csv",index=False)
    work=[]
    for p in PROTOCOLS:
        for r in range(3):
            for f in range(5):
                for arm in ("without_bo","with_bo"):
                    settings=tables["settings"]
                    s=settings[settings.protocol.eq(p)&settings.repeat.eq(r)&settings.fold.eq(f)&settings.arm.eq(arm)&settings.strategy.eq("forward10")]
                    if len(s)!=1:
                        raise ValueError("Missing nested model")
                    for kind,choice,source in (("nested_forward10",s.iloc[0].to_dict(),tables["predictions"]),
                                               ("task1_frozen10",spec,frozen)):
                        expected=source[source.protocol.eq(p)&source.repeat.eq(r)&source.fold.eq(f)&source.arm.eq(arm)]
                        if kind=="nested_forward10":
                            expected=expected[expected.strategy.eq("forward10")]
                        work.append(((p,r,f),arm,kind,choice,expected))
    results=[]
    with parallel_config(backend="loky",inner_max_num_threads=1):
        for result in Parallel(n_jobs=args.jobs,return_as="generator_unordered")(
            delayed(job)(key,arm,kind,choice,expected,data,assignments,elements,output,fingerprint)
            for key,arm,kind,choice,expected in work):
            results.append(result)
            if len(results)%15==0:
                print(f"Missing-reference sensitivity {len(results)}/180 jobs (two refits each)",flush=True)
    keys=["protocol","repeat","fold","arm","kind"]
    for category in ("predictions","coverage","exclusions"):
        pd.DataFrame([dict(**{k:r[k] for k in keys},**row) for r in results for row in r[category]]).to_csv(output / f"{category}.csv",index=False)
    predictions=pd.read_csv(output / "predictions.csv")
    groups=["protocol","arm","kind","policy"]
    repeats,summary=aggregate_predictions(predictions,groups+["fit"])
    repeats.to_csv(output / "repeat_metrics.csv",index=False)
    summary.to_csv(output / "summary.csv",index=False)
    differences=[]
    for key,part in repeats.groupby(groups+["repeat"]):
        part=part.set_index("fit")
        if part.loc["baseline_same_test","n"]!=part.loc["restricted_refit","n"]:
            raise ValueError("Unmatched sensitivity test sets")
        differences.append(dict(zip(groups+["repeat"],key),n=int(part.loc["restricted_refit","n"]),
            r2_delta=part.loc["restricted_refit","r2"]-part.loc["baseline_same_test","r2"],
            rmse_delta=part.loc["restricted_refit","rmse"]-part.loc["baseline_same_test","rmse"]))
    delta=pd.DataFrame(differences)
    delta.to_csv(output / "paired_repeat_differences.csv",index=False)
    ds=delta.groupby(groups,as_index=False).agg(r2_delta_mean=("r2_delta","mean"),r2_delta_sd=("r2_delta","std"),
        n_min=("n","min"),n_max=("n","max"))
    ds.to_csv(output / "paired_summary.csv",index=False)
    plt.rcParams.update({"font.family":"Arial","font.size":11})
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    fig.subplots_adjust(left=.18,right=.97,bottom=.22,top=.80,wspace=.18)
    for ax,policy in zip(axes,POLICIES):
        for kind,offset,color,name in (("nested_forward10",-.13,"#268787","Nested forward-10"),
                                     ("task1_frozen10",.13,"#c65c47","Frozen Task-1 sensitivity")):
            part=ds[ds.arm.eq("with_bo")&ds.kind.eq(kind)&ds.policy.eq(policy)].set_index("protocol").loc[PROTOCOLS]
            ax.errorbar(part.r2_delta_mean,np.arange(3)+offset,xerr=part.r2_delta_sd,fmt="o",capsize=4,color=color,label=name)
        ax.set_yticks(range(3),[NAMES[p] for p in PROTOCOLS]); ax.invert_yaxis()
        ax.axvline(0,color="#777777",lw=1); ax.set_title(POLICY_LABELS[policy]); ax.set_xlabel("R2 change on identical test samples")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False)
    fig.text(.5,.03,"With-BO training. Mean and SD across three repeats. Feature choices/configurations are held fixed, not reselected.",ha="center",fontsize=10)
    for ext in ("png","pdf"):
        fig.savefig(output / f"missing_reference_sensitivity.{ext}",dpi=200)
    plt.close(fig)
    lines=["# Missing-reference sensitivity","",f"Unlisted reference elements: {', '.join(absent)}. Affected samples: {int((~eligibility(data,[],'listed_elements_only',elements)).sum())}.","",
        "360 refits: two restrictions, two current model specifications, two BO training arms, three tasks, three repeats, five folds. Original feature identities and hyperparameters are held fixed; this tests refitting/coverage sensitivity, not selection stability after changing the dataset.","",
        "Each restricted refit is compared against the original model's predictions on exactly the same retained test samples. Different test populations must not be compared directly. Full per-fold exclusions and sample counts are exported. Complete-selected-property eligibility can vary by fold because nested features vary.","",
        "The frozen model's features/config were chosen with Task-1 outcomes, including BO, so its results remain retrospective, including the initial-only refit arm. No missing Eu/Nb references were invented. Known zero component fractions need no reference value for that absent component in a weighted property.","",
        "| With BO | Restriction | Task | n per repeat | Paired R2 change |","|---|---|---|---|---:|"]
    for r in ds[ds.arm.eq("with_bo")].itertuples():
        lines.append(f"| {r.kind} | {r.policy} | {NAMES[r.protocol]} | {r.n_min}-{r.n_max} | {r.r2_delta_mean:+.3f} +/- {r.r2_delta_sd:.3f} |")
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save_json(output / "verification.json",dict(complete=True,jobs=len(results),refits=2*len(results),same_test_pairs=True,
        original_models_unchanged=True,unlisted_elements=absent))
    print(ds[ds.arm.eq("with_bo")].to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
