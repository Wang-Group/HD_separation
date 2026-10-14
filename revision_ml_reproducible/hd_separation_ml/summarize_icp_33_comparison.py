"""Paired performance and selection-stability report for the 33-pool ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_followup_common import (ROOT,FEATURES,BASES,PROTOCOLS,NAMES,load_current,
    aggregate_predictions,save_json,sha,label)
from icp_33_result_io import load_reduced
from icp_forward_33_comparison import OUTPUT,POOL,REMOVED,KEYS

STRATEGIES=["full_pool","forward10","forward_best","forward_one_se","task1_frozen10"]
METHODS={"full_pool":"Full pool","forward10":"Forward 10","forward_best":"Inner-best",
         "forward_one_se":"One-SE","task1_frozen10":"Frozen re-selection"}
COLORS={39:"#777d84",33:"#178787"}


def compare(predictions):
    keys=["arm","protocol","repeat","fold","strategy","sample"]
    a=predictions[predictions.pool.eq(39)].set_index(keys).sort_index()
    b=predictions[predictions.pool.eq(33)].set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index,b.index)
    np.testing.assert_allclose(a.y_true,b.y_true,atol=1e-12,rtol=0)
    repeats,summary=aggregate_predictions(predictions,["pool","arm","protocol","strategy"])
    repeat_keys=["arm","protocol","strategy","repeat"]
    pairs=repeats[repeats.pool.eq(39)].merge(repeats[repeats.pool.eq(33)],on=repeat_keys,suffixes=("_39","_33"),validate="one_to_one")
    if not pairs.n_39.eq(pairs.n_33).all():
        raise ValueError("Different evaluation populations")
    for metric in ("r2","rmse","mae"):
        pairs[metric+"_delta"]=pairs[metric+"_33"]-pairs[metric+"_39"]
    delta=pairs.groupby(repeat_keys[:-1],as_index=False).agg(
        r2_delta_mean=("r2_delta","mean"),r2_delta_sd=("r2_delta","std"),
        rmse_delta_mean=("rmse_delta","mean"),rmse_delta_sd=("rmse_delta","std"))
    return repeats,summary,pairs,delta


def frequencies(settings):
    feature_rows,family_rows=[],[]
    for key,part in settings[settings.strategy.ne("full_pool")].groupby(["pool","arm","protocol","strategy"]):
        common=dict(zip(["pool","arm","protocol","strategy"],key))
        subsets=[set(f) for f in part.features]
        if len(subsets)!=15:
            raise ValueError("Expected 15 selected-feature fits")
        for c in FEATURES:
            feature_rows.append(dict(**common,feature=c,in_candidate_pool=key[0]==39 or c in POOL,
                selected_fits=sum(c in s for s in subsets),total_fits=15))
        for b in BASES:
            variants={b+"_element1",b+"_element2","icp_weighted_"+b}
            family_rows.append(dict(**common,property=b,in_candidate_pool=key[0]==39 or bool(variants&set(POOL)),
                selected_fits=sum(bool(variants&s) for s in subsets),total_fits=15))
        vibration={c for c in FEATURES if c.startswith(("Frequency_H_","Frequency_D_","delta_ZPE_H_")) or
                   c in {"icp_weighted_Frequency_H","icp_weighted_Frequency_D","icp_weighted_delta_ZPE_H"}}
        family_rows.append(dict(**common,property="shared_H_D_vibration_information",in_candidate_pool=True,
            selected_fits=sum(bool(vibration&s) for s in subsets),total_fits=15))
    return pd.DataFrame(feature_rows),pd.DataFrame(family_rows)


def figures(summary,frequency):
    plt.rcParams.update({"font.family":"Arial","font.size":10,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
    fig.subplots_adjust(left=.07,right=.985,bottom=.17,top=.89,hspace=.48,wspace=.16)
    for i,arm in enumerate(("without_bo","with_bo")):
        for j,protocol in enumerate(PROTOCOLS):
            ax=axes[i,j]
            for pool,offset in ((39,-.12),(33,.12)):
                part=summary[summary.arm.eq(arm)&summary.protocol.eq(protocol)&summary.pool.eq(pool)].set_index("strategy").loc[STRATEGIES]
                ax.errorbar(np.arange(5)+offset,part.r2_mean,yerr=part.r2_sd,fmt="o",capsize=3,color=COLORS[pool],label=f"{pool} candidates")
                other=summary[summary.arm.eq(arm)&summary.protocol.eq(protocol)&summary.pool.ne(pool)].set_index("strategy")
                for x,r in zip(np.arange(5)+offset,part.itertuples()):
                    above=r.r2_mean>other.loc[r.Index,"r2_mean"] or (r.r2_mean==other.loc[r.Index,"r2_mean"] and pool==33)
                    edge=r.r2_mean+(r.r2_sd if above else -r.r2_sd)
                    ax.annotate(f"{r.r2_mean:.3f}",(x,edge),xytext=(0,5 if above else -10),textcoords="offset points",ha="center",fontsize=7.5,color=COLORS[pool])
            ax.axhline(0,color="#bbbbbb",lw=.8); ax.grid(axis="y",alpha=.15)
            ax.set_xticks(range(5),[METHODS[s] for s in STRATEGIES],rotation=30,ha="right")
            ax.set_title(NAMES[protocol]); ax.set_ylim(min(-.22,float((summary.r2_mean-summary.r2_sd).min())-.10),.70)
            if j==0:
                ax.set_ylabel(("Without BO training" if arm=="without_bo" else "With BO training")+"\nOut-of-fold R2")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.5,.98))
    fig.text(.5,.025,"Identical tests and splits; mean +/- SD over three repeats. Frozen re-selection is retrospective and BO-informed in both refit arms.",ha="center",fontsize=9)
    for ext in ("png","pdf"):
        fig.savefig(OUTPUT / f"performance_39_vs_33.{ext}",dpi=210)
    plt.close(fig)
    bases=[b for b in BASES if b not in ("Frequency_D","delta_ZPE_H")]
    cols=[(p,pool) for p in PROTOCOLS for pool in (39,33)]
    fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True)
    fig.subplots_adjust(left=.14,right=.92,bottom=.19,top=.89,wspace=.12)
    part=frequency[frequency.arm.eq("with_bo")&frequency.strategy.eq("forward10")]
    for ax,variant,title in zip(axes,("element1","element2","weighted"),("M1 properties","M2 properties","ICP-weighted properties")):
        features=["icp_weighted_"+b if variant=="weighted" else b+"_"+variant for b in bases]
        array=part.pivot(index="feature",columns=["protocol","pool"],values="selected_fits").loc[features,cols].to_numpy()
        im=ax.imshow(array,vmin=0,vmax=15,cmap="YlGnBu",aspect="auto")
        for i in range(len(bases)):
            for j in range(6):
                ax.text(j,i,str(int(array[i,j])),ha="center",va="center",color="white" if array[i,j]>=10 else "black")
        ax.set_xticks(range(6),[f"{task}\n{pool}" for task in ("Random","Unseen pairs","Composition") for pool in (39,33)],rotation=35,ha="right")
        ax.set_yticks(range(len(bases)),[label(b+"_element1")[3:] for b in bases]); ax.set_title(title)
    fig.colorbar(im,cax=fig.add_axes([.94,.26,.012,.46]),label="Selected fits / 15")
    fig.suptitle("Forward-10 selection frequency: original 39 versus reduced 33",fontsize=14)
    fig.text(.5,.025,"With BO training. H frequency represents the shared H/D vibration information; D frequency and ZPE are removed only from pool 33.",ha="center",fontsize=9.5)
    for ext in ("png","pdf"):
        fig.savefig(OUTPUT / f"feature_frequency_39_vs_33.{ext}",dpi=210)
    plt.close(fig)


def main():
    data,old,_,old_spec,old_frozen,_=load_current()
    new,manifest,provenance=load_reduced(OUTPUT)
    frozen_folder=OUTPUT / "frozen33"
    frozen_manifest=json.loads((frozen_folder / "analysis_manifest.json").read_text(encoding="utf-8"))
    for path,expected in frozen_manifest["dependencies"].items():
        if sha(Path(path))!=expected:
            raise ValueError(f"Changed frozen dependency: {path}")
    if not json.loads((frozen_folder / "verification.json").read_text(encoding="utf-8"))["complete"]:
        raise ValueError("Frozen33 not complete")
    new_spec=json.loads((frozen_folder / "frozen_specification.json").read_text(encoding="utf-8"))
    frozen33=pd.read_csv(frozen_folder / "predictions.csv")
    predictions=pd.concat([old["predictions"].assign(pool=39),new["predictions"].assign(pool=33),
        old_frozen.assign(pool=39,strategy="task1_frozen10"),frozen33.assign(pool=33,strategy="task1_frozen10")],ignore_index=True)
    predictions["strategy"]=predictions.strategy.replace({"all39":"full_pool","all33":"full_pool"})
    np.testing.assert_allclose(predictions.y_true,data.loc[predictions["sample"],"separation_coefficient"],atol=1e-12,rtol=0)
    repeats,summary,paired,delta=compare(predictions)
    settings=pd.concat([old["settings"].assign(pool=39),new["settings"].assign(pool=33)],ignore_index=True)
    settings["strategy"]=settings.strategy.replace({"all39":"full_pool","all33":"full_pool"})
    feature,family=frequencies(settings)
    counts=settings.groupby(["pool","arm","protocol","strategy"],as_index=False).agg(
        n_features_mean=("n_features","mean"),n_features_min=("n_features","min"),n_features_max=("n_features","max"))
    for name,frame in (("comparison_predictions",predictions),("comparison_repeat_metrics",repeats),
        ("comparison_summary",summary),("paired_repeat_differences",paired),("paired_summary",delta),
        ("feature_frequency_comparison",feature),("property_frequency_comparison",family),("selected_feature_counts",counts)):
        frame.to_csv(OUTPUT / f"{name}.csv",index=False)
    # Independently check new final fits when selected inputs and hyperparameters stayed identical.
    identical=[]
    a=settings[settings.pool.eq(39)].set_index(KEYS+["strategy"])
    b=settings[settings.pool.eq(33)].set_index(KEYS+["strategy"])
    for key,row in b.iterrows():
        reference=a.loc[key]
        same=row.features==reference.features and row.model==reference.model and row.config==reference.config
        if same:
            subset=predictions.copy()
            for k,v in zip(KEYS+["strategy"],key):
                subset=subset[subset[k].eq(v)]
            x=subset[subset.pool.eq(39)].set_index("sample").y_pred.sort_index()
            y=subset[subset.pool.eq(33)].set_index("sample").y_pred.sort_index()
            np.testing.assert_allclose(x,y,atol=1e-10,rtol=0)
            identical.append(dict(zip(KEYS+["strategy"],key),max_prediction_difference=float(np.max(np.abs(x-y)))))
    pd.DataFrame(identical).to_csv(OUTPUT / "identical_model_reproduction.csv",index=False)
    figures(summary,feature)
    run=json.loads((OUTPUT / "verification.json").read_text(encoding="utf-8"))
    headline=[]
    for protocol in PROTOCOLS:
        pair=summary[summary.arm.eq("with_bo")&summary.protocol.eq(protocol)&summary.strategy.eq("forward10")].set_index("pool")
        headline.append(f"{NAMES[protocol]} {pair.loc[39,'r2_mean']:.3f} to {pair.loc[33,'r2_mean']:.3f}")
    frozen_composition=summary[summary.arm.eq("with_bo")&summary.protocol.eq("known_pair_new_composition")&summary.strategy.eq("task1_frozen10")].set_index("pool")
    frozen_tuning=pd.read_csv(frozen_folder / "candidate_tuning_summary.csv").sort_values("inner_r2_mean",ascending=False)
    tuning_gap=frozen_tuning.iloc[0].inner_r2_mean-frozen_tuning.iloc[1].inner_r2_mean
    overlap=len(set(old_spec["features"])&set(new_spec["features"]))
    lines=["# Original 39 versus reduced 33 descriptors","",
        "This comparison removes D vibration and H/D ZPE in M1, M2 and ICP-weighted form, retaining H vibration as the representative of their shared information. The remaining 33 inputs are unchanged. It is a fixed descriptor-family ablation, not general correlation-threshold pruning.","",
        "## Main findings","",
        "With BO training, nested forward-10 R2 changes as follows: "+"; ".join(headline)+". The reduced pool retains predictive performance and the main selection patterns, with a modest composition-task improvement. The three repeats do not establish a statistically significant improvement.","",
        "Without BO training, the composition task remains weak despite redundancy removal. Full-pool models improve modestly in all six task/arm comparisons, whereas the one-SE results are mixed. Redundancy removal is justified as a simpler candidate definition, not a universal performance gain.","",
        f"The separate frozen re-selection retains {overlap}/10 feature identities but changes the feature order and tuned model settings. Its with-BO composition R2 changes from {frozen_composition.loc[39,'r2_mean']:.3f} to {frozen_composition.loc[33,'r2_mean']:.3f}. This is tuning/transfer sensitivity, not evidence that deleting D frequency and ZPE removes needed inputs from the original frozen model: neither was in that model.","",
        "## Matched design","",
        "The same 665-sample snapshot, outer/inner memberships, 3 repeats x 5 outer folds, 3 inner folds, RF/Extra Trees grid and seeds are used. Random and unseen-pair tests contain 554 initial samples per repeat; new-composition tests contain 111 independent BO samples. Both with-BO and without-BO training arms are retained. In the no-BO composition task, inner tuning uses the existing initial-only unseen-pair proxy; no BO outcomes enter its selection.","",
        "Forward selection is repeated in the 33-input pool through 16 steps. Candidate ordering, tie-breaking, 120-tree selection models, 80-tree tuning models and 400-tree final models match the original workflow. Full-pool, forward-10, inner-best and one-SE variants are compared. Exact cached CV evaluations are reused only for identical ordered input subsets, memberships, configurations and seeds. Changed subsets are fitted anew; the first cached candidate is numerically replayed in every job.","",
        f"New forward candidate evaluations: {run['new_forward_trials']}; reused exact evaluations: {run['reused_forward_trials']}. Each evaluation contains two models x three inner folds. All 90 outer training-arm jobs completed. {len(identical)} unchanged final model specifications also reproduce their original predictions to 1e-10.","",
        "## Performance","",
        "Mean R2 +/- SD across three pooled out-of-fold repeats. Differences are paired 33-minus-39; these are not independent-replicate confidence intervals.","",
        "| Training | Task | Strategy | 39 candidates | 33 candidates | Paired change |","|---|---|---|---:|---:|---:|"]
    for arm in ("with_bo","without_bo"):
        for protocol in PROTOCOLS:
            for strategy in STRATEGIES:
                pair=summary[summary.arm.eq(arm)&summary.protocol.eq(protocol)&summary.strategy.eq(strategy)].set_index("pool")
                change=delta[delta.arm.eq(arm)&delta.protocol.eq(protocol)&delta.strategy.eq(strategy)].iloc[0]
                lines.append(f"| {arm} | {NAMES[protocol]} | {METHODS[strategy]} | {pair.loc[39,'r2_mean']:.3f} +/- {pair.loc[39,'r2_sd']:.3f} | {pair.loc[33,'r2_mean']:.3f} +/- {pair.loc[33,'r2_sd']:.3f} | {change.r2_delta_mean:+.3f} |")
    lines += ["","[Performance figure](performance_39_vs_33.png)","",
        "## Selection stability","",
        "Selection-frequency files distinguish M1, M2 and weighted variants and include both BO training arms, all tasks, and all three subset-selection rules. Counts are out of 15 outer fits. Removed features are explicitly marked as unavailable in pool 33, not interpreted as rejected by selection. A shared H/D vibration family row permits a like-for-like family comparison.","",
        "| With-BO forward-10 | Task | 39 pool family count | 33 pool family count |","|---|---|---:|---:|"]
    for base in ("work_function","E_std","FE_ads_avg","delta_EOH","electronegativity","shared_H_D_vibration_information"):
        for p in PROTOCOLS:
            f=family[family.arm.eq("with_bo")&family.strategy.eq("forward10")&family.protocol.eq(p)&family.property.eq(base)].set_index("pool")
            lines.append(f"| {base} | {NAMES[p]} | {f.loc[39,'selected_fits']}/15 | {f.loc[33,'selected_fits']}/15 |")
    lines += ["","[M1/M2/weighted frequency comparison](feature_frequency_39_vs_33.png)","",
        "## Frozen Task 1 sensitivity","",
        "A new consensus of 10 features is selected from the reduced-pool Task-1 with-BO forward-10 fits. The same consensus-frequency and mean-forward-rank rules and the same global Task-1 inner-CV model grid are used. This single specification is then refitted on all three tasks and both training arms. Like the original frozen model, it is retrospective and not independent validation. Its initial-only refit arm is still BO-informed at specification selection.","",
        f"Original model: {old_spec['model']}, configuration {old_spec['config']}. Reduced-pool model: {new_spec['model']}, configuration {new_spec['config']}.","",
        f"The two leading configurations in the reduced-pool Task-1 tuning differ by only {tuning_gap:.6f} mean inner R2. Randomized tree fitting can depend on input-column order; the two exchanged feature positions change the exact random realization. The near-tied tuning winner is therefore not strong evidence of a better transferable specification. Do not choose between these settings using Task-3 test scores.","",
        f"Feature identity overlap is {overlap}/10; table row order is the model input order, not a change of feature identity.","",
        "| Original frozen feature | Reduced-pool frozen feature |","|---|---|"]
    for a,b in zip(old_spec["features"],new_spec["features"]):
        lines.append(f"| {a} | {b} |")
    lines += ["",
        "The original frozen feature list already contains none of the six removed descriptors. Simply applying the 33-column restriction to that already-frozen specification would leave its predictions unchanged. Differences in the newly selected frozen model arise from repeating consensus selection/tuning, not from deleting an input it previously used.","",
        "## Scope","",
        "Other correlations remain, including relations between H/D bond descriptors and the geometry of weighted properties within a fixed metal pair. Removing duplicate vibration descriptors does not establish causal feature importance. Any additional pruning threshold would be a separate training-fold-only comparison. Original completed 39-pool runs, data and manuscript figures are unchanged.","",
        "## Reproduction","",
        "From the repository root, using the original ML environment:","",
        "```powershell",
        "$py = 'C:/Users/yibinjiang/anaconda3/python.exe'",
        "$env:LOKY_MAX_CPU_COUNT = '10'",
        "& $py hd_separation_ml/icp_forward_33_comparison.py --jobs 10",
        "& $py hd_separation_ml/icp_frozen33_sensitivity.py --jobs 10",
        "& $py hd_separation_ml/summarize_icp_33_comparison.py",
        "& $py -m unittest discover -s hd_separation_ml -p test_icp_forward_33_comparison.py -v",
        "```","",
        "The source CV artifacts are required. Code/data hashes and manifests prevent silently combining changed inputs with cached fits. Individual path steps, tuning steps and final models are checkpointed for resumption. For new response values, regenerate the upstream 39-pool workflow before this matched comparison."]
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save_json(OUTPUT / "comparison_verification.json",dict(complete=True,matched_tests=True,matched_targets=True,
        verified_reduced_checkpoints=len(provenance)//3,verified_path_trials=90*408,
        identical_model_refits=len(identical),max_identical_prediction_difference=max([r["max_prediction_difference"] for r in identical],default=0.),
        checkpoint_hashes=provenance,report_script_sha256=sha(Path(__file__))))
    print(summary[summary.arm.eq("with_bo")&summary.strategy.isin(["forward10","task1_frozen10"])].to_string(index=False))
    print(OUTPUT / "RESULTS.md")


if __name__=="__main__":
    main()
