"""Retrospective Task-1 consensus specification, frozen across all three tasks.

This is a post-selection sensitivity analysis, not independent nested validation.
The original nested-CV checkpoints are read-only comparators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config

from icp_bo_augmentation_cv import representation
from icp_electrochemical_exclusion import FEATURES, check_features, clean_inputs, reference_audit
from icp_forward_bo_ablation import ARMS, PROTOCOLS, matched_plan, seed_for
from icp_forward_selection_revision import save_json
from icp_forward_three_tasks import check_split
from icp_intrinsic_descriptor_analysis import LABELS, scores
from icp_nested_group_cv import make_model
from icp_revision_inputs import sha
from summarize_icp_forward_bo_ablation import load_ablation

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/forward_bo_ablation_20261013"
OUTPUT = ROOT / "outputs/task1_frozen_sensitivity_20261013"
TASK1 = "stratified_random"
NAMES = {TASK1:"Random entities", "unseen_pair":"Unseen metal pairs",
         "known_pair_new_composition":"New compositions"}
KEYS = ["arm", "protocol", "repeat", "fold"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def task1_only(frame):
    return frame[frame.arm.eq("with_bo") & frame.protocol.eq(TASK1)].copy()


def consensus_features(settings, paths, n=10):
    source = task1_only(settings)
    source = source[source.strategy.eq("forward10")]
    if len(source) != 15 or source.duplicated(["repeat", "fold"]).any():
        raise ValueError("Need 15 Task-1 with-BO forward-10 settings")
    if set(zip(source.repeat, source.fold)) != {(r, f) for r in range(3) for f in range(5)}:
        raise ValueError("Unexpected Task-1 split keys")
    path = task1_only(paths)
    ranks, feature_sets = [], []
    for row in source.itertuples():
        check_features(row.features)
        if len(row.features) != 10:
            raise ValueError("Source strategy is not forward-10")
        p = path[(path.repeat==row.repeat) & (path.fold==row.fold)]
        if len(p) != 16 or p.added_feature.duplicated().any() or set(p.n_features) != set(range(1,17)):
            raise ValueError("Incomplete source path")
        if p.sort_values("n_features").head(10).added_feature.tolist() != row.features:
            raise ValueError("Selected features disagree with source path")
        feature_sets.append(set(row.features))
        ranks.append(dict(zip(p.added_feature, p.n_features)))
    result = pd.DataFrame([dict(feature=c, source_selected_fits=sum(c in s for s in feature_sets),
        total_fits=15, mean_path_rank=float(np.mean([r.get(c,17) for r in ranks])),
        original_column_index=i) for i,c in enumerate(FEATURES)])
    result = result.sort_values(["source_selected_fits", "mean_path_rank", "original_column_index"],
        ascending=[False, True, True], kind="stable").reset_index(drop=True)
    result["consensus_rank"] = np.arange(1,len(result)+1)
    result["frozen_selected"] = result.consensus_rank <= n
    return result.loc[result.frozen_selected,"feature"].tolist(), result


def choose_global_config(candidate_scores, configs):
    if set(candidate_scores.protocol) != {TASK1} or set(candidate_scores.arm) != {"with_bo"}:
        raise ValueError("Only Task-1 with-BO scores may choose the specification")
    summary = candidate_scores.groupby(["model", "config_id"], as_index=False).agg(
        inner_r2_mean=("r2","mean"), inner_r2_sd=("r2","std"), n_inner_folds=("r2","size"))
    expected = {(m,c) for m in ("RF","ET") for c in range(len(configs))}
    if set(zip(summary.model,summary.config_id)) != expected or not summary.n_inner_folds.eq(45).all():
        raise ValueError("Expected 12 configurations, each evaluated on 45 inner folds")
    keys = ["model","config_id","repeat","fold","inner_fold"]
    if candidate_scores.duplicated(keys).any() or not np.isfinite(candidate_scores.r2).all():
        raise ValueError("Duplicate or invalid tuning scores")
    summary = summary.sort_values(["inner_r2_mean","model","config_id"], ascending=[False,True,True])
    winner = summary.iloc[0]
    return dict(model=winner.model, config_id=int(winner.config_id),
        config=configs[int(winner.config_id)], tuning_mean_r2=float(winner.inner_r2_mean)), summary


def tune_job(key, data, assignments, memberships, features, options, output, fingerprint):
    repeat, fold = map(int,key)
    destination = output / "tuning_checkpoints" / f"r{repeat}_f{fold}.json"
    if destination.exists():
        result = json.loads(destination.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Stale tuning checkpoint")
        return result
    _, outer_train, outer_test = matched_plan((TASK1,repeat,fold),data,assignments)
    selected = memberships[(memberships.repeat==repeat) & (memberships.fold==fold)]
    rows, audit = [], []
    if set(selected.inner_fold) != {0,1,2}:
        raise ValueError("Expected three inner folds")
    for inner_fold, part in selected.groupby("inner_fold"):
        train = data.loc[part.loc[part.role.eq("train"),"sample"]].reset_index(drop=True)
        valid = data.loc[part.loc[part.role.eq("validation"),"sample"]].reset_index(drop=True)
        ids = set(train["sample"]) | set(valid["sample"])
        if ids - set(outer_train["sample"]) or ids & set(outer_test["sample"]):
            raise ValueError("Tuning membership violates source outer split")
        audit.append(dict(inner_fold=int(inner_fold), **check_split(train,valid,TASK1)))
        transform = representation("hybrid39").fit(clean_inputs(train))
        X = transform.transform(clean_inputs(train))[features].to_numpy()
        V = transform.transform(clean_inputs(valid))[features].to_numpy()
        for name in ("RF","ET"):
            for config_id, config in enumerate(options["configs"]):
                estimator = make_model(name,config,options["search_trees"],seed_for((TASK1,repeat,fold))+int(inner_fold))
                estimator.fit(X,train.separation_coefficient)
                rows.append(dict(arm="with_bo",protocol=TASK1,repeat=repeat,fold=fold,
                    inner_fold=int(inner_fold),model=name,config_id=config_id,
                    **scores(valid.separation_coefficient,estimator.predict(V))))
    result = dict(fingerprint=fingerprint,scores=rows,audit=audit)
    save_json(destination,result)
    print(f"Task-1 specification tuning: repeat {repeat+1}, fold {fold+1}",flush=True)
    return result


def fit_frozen(train, test, specification, seed):
    features = specification["features"]
    check_features(features)
    transform = representation("hybrid39").fit(clean_inputs(train))
    X = transform.transform(clean_inputs(train))[features].to_numpy()
    V = transform.transform(clean_inputs(test))[features].to_numpy()
    if not np.isfinite(X).all() or not np.isfinite(V).all():
        raise ValueError("Non-finite transformed inputs")
    estimator = make_model(specification["model"],specification["config"],specification["trees"],seed)
    estimator.fit(X,train.separation_coefficient)
    return estimator.predict(V)


def refit_job(key, data, assignments, specification, exposed, output, fingerprint):
    arm, protocol, repeat, fold = key
    repeat,fold = int(repeat),int(fold)
    destination = output / "refit_checkpoints" / f"{arm}_{protocol}_r{repeat}_f{fold}.json"
    if destination.exists():
        result = json.loads(destination.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint or result["specification_hash"] != digest(specification):
            raise ValueError("Stale frozen-model checkpoint")
        return result
    original, augmented, test = matched_plan((protocol,repeat,fold),data,assignments)
    train = original if arm=="without_bo" else augmented
    audit = check_split(train,test,protocol)
    audit.update(n_train_initial=int(train.cohort.eq("initial").sum()),n_train_bo=int(train.cohort.eq("BO").sum()),
        n_test_previously_in_task1_selection=int(test["sample"].isin(exposed).sum()),
        specification_selection_included_bo=True, independent_validation=False)
    pred = fit_frozen(train,test,specification,seed_for((protocol,repeat,fold)))
    result = dict(arm=arm,protocol=protocol,repeat=repeat,fold=fold,fingerprint=fingerprint,
        specification_hash=digest(specification),model=specification["model"],features=specification["features"],
        config=specification["config"],audit=audit,predictions=[dict(sample=r["sample"],metal_pair=r.metal_pair,
        y_true=float(r.separation_coefficient),y_pred=float(p)) for (_,r),p in zip(test.iterrows(),pred)])
    save_json(destination,result)
    return result


def metric_tables(predictions):
    keys = ["arm","protocol","strategy","repeat"]
    rows = []
    for key, part in predictions.groupby(keys):
        if part["sample"].duplicated().any():
            raise ValueError("Duplicate sample in pooled repeat predictions")
        rows.append(dict(zip(keys,key),n=len(part),**scores(part.y_true,part.y_pred)))
    repeats = pd.DataFrame(rows)
    summary = repeats.groupby(keys[:-1],as_index=False).agg(n=("n","first"),
        r2_mean=("r2","mean"),r2_sd=("r2","std"),rmse_mean=("rmse","mean"),rmse_sd=("rmse","std"),
        mae_mean=("mae","mean"),mae_sd=("mae","std"))
    return repeats,summary


def label(feature):
    if feature.startswith("icp_weighted_"):
        return "ICP-weighted "+LABELS[feature[len("icp_weighted_"):]]
    base, side = feature.rsplit("_element",1)
    return "M"+side+" "+LABELS[base]


def figures(summary, frequency, features, output):
    plt.rcParams.update({"font.family":"Arial","font.size":11,"pdf.fonttype":42,
        "axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(13.5,5.8),sharey=True)
    fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.79,wspace=.13)
    strategies = ["all39","forward10","task1_frozen10"]
    colors = ["#8c9398","#248385","#c65546"]
    names = ["Nested: all 39","Nested: forward 10","Task-1 frozen 10"]
    for ax,arm,title in zip(axes,["with_bo","without_bo"],["With BO training","Initial-only refitting"]):
        for i,(strategy,color,name) in enumerate(zip(strategies,colors,names)):
            p = summary[(summary.arm==arm)&(summary.strategy==strategy)].set_index("protocol").loc[PROTOCOLS]
            x = np.arange(3)+(i-1)*.24
            ax.bar(x,p.r2_mean,width=.225,color=color,label=name)
            ax.errorbar(x,p.r2_mean,yerr=p.r2_sd,fmt="none",color="#222222",capsize=3,lw=1)
            for xx,r in zip(x,p.itertuples()):
                ax.text(xx,r.r2_mean+r.r2_sd+.012,f"{r.r2_mean:.3f}",ha="center",va="bottom",fontsize=9,rotation=55)
        ax.set_xticks(range(3),["Random","Unseen pairs","New compositions"])
        ax.set_title(title,fontsize=13,pad=14)
        ax.axhline(0,color="#555555",lw=.7)
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Pooled repeat R-squared (mean and SD)")
    axes[0].set_ylim(min(-.15,float((summary.r2_mean-summary.r2_sd).min())-.08),
                     max(.76,float((summary.r2_mean+summary.r2_sd).max())+.15))
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.52,.97),ncol=3,frameon=False)
    fig.text(.5,.085,"Frozen specification selected using Task 1 with BO. Initial-only refits still use that BO-informed specification.",
        ha="center",fontsize=10)
    fig.text(.5,.035,"Retrospective sensitivity, not independent validation. Error bars describe variability over three repeats.",
        ha="center",fontsize=10)
    for ext in ("png","pdf"):
        fig.savefig(output / ("frozen_vs_nested_performance."+ext),dpi=210)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(9.2,6.5))
    fig.subplots_adjust(left=.40,right=.90,bottom=.15,top=.85)
    mat = frequency.pivot(index="feature",columns="protocol",values="selected_fits").loc[features,PROTOCOLS].to_numpy()
    heat = ax.imshow(mat,aspect="auto",cmap="YlGnBu",vmin=0,vmax=15)
    for i in range(len(features)):
        for j in range(3):
            ax.text(j,i,str(int(mat[i,j])),ha="center",va="center",color="white" if mat[i,j]>=10 else "black")
    ax.set_xticks(range(3),["Random","Unseen pairs","New compositions"])
    ax.set_yticks(range(len(features)),[label(f) for f in features])
    ax.tick_params(length=0)
    fig.colorbar(heat,ax=ax,pad=.04,shrink=.8,ticks=[0,5,10,15]).set_label("Selected runs / 15")
    fig.suptitle("Frozen feature set: selection frequencies in the original nested CV",fontsize=13,y=.96)
    fig.text(.5,.04,"With-BO forward-10 runs. These are nested selection frequencies, not the forced inclusion in the frozen model.",
        ha="center",fontsize=9)
    for ext in ("png","pdf"):
        fig.savefig(output / ("frozen_features_nested_frequency."+ext),dpi=210)
    plt.close(fig)


def report(output,specification,ranking,summary,deltas,audit):
    lines = ["# Task-1 Frozen Model Sensitivity", "", "## Interpretation", "",
        "One exact ten-feature list, model family and hyperparameter configuration were chosen from Task 1 "
        "with BO in training, then frozen for all tasks and both refitting arms. No Task-2/3 score was used "
        "to choose or revise the specification. However, the same samples recur across tasks and folds. "
        "Their responses can already have influenced Task-1 selection. This is a retrospective post-selection "
        "sensitivity check, NOT independent validation or a replacement for the original nested CV.", "",
        "The initial-only refit arm excludes BO rows from estimator training, but its specification is BO-informed. "
        "It must NOT be presented as a complete BO-free model-selection workflow.", "",
        "## Specification", "",
        f"Model: {specification['model']}; config: {json.dumps(specification['config'])}; trees: {specification['trees']}.",
        "Ten features were chosen by descending selection frequency in the 15 Task-1 with-BO forward-10 runs. "
        "Ties use ascending mean forward-entry rank (17 for absent from the 16-step path), then original feature-column order. "
        "The feature count of ten was fixed in advance; no Task-2/3 result selected the count.",
        "RF and Extra Trees, each with the same six archived configurations, were compared over the 45 saved "
        "Task-1 inner folds using this common feature set and 80 trees. The largest mean inner-fold R2 selected "
        "the model/configuration; ties break by model name and config index. These tuning scores are selection "
        "criteria, not unbiased performance estimates because the consensus features were chosen globally.", "",
        "| Feature | Task-1 selected runs / 15 |", "|---|---:|"]
    for row in ranking[ranking.frozen_selected].itertuples():
        lines.append(f"| {row.feature} | {row.source_selected_fits} |")
    lines += ["", "## Evaluation", "",
        "The existing 3 repeats x 5 outer folds and matched test samples are retained. Imputation is refitted "
        "on each training fold only, as are the estimator parameters. Feature identities, model family and "
        "hyperparameters never change. Seeds and 400-tree final fits match the original workflow.",
        "Allowed inputs are unshifted elemental references and ICP fractions. Operating current, eta10, "
        "EIS resistance and applied-potential corrections are excluded. Source-reference missingness and "
        "training-median imputation are unchanged from the original ML.",
        "Task 1 and Task 2 test 554 initial samples per repeat. Task 3 tests 111 independent BO samples. "
        "Scores pool all test predictions within a repeat before averaging the three repeats. SD describes "
        "split variability, not a confidence interval or experimental uncertainty.", "",
        "| Task | Refitting data | Nested forward-10 R2 | Frozen R2 | Frozen minus nested | Frozen RMSE |",
        "|---|---|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for arm in ("with_bo","without_bo"):
            p = summary[(summary.protocol==protocol)&(summary.arm==arm)].set_index("strategy")
            n,f = p.loc["forward10"],p.loc["task1_frozen10"]
            lines.append(f"| {NAMES[protocol]} | {'Initial + BO' if arm=='with_bo' else 'Initial only, BO-informed specification'} | "
                f"{n.r2_mean:.3f} +/- {n.r2_sd:.3f} | {f.r2_mean:.3f} +/- {f.r2_sd:.3f} | "
                f"{f.r2_mean-n.r2_mean:+.3f} | {f.rmse_mean:.3f} +/- {f.rmse_sd:.3f} |")
    lines += ["", "A performance gain in this sensitivity check is not evidence that it generalizes better "
        "than nested selection: the frozen specification has retrospective access to information from the dataset. "
        "A decline indicates a transfer cost for this particular Task-1 consensus specification, not that no "
        "unified model can work. Task-specific selection frequencies remain the nested-CV feature-stability evidence.", "",
        f"Selection exposure audit: {int(audit.n_test_previously_in_task1_selection.sum())} of "
        f"{int(audit.n_test.sum())} repeated outer-test appearances were previously present in Task-1 "
        "selection/tuning inputs. See split_and_selection_exposure_audit.csv; refit train/test isolation still holds.", "",
        "## Reproduction", "", "```powershell",
        "& 'C:\\Users\\yibinjiang\\anaconda3\\python.exe' hd_separation_ml/icp_task1_frozen_sensitivity.py --jobs 10",
        "```", "", "Inputs, upstream checkpoints and code are hashed. Changed targets require regenerating "
        "the upstream Task-1 selection and nested comparisons, then using a fresh output directory. "
        "No original datasets, nested-CV outputs or manuscript figures were overwritten.", "",
        "Main outputs: frozen_specification.json, consensus_feature_ranking.csv, candidate_tuning_summary.csv, "
        "summary.csv, repeat_metrics.csv, paired_repeat_differences.csv, predictions.csv, "
        "nested_feature_frequency.csv and the two PNG/PDF figures."]
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--jobs",type=int,default=10)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    start = time.perf_counter()
    source,output = args.source.resolve(),args.output.resolve()
    data,tables,_,previous = load_ablation(source)
    versions = dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__)
    if versions != previous["versions"]:
        raise ValueError("Use the original ML runtime for matched model comparisons")
    features,ranking = consensus_features(tables["settings"],tables["path"])
    reference = ROOT / "revision_inputs/elemental_properties.csv"
    reference_audit(data,pd.read_csv(reference))
    assignment_file = Path(previous["source"]) / "outer_assignments.csv"
    assignments = pd.read_csv(assignment_file)
    memberships = task1_only(tables["inner_assignments"])
    if memberships.duplicated(["repeat","fold","inner_fold","role","sample"]).any():
        raise ValueError("Duplicate source inner membership")
    exposed = set(memberships["sample"])
    dependencies = {**previous["source_hashes"],**previous["code_hashes"]}
    provenance = pd.read_csv(source / "checkpoint_provenance.csv")
    dependencies.update({r.checkpoint:r.sha256 for r in provenance.itertuples()})
    for file in [Path(__file__),ROOT / "test_icp_task1_frozen_sensitivity.py",
        ROOT / "icp_electrochemical_exclusion.py",ROOT / "summarize_icp_forward_bo_ablation.py",
        reference,assignment_file,source / "analysis_manifest.json",source / "checkpoint_provenance.csv"]:
        dependencies[str(file.resolve())] = sha(file)
    for path,expected in dependencies.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Changed dependency: {path}")
    manifest = dict(source=str(source),dependencies=dependencies,features=features,versions=versions,
        selection_source="Task 1, with BO, fixed-ten consensus",search_options=previous["search_options"],
        feature_tiebreak="mean path rank, missing=17; then original column index",
        fixed_feature_count=10,independent_validation=False,selection_uses_bo=True,
        evaluation="Same frozen specification for three tasks and both refitting arms")
    manifest["fingerprint"] = digest(manifest)
    output.mkdir(parents=True,exist_ok=True)
    file = output / "analysis_manifest.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8"))["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Changed inputs/code/options: choose a fresh output directory")
    save_json(file,manifest)
    for folder in ("tuning_checkpoints","refit_checkpoints"):
        (output / folder).mkdir(exist_ok=True)
    ranking.to_csv(output / "consensus_feature_ranking.csv",index=False)
    memberships.to_csv(output / "task1_selection_memberships.csv",index=False)
    print("Frozen consensus features: "+", ".join(features),flush=True)
    options = previous["search_options"]
    with parallel_config(backend="loky",inner_max_num_threads=1):
        tuned = Parallel(n_jobs=args.jobs)(delayed(tune_job)((r,f),data,assignments,memberships,features,
            options,output,manifest["fingerprint"]) for r in range(3) for f in range(5))
    candidate_scores = pd.DataFrame([s for result in tuned for s in result["scores"]])
    choice,tuning_summary = choose_global_config(candidate_scores,options["configs"])
    candidate_scores.to_csv(output / "candidate_tuning_scores.csv",index=False)
    tuning_summary.to_csv(output / "candidate_tuning_summary.csv",index=False)
    specification = dict(**choice,features=features,trees=options["final_trees"],
        selection_source="Task 1 with BO",interpretation="Retrospective sensitivity, not independent validation")
    save_json(output / "frozen_specification.json",specification)
    print("Model/configuration frozen: "+json.dumps(choice),flush=True)
    keys = [(arm,p,r,f) for arm in ARMS for p in PROTOCOLS for r in range(3) for f in range(5)]
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results = Parallel(n_jobs=args.jobs)(delayed(refit_job)(key,data,assignments,specification,exposed,
            output,manifest["fingerprint"]) for key in keys)
    if len(results)!=90 or {r["specification_hash"] for r in results}!={digest(specification)}:
        raise ValueError("Specification changed between refits")
    frozen = pd.DataFrame([dict(**{k:r[k] for k in KEYS},strategy="task1_frozen10",**p)
        for r in results for p in r["predictions"]])
    baseline = tables["predictions"].copy()
    idx = KEYS+["sample"]
    a = frozen.set_index(idx).sort_index()
    b = baseline[baseline.strategy.eq("forward10")].set_index(idx).sort_index()
    pd.testing.assert_index_equal(a.index,b.index)
    np.testing.assert_allclose(a.y_true,b.y_true,atol=1e-12,rtol=0)
    all_predictions = pd.concat([frozen,baseline],ignore_index=True)
    repeats,summary = metric_tables(all_predictions)
    for protocol,expected in [(TASK1,554),("unseen_pair",554),("known_pair_new_composition",111)]:
        if not repeats.loc[repeats.protocol.eq(protocol),"n"].eq(expected).all():
            raise ValueError("Incomplete pooled-repeat test coverage")
    comparison_keys = ["arm","protocol","repeat"]
    left = repeats[repeats.strategy.eq("task1_frozen10")].drop(columns="strategy")
    right = repeats[~repeats.strategy.eq("task1_frozen10")]
    deltas = left.merge(right,on=comparison_keys,suffixes=("_frozen","_nested"),validate="one_to_many")
    for metric in ("r2","rmse","mae"):
        deltas["frozen_minus_nested_"+metric] = deltas[metric+"_frozen"]-deltas[metric+"_nested"]
    audit = pd.DataFrame([dict(**{k:r[k] for k in KEYS},**r["audit"]) for r in results])
    frequency = []
    settings = tables["settings"]
    for protocol in PROTOCOLS:
        selected = settings[settings.arm.eq("with_bo") & settings.protocol.eq(protocol) & settings.strategy.eq("forward10")]
        for feature in FEATURES:
            frequency.append(dict(protocol=protocol,feature=feature,total_fits=len(selected),
                selected_fits=sum(feature in v for v in selected.features),frozen_selected=feature in features))
    frequency = pd.DataFrame(frequency)
    for name,frame in [("predictions",frozen),("repeat_metrics",repeats),("summary",summary),
        ("paired_repeat_differences",deltas),("split_and_selection_exposure_audit",audit),
        ("nested_feature_frequency",frequency)]:
        frame.to_csv(output / (name+".csv"),index=False)
    figures(summary,frequency,features,output)
    report(output,specification,ranking,summary,deltas,audit)
    verification = dict(complete=True,n_consensus_features=len(features),n_tuning_fits=len(candidate_scores),
        n_refits=len(results),unique_specifications=1,paired_test_memberships=True,
        independent_validation=False,nested_results_unchanged=True,
        source_selection_samples=len(exposed),source_selection_bo_samples=int(data.loc[sorted(exposed),"cohort"].eq("BO").sum()),
        elapsed_seconds=time.perf_counter()-start)
    save_json(output / "verification.json",verification)
    print(summary[summary.strategy.isin(["forward10","task1_frozen10"])].to_string(index=False),flush=True)
    print(json.dumps(verification,indent=2),flush=True)


if __name__ == "__main__":
    main()
