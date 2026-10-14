"""Portable composition controls and observed-descriptor figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from inputs import ROOT, KEYS, read_json, write_json
from modeling import selected_settings
from revision_composition_bridge33 import job as bridge_job, report as bridge_report
from revision_remaining_common import observed_weighted
from revision_electronegativity_branches import BASES, shape_summary, binned_summary, draw_response, draw_geometry


def composition(keys,data,assignments,source,config,output,fingerprint,jobs):
    predictions=pd.read_csv(source / "predictions.csv")
    work=[]
    for key in keys:
        expected=predictions.copy()
        for field,value in zip(KEYS,key): expected=expected[expected[field].eq(value)]
        settings=selected_settings(source,key)
        if not {"all33","forward10"}<=set(settings.strategy):
            raise ValueError("Composition comparison requires a complete 33-candidate training source")
        work.append((key,settings,expected))
    with parallel_config(backend="loky",inner_max_num_threads=1):
        results=Parallel(n_jobs=jobs)(delayed(bridge_job)(k,data,assignments,s,e,config["search_options"],output,fingerprint) for k,s,e in work)
    for category in ("predictions","settings","tuning","memberships","split_audit"):
        table=pd.DataFrame([dict(**{k:r[k] for k in KEYS},**v) for r in results for v in r[category]])
        table.to_csv(output / (category+".csv"),index=False)
    if len(keys)==90: bridge_report(pd.read_csv(output / "predictions.csv"),output)
    return dict(jobs=len(results),max_baseline_prediction_error=max(e for r in results for e in r["reproduction_errors"]))


def descriptor_figures(data,output,bootstrap=1000):
    d=data.copy(); d["system"]=d.element1+"-"+d.element2
    columns=[m+"_"+b for b in BASES for m in ("precursor","icp")]
    for base in BASES:
        for mode in ("precursor","icp"): d[mode+"_"+base]=observed_weighted(d,base,mode)
    d[["sample","cohort","metal_pair","system","separation_coefficient",*columns]].to_csv(output / "observed_descriptors.csv",index=False)
    complete=d[np.isfinite(d[[m+"_electronegativity" for m in ("precursor","icp")]]).all(axis=1)]
    initial=complete[complete.cohort.eq("initial")]; bo=complete[complete.cohort.eq("BO")]; four=bo[bo.system.ne("Co-Fe")]
    sets=[("initial",initial),("original_four_bo",four),("all_five_bo",bo)]+list(bo.groupby("system"))
    shapes=[]; bins=[]
    for i,(name,part) in enumerate(sets):
        for mode in ("precursor","icp"):
            column=mode+"_electronegativity"
            shapes.append(dict(cohort=name,weighting=mode,**shape_summary(part,column,bootstrap,731+i,"pair_cluster" if name=="initial" else "within_pair")))
            b=binned_summary(part,column,10 if name=="initial" else 7,12 if name=="initial" else 6)
            if len(b): bins.append(b.assign(cohort=name,weighting=mode))
    shapes=pd.DataFrame(shapes); bins=pd.concat(bins,ignore_index=True)
    shapes.to_csv(output / "descriptive_shape_bootstrap.csv",index=False)
    bins.to_csv(output / "empirical_mean_q90_bins.csv",index=False)
    plt.rcParams.update({"font.family":"Arial","font.size":10,"pdf.fonttype":42})
    draw_response(complete,four,bins,shapes,output,"original_four_bo")
    draw_response(complete,bo,bins,shapes,output,"all_five_bo")
    draw_geometry(d[np.isfinite(d[columns]).all(axis=1)],output)
    return dict(initial_en=len(initial),bo_en=len(bo),bootstrap=bootstrap,no_imputation=True)
