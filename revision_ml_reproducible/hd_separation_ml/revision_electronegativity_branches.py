"""Observed-composition check of electronegativity shape and BO trajectories."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from revision_remaining_common import load_final, add_recipes, observed_weighted, OUTPUT, REDUCED
from revision_followup_common import start_output, save_json

COLORS = {"Co-Fe":"#c53949", "Co-Mo":"#d56618", "Co-Ni":"#277bb2", "Co-V":"#278267", "Fe-Ni":"#8654ae"}
MARKERS = dict(zip(COLORS, ("o", "s", "^", "D", "P")))
BASES = ("electronegativity", "FE_ads_avg", "work_function")
LABELS = {"electronegativity":"Electronegativity", "FE_ads_avg":"H adsorption energy (eV)", "work_function":"Work function (eV)"}


def quadratic(x, y):
    """Unconstrained descriptive quadratic; no forced interior maximum."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 6 or len(np.unique(x)) < 3 or np.std(x) < 1e-10:
        return None
    center, scale = float(np.mean(x)), float(np.std(x))
    z = (x-center)/scale
    c = np.polynomial.polynomial.polyfit(z, y, 2)
    residual = np.sum((y-np.polynomial.polynomial.polyval(z,c))**2)
    total = np.sum((y-y.mean())**2)
    vertex = center-scale*c[1]/(2*c[2]) if abs(c[2]) > 1e-12 else np.nan
    interior = bool(c[2] < -1e-12 and x.min() < vertex < x.max())
    return dict(n=len(x), xmin=float(x.min()), xmax=float(x.max()), center=center, scale=scale,
        intercept=float(c[0]), linear=float(c[1]), quadratic=float(c[2]),
        curvature_per_descriptor_squared=float(c[2]/scale**2), vertex=float(vertex),
        interior_maximum=interior, descriptive_r2=float(1-residual/total) if total>0 else np.nan)


def bootstrap_indices(groups, rng, mode):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    blocks = [np.flatnonzero(groups == g) for g in unique]
    if mode == "pair_cluster":
        return np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
    if mode == "within_pair":
        return np.concatenate([rng.choice(b, size=len(b), replace=True) for b in blocks])
    raise ValueError(mode)


def shape_summary(frame, column, bootstrap, seed, mode):
    d = frame[np.isfinite(frame[column])].reset_index(drop=True)
    fit = quadratic(d[column], d.separation_coefficient)
    if fit is None:
        return dict(n=len(d), fit_available=False)
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(bootstrap):
        part = d.iloc[bootstrap_indices(d.metal_pair, rng, mode)]
        value = quadratic(part[column], part.separation_coefficient)
        if value is not None:
            fits.append(value)
    boots = pd.DataFrame(fits)
    return dict(fit, fit_available=True, bootstrap_mode=mode, bootstrap_valid=len(boots),
        curvature_p025=float(boots.curvature_per_descriptor_squared.quantile(.025)),
        curvature_p975=float(boots.curvature_per_descriptor_squared.quantile(.975)),
        interior_maximum_bootstrap_fraction=float(boots.interior_maximum.mean()))


def binned_summary(frame, column, bins, minimum):
    d = frame[np.isfinite(frame[column])].copy()
    edges = np.unique(np.quantile(d[column], np.linspace(0,1,bins+1)))
    if len(edges) < 2:
        return pd.DataFrame()
    d["bin"] = pd.cut(d[column], edges, include_lowest=True, labels=False)
    rows = []
    for b, part in d.groupby("bin"):
        if len(part) >= minimum:
            rows.append(dict(bin=int(b), n=len(part), pairs=part.metal_pair.nunique(),
                x=float(part[column].median()), xmin=float(part[column].min()), xmax=float(part[column].max()),
                mean=float(part.separation_coefficient.mean()), q90=float(part.separation_coefficient.quantile(.9))))
    return pd.DataFrame(rows)


def save_figure(fig, output, name):
    for ext in ("png", "pdf"):
        fig.savefig(output / (name+"."+ext), dpi=200)
    plt.close(fig)


def draw_response(data, bo, bins, shapes, output, name):
    fig, axes = plt.subplots(2,2,figsize=(13,9), sharey=True)
    fig.subplots_adjust(left=.07, right=.98, top=.88, bottom=.1, hspace=.3, wspace=.18)
    for row, (cohort, part) in enumerate((("initial",data[data.cohort.ne("BO")]), (name,bo))):
        for col, mode in enumerate(("precursor", "icp")):
            ax = axes[row,col]
            x = mode+"_electronegativity"
            if row == 0:
                ax.scatter(part[x],part.separation_coefficient,s=14,color="#858b91",alpha=.35,linewidths=0)
            else:
                for pair, group in part.groupby("system"):
                    ax.scatter(group[x],group.separation_coefficient,s=40,color=COLORS[pair],marker=MARKERS[pair],edgecolor="white",linewidth=.5)
            b = bins[bins.cohort.eq(cohort)&bins.weighting.eq(mode)]
            ax.plot(b.x,b["mean"],"o-",color="#222222",markersize=4,label="Binned mean")
            ax.plot(b.x,b.q90,"s-",color="#bb2789",markersize=4,label="Binned 90th percentile")
            fit = shapes[shapes.cohort.eq(cohort)&shapes.weighting.eq(mode)].iloc[0]
            if fit.fit_available:
                xx=np.linspace(fit.xmin,fit.xmax,200)
                yy=np.polynomial.polynomial.polyval((xx-fit.center)/fit.scale,[fit.intercept,fit.linear,fit.quadratic])
                ax.plot(xx,yy,"--",color="#666666",lw=1,label="Unconstrained quadratic mean")
            ax.axvline(1.85,color="#aaaaaa",lw=.8,ls=":")
            ax.set_title(("Initial screening" if row==0 else f"BO: {bo.system.nunique()} metal pairs")+f" | {mode} | n={len(part)}",fontsize=11)
            ax.set_xlabel("Nominal precursor-weighted electronegativity" if mode=="precursor" else "ICP-weighted electronegativity")
            ax.grid(alpha=.13); ax.set_ylim(1.5,12.2)
            if col==0: ax.set_ylabel("H/D separation coefficient")
    axes[0,0].legend(fontsize=9,loc="lower center",frameon=False)
    handles=[Line2D([],[],ls="",marker=MARKERS[p],color=COLORS[p],label=p) for p in COLORS if p in set(bo.system)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.944),ncol=5,frameon=False)
    fig.suptitle("Electronegativity: observed response, mean and upper envelope",fontsize=15,y=.988)
    fig.text(.5,.025,"Same complete observations in both columns; no descriptor imputation. Dotted line: historical 1.85 reference, not a fitted optimum.",ha="center",fontsize=9)
    save_figure(fig,output,name+"_electronegativity")


def draw_geometry(data, output):
    pairs=(("electronegativity","FE_ads_avg"),("electronegativity","work_function"),("FE_ads_avg","work_function"))
    bo=data[data.cohort.eq("BO")]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    fig.subplots_adjust(left=.075,right=.98,bottom=.12,top=.86,wspace=.28,hspace=.42)
    for col,(a,b) in enumerate(pairs):
        xr=bo[[m+"_"+a for m in ("precursor","icp")]].to_numpy().ravel()
        yr=bo[[m+"_"+b for m in ("precursor","icp")]].to_numpy().ravel()
        xp=max(np.ptp(xr)*.12,.015); yp=max(np.ptp(yr)*.12,.015)
        for row,mode in enumerate(("precursor","icp")):
            ax=axes[row,col]; x,y=mode+"_"+a,mode+"_"+b
            initial=data[data.cohort.ne("BO")]
            ax.scatter(initial[x],initial[y],s=9,color="#bbbbbb",alpha=.35,linewidths=0)
            for pair,group in bo.groupby("system"):
                g=group.sort_values("actual_frac_element1" if mode=="icp" else "nominal_frac_element1")
                ax.plot(g[x],g[y],color=COLORS[pair],lw=.7,alpha=.6)
                ax.scatter(g[x],g[y],color=COLORS[pair],marker=MARKERS[pair],s=29,edgecolor="white",lw=.4)
            ax.set(xlim=(xr.min()-xp,xr.max()+xp),ylim=(yr.min()-yp,yr.max()+yp),xlabel=LABELS[a],ylabel=LABELS[b])
            ax.set_title("Nominal precursor weights" if mode=="precursor" else "ICP molar weights",fontsize=11)
            ax.grid(alpha=.12)
    handles=[Line2D([],[],ls="",marker=MARKERS[p],color=COLORS[p],label=p) for p in COLORS]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.946),ncol=5,frameon=False)
    fig.suptitle("BO trajectories in the original descriptor spaces | zoomed to BO ranges",fontsize=15,y=.99)
    fig.text(.5,.027,"For each fixed metal pair, every weighted descriptor is affine in the same molar fraction. Lines show geometry, not fitted mechanisms.",ha="center",fontsize=9)
    save_figure(fig,output,"bo_pair_trajectories")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT / "electronegativity_branches")
    parser.add_argument("--bootstrap",type=int,default=1000)
    args=parser.parse_args()
    if args.bootstrap<20: raise ValueError("Use at least 20 bootstrap draws")
    data,_,_,_,_,context=load_final()
    data=add_recipes(data)
    freq_file=REDUCED / "feature_frequency_comparison.csv"
    shap_file=OUTPUT / "shap33/feature_summary.csv"
    output=args.output.resolve()
    start_output(output,__file__,context,dict(bootstrap=args.bootstrap,seed=731,
        shape="unconstrained quadratic; descriptive in-sample R2, not CV",initial_resampling="unordered-pair cluster bootstrap",
        bo_resampling="within-pair bootstrap; adaptive BO dependence not modelled",bins="10 initial / 7 pooled BO equal-count bins",
        historical_reference=1.85),extra_files=[freq_file,shap_file])
    data["system"]=data.element1+"-"+data.element2
    for base in BASES:
        for mode in ("precursor","icp"):
            data[mode+"_"+base]=observed_weighted(data,base,mode)
    columns=[m+"_"+b for b in BASES for m in ("precursor","icp")]
    data[["sample","cohort","metal_pair","system","actual_frac_element1","nominal_frac_element1","separation_coefficient",*columns]].to_csv(output / "observed_descriptors.csv",index=False)
    complete=data[np.isfinite(data[[m+"_electronegativity" for m in ("precursor","icp")]]).all(axis=1)]
    initial=complete[complete.cohort.ne("BO")]
    bo=complete[complete.cohort.eq("BO")]
    four=bo[bo.system.ne("Co-Fe")]
    sets=[("initial",initial),("original_four_bo",four),("all_five_bo",bo)]
    sets += [(p,g) for p,g in bo.groupby("system")]
    shapes=[]; binned=[]; ranges=[]
    for i,(name,part) in enumerate(sets):
        for mode in ("precursor","icp"):
            x=mode+"_electronegativity"
            shapes.append(dict(cohort=name,weighting=mode,**shape_summary(part,x,args.bootstrap,731+i,"pair_cluster" if name=="initial" else "within_pair")))
            b=binned_summary(part,x,10 if name=="initial" else 7,12 if name=="initial" else 6)
            if not b.empty: binned.append(b.assign(cohort=name,weighting=mode))
            ranges.append(dict(cohort=name,weighting=mode,n=len(part),xmin=part[x].min(),xmax=part[x].max(),
                below_1_85=int((part[x]<1.85).sum()),above_1_85=int((part[x]>1.85).sum()),
                equal_1_85=int(np.isclose(part[x],1.85,atol=1e-10,rtol=0).sum())))
    shapes=pd.DataFrame(shapes); bins=pd.concat(binned,ignore_index=True)
    shapes.to_csv(output / "descriptive_shape_bootstrap.csv",index=False)
    bins.to_csv(output / "empirical_mean_q90_bins.csv",index=False)
    pd.DataFrame(ranges).to_csv(output / "pair_en_ranges.csv",index=False)
    freq=pd.read_csv(freq_file)
    freq=freq[freq.pool.eq(33)&freq.strategy.eq("forward10")&freq.feature.str.contains("electronegativity")]
    shap=pd.read_csv(shap_file)
    shap=shap[shap.kind.eq("nested33_forward10")&shap.feature.str.contains("electronegativity")]
    importance=freq.merge(shap,on=["arm","protocol","feature"],validate="one_to_one")
    importance.to_csv(output / "electronegativity_selection_shap.csv",index=False)
    plt.rcParams.update({"font.family":"Arial","font.size":10,"pdf.fonttype":42})
    for name,part in (("original_four_bo",four),("all_five_bo",bo)):
        draw_response(complete,part,bins,shapes,output,name)
    geometry=data[np.isfinite(data[columns]).all(axis=1)]
    draw_geometry(geometry,output)
    save_json(output / "verification.json",dict(complete=True,all_rows=len(data),initial_en=len(initial),
        original_four_bo=len(four),all_five_bo=len(bo),initial_geometry=int(geometry.cohort.ne("BO").sum()),
        bo_geometry=int(geometry.cohort.eq("BO").sum()),identical_rows_per_weighting=True,no_imputation=True,
        shape_metrics_are_descriptive_not_predictive=True))
    print(shapes.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
