"""Reviewer correlation matrices, using observed properties and audited eta10."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, false_discovery_control

from revision_followup_common import ROOT, BASES, FEATURES, label, load_current, start_output, save_json

HISTORICAL = ROOT / "outputs/feature_selection_561_rs_from_eis_no_eis_fitting/feature_matrix_561_no_eis_fitting.csv"
ETA = ROOT.parent / "0-Figures/Figure1/data/figure1_this_work_eta10_censoring.csv"
COVARIATES = ["I_raw", "Rs_from_EIS", "E_IR_corrected_final"]
OUTCOMES = ["separation_coefficient", "eta10", *COVARIATES]
OUTCOME_LABELS = ["H/D separation", "Exact eta10", "Current at -1 V", "EIS Rs", "Corrected potential"]


def observed_features(data):
    out = data[[f"{b}_{s}" for b in BASES for s in ("element1", "element2")]].copy()
    f = data.actual_frac_element1.to_numpy(float)
    for b in BASES:
        a, c = data[f"{b}_element1"].to_numpy(float), data[f"{b}_element2"].to_numpy(float)
        # A missing property of a genuinely absent component does not affect its mean.
        out[f"icp_weighted_{b}"] = np.where(f == 0, 0., f*a) + np.where(f == 1, 0., (1-f)*c)
    return out[FEATURES]


def audited_initial(data):
    data = data[data.cohort.eq("initial")].copy()
    old = pd.read_csv(HISTORICAL).set_index("sample")
    eta = pd.read_csv(ETA).set_index("sample")
    if not old.index.is_unique or not eta.index.is_unique:
        raise ValueError("Nonunique source identifiers")
    old, eta = old.loc[data["sample"]], eta.loc[data["sample"]]
    np.testing.assert_allclose(data.separation_coefficient, old.separation_coefficient, atol=1e-10)
    np.testing.assert_allclose(data.separation_coefficient, eta.separation_coefficient, atol=1e-10)
    for c in ("I_raw", "Rs_from_EIS"):
        np.testing.assert_allclose(old[c], old[c+".1"], atol=1e-12)
    np.testing.assert_allclose(old.E_IR_corrected_final, old.E_actual + .9*.05*old.Rs_from_EIS)
    for c in COVARIATES:
        if c in data:
            np.testing.assert_allclose(data[c], old[c], atol=1e-10)
        data[c] = old[c].to_numpy()
    if not np.isfinite(data[COVARIATES].to_numpy(float)).all():
        raise ValueError("Missing operating covariates")
    data["potential_offset"] = data.E_IR_corrected_final + .098
    for c in FEATURES[:26]:
        known = data[c].notna()
        shifted = data[c].copy()
        if c.startswith("work_function_"):
            shifted += data.potential_offset
        elif c.startswith(("pzc_", "E_std_")):
            shifted -= data.potential_offset
        np.testing.assert_allclose(shifted[known], old.loc[data.loc[known,"sample"],c], atol=1e-9)
    data["eta10_status"] = eta.eta10_status.to_numpy()
    exact = eta.eta10_status.eq("exact_crossing") & ~eta.is_eta10_below_10mV_excluded_from_pareto
    data["eta10"] = eta.eta10_exact_mV.where(exact).to_numpy()
    turning = eta.eta10_status.eq("left_censored_turning_point_estimate")
    estimates = eta.eta10_exact_mV.where(exact).combine_first(eta.turning_point_eta_estimate_mV.where(turning))
    if int(estimates.notna().sum()) != int(exact.sum()+turning.sum()):
        raise ValueError("Missing eta10 estimate in source column")
    data["eta10_including_turning_estimates"] = estimates.to_numpy()
    data["eta10_plot_position_not_measurement"] = eta.plot_x_overpotential_mV.to_numpy()
    return data


def shifted_features(features, offset):
    out = features.copy()
    for base, sign in (("work_function",1), ("pzc",-1), ("E_std",-1)):
        for c in (f"{base}_element1", f"{base}_element2", f"icp_weighted_{base}"):
            out[c] += sign*np.asarray(offset)
    return out


def correlations(frame, columns):
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i+1:]:
            part = frame[[a,b]].dropna()
            for method, fun in (("pearson",pearsonr), ("spearman",spearmanr)):
                valid = len(part) >= 3 and part[a].nunique() > 1 and part[b].nunique() > 1
                r, p = fun(part[a], part[b]) if valid else (np.nan,np.nan)
                rows.append(dict(feature_a=a,feature_b=b,method=method,n=len(part),r=float(r),p_nominal=float(p)))
    result = pd.DataFrame(rows)
    result["q_bh_exploratory"] = np.nan
    for _, part in result.groupby("method"):
        ids = part.index[part.p_nominal.notna()]
        result.loc[ids,"q_bh_exploratory"] = false_discovery_control(result.loc[ids,"p_nominal"].to_numpy())
    return result


def plot(data, output):
    plt.rcParams.update({"font.family":"Arial","font.size":10})
    fig, axes = plt.subplots(1,2,figsize=(13,14))
    fig.subplots_adjust(left=.25,right=.90,bottom=.12,top=.94,wspace=.10)
    for ax, method in zip(axes, ("pearson","spearman")):
        arr = data[data.method.eq(method)].pivot(index="feature_a",columns="feature_b",values="r").reindex(index=FEATURES,columns=OUTCOMES)
        im = ax.imshow(arr, cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
        ax.set_xticks(range(5),OUTCOME_LABELS,rotation=40,ha="right")
        ax.set_yticks(range(39),[label(f) for f in FEATURES] if ax is axes[0] else [])
        for i in range(39):
            for j in range(5):
                v = arr.iloc[i,j]
                if np.isfinite(v):
                    ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8,color="white" if abs(v)>.65 else "black")
        ax.set_title(method.title())
    fig.colorbar(im,cax=fig.add_axes([.925,.3,.014,.4]),label="Correlation")
    fig.suptitle("Observed intrinsic descriptors and electrochemical measurements",y=.985,fontsize=14)
    fig.text(.5,.015,"Initial screening; pairwise complete values, no imputation. Exact eta10 only; pairwise sample counts are exported.",ha="center",fontsize=10)
    for ext in ("png","pdf"):
        fig.savefig(output / f"descriptor_outcome_correlations.{ext}",dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT / "outputs/reviewer_stage2_correlations_20261013")
    args = parser.parse_args()
    data,_,_,_,_,context = load_current()
    output = args.output.resolve()
    start_output(output,__file__,context,dict(cohort="initial",missing="pairwise observed; no imputation",
        eta="exact crossings only; estimated/censored values excluded",historical_shift="WF + offset; PZC and E_std - offset; offset=E_IR+.098"),
        extra_files=[HISTORICAL,ETA,ROOT / "icp_historical_model_bridge.py"])
    data = audited_initial(data)
    data.to_csv(output / "audited_initial_dataset.csv",index=False)
    data.groupby("eta10_status").size().rename("n").to_csv(output / "eta10_status_counts.csv")
    blocks=[]
    for kind, features in (("intrinsic",observed_features(data)),
                           ("potential_shifted",shifted_features(observed_features(data),data.potential_offset))):
        frame = pd.concat([features,data[OUTCOMES]],axis=1)
        frame.to_csv(output / f"{kind}_correlation_inputs.csv",index_label="sample")
        corr = correlations(frame,FEATURES+OUTCOMES)
        corr["representation"] = kind
        blocks.append(corr)
        for method in ("pearson","spearman"):
            frame.corr(method=method,min_periods=3).to_csv(output / f"{kind}_{method}_matrix.csv")
        frame.notna().astype(int).T.dot(frame.notna().astype(int)).to_csv(output / f"{kind}_pairwise_n.csv")
    result = pd.concat(blocks,ignore_index=True)
    result.to_csv(output / "all_correlations_long.csv",index=False)
    cross = result[result.feature_a.isin(FEATURES) & result.feature_b.isin(OUTCOMES)]
    cross.to_csv(output / "descriptor_outcome_correlations.csv",index=False)
    plot(cross[cross.representation.eq("intrinsic")],output)
    secondary = pd.concat([observed_features(data),data[["eta10_including_turning_estimates"]]],axis=1)
    correlations(secondary,FEATURES+["eta10_including_turning_estimates"]).query("feature_b == 'eta10_including_turning_estimates'").to_csv(
        output / "eta10_estimate_sensitivity.csv",index=False)
    direct = result[result.representation.eq("intrinsic") & result.feature_a.eq("separation_coefficient")]
    lines = ["# Reviewer correlation audit", "",f"Initial cohort: {len(data)}. Exact eta10: {data.eta10.notna().sum()}. Including four turning estimates: {data.eta10_including_turning_estimates.notna().sum()}.","",
        "All source joins are one-to-one; H/D targets agree exactly. Duplicate historical current/Rs columns agree. Historical potential shifts reproduce archived values.","",
        "Correlations are descriptive, not evidence of causality or leakage. Pairwise n varies with missing elemental references and eta10 censoring. Nominal p and exploratory BH q assume independent rows; related metal pairs violate that assumption, so they must not be used as confirmatory significance claims.","",
        "I_raw is the archived signed, area-normalized LSV current near -1 V. Its historical extraction is retained, not reinterpreted as the -50 mA measurement current. E_IR = E_actual + 0.9*0.05*Rs. Offset = E_IR + 0.098 V. Potential-shifted means are retrospective diagnostics, not intrinsic properties.","",
        "| Variable | Pearson r | Spearman rho | n |","|---|---:|---:|---:|"]
    for f in OUTCOMES[1:]:
        p=direct[direct.feature_b.eq(f)].set_index("method")
        lines.append(f"| {f} | {p.loc['pearson','r']:.3f} | {p.loc['spearman','r']:.3f} | {p.loc['pearson','n']} |")
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save_json(output / "verification.json",dict(complete=True,n_initial=len(data),n_eta_exact=int(data.eta10.notna().sum()),
        n_eta_with_estimates=int(data.eta10_including_turning_estimates.notna().sum()),source_targets_equal=True,covariates_complete=True,
        original_models_unchanged=True))
    print("\n".join(lines),flush=True)


if __name__ == "__main__":
    main()
