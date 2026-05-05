
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.inspection import permutation_importance

ROOT = Path(r"C:\Users\yibinjiang\Documents\GitHub\HD_separation")
TMP = ROOT / "_tmp_bo_descriptor_validation"
TMP.mkdir(exist_ok=True)

full = pd.read_csv(ROOT / "hd_separation_ml/outputs/feature_selection_561_rs_from_eis_no_eis_fitting/feature_matrix_561_no_eis_fitting.csv")
final8 = pd.read_csv(ROOT / "hd_separation_ml/outputs/final_model_8features_561_seed17_rs_from_eis/feature_matrix_8features.csv")
bo = pd.read_csv(ROOT / "0-Figures/Figure4/curated/figure4_bo_weighted_ml_top3_descriptors.csv")

# Build weighted descriptors for the no-BO dataset. For initial bimetallic screening the nominal metal ratio is 6:1.
def frac_initial(row):
    if row["element1"] == row["element2"]:
        return 1.0, 0.0
    return 6/7, 1/7

fracs = full.apply(frac_initial, axis=1, result_type="expand")
full["frac1"] = fracs[0]
full["frac2"] = fracs[1]
full["weighted_electronegativity"] = full["frac1"]*full["electronegativity_element1"] + full["frac2"]*full["electronegativity_element2"]
full["weighted_FE_ads_avg"] = full["frac1"]*full["FE_ads_avg_element1"] + full["frac2"]*full["FE_ads_avg_element2"]
full["weighted_delta_EOH"] = full["frac1"]*full["delta_EOH_element1"] + full["frac2"]*full["delta_EOH_element2"]

weighted_features = ["weighted_electronegativity", "weighted_FE_ads_avg", "weighted_delta_EOH"]
y = full["separation_coefficient"].astype(float)
Xw = full[weighted_features].astype(float)

# Projection-level validation: how much of the no-BO ML dataset can be represented by these three weighted descriptors?
Xtr, Xte, ytr, yte = train_test_split(Xw, y, test_size=0.2, random_state=17)
models = {
    "quadratic_ridge": make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), RidgeCV(alphas=np.logspace(-4, 4, 41))),
    "extra_trees_3desc": ExtraTreesRegressor(n_estimators=500, random_state=17, min_samples_leaf=3),
    "random_forest_3desc": RandomForestRegressor(n_estimators=500, random_state=17, min_samples_leaf=3),
}
rows=[]
for name, model in models.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    cv = cross_val_score(model, Xw, y, cv=KFold(n_splits=5, shuffle=True, random_state=17), scoring="r2")
    rows.append({
        "model": name,
        "test_R2": r2_score(yte, pred),
        "test_RMSE": np.sqrt(mean_squared_error(yte, pred)),
        "cv_R2_mean": cv.mean(),
        "cv_R2_std": cv.std(),
    })
model_summary = pd.DataFrame(rows).sort_values("test_R2", ascending=False)
model_summary.to_csv(TMP / "weighted_descriptor_model_validation.csv", index=False)
print("3-descriptor projection validation")
print(model_summary.to_string(index=False))

# Use best projection surrogate for volcano-like visualization.
best_name = model_summary.iloc[0]["model"]
best = models[best_name]
best.fit(Xw, y)

# Reference final 8-feature model validation, only to document that this is a projection surrogate, not the full model.
features8 = ["I_raw", "work_function_element2", "MM_bond_element2", "FE_ads_avg_element1", "delta_EOH_element1", "delta_EOH_element2", "electronegativity_element1", "electronegativity_element2"]
X8 = pd.DataFrame({
    "I_raw": final8["I_raw.1"],
    "work_function_element2": final8["work_function_element2"],
    "MM_bond_element2": final8["MM_bond_element2"],
    "FE_ads_avg_element1": final8["FE_ads_avg_element1"],
    "delta_EOH_element1": final8["delta_EOH_element1"],
    "delta_EOH_element2": final8["delta_EOH_element2"],
    "electronegativity_element1": final8["electronegativity_element1"],
    "electronegativity_element2": final8["electronegativity_element2"],
})
y8 = final8["separation_coefficient"]
y_bins = pd.qcut(y8, q=10, labels=False, duplicates="drop")
X8tr, X8te, y8tr, y8te = train_test_split(X8, y8, test_size=0.2, random_state=17, stratify=y_bins)
model8 = ExtraTreesRegressor(random_state=17, max_depth=15, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
model8.fit(X8tr, y8tr)
p8 = model8.predict(X8te)
full8_summary = pd.DataFrame([{"model":"current_final_8feature_extra_trees", "test_R2":r2_score(y8te,p8), "test_RMSE":np.sqrt(mean_squared_error(y8te,p8))}])
full8_summary.to_csv(TMP / "current_8feature_model_reference.csv", index=False)

# Build dense grids in each BO system's composition space.
# Constraint follows platform: v1 in 1-19, v2 in 1-19, Na3Cit in 0-18, water >= 0, total 20 mL.
# We sample at 0.5 mL resolution to keep file manageable.
systems = [("Co-Mo", "Co", "Mo"), ("Co-V", "Co", "V"), ("Co-Ni", "Co", "Ni"), ("Fe-Ni", "Fe", "Ni")]
# Element descriptor lookup from the full initial matrix.
lookup = {}
for _, r in full.iterrows():
    lookup.setdefault(r["element1"], {
        "electronegativity": r["electronegativity_element1"],
        "FE_ads_avg": r["FE_ads_avg_element1"],
        "delta_EOH": r["delta_EOH_element1"],
    })
    lookup.setdefault(r["element2"], {
        "electronegativity": r["electronegativity_element2"],
        "FE_ads_avg": r["FE_ads_avg_element2"],
        "delta_EOH": r["delta_EOH_element2"],
    })

def weighted_row(system, e1, e2, v1, v2, vcit):
    f1 = v1/(v1+v2)
    f2 = v2/(v1+v2)
    return {
        "system": system, "substance1": e1, "substance2": e2,
        "volume1": v1, "volume2": v2, "volume3": vcit,
        "frac1": f1, "frac2": f2,
        "weighted_electronegativity": f1*lookup[e1]["electronegativity"] + f2*lookup[e2]["electronegativity"],
        "weighted_FE_ads_avg": f1*lookup[e1]["FE_ads_avg"] + f2*lookup[e2]["FE_ads_avg"],
        "weighted_delta_EOH": f1*lookup[e1]["delta_EOH"] + f2*lookup[e2]["delta_EOH"],
    }

grid_rows=[]
vals = np.round(np.arange(1.0, 19.0+1e-9, 0.5), 2)
cit_vals = np.round(np.arange(0.0, 18.0+1e-9, 0.5), 2)
for system,e1,e2 in systems:
    for v1 in vals:
        for v2 in vals:
            rem = 20 - v1 - v2
            if rem < -1e-9:
                continue
            for vcit in cit_vals[cit_vals <= min(18, rem)+1e-9]:
                grid_rows.append(weighted_row(system,e1,e2,float(v1),float(v2),float(vcit)))
grid = pd.DataFrame(grid_rows)
grid["predicted_SHD_projection_model"] = best.predict(grid[weighted_features])
grid.to_csv(TMP / "bo_space_grid_weighted_descriptor_predictions.csv", index=False)

# Prediction summaries by system and top candidates from the descriptor surface.
top_grid = grid.sort_values("predicted_SHD_projection_model", ascending=False).groupby("system").head(10)
top_grid.to_csv(TMP / "top_predicted_grid_points_by_projection_model.csv", index=False)

# BO measured data vs projection model.
bo = bo.copy()
bo["predicted_SHD_projection_model"] = best.predict(bo[weighted_features])
bo_eval=[]
for sys, d in bo.groupby("system"):
    corr = d[["separation_coefficient", "predicted_SHD_projection_model"]].corr().iloc[0,1]
    bo_eval.append({
        "system": sys,
        "n_BO": len(d),
        "BO_measured_vs_projection_pred_corr": corr,
        "BO_measured_SHD_max": d["separation_coefficient"].max(),
        "projection_pred_SHD_max_on_BO_points": d["predicted_SHD_projection_model"].max(),
        "measured_best_sample": d.loc[d["separation_coefficient"].idxmax(), "sample"],
        "predicted_best_BO_sample": d.loc[d["predicted_SHD_projection_model"].idxmax(), "sample"],
    })
bo_eval = pd.DataFrame(bo_eval)
bo_eval.to_csv(TMP / "bo_measured_vs_projection_model_eval.csv", index=False)
print("\nBO measured vs projection model")
print(bo_eval.to_string(index=False))

# Plot helper.
pairs = [
    ("weighted_electronegativity", "weighted_FE_ads_avg", "Weighted electronegativity", "Weighted H adsorption energy / eV"),
    ("weighted_electronegativity", "weighted_delta_EOH", "Weighted electronegativity", "Weighted OH adsorption energy / eV"),
    ("weighted_FE_ads_avg", "weighted_delta_EOH", "Weighted H adsorption energy / eV", "Weighted OH adsorption energy / eV"),
]

for xcol,ycol,xlab,ylab in pairs:
    fig, axes = plt.subplots(1,4, figsize=(18,4.2), sharex=False, sharey=False)
    for ax,(system,e1,e2) in zip(axes, systems):
        gd = grid[grid.system==system]
        bd = bo[bo.system==system]
        sc = ax.scatter(gd[xcol], gd[ycol], c=gd["predicted_SHD_projection_model"], cmap="viridis", s=8, alpha=0.22, linewidths=0)
        # measured BO samples overlay
        ax.scatter(bd[xcol], bd[ycol], c=bd["separation_coefficient"], cmap="Reds", s=45, edgecolors="black", linewidths=0.45, alpha=0.9)
        # high-separation BO points: top 20% within system
        q = bd["separation_coefficient"].quantile(0.8)
        hd = bd[bd["separation_coefficient"]>=q]
        ax.scatter(hd[xcol], hd[ycol], marker="*", s=130, facecolors="none", edgecolors="black", linewidths=1.1)
        ax.set_title(system, fontweight="bold")
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(ylab)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9, pad=0.01)
    cbar.set_label("Predicted SH/D from 3-descriptor projection model")
    fig.suptitle(f"Projected BO composition space: {xlab} vs {ylab}\nTransparent field = ML-derived descriptor surface; red points = measured BO samples; stars = top 20% measured SH/D", y=1.05)
    fig.savefig(TMP / f"projection_surface_{xcol}_vs_{ycol}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Aggregate 3D pair plot for all systems together.
fig, axes = plt.subplots(1,3, figsize=(15,4.4))
for ax,(xcol,ycol,xlab,ylab) in zip(axes,pairs):
    sc = ax.scatter(grid[xcol], grid[ycol], c=grid["predicted_SHD_projection_model"], cmap="viridis", s=5, alpha=0.07, linewidths=0)
    ax.scatter(bo[xcol], bo[ycol], c=bo["separation_coefficient"], cmap="Reds", s=45, edgecolors="black", linewidths=0.4)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.2)
fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01, label="Predicted SH/D from projection model")
fig.suptitle("All four BO systems in ML-derived weighted-descriptor space", y=1.03)
fig.savefig(TMP / "all_systems_projection_surface_three_pairs.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Save concise interpretation notes.
notes = []
notes.append("This temporary validation distinguishes the final 8-feature model from a low-dimensional projection surrogate.\n")
notes.append("The final 8-feature model does not contain precursor volume or metal fraction inputs, so it cannot directly produce a continuous prediction field over BO composition ratios without additional assumptions.\n")
notes.append(f"Best 3-descriptor projection model: {best_name}. See weighted_descriptor_model_validation.csv for held-out and CV R2.\n")
notes.append("The projection surrogate is useful only as a qualitative descriptor-map/volcano check, not as the formal predictive model used in the manuscript.\n")
(TMP / "README_interpretation.txt").write_text(''.join(notes), encoding="utf-8")
print(f"\nOutputs written to: {TMP}")
