"""Exploratory ICP descriptor comparison without electrochemical covariates.

Use the latest composition policy, original elemental properties, training-fold
imputation, and repeated random / unordered-metal-pair CV. No feature selection
or hyperparameter search is performed. Interpretation is predictive, not causal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from icp_composition_weighted_sensitivity import COMPOSITION_TABLE, FEATURE_MATRIX
from icp_master import load_initial_composition

ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "EH_extra.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs" / "icp_intrinsic_descriptors_20260928"
BASES = [
    "work_function", "pzc", "MM_bond", "E_std", "FE_ads_avg", "delta_EOH",
    "electronegativity", "MH_bond_kJ", "MD_bond_kJ", "Frequency_OH",
    "Frequency_H", "Frequency_D", "delta_ZPE_H",
]
LABELS = {
    "work_function": "Work function", "pzc": "PZC", "MM_bond": "M-M bond",
    "E_std": "Standard potential", "FE_ads_avg": "H adsorption",
    "delta_EOH": "OH adsorption", "electronegativity": "Electronegativity",
    "MH_bond_kJ": "M-H bond", "MD_bond_kJ": "M-D bond",
    "Frequency_OH": "OH vibration", "Frequency_H": "H vibration",
    "Frequency_D": "D vibration", "delta_ZPE_H": "H/D ZPE difference",
}
SCENARIOS = {
    "nominal13": "Precursor-weighted 13",
    "icp13": "ICP-weighted 13",
    "raw26": "Separate elemental 26",
    "raw26_fraction": "Separate 26 + ICP fraction",
}
FAMILIES = {
    **{LABELS[b]: [b] for b in BASES if b not in
       {"MH_bond_kJ", "MD_bond_kJ", "Frequency_H", "Frequency_D", "delta_ZPE_H"}},
    "H/D bond energies": ["MH_bond_kJ", "MD_bond_kJ"],
    "H/D vibration and ZPE": ["Frequency_H", "Frequency_D", "delta_ZPE_H"],
}
# Same reference values as notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb.
E_STD = {
    "Ag": 0.80, "Al": -1.66, "Au": 1.52, "Ba": -2.91, "Ca": -2.87,
    "Cd": -0.40, "Ce": -2.34, "Co": -0.28, "Cr": -0.74, "Cu": 0.34,
    "Eu": -2.81, "Fe": -0.44, "Ga": -0.53, "In": -0.34, "La": -2.38,
    "Mg": -2.37, "Mn": -1.19, "Mo": -0.20, "Nb": -1.10, "Ni": -0.25,
    "Pb": -0.13, "Pd": 0.92, "Pt": 1.20, "Ru": 0.455, "Sc": -2.08,
    "Sr": -2.90, "V": -1.18, "W": -0.12, "Y": -2.37, "Zn": -0.76,
}


def load_data():
    elemental = pd.read_excel(WORKBOOK, sheet_name="Sheet1").rename(columns={
        "Unnamed: 0": "element", "Work function (eV)": "work_function",
        "M-M": "MM_bond", "E(M-H) (kJ/mol)": "MH_bond_kJ",
        "E(M-D) (kJ/mol)": "MD_bond_kJ", "delta_ZPE_H (eV)": "delta_ZPE_H",
    })
    adsorption = pd.read_excel(WORKBOOK, sheet_name="Sheet2").rename(
        columns={"Unnamed: 0": "element"})
    adsorption["element"] = adsorption["element"].ffill()
    averages = adsorption.groupby("element")["Free Energy of Adsorption (eV)"].mean()
    elemental["FE_ads_avg"] = elemental["element"].map(averages)
    elemental["E_std"] = elemental["element"].map(E_STD)
    elemental = elemental.set_index("element")[BASES].apply(pd.to_numeric, errors="raise")
    if not elemental.index.is_unique:
        raise ValueError("Elemental property table contains duplicate elements.")

    composition = load_initial_composition(COMPOSITION_TABLE)
    meta = ["sample", "element1", "element2", "separation_coefficient"]
    frac = ["actual_frac_element1", "actual_frac_element2",
            "nominal_frac_element1", "nominal_frac_element2"]
    data = composition[[*meta, *frac, "merge_status", "matched_zero_component_accepted"]].copy()
    if data[meta + frac].isna().any().any():
        raise ValueError("Missing response, identity, or composition.")
    if not ((data[frac] >= 0) & (data[frac] <= 1)).all().all():
        raise ValueError("Fractions must lie in [0, 1].")
    for prefix in ("actual", "nominal"):
        np.testing.assert_allclose(data[f"{prefix}_frac_element1"] +
            data[f"{prefix}_frac_element2"], 1, atol=1e-10)
    data["metal_pair"] = data[["element1", "element2"]].apply(
        lambda row: "-".join(sorted(row)), axis=1)
    needed_elements = sorted(set(data.element1) | set(data.element2))
    elemental = elemental.reindex(needed_elements)
    for side in ("element1", "element2"):
        for base in BASES:
            data[f"{base}_{side}"] = data[side].map(elemental[base])

    # Independently check the unshifted values against the existing raw-property export.
    archived = pd.read_csv(ROOT / "outputs" / "03_eta10_activity_model_raw_descriptors" /
        "feature_matrix_561_with_eta10_raw_electronic_descriptors.csv").set_index("sample")
    for base in BASES:
        for side in ("element1", "element2"):
            col = f"{base}_{side}"
            ok = data[col].notna()
            np.testing.assert_allclose(data.loc[ok, col],
                archived.loc[data.loc[ok, "sample"], col], atol=1e-9)
    return data, elemental


class DescriptorRepresentation(BaseEstimator, TransformerMixin):
    def __init__(self, scenario):
        self.scenario = scenario

    def fit(self, X, y=None):
        self.medians_ = {}
        for base in BASES:
            values = X[[f"{base}_element1", f"{base}_element2"]].to_numpy(float)
            self.medians_[base] = float(np.nanmedian(values))
            if not np.isfinite(self.medians_[base]):
                raise ValueError(f"No training values for {base}")
        return self

    def transform(self, X):
        out = {}
        for base in BASES:
            v1 = X[f"{base}_element1"].fillna(self.medians_[base]).to_numpy(float)
            v2 = X[f"{base}_element2"].fillna(self.medians_[base]).to_numpy(float)
            if self.scenario in ("icp13", "nominal13"):
                prefix = "actual" if self.scenario == "icp13" else "nominal"
                f = X[f"{prefix}_frac_element1"].to_numpy(float)
                out[base] = f * v1 + (1 - f) * v2
            else:
                out[f"{base}_element1"] = v1
                out[f"{base}_element2"] = v2
        if self.scenario == "raw26_fraction":
            out["actual_frac_element1"] = X["actual_frac_element1"].to_numpy(float)
        return pd.DataFrame(out, index=X.index)


def scores(y, pred):
    return {"r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred))}


def interpret(model, X, y, seed):
    # Permute correlated isotope-derived quantities together as well as individually.
    groups = [("individual", b, [b]) for b in BASES]
    groups += [("family", name, bases) for name, bases in FAMILIES.items()]
    rng = np.random.default_rng(seed)
    batch, keys = [], []
    arr = X.to_numpy(float)
    for kind, name, bases in groups:
        columns = [X.columns.get_loc(b) for b in bases]
        for rep in range(5):
            changed = arr.copy()
            changed[:, columns] = arr[rng.permutation(len(arr))][:, columns]
            batch.append(changed)
            keys.append((kind, name, rep))
    base_score = r2_score(y, model.predict(arr))
    prediction = model.predict(np.concatenate(batch)).reshape(len(batch), len(arr))
    permutation_rows = [{"kind": kind, "feature": name, "permutation": rep,
        "r2_drop": float(base_score - r2_score(y, pred))}
        for (kind, name, rep), pred in zip(keys, prediction)]
    explanation = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    values = np.asarray(explanation.shap_values(arr, check_additivity=True))
    np.testing.assert_allclose(values.sum(axis=1) + float(np.asarray(
        explanation.expected_value).ravel()[0]), model.predict(arr), atol=1e-6)
    individual = pd.Series(np.abs(values).mean(axis=0), index=BASES)
    ranks = individual.rank(ascending=False, method="min")
    shap_rows = [{"feature": b, "mean_abs_shap": float(individual[b]),
        "rank": float(ranks[b]), "top3": bool(ranks[b] <= 3)} for b in BASES]
    return permutation_rows, shap_rows


def run(data, args):
    y = data.separation_coefficient.to_numpy(float)
    strata = pd.qcut(y, 5, labels=False, duplicates="drop")
    groups = data.metal_pair.to_numpy()
    metrics, predictions, permutation, shap_rows, split_rows = [], [], [], [], []
    for protocol in ("random", "metal_pair_grouped"):
        for repeat in range(args.repeats):
            seed = 42 + repeat
            if protocol == "random":
                splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
                folds = splitter.split(data, strata)
            else:
                splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                folds = splitter.split(data, strata, groups)
            for fold, (train, test) in enumerate(folds):
                common = {"protocol": protocol, "repeat": repeat, "fold": fold}
                overlap = len(set(groups[train]) & set(groups[test]))
                if protocol == "metal_pair_grouped" and overlap:
                    raise ValueError("Metal-pair leakage in grouped CV.")
                for idx in test:
                    split_rows.append({**common, "sample": data.iloc[idx]["sample"],
                        "metal_pair": groups[idx], "train_test_pair_overlap": overlap})
                baseline = DummyRegressor().fit(np.zeros((len(train), 1)), y[train])
                pred = baseline.predict(np.zeros((len(test), 1)))
                metrics.append({**common, "scenario": "dummy", "model": "Mean baseline",
                    "n_train": len(train), "n_test": len(test), **scores(y[test], pred)})
                for scenario in SCENARIOS:
                    transform = DescriptorRepresentation(scenario).fit(data.iloc[train])
                    train_x, test_x = transform.transform(data.iloc[train]), transform.transform(data.iloc[test])
                    for model_name, cls in [("RF", RandomForestRegressor), ("ET", ExtraTreesRegressor)]:
                        model = cls(n_estimators=args.trees, max_features="sqrt",
                            random_state=seed, n_jobs=4)
                        model.fit(train_x.to_numpy(), y[train])
                        model.n_jobs = 1
                        pred = model.predict(test_x.to_numpy())
                        key = {**common, "scenario": scenario, "model": model_name}
                        metrics.append({**key, "n_train": len(train), "n_test": len(test),
                            **scores(y[test], pred)})
                        for idx, estimate in zip(test, pred):
                            predictions.append({**key, "sample": data.iloc[idx]["sample"],
                                "y_true": y[idx], "y_pred": float(estimate)})
                        if protocol == "metal_pair_grouped" and scenario == "icp13":
                            p, s = interpret(model, test_x, y[test], seed * 100 + fold)
                            permutation.extend({**key, **row} for row in p)
                            shap_rows.extend({**key, **row} for row in s)
                print(f"Completed {protocol} repeat {repeat + 1}/{args.repeats}, fold {fold + 1}/5", flush=True)
            pd.DataFrame(metrics).to_csv(args.output / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output / "cv_assignments.csv", index=False)
    pd.DataFrame(permutation).to_csv(args.output / "permutation_importance_folds.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(args.output / "shap_importance_folds.csv", index=False)
    summary = pd.DataFrame(metrics).groupby(["protocol", "scenario", "model"], as_index=False).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"), rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"), n_folds=("r2", "size"))
    summary.to_csv(args.output / "model_summary.csv", index=False)
    p = pd.DataFrame(permutation).groupby(
        ["model", "repeat", "fold", "kind", "feature"], as_index=False).r2_drop.mean()
    p = p.groupby(["model", "kind", "feature"], as_index=False).agg(
        r2_drop_mean=("r2_drop", "mean"), r2_drop_std=("r2_drop", "std"))
    p.to_csv(args.output / "permutation_importance_summary.csv", index=False)
    s = pd.DataFrame(shap_rows).groupby(["model", "feature"], as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"), std_abs_shap=("mean_abs_shap", "std"),
        mean_rank=("rank", "mean"), top3_fraction=("top3", "mean"))
    s.to_csv(args.output / "shap_importance_summary.csv", index=False)
    return summary, p, s


def plot(summary, permutation, shap_summary, output):
    plt.rcParams.update({"font.family": "Arial", "font.size": 11,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.0), layout="constrained")
    colors = {"RF": "#237b95", "ET": "#d47c27"}
    y = np.arange(len(SCENARIOS))
    for ax, protocol, title in zip(axes[0], ("random", "metal_pair_grouped"),
            ("Random cross-validation", "Metal-pair grouped cross-validation")):
        for i, model in enumerate(colors):
            sub = summary[(summary.protocol == protocol) & (summary.model == model)].set_index(
                "scenario").reindex(SCENARIOS)
            ax.errorbar(sub.r2_mean, y + (i - .5) * .18, xerr=sub.r2_std,
                fmt="o", markersize=6, capsize=3, color=colors[model], label=model)
        ax.set_yticks(y, list(SCENARIOS.values()))
        ax.invert_yaxis()
        ax.set_xlim(.28, .62)
        ax.set_xlabel(r"Validation $R^2$ (mean and fold SD)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=.22)
        ax.legend(frameon=False)
    ordering = shap_summary.groupby("feature").mean_abs_shap.mean().sort_values().index
    for i, model in enumerate(colors):
        sub = shap_summary[shap_summary.model == model].set_index("feature").reindex(ordering)
        axes[1, 0].barh(np.arange(len(ordering)) + (i - .5) * .35, sub.mean_abs_shap,
            height=.34, color=colors[model], label=model)
    axes[1, 0].set_yticks(np.arange(len(ordering)), [LABELS[b] for b in ordering])
    axes[1, 0].set_xlabel("Mean absolute validation SHAP value")
    axes[1, 0].set_title("ICP-weighted descriptors: grouped CV", fontweight="bold")
    fam = permutation[permutation.kind == "family"]
    ordering = fam.groupby("feature").r2_drop_mean.mean().sort_values().index
    for i, model in enumerate(colors):
        sub = fam[fam.model == model].set_index("feature").reindex(ordering)
        axes[1, 1].errorbar(sub.r2_drop_mean, np.arange(len(ordering)) + (i - .5) * .18,
            xerr=sub.r2_drop_std, fmt="o", markersize=5, capsize=2, color=colors[model])
    axes[1, 1].set_yticks(np.arange(len(ordering)), ordering)
    axes[1, 1].axvline(0, color="0.55", linewidth=.8)
    axes[1, 1].set_xlabel(r"Validation $R^2$ drop after family permutation")
    axes[1, 1].set_title("Predictive importance of descriptor families", fontweight="bold")
    fig.suptitle("ICP composition and intrinsic elemental descriptors | 554 samples", fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"icp_intrinsic_descriptor_comparison.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    data, elemental = load_data()
    print(f"Samples={len(data)}, metal pairs={data.metal_pair.nunique()}, "
        f"pure metals={data.element1.eq(data.element2).sum()}", flush=True)
    print("Elemental-property NaNs:\n" + elemental.isna().sum().to_string(), flush=True)
    if args.audit_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / "analysis_dataset_unimputed.csv", index=False)
    elemental.to_csv(args.output / "elemental_properties_unimputed.csv")
    policy = {
        "n_samples": len(data), "n_unordered_pairs": data.metal_pair.nunique(),
        "pure_metals": int(data.element1.eq(data.element2).sum()),
        "zero_component_rows": int(data.matched_zero_component_accepted.astype(bool).sum()),
        "repeats": args.repeats, "folds": 5, "seeds": list(range(42, 42 + args.repeats)),
        "models": {"trees": args.trees, "max_features": "sqrt", "tuning": "none"},
        "scenarios": SCENARIOS,
        "missing_values": "Original workbook NaNs; pooled endpoint median learned on training fold before weighting.",
        "electrochemical_covariates": "Excluded: I_raw, Rs, eta10, sample potentials and potential shifts.",
        "target": "H/D separation coefficient; initial screening only; BO excluded.",
        "weighting": "Metal-normalized molar fraction; empirical linear descriptor; not measured surface property.",
        "composition_policy": "Latest 20260927 policy; 47 single-zero rows retained; W retained except W-Nb; 6 Au bimetals excluded.",
        "shap": "Tree-path-dependent SHAP on grouped validation folds; no post-hoc feature selection.",
        "permutation": "Validation R2 drop, 5 shuffles per fold, individual and isotope-related families.",
        "uncertainty": "Fold SD, not confidence intervals; repeated CV folds overlap in training data.",
        "status": "Exploratory fixed-model comparison; not a complete reviewer robustness study or causal identification.",
        "versions": {"sklearn": sklearn.__version__, "shap": shap.__version__},
        "inputs_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [WORKBOOK, FEATURE_MATRIX, COMPOSITION_TABLE]},
    }
    (args.output / "analysis_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    summary, permutation, shap_summary = run(data, args)
    plot(summary, permutation, shap_summary, args.output)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\nGrouped-validation SHAP ranking:", flush=True)
    print(shap_summary.sort_values(["model", "mean_rank"]).to_string(index=False), flush=True)
    print(f"Outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
