"""Repro script for delta_EOH and PZC modeling.

This version intentionally excludes delta_EH_0.25ML from:
1) delta-model targets (only delta_EOH is modeled)
2) PZC metal feature set

Usage (PowerShell):
  C:/Users/yibinjiang/anaconda3/python.exe pzc_ml_pipeline.py

Required files in current folder:
  - EH_extra_12_15.xlsx
  - potential_of_zero_charge.xlsx
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

BASE = Path(".")
EH_XLSX = BASE / "../EH_extra.xlsx"
PZC_XLSX = BASE / "potential_of_zero_charge.xlsx"


def to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("–", "-").replace("−", "-")
    s = re.sub(r"[^0-9\-\.]", "", s)
    if s in ("", "-", "--"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def base_metal(e):
    if pd.isna(e):
        return None
    m = re.match(r"[A-Z][a-z]?", str(e).strip())
    return m.group(0) if m else None


def parse_facet(e):
    if pd.isna(e):
        return None
    m = re.search(r"\(([^\)]+)\)", str(e))
    return m.group(1).strip() if m else None


def parse_solution(sol):
    if pd.isna(sol):
        return {}
    s = str(sol).strip()
    s_lower = s.lower()

    conc_val = np.nan
    conc_unit = None
    m = re.search(r"([0-9]*\.?[0-9]+)\s*(mM|M|N)", s, flags=re.IGNORECASE)
    if m:
        conc_val = float(m.group(1))
        conc_unit = m.group(2).upper()
        if conc_unit == "MM":
            conc_val /= 1000.0
            conc_unit = "M"

    solvent = None
    if "/" in s:
        parts = s.split("/")
        if len(parts) >= 2:
            solvent = parts[-1].strip()
    elif " in " in s_lower:
        solvent = s_lower.split(" in ")[-1].strip()

    s_no_conc = re.sub(r"^[0-9]*\.?[0-9]+\s*(mM|M|N)\s*", "", s, flags=re.IGNORECASE)
    electrolyte = re.split(r"/|,|;", s_no_conc)[0].strip()

    solvent_class = None
    if solvent:
        if "h2o" in solvent.lower() or "water" in solvent.lower():
            solvent_class = "aqueous"
        else:
            solvent_class = "non-aqueous"

    return {
        "solution_raw": s,
        "concentration_M": conc_val,
        "concentration_unit": conc_unit,
        "electrolyte": electrolyte if electrolyte else None,
        "solvent": solvent,
        "solvent_class": solvent_class,
    }


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_eh_sheet():
    return pd.read_excel(EH_XLSX, sheet_name="Sheet1").rename(
        columns={"Unnamed: 0": "metal", "delta EH\n 0.25ML": "delta_EH_0.25ML"}
    )


def load_sheet2_stats():
    sheet2 = pd.read_excel(EH_XLSX, sheet_name="Sheet2").rename(columns={"Unnamed: 0": "metal"})
    sheet2["metal"] = sheet2["metal"].ffill()
    agg_cols = ["Adsorption Energies (eV)", "Free Energy of Adsorption (eV)"]
    stats = sheet2.groupby("metal")[agg_cols].agg(["count", "mean", "std", "min", "max"])
    stats.columns = [f"sheet2_{c[0].split(' (')[0].replace(' ', '_')}_{c[1]}" for c in stats.columns]
    return stats.reset_index()


def load_sheet3():
    return pd.read_excel(EH_XLSX, sheet_name="Sheet3").rename(
        columns={
            "Unnamed: 0": "metal",
            "Fundamental Frequency": "freq_fundamental",
            "Harmonic Frequency": "freq_harmonic",
        }
    )


def load_pzc_table():
    pzc = pd.read_excel(PZC_XLSX, sheet_name="table_1")
    pzc["metal"] = pzc["Electrode"].apply(base_metal)
    pzc["facet"] = pzc["Electrode"].apply(parse_facet)
    pzc["Epzc_num"] = pzc["Epzc"].apply(to_float)
    solution_df = pd.json_normalize(pzc["Solution 1)"].apply(parse_solution))
    return pd.concat([pzc, solution_df], axis=1)


def run_delta_eoh_model():
    sheet1 = load_eh_sheet()
    pzc = load_pzc_table()

    pzc_stats = (
        pzc.dropna(subset=["metal", "Epzc_num"])
        .groupby("metal")["Epzc_num"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    pzc_stats = pzc_stats.rename(columns={c: f"pzc_{c}" for c in pzc_stats.columns if c != "metal"})

    sheet2_stats = load_sheet2_stats()
    sheet3 = load_sheet3()

    all_df = sheet1.merge(pzc_stats, on="metal", how="left")
    all_df = all_df.merge(sheet2_stats, on="metal", how="left")
    all_df = all_df.merge(sheet3, on="metal", how="left")

    base6 = [
        "Work function (eV)",
        "electronegativity",
        "M-M",
        "E(M-H) (kcal/mol)",
        "Frequency_H",
        "Frequency_OH",
    ]
    kJ = ["E(M-H) (kJ/mol)"]
    pzc_feat = [c for c in all_df.columns if c.startswith("pzc_")]
    sheet2_feat = [c for c in all_df.columns if c.startswith("sheet2_")]
    sheet3_feat = [c for c in ("freq_fundamental", "freq_harmonic") if c in all_df.columns]

    feature_sets = {
        "base6": base6,
        "base6+kJ": base6 + kJ,
        "base6+pzc": base6 + pzc_feat,
        "base6+sheet2": base6 + sheet2_feat,
        "base6+sheet2+sheet3": base6 + sheet2_feat + sheet3_feat,
        "base6+pzc+sheet2+sheet3+kJ": base6 + pzc_feat + sheet2_feat + sheet3_feat + kJ,
    }

    models_scaled = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
        "SVR_RBF": SVR(C=10.0, gamma="scale", epsilon=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=3, weights="distance"),
    }
    models_tree = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=0),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=0),
        "GBR": GradientBoostingRegressor(random_state=0),
    }

    rows = []
    target_col = "delta_EOH"
    for fs_name, feats in feature_sets.items():
        mask = all_df[target_col].notna()
        X = all_df.loc[mask, feats]
        y = all_df.loc[mask, target_col]
        if len(y) < 5:
            continue
        cv = KFold(n_splits=5, shuffle=True, random_state=0)

        for name, model in models_scaled.items():
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )
            preds = cross_val_predict(pipe, X, y, cv=cv)
            rows.append(
                {
                    "target": target_col,
                    "feature_set": fs_name,
                    "model": name,
                    "scaled": True,
                    "mae": mean_absolute_error(y, preds),
                    "rmse": rmse(y, preds),
                    "r2": r2_score(y, preds),
                    "n": len(y),
                }
            )

        for name, model in models_tree.items():
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
            preds = cross_val_predict(pipe, X, y, cv=cv)
            rows.append(
                {
                    "target": target_col,
                    "feature_set": fs_name,
                    "model": name,
                    "scaled": False,
                    "mae": mean_absolute_error(y, preds),
                    "rmse": rmse(y, preds),
                    "r2": r2_score(y, preds),
                    "n": len(y),
                }
            )

    metrics_df = pd.DataFrame(rows).sort_values("mae")
    best = metrics_df.iloc[0]
    best_feats = feature_sets[best["feature_set"]]
    best_model = models_scaled[best["model"]] if bool(best["scaled"]) else models_tree[best["model"]]

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if bool(best["scaled"]):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", best_model))
    pipe = Pipeline(steps)

    mask = all_df[target_col].notna()
    pipe.fit(all_df.loc[mask, best_feats], all_df.loc[mask, target_col])
    all_df["delta_EOH_pred_best"] = pipe.predict(all_df[best_feats])

    out_path = BASE / "delta_EOH_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="features_with_preds", index=False)
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        pzc_stats.to_excel(writer, sheet_name="pzc_stats", index=False)
        sheet2_stats.to_excel(writer, sheet_name="sheet2_stats", index=False)
        sheet3.to_excel(writer, sheet_name="sheet3", index=False)

    print("Wrote", out_path)


def run_pzc_models():
    sheet1 = load_eh_sheet()
    sheet2_stats = load_sheet2_stats()
    sheet3 = load_sheet3()
    pzc = load_pzc_table()

    all_df = sheet1.merge(sheet2_stats, on="metal", how="left")
    all_df = all_df.merge(sheet3, on="metal", how="left")

    pzc_ml = pzc.merge(all_df, on="metal", how="inner")
    pzc_ml = pzc_ml[pzc_ml["Epzc_num"].notna()].copy()

    metal_features = [
        "Work function (eV)",
        "electronegativity",
        "M-M",
        "E(M-H) (kcal/mol)",
        "E(M-H) (kJ/mol)",
        "Frequency_H",
        "Frequency_OH",
        "delta_EOH",
    ] + [c for c in all_df.columns if c.startswith("sheet2_")] + [
        "freq_fundamental",
        "freq_harmonic",
    ]
    metal_features = [c for c in dict.fromkeys(metal_features) if c in pzc_ml.columns]

    condition_features_num = ["concentration_M"]
    condition_features_cat = ["concentration_unit", "electrolyte", "solvent", "solvent_class", "methods", "facet"]

    num_features = metal_features + condition_features_num
    cat_features = [c for c in condition_features_cat if c in pzc_ml.columns]

    X = pzc_ml[num_features + cat_features]
    y = pzc_ml["Epzc_num"]

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=0),
        "GBR": GradientBoostingRegressor(random_state=0),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=0),
        "Ridge": Ridge(alpha=1.0),
    }

    def make_pipe(model):
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_features,
                ),
            ],
            remainder="drop",
        )
        return Pipeline([("pre", pre), ("model", model)])

    groups = pzc_ml["metal"]
    cv = GroupKFold(n_splits=min(5, groups.nunique()))

    rows = []
    for name, model in models.items():
        pipe = make_pipe(model)
        preds = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(y, preds),
                "rmse": rmse(y, preds),
                "r2": r2_score(y, preds),
                "n": len(y),
                "n_metals": groups.nunique(),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("mae")
    best_model_name = metrics_df.iloc[0]["model"]
    pipe = make_pipe(models[best_model_name])
    pipe.fit(X, y)

    baseline_solution = "1 N NaClO4/H2O"
    baseline_method = "S"
    base_mask = (pzc["Solution 1)"].astype(str).str.strip() == baseline_solution) & (
        pzc["methods"].astype(str).str.strip() == baseline_method
    )
    if pzc[base_mask].empty:
        raise SystemExit("Baseline condition not found in PZC table")
    base_cond = pzc[base_mask].iloc[0]

    rows_pred = []
    for metal in all_df["metal"].unique():
        feat_row = all_df[all_df["metal"] == metal][metal_features].iloc[0].to_dict()
        row = {}
        row.update(feat_row)
        row["concentration_M"] = base_cond.get("concentration_M", np.nan)
        row["concentration_unit"] = base_cond.get("concentration_unit", None)
        row["electrolyte"] = base_cond.get("electrolyte", None)
        row["solvent"] = base_cond.get("solvent", None)
        row["solvent_class"] = base_cond.get("solvent_class", None)
        row["methods"] = baseline_method
        row["facet"] = None
        row["metal"] = metal
        rows_pred.append(row)

    pred_df = pd.DataFrame(rows_pred)
    pred_df["Epzc_pred"] = pipe.predict(pred_df[num_features + cat_features])

    out_path = BASE / "pzc_baseline_prediction.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)
        pred_df.to_excel(writer, sheet_name="pred_all_metals", index=False)
        pzc[base_mask].to_excel(writer, sheet_name="raw_condition_rows", index=False)

    print("Wrote", out_path)


if __name__ == "__main__":
    run_delta_eoh_model()
    run_pzc_models()
