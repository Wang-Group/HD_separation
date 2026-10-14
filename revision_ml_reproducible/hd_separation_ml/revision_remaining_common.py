"""Verified inputs for the final three reviewer ML follow-ups."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_followup_common import ROOT, BASES, load_current, sha
from icp_33_result_io import load_reduced
from icp_forward_33_comparison import OUTPUT as REDUCED, POOL

OUTPUT = ROOT / "outputs/reviewer_ml_completion_20261014"
RECIPE4 = ROOT.parent / "0-Figures/Figure4/curated/figure4_bo_complete_112.csv"
RECIPE_COFE = ROOT / "revision_inputs/cofe/initial_sampling_Co_Fe_1.json"
RAW22 = [c for c in POOL if not c.startswith("icp_weighted_")]
BASE11 = [b for b in BASES if b not in ("Frequency_D", "delta_ZPE_H")]


def load_final():
    data,old,assignments,old_spec,old_frozen,context = load_current()
    reduced,manifest,provenance = load_reduced(REDUCED)
    folder = REDUCED / "frozen33"
    frozen_manifest = json.loads((folder / "analysis_manifest.json").read_text(encoding="utf-8"))
    dependencies = {**context["dependencies"], **manifest["dependencies"], **frozen_manifest["dependencies"]}
    for entry in provenance:
        dependencies[entry["path"]] = entry["sha256"]
    files = [Path(__file__), RECIPE4, RECIPE_COFE]
    files += [folder / n for n in ("predictions.csv", "frozen_specification.json", "verification.json")]
    for file in files:
        dependencies[str(file.resolve())] = sha(file)
    for path,expected in dependencies.items():
        if sha(path) != expected:
            raise ValueError(f"Changed dependency: {path}")
    if not json.loads((folder / "verification.json").read_text(encoding="utf-8"))["complete"]:
        raise ValueError("Reduced frozen run is incomplete")
    spec = json.loads((folder / "frozen_specification.json").read_text(encoding="utf-8"))
    frozen = pd.read_csv(folder / "predictions.csv")
    return data,reduced,assignments,{"original_frozen10":old_spec,"reselected_frozen10":spec}, \
        {"original_frozen10":old_frozen,"reselected_frozen10":frozen}, \
        dict(context,dependencies=dependencies)


def add_recipes(data):
    four = pd.read_csv(RECIPE4)
    four["sample"] = four.system+four.row_id.astype(str)
    four["recipe_source"] = str(RECIPE4.relative_to(ROOT.parent))
    cofe = pd.DataFrame(json.loads(RECIPE_COFE.read_text(encoding="utf-8")))
    cofe["sample"] = "Co-Fe"+cofe.index.astype(str)
    cofe["recipe_source"] = str(RECIPE_COFE.relative_to(ROOT.parent))
    recipes = pd.concat([four,cofe],ignore_index=True).set_index("sample",verify_integrity=True)
    result = data.copy()
    result["recipe_source"] = "revision_inputs/sample_metadata.csv"
    for sample,row in result[result.cohort.eq("BO")].iterrows():
        recipe = recipes.loc[row["sample"]]
        if (recipe.substance1,recipe.substance2)!=(row.element1,row.element2):
            raise ValueError(f"Recipe element orientation mismatch: {sample}")
        np.testing.assert_allclose(recipe.separation_coefficient,row.separation_coefficient,atol=1e-7,rtol=0)
        volumes = np.asarray([recipe.volume1,recipe.volume2],float)
        if not np.isfinite(volumes).all() or (volumes<0).any() or volumes.sum()<=0:
            raise ValueError(f"Invalid precursor volumes: {sample}")
        result.loc[sample,"nominal_frac_element1"] = volumes[0]/volumes.sum()
        result.loc[sample,"nominal_frac_element2"] = volumes[1]/volumes.sum()
        result.loc[sample,"recipe_source"] = recipe.recipe_source
        result.loc[sample,"volume1"] = volumes[0]
        result.loc[sample,"volume2"] = volumes[1]
    for prefix in ("actual","nominal"):
        f=result[prefix+"_frac_element1"]
        if not f.between(0,1).all():
            raise ValueError(f"Invalid {prefix} fractions")
        np.testing.assert_allclose(f+result[prefix+"_frac_element2"],1,atol=1e-10,rtol=0)
    return result


def reweight(frame,data,mode):
    """Use the same fold-imputed endpoints; change only the weighting fraction."""
    result=frame.copy()
    f=data[("nominal" if mode=="precursor" else "actual")+"_frac_element1"].to_numpy(float)
    if mode not in ("precursor","icp") or not np.isfinite(f).all():
        raise ValueError("Invalid weighting mode or fraction")
    for base in BASE11:
        result["icp_weighted_"+base]=f*result[base+"_element1"].to_numpy()+(1-f)*result[base+"_element2"].to_numpy()
    return result


def observed_weighted(data,base,mode):
    f=data[("nominal" if mode=="precursor" else "actual")+"_frac_element1"].to_numpy(float)
    return np.where(f==0,0.,f*data[base+"_element1"].to_numpy(float))+np.where(f==1,0.,(1-f)*data[base+"_element2"].to_numpy(float))
