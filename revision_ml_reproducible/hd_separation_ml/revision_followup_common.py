"""Read-only access to the completed ML runs for reviewer follow-up analyses."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from icp_bo_augmentation_cv import representation
from icp_electrochemical_exclusion import FEATURES, clean_inputs, reference_audit
from icp_forward_bo_ablation import PROTOCOLS, matched_plan, seed_for
from icp_forward_selection_revision import save_json
from icp_intrinsic_descriptor_analysis import BASES, LABELS, scores
from icp_nested_group_cv import make_model
from icp_revision_inputs import sha
from icp_task1_frozen_sensitivity import digest, label, NAMES
from summarize_icp_forward_bo_ablation import load_ablation

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/forward_bo_ablation_20261013"
FROZEN = ROOT / "outputs/task1_frozen_sensitivity_20261013"


def load_current():
    data,tables,_,source = load_ablation(SOURCE)
    frozen_manifest = json.loads((FROZEN / "analysis_manifest.json").read_text(encoding="utf-8"))
    dependencies = {**source["source_hashes"],**source["code_hashes"],**frozen_manifest["dependencies"]}
    files = [FROZEN / n for n in ("frozen_specification.json","predictions.csv","analysis_manifest.json")]
    files += [ROOT / "revision_inputs/elemental_properties.csv",Path(__file__),
              Path(source["source"]) / "outer_assignments.csv",SOURCE / "checkpoint_provenance.csv"]
    dependencies.update({str(f.resolve()):sha(f) for f in files})
    for path,expected in dependencies.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Changed completed-run dependency: {path}")
    versions = dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__)
    if versions != source["versions"]:
        raise ValueError("Use the original ML runtime")
    reference_audit(data,pd.read_csv(ROOT / "revision_inputs/elemental_properties.csv"))
    spec = json.loads((FROZEN / "frozen_specification.json").read_text(encoding="utf-8"))
    frozen = pd.read_csv(FROZEN / "predictions.csv")
    assignment = pd.read_csv(Path(source["source"]) / "outer_assignments.csv")
    return data,tables,assignment,spec,frozen,dict(dependencies=dependencies,versions=versions,
        options=source["search_options"],source=str(SOURCE),frozen_source=str(FROZEN))


def start_output(output,script,context,design,extra_files=()):
    deps = dict(context["dependencies"])
    for f in [Path(script),*map(Path,extra_files)]:
        deps[str(f.resolve())] = sha(f)
    manifest = dict(dependencies=deps,versions=context["versions"],design=design)
    manifest["fingerprint"] = digest(manifest)
    file = output / "analysis_manifest.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8"))["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Changed analysis inputs/code; use a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    save_json(file,manifest)
    return manifest["fingerprint"]


def fit_model(train,test,features,name,config,trees,seed):
    transform = representation("hybrid39").fit(clean_inputs(train))
    X,V = transform.transform(clean_inputs(train)),transform.transform(clean_inputs(test))
    model = make_model(name,config,trees,seed).fit(X[features].to_numpy(),train.separation_coefficient)
    return model,V,model.predict(V[features].to_numpy())


def aggregate_predictions(predictions,groups):
    rows = []
    for key,part in predictions.groupby(groups+["repeat"]):
        if part["sample"].duplicated().any():
            raise ValueError("Duplicate sample within repeat")
        rows.append(dict(zip(groups+["repeat"],key),n=len(part),**scores(part.y_true,part.y_pred)))
    repeats = pd.DataFrame(rows)
    summary = repeats.groupby(groups,as_index=False).agg(n_min=("n","min"),n_max=("n","max"),
        r2_mean=("r2","mean"),r2_sd=("r2","std"),rmse_mean=("rmse","mean"),rmse_sd=("rmse","std"),
        mae_mean=("mae","mean"),mae_sd=("mae","std"))
    return repeats,summary
