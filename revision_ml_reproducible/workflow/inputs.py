"""Portable input rebuilding, split regeneration and content-addressed runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from icp_revision_inputs import build_revision_data, sha
from icp_cofe_inclusion import load_cofe, comparison_partitions
from icp_two_prediction_tasks import design
from icp_electrochemical_exclusion import RAW, reference_audit
from revision_remaining_common import add_recipes

ROOT=Path(__file__).resolve().parents[1]
ML=ROOT / "hd_separation_ml"
REFERENCE=ROOT / "reference"
KEYS=["arm","protocol","repeat","fold"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def versions():
    return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        sklearn=sklearn.__version__,joblib=joblib.__version__)


def load_data(targets=None):
    initial,bo,audit=build_revision_data()
    extra,extra_audit=load_cofe(initial,ML / "revision_inputs/cofe/ICP-OES_received.xlsx",
        ML / "revision_inputs/cofe/targets.csv","separation_coefficient",ML / "revision_inputs/elemental_properties.csv")
    data=pd.concat([initial,bo,extra],ignore_index=True).set_index("sample",drop=False)
    expected=pd.read_csv(REFERENCE / "analysis_dataset.csv").set_index("sample",drop=False)
    if set(data.index)!=set(expected.index): raise ValueError("Changed cohort; build a newly reviewed package snapshot")
    data=data.loc[expected.index].copy()
    numeric=RAW+["actual_frac_element1","actual_frac_element2","separation_coefficient"]
    np.testing.assert_allclose(data[numeric],expected[numeric],atol=1e-12,rtol=0,equal_nan=True)
    for c in ("element1","element2","metal_pair","cohort","icp_group","composition_group"):
        if not data[c].equals(expected[c]): raise ValueError(f"Input rebuild mismatch: {c}")
    reference_audit(data,pd.read_csv(ML / "revision_inputs/elemental_properties.csv"))
    data=add_recipes(data)
    if targets is not None:
        table=pd.read_csv(targets)
        if not {"sample","separation_coefficient"}<=set(table) or table["sample"].duplicated().any():
            raise ValueError("New target table needs unique sample IDs and separation_coefficient")
        if set(table["sample"])!=set(data.index): raise ValueError("New targets must cover exactly the 665 included samples")
        y=pd.to_numeric(table.set_index("sample").loc[data.index,"separation_coefficient"],errors="raise")
        if not np.isfinite(y).all() or y.nunique()<3: raise ValueError("Targets must be finite and varying")
        data["separation_coefficient"]=y
    return data,pd.concat([audit,extra_audit],ignore_index=True)


def regenerate_splits(data, output):
    initial=data[data.cohort.eq("initial")].reset_index(drop=True)
    four=data[data.cohort.eq("BO")&data.metal_pair.ne("Co-Fe")].reset_index(drop=True)
    cofe=data[data.cohort.eq("BO")&data.metal_pair.eq("Co-Fe")].reset_index(drop=True)
    stage=output / "initial_split_design"; stage.mkdir(parents=True,exist_ok=True)
    options=read_json(ROOT / "config.json")["search_options"]
    design(initial,four,stage,options)
    base=pd.read_csv(stage / "outer_assignments.csv")
    plans=comparison_partitions(initial,four,cofe,base)
    rows=[]
    for plan in plans:
        common={k:plan[k] for k in ("protocol","repeat","fold")}
        for role in ("original","existing","augmented","testing"):
            for _,row in plan[role].iterrows():
                rows.append(dict(**common,role=role,sample=row["sample"]))
    result=pd.DataFrame(rows)
    result.to_csv(output / "outer_assignments.csv",index=False)
    return result


def data_digest(data):
    return hashlib.sha256(data.to_csv(index=False,float_format="%.17g").encode()).hexdigest()


def start_run(args, data):
    output=args.output.resolve() if args.output else ROOT / "runs" / (args.command+"_"+str(getattr(args,"pool",33)))
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Output must be inside this package's runs/ directory; source/reference files are read-only")
    config=read_json(ROOT / "config.json")
    code=[ROOT / "run.py",ROOT / "config.json",*sorted((ROOT / "workflow").glob("*.py")),*sorted(ML.glob("*.py"))]
    inputs=[p for p in read_json(ROOT / "source_inventory.json")["files"] if p["role"]=="input"]
    arguments={k:v for k,v in vars(args).items() if k not in ("output","jobs","targets")}
    if "source" in arguments and arguments["source"] is not None:
        arguments["source"]=str(Path(arguments["source"]).resolve().relative_to(ROOT))
    manifest=dict(schema="portable-hd-run-v1",versions=versions(),options=arguments,config=config,
        input_hashes={p["path"]:sha(ROOT / p["path"]) for p in inputs},data_sha256=data_digest(data),
        custom_target_sha256=sha(args.targets) if args.targets else None,
        code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in code})
    source=resolve_source(getattr(args,"source",None),getattr(args,"pool",33))
    source_files=[p for p in source.glob("*") if p.is_file() and p.suffix in (".json",".csv")]
    manifest["model_source_hashes"]={p.relative_to(ROOT).as_posix():sha(p) for p in source_files}
    manifest["frozen_reference_hashes"]={p.relative_to(ROOT).as_posix():sha(p) for p in REFERENCE.glob("*frozen*.json")}
    manifest["fingerprint"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    file=output / "run_manifest.json"
    if file.exists() and read_json(file)["fingerprint"]!=manifest["fingerprint"]:
        raise ValueError("Changed code/data/settings; choose a new --output within runs/")
    output.mkdir(parents=True,exist_ok=True); write_json(file,manifest)
    for folder in ("checkpoints","paths","tuning","progress","tuning_checkpoints"):
        (output / folder).mkdir(exist_ok=True)
    return output,manifest["fingerprint"]


def resolve_source(source,pool):
    path=Path(source).resolve() if source else REFERENCE / f"pool{pool}"
    if not path.is_relative_to(ROOT): raise ValueError("Model source must be inside the package")
    return path
