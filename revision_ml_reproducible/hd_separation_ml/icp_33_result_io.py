"""Read and verify completed reduced-pool artifacts without modifying them."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_followup_common import PROTOCOLS,sha
from icp_forward_33_comparison import POOL,KEYS,CATEGORIES
from icp_forward_selection_revision import select_models


def load_reduced(output):
    manifest=json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    for path,expected in manifest["dependencies"].items():
        if sha(Path(path))!=expected:
            raise ValueError(f"Changed reduced-pool dependency: {path}")
    verification=json.loads((output / "verification.json").read_text(encoding="utf-8"))
    if not verification["complete"] or verification["jobs"]!=90:
        raise ValueError("Reduced-pool fitting is incomplete")
    records,provenance=[],[]
    for arm in ("without_bo","with_bo"):
        for protocol in PROTOCOLS:
            for repeat in range(3):
                for fold in range(5):
                    key=(arm,protocol,repeat,fold)
                    name=f"{arm}_{protocol}_r{repeat}_f{fold}.json"
                    files=[output / category / name for category in ("checkpoints","paths","tuning")]
                    record,state,tuning=[json.loads(f.read_text(encoding="utf-8")) for f in files]
                    if any(r["fingerprint"]!=manifest["fingerprint"] for r in (record,state,tuning)):
                        raise ValueError("Reduced-pool fingerprint mismatch")
                    if tuple(record[k] for k in KEYS)!=key:
                        raise ValueError("Wrong model identity")
                    selected=[]
                    if len(state["steps"])!=16 or len(state["trials"])!=408:
                        raise ValueError("Incomplete reduced forward search")
                    for size,winner in enumerate(state["steps"],1):
                        trials=[t for t in state["trials"] if t["n_features"]==size]
                        if {t["added_index"] for t in trials}!=set(range(33))-set(selected) or len(trials)!=34-size:
                            raise ValueError("Missing/extra reduced candidate")
                        for t in trials:
                            if t["mask"]!=selected+[t["added_index"]] or t["added_feature"]!=POOL[t["added_index"]]:
                                raise ValueError("Reduced descriptor order changed")
                            if abs(t["mean_r2"]-np.mean([r["r2"] for r in t["inner_scores"]]))>1e-12:
                                raise ValueError("Incorrect candidate score")
                        best=sorted(trials,key=lambda r:(-r["mean_r2"],r["added_index"]))[0]
                        if best!=winner:
                            raise ValueError("Incorrect reduced path winner")
                        selected=winner["mask"]
                    if state["selected"]!=selected:
                        raise ValueError("Wrong final reduced subset")
                    if record["path"]!=[{k:v for k,v in s.items() if k!="inner_scores"} for s in state["steps"]]:
                        raise ValueError("Reported path differs from scored path")
                    if len(tuning["candidates"])!=204 or set(tuning["completed_sizes"])!=set(range(1,17))|{33}:
                        raise ValueError("Incomplete tuning grid")
                    candidates=tuning["candidates"]
                    grid={(c["model"],c["config_id"],c["n_features"]) for c in candidates}
                    if grid!={(m,c,n) for m in ("RF","ET") for c in range(6) for n in [*range(1,17),33]}:
                        raise ValueError("Missing model/config/subset combination")
                    score_groups=pd.DataFrame(tuning["tuning_scores"]).groupby(["model","config_id","n_features"])
                    for c in candidates:
                        rows=score_groups.get_group((c["model"],c["config_id"],c["n_features"]))
                        if len(rows)!=3 or set(rows.inner_fold)!={0,1,2}:
                            raise ValueError("Incorrect tuning fold count")
                        np.testing.assert_allclose([c["inner_r2_mean"],c["inner_r2_std"]],[rows.r2.mean(),rows.r2.std()],atol=1e-12,rtol=0)
                        expected=POOL if c["n_features"]==33 else [POOL[i] for i in state["steps"][c["n_features"]-1]["mask"]]
                        if c["features"]!=expected:
                            raise ValueError("Tuning features differ from forward path")
                    choices,_=select_models([c for c in candidates if c["n_features"]!=33],[c for c in candidates if c["n_features"]==33])
                    choices["all33"]=choices.pop("all39")
                    for setting in record["settings"]:
                        choice=choices[setting["strategy"]]
                        for field in ("features","config","model","n_features"):
                            if setting[field]!=choice[field]:
                                raise ValueError("Saved model disagrees with inner tuning")
                        if not set(setting["features"])<=set(POOL):
                            raise ValueError("Excluded descriptor selected")
                    records.append(record)
                    provenance.extend(dict(path=str(f),sha256=sha(f)) for f in files)
    tables={category:pd.DataFrame([dict(**{k:r[k] for k in KEYS},**entry) for r in records for entry in r[category]]) for category in CATEGORIES}
    return tables,manifest,provenance
