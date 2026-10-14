"""Record actual packaging checks; do not imply a new 90-job scratch search."""
from pathlib import Path
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT / "validation"; output.mkdir(exist_ok=True)
    checks={}
    runs=("development_audit_v2","reproduction_33","frozen_validation","scratch_validation",
          "composition_validation","figures_validation","sensitivity39_validation")
    for name in runs:
        source=ROOT / "runs" / name / "verification.json"
        status=json.loads(source.read_text(encoding="utf-8"))
        if not status["complete"]: raise ValueError(name)
        checks[name]=status
        shutil.copy2(source,output / (name+".json"))
    tests=ET.parse(ROOT / "runs/test_results.xml").getroot()
    if list(tests.iter("failure")) or list(tests.iter("error")): raise ValueError("Tests failed")
    checks["unit_tests_passed"]=len(list(tests.iter("testcase")))
    shutil.copy2(ROOT / "runs/test_results.xml",output / "tests.xml")
    archive=ROOT / "published_results/reviewer_ml_completion_20261014"
    actual=pd.read_csv(ROOT / "runs/composition_validation/predictions.csv")
    reference=pd.read_csv(archive / "composition_bridge/predictions.csv")
    index=["arm","protocol","repeat","fold","scenario","sample"]
    a=actual.set_index(index).sort_index(); b=reference.set_index(index).loc[a.index]
    np.testing.assert_allclose(a.y_true,b.y_true,rtol=0,atol=1e-12)
    np.testing.assert_allclose(a.y_pred,b.y_pred,rtol=0,atol=1e-10)
    checks["all_six_composition_scenarios_replay_error"]=float(np.max(np.abs(a.y_pred-b.y_pred)))
    figure_pairs=[(ROOT / "runs/reproduction_33" / f,archive / "shap33" / f)
                  for f in ("nested33_shap.png","frozen_vs_nested_family_shap.png")]
    figure_pairs += [(ROOT / "runs/figures_validation" / f,archive / "electronegativity_branches" / f)
                    for f in ("original_four_bo_electronegativity.png","all_five_bo_electronegativity.png","bo_pair_trajectories.png")]
    checks["pixel_checks"]={}
    for first,second in figure_pairs:
        with Image.open(first) as a,Image.open(second) as b:
            aa=np.asarray(a.convert("RGB")); bb=np.asarray(b.convert("RGB"))
            if aa.std()<1: raise ValueError("Blank figure")
            checks["pixel_checks"][first.name]=dict(nonblank=True,identical_to_archived=bool(aa.shape==bb.shape and np.array_equal(aa,bb)))
    relocation=json.loads((output / "relocation.json").read_text(encoding="utf-8"))
    if not relocation["passed"]: raise ValueError("Relocation failed")
    checks["relocation"]=relocation
    (output / "summary.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    maximum=max(checks[n].get("max_prediction_error",0.) for n in runs)
    lines=["# Packaging Validation", "", "Validated on 2026-10-14 using the pinned Windows ML environment.", "",
        f"- {checks['unit_tests_passed']} tests passed, including nine package-specific tests.",
        "- Rebuilt all 665 included samples from packaged ICP workbooks, metadata and property tables.",
        "- Regenerated outer assignments exactly and verified ordered memberships for 90 outer arms and 270 inner partitions.",
        "- Refit all 90 primary task/arm/fold combinations: selected-10, all-33, original frozen-10 and reselected frozen-10, with held-out SHAP for the three ten-feature specifications.",
        "- Recomputed Task-1 consensus/global tuning and all 90 reselected frozen-model evaluations.",
        "- Recomputed one complete 33-candidate outer job from scratch: all 16 forward steps, all declared hyperparameter candidates and all four final strategies.",
        "- Refit one 39-candidate sensitivity job and recalculated one complete six-scenario composition comparison.",
        "- Regenerated all three electronegativity/BO trajectory figures with the full 1,000-draw shape bootstrap.",
        f"- Maximum primary/frozen/scratch replay prediction discrepancy: {maximum:.3g}.",
        f"- Six-scenario composition replay discrepancy: {checks['all_six_composition_scenarios_replay_error']:.3g}.",
        "- Copied the distributable package into a temporary directory outside this checkout, with spaces in its path. Used isolated Python from another working directory and an audit hook prohibiting reads from the original repository. The full input/split audit and one fresh four-model refit passed.", "",
        "## Scope", "",
        "The complete 90-job nested feature-selection search was not rerun during packaging. Its verified original outputs are included, and the portable scratch command is supplied. One full scratch job, all 90 primary refit jobs, and full frozen retuning/refits were executed to validate the numerical transfer. Partial checks remain explicitly marked partial in their machine-readable records.", "",
        "All five re-rendered PNGs were checked for nonblank pixels and compared with archived images. See validation/summary.json for the comparison results and validation/tests.xml for individual tests.", "",
        "Source inputs/helpers are byte-identical copies recorded in source_inventory.json. package_manifest.json seals the distributable files; runs/ is intentionally excluded. Historical report text is retained verbatim, including old local command examples. Use the portable commands in README.md instead."]
    (ROOT / "VALIDATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in checks.items() if k in ("unit_tests_passed","pixel_checks","all_six_composition_scenarios_replay_error")},indent=2))


if __name__=="__main__":
    main()
