"""Reproducible revision analysis driven by English ICP and a replaceable y CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config

from icp_revision_inputs import ROOT, WORKBOOK, TARGETS, METADATA, PROPERTIES, build_revision_data, sha
from icp_two_prediction_tasks import design, run_job, aggregate

PROTOCOLS = ["stratified_random", "unseen_pair", "known_pair_new_composition"]
SCENARIOS = ["nominal13", "raw26", "icp13", "hybrid39"]


def analysis_options(smoke=False, repeats=3):
    configs = [dict(max_depth=d, min_samples_leaf=l, max_features=f) for d, l, f in
               [(None, 1, "sqrt"), (None, 3, 1.0), (None, 6, "sqrt"),
                (8, 1, 1.0), (8, 3, "sqrt"), (8, 6, 1.0)]]
    return dict(protocols=PROTOCOLS, scenarios=["icp13"] if smoke else SCENARIOS,
                repeats=1 if smoke else repeats, nested_raw=True,
                selector_trees=8 if smoke else 64, search_trees=8 if smoke else 80,
                final_trees=16 if smoke else 400, configs=configs[:2] if smoke else configs,
                feature_counts={s: ([4, p] if smoke else sorted({4, 8, 12, p}))
                                for s, p in [("nominal13", 13), ("icp13", 13), ("raw26", 26), ("hybrid39", 39)]})


def manifest_for(args, initial, bo, audit, options):
    modules = ["icp_revision_ml", "icp_revision_inputs", "icp_two_prediction_tasks",
               "icp_bo_augmentation_cv", "icp_nested_group_cv", "icp_hybrid_descriptor_cv",
               "icp_intrinsic_descriptor_analysis", "icp_parsimony_sensitivity", "icp_master"]
    inputs = {"icp_workbook": args.workbook, "targets": args.targets,
              "sample_metadata": args.metadata, "elemental_properties": args.properties}
    manifest = dict(schema="english-icp-revision-1", target_column=args.target, target_label=args.target_label,
        input_hashes={role: sha(p) for role, p in inputs.items()},
        code_hashes={m + ".py": sha(ROOT / (m + ".py")) for m in modules},
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
        n_initial=len(initial), n_bo=len(bo), inclusion_counts=audit.status.value_counts().to_dict(),
        options=options, outer_folds=5, inner_folds=3,
        shared_icp="One initial-priority representative per measured ICP group; repeated labels retained for sensitivity.",
        imputation="Element-property medians learned from training rows only, before weighting.",
        target_handling="Only the requested CSV column supplies y; missing y excluded and recorded; no old checkpoint reuse.",
        random_protocol="Stratified interpolation benchmark; metal pairs may overlap. Shared ICP representatives remain unique.",
        grouped_protocol="Unordered metal pairs isolated in both outer and inner folds.",
        known_pair_protocol="5 mol% composition bins held out within each BO metal pair; initial matching bins removed.",
        inference="Retrospective internal validation, not external prospective validation.",
        selection="Training-only RFE and six forest configurations; best inner mean R2 and one-SE rules.",
        fixed_control="Matched test sets and common fixed settings before/after BO; known-pair initial-only arm has no BO-label tuning.",
        metric_summary="Pooled OOF R2/RMSE/MAE per repeat; repeat SD is split variability, not an independent-sample confidence interval.",
        smoke_only=args.smoke)
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["input_paths"] = {role: str(Path(p).resolve()) for role, p in inputs.items()}
    return manifest


def baseline_predictions(initial, bo, assignments, output):
    data = pd.concat([initial, bo], ignore_index=True).set_index("sample")
    rows = []
    for (protocol, repeat, fold), part in assignments.groupby(["protocol", "repeat", "fold"]):
        testing = data.loc[part.loc[part.role.eq("test"), "sample"]]
        for arm in ("initial_only", "initial_plus_bo"):
            roles = ["train_initial", "train_bo"] if arm == "initial_plus_bo" else ["train_initial"]
            training = data.loc[part.loc[part.role.isin(roles), "sample"]]
            mean = float(training.separation_coefficient.mean())
            pair_means = training.groupby("metal_pair").separation_coefficient.mean()
            for sample, row in testing.iterrows():
                rows.append(dict(protocol=protocol, repeat=repeat, fold=fold, training=arm, sample=sample,
                    cohort=row.cohort, metal_pair=row.metal_pair, y_true=float(row.separation_coefficient),
                    train_mean=mean, train_pair_mean=float(pair_means.get(row.metal_pair, mean))))
    pd.DataFrame(rows).to_csv(output / "baseline_predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--target", default="separation_coefficient")
    parser.add_argument("--target-label", default=None)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--properties", type=Path, default=PROPERTIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.repeats < 1:
        parser.error("jobs and repeats must be positive")
    args.target_label = args.target_label or args.target.replace("_", " ")
    initial, bo, audit = build_revision_data(args.workbook, args.targets, args.target, args.metadata, args.properties)
    options = analysis_options(args.smoke, args.repeats)
    manifest = manifest_for(args, initial, bo, audit, options)
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", args.target)
    output = args.output or ROOT / "outputs" / f"revision_{slug}_{manifest['fingerprint'][:12]}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "analysis_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Inputs/code/settings changed. Use a new output directory; old checkpoints are not reusable.")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "checkpoints").mkdir(exist_ok=True)
    initial.to_csv(output / "initial_dataset.csv", index=False)
    bo.to_csv(output / "bo_dataset.csv", index=False)
    audit.to_csv(output / "sample_inclusion_audit.csv", index=False)
    jobs = design(initial, bo, output, options)
    assignments = pd.read_csv(output / "outer_assignments.csv")
    baseline_predictions(initial, bo, assignments, output)
    print(f"INPUT AUDIT: {len(initial)} initial + {len(bo)} BO; {len(jobs)} outer jobs; output={output}", flush=True)
    if args.audit_only:
        return
    start = time.perf_counter()
    if args.aggregate_only:
        results = []
        for protocol, _, _, repeat, fold, scenario, model in jobs:
            path = output / "checkpoints" / f"{protocol}_r{repeat}_f{fold}_{scenario}_{model}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["fingerprint"] != manifest["fingerprint"]:
                raise ValueError(f"Checkpoint fingerprint mismatch: {path.name}")
            results.append(result)
    else:
        results = []
        with parallel_config(backend="loky", inner_max_num_threads=1):
            iterator = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(run_job)(initial, bo, *job, manifest["fingerprint"], output, options) for job in jobs)
            for result in iterator:
                results.append(result)
                print(f"{len(results)}/{len(jobs)} {result['protocol']} {result['scenario']} {result['model']} "
                      f"r{result['repeat']}f{result['fold']} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    aggregate(results, output)
    (output / "runtime.json").write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        completed_jobs=len(results), workers=args.jobs), indent=2), encoding="utf-8")
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
