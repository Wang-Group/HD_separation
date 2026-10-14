"""Audit descriptor provenance and replay nested-CV fits without operating inputs.

Feature searches are verified and reused, NOT recomputed. Outer estimators are
refitted on an explicit input allowlist with the original fold-local choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config

from icp_bo_augmentation_cv import representation
from icp_forward_bo_ablation import ARMS, PROTOCOLS, matched_plan, seed_for
from icp_forward_selection_revision import save_json
from icp_intrinsic_descriptor_analysis import BASES
from icp_nested_group_cv import make_model
from icp_revision_inputs import sha
from summarize_icp_forward_bo_ablation import load_ablation
from summarize_icp_forward_three_tasks import METHODS, NAMES, performance_tables, selection_tables

ROOT = Path(__file__).resolve().parent
RAW = [f"{base}_{side}" for base in BASES for side in ("element1", "element2")]
FEATURES = RAW + ["icp_weighted_" + base for base in BASES]
INPUTS = RAW + ["actual_frac_element1"]
EXCLUDED = ["I_raw", "R_S", "Rs_from_EIS", "eta10", "E_actual",
            "E_IR_corrected_final", "sample_potential_shift"]
KEYS = ["arm", "protocol", "repeat", "fold"]
TOLERANCE = 1e-10


def check_features(features):
    if not features or len(features) != len(set(features)) or set(features) - set(FEATURES):
        raise ValueError(f"Invalid or forbidden model features: {features}")


def clean_inputs(data):
    """Only reference properties and ICP composition can enter the transformer."""
    return data.loc[:, INPUTS].copy()


def reference_audit(data, reference):
    if reference.element.duplicated().any():
        raise ValueError("Duplicate elemental reference")
    reference = reference.set_index("element")
    if data[["element1", "element2"]].isna().any().any():
        raise ValueError("Missing element identity")
    rows = []
    for base in BASES:
        for side in ("element1", "element2"):
            column = f"{base}_{side}"
            expected = data[side].map(reference[base]).to_numpy(float)
            observed = data[column].to_numpy(float)
            if not np.allclose(observed, expected, atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError(f"Shifted or inconsistent reference descriptor: {column}")
            rows.append(dict(feature=column, source="elemental_properties.csv",
                n_rows=len(data), n_reference_missing=int(np.isnan(expected).sum()),
                n_unlisted_element_rows=int((~data[side].isin(reference.index)).sum()),
                reference_matches=True, sample_potential_shift=False))
        rows.append(dict(feature="icp_weighted_"+base,
            source=f"ICP molar fraction * {base}_M1 + (1 - fraction) * {base}_M2",
            n_rows=len(data), reference_matches=True, sample_potential_shift=False))
    return pd.DataFrame(rows)


def check_path(state, fingerprint, max_features):
    if state["fingerprint"] != fingerprint or len(state["steps"]) != max_features:
        raise ValueError("Incomplete or stale forward path")
    selected = []
    for size, winner in enumerate(state["steps"], 1):
        trials = [r for r in state["trials"] if r["n_features"] == size]
        expected = set(range(len(FEATURES))) - set(selected)
        if len(trials) != len(expected) or {r["added_index"] for r in trials} != expected:
            raise ValueError("Forward candidate pool is not the allowed 39 descriptors")
        for trial in trials:
            index = trial["added_index"]
            if trial["added_feature"] != FEATURES[index] or trial["mask"] != selected + [index]:
                raise ValueError("Forward path has a forbidden or inconsistent feature")
        best = sorted(trials, key=lambda r: (-r["mean_r2"], r["added_index"]))[0]
        if winner != best:
            raise ValueError("Saved path winner does not match inner-CV trials")
        selected = winner["mask"]
    if state["selected"] != selected or len(state["trials"]) != sum(39-i for i in range(max_features)):
        raise ValueError("Incomplete forward path trials")


def poison_covariates(data, seed):
    result = data.copy()
    rng = np.random.default_rng(seed)
    # Also perturb the response column to catch its accidental use by transform.
    for column in EXCLUDED + ["separation_coefficient"]:
        result[column] = rng.uniform(-1e6, 1e6, len(result))
    return result


def replay_job(key, data, assignments, settings, expected, trees, output, fingerprint):
    arm, protocol, repeat, fold = key
    destination = output / "checkpoints" / f"{arm}_{protocol}_r{repeat}_f{fold}.json"
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Stale exclusion replay; use a fresh output directory")
        return saved
    start = time.perf_counter()
    original, augmented, test = matched_plan((protocol, repeat, fold), data, assignments)
    train = original if arm == "without_bo" else augmented
    seed = seed_for((protocol, repeat, fold))
    transformer = representation("hybrid39").fit(clean_inputs(train))
    X, V = transformer.transform(clean_inputs(train)), transformer.transform(clean_inputs(test))
    if list(X.columns) != FEATURES or not np.isfinite(X.to_numpy()).all() or not np.isfinite(V.to_numpy()).all():
        raise ValueError("Invalid transformed descriptor matrix")
    perturbed = representation("hybrid39").fit(poison_covariates(train, seed))
    for rows, clean in ((train, X), (test, V)):
        pd.testing.assert_frame_equal(clean, perturbed.transform(poison_covariates(rows, seed+1)), check_exact=True)
    result = dict(arm=arm, protocol=protocol, repeat=int(repeat), fold=int(fold),
        fingerprint=fingerprint, predictions=[], models=[],
        n_train=len(train), n_test=len(test), covariate_perturbation_invariant=True)
    cache = {}
    for choice in settings.to_dict("records"):
        selected = choice["features"]
        check_features(selected)
        cache_key = (choice["model"], tuple(selected), json.dumps(choice["config"], sort_keys=True))
        if cache_key not in cache:
            model = make_model(choice["model"], choice["config"], trees, seed)
            model.fit(X[selected].to_numpy(), train.separation_coefficient.to_numpy())
            cache[cache_key] = model.predict(V[selected].to_numpy())
        pred = cache[cache_key]
        old = expected[expected.strategy.eq(choice["strategy"])].set_index("sample")
        if set(old.index) != set(test["sample"]):
            raise ValueError("Replay test membership differs")
        delta = np.abs(pred - old.loc[test["sample"], "y_pred"].to_numpy())
        np.testing.assert_allclose(pred, old.loc[test["sample"], "y_pred"], atol=TOLERANCE, rtol=0)
        result["models"].append(dict(strategy=choice["strategy"], model=choice["model"],
            n_features=len(selected), max_prediction_difference=float(delta.max())))
        result["predictions"].extend(dict(strategy=choice["strategy"], sample=row["sample"],
            metal_pair=row.metal_pair, y_true=float(row.separation_coefficient), y_pred=float(p))
            for (_, row), p in zip(test.iterrows(), pred))
    result.update(unique_estimators_refitted=len(cache), elapsed_seconds=time.perf_counter()-start)
    save_json(destination, result)
    return result


def report(output, source, summary, verification):
    lines = ["# Electrochemical-Covariate Exclusion", "",
        "## What Was Done", "",
        "The completed nested forward-selection runs ALREADY excluded the questioned electrochemical inputs. "
        "This run independently refits their outer estimators after explicitly stripping all non-allowed columns. "
        "It verifies and reuses the saved fold-local feature searches and hyperparameters; it is not a new feature search or an independent validation dataset.", "",
        "Allowed model features: 26 unshifted M1/M2 reference properties and 13 ICP-molar-weighted properties. "
        "ICP fraction constructs the weighted properties; it is not an additional 40th feature. "
        "Missing reference values are imputed using outer-training data only during refitting, "
        "and inner-training data only during the original nested selection.", "",
        f"Reference coverage limitation: {', '.join(verification['unlisted_reference_elements'])} "
        f"have no property-table entries, affecting {verification['samples_with_unlisted_elements']} samples. "
        "Their endpoint descriptors are missing and therefore training-median imputed, not measured or "
        "element-specific reference values. This pre-existing limitation is preserved for the matched replay; "
        "see reference_coverage_missing.csv. A separate coverage sensitivity analysis is warranted.", "",
        "Excluded: I_raw, R_S/Rs_from_EIS, eta10, sample potentials, and sample-specific potential shifts. "
        "Work function, PZC and standard potential are matched against the unshifted elemental reference, "
        "not merely checked by feature name. The response and sample IDs are never part of X.", "",
        "Reference adsorption/bond energies are retained. Being related to catalysis does not by itself make "
        "a reference property a post-measurement performance proxy. ICP composition is measured after synthesis; "
        "these are composition-informed models, not purely presynthesis predictions.", "",
        f"Verified {verification['forward_paths']} full forward paths and refitted/replayed "
        f"{verification['model_settings']} selected model settings across {verification['outer_arm_jobs']} outer arm/fold jobs. "
        f"Maximum prediction difference from the original run: {verification['max_prediction_difference']:.3g}. "
        "Perturbing/inserting excluded inputs and the response left the descriptor matrices unchanged in every job.", "",
        "## Results", "",
        "R2 is calculated on pooled out-of-fold predictions for each repeat, then summarized as mean +/- SD "
        "across 3 repeats of 5-fold outer CV. SD is split variability, not experimental uncertainty.", "",
        "| Task | Strategy | Without BO training | With BO training |",
        "|---|---|---:|---:|"]
    for protocol in PROTOCOLS:
        for strategy in ("all39", "forward10", "forward_best", "forward_one_se"):
            part = summary[(summary.protocol==protocol) & (summary.strategy==strategy)].set_index("arm")
            values = [f"{part.loc[arm,'r2_mean']:.3f} +/- {part.loc[arm,'r2_sd']:.3f}" for arm in ARMS]
            lines.append(f"| {NAMES[protocol]} | {METHODS[strategy]} | " + " | ".join(values) + " |")
    lines += ["", "The first two tasks test initial-screening samples. The composition task tests held-out BO "
        "compositions in BOTH arms: without BO means no BO training, not no BO testing. "
        "For that task, initial-only tuning uses grouped-pair inner CV, while augmented tuning uses composition groups. "
        "The original matched-analysis report includes a locked-model control for this distinction.", "",
        "## Reviewer Interpretation", "",
        "The predictive signal in this updated workflow does not require the questioned electrochemical "
        "covariates or sample-specific descriptor shifts. Forward-10 retains broadly comparable performance "
        "with fewer inputs. These controls support a composition/property-based empirical model, not a causal mechanism.", "",
        "This exclusion replay alone does not quantify the benefit of ADDING electrochemical covariates, "
        "nor establish unchanged historical SHAP rankings. The earlier initial-only fixed-model comparison "
        "is a separate diagnostic. A matched covariate-on/off nested-selection comparison and the requested "
        "Pearson/Spearman audit would need to be reported separately to address those parts of the reviewer comment.", "",
        "The combined input contains I_raw and E_IR_corrected_final for 554 initial samples, "
        "but not the 111 BO samples. Their absence here is not evidence that those experiments were never performed. "
        "No missing BO operating variables were imputed to manufacture a covariate-included comparison.", "",
        "## Reproduction", "", "From the repository root:", "", "```powershell",
        "& 'C:\\Users\\yibinjiang\\anaconda3\\python.exe' hd_separation_ml/icp_electrochemical_exclusion.py --jobs 10",
        "```", "", f"Original feature-selection results: `{source}`.",
        "The manifest hashes inputs, source code, selection paths and checkpoints. Changed targets or "
        "descriptors require rerunning the upstream nested workflow, not replaying old selected features. "
        "To force fresh outer refits with unchanged dependencies, provide a new --output directory.", "",
        "Files: summary.csv; repeat_metrics.csv; predictions.csv; model_replay_audit.csv; "
        "descriptor_source_audit.csv; excluded_inputs.csv; forward_path_audit.csv; "
        "feature_selection_frequency.csv; property_selection_frequency.csv; verification.json."]
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/forward_bo_ablation_20261013")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/electrochemical_exclusion_verified_20261013")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    source, output = args.source.resolve(), args.output.resolve()
    data, tables, _, previous = load_ablation(source)
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)
    if versions != previous["versions"]:
        raise ValueError("Use the original Python/package versions for an exact estimator replay")
    reference_file = ROOT / "revision_inputs/elemental_properties.csv"
    reference = pd.read_csv(reference_file)
    descriptor_audit = reference_audit(data, reference)
    missing_reference = data.melt(id_vars=["sample", "cohort"], value_vars=["element1", "element2"],
        var_name="endpoint", value_name="element")
    missing_reference = missing_reference[~missing_reference.element.isin(reference.element)]
    for features in tables["settings"].features:
        check_features(features)
    provenance = pd.read_csv(source / "checkpoint_provenance.csv")
    path_rows = []
    dependencies = {**previous["source_hashes"], **previous["code_hashes"]}
    for row in provenance.itertuples(index=False):
        checkpoint = Path(row.checkpoint)
        file = checkpoint.parent.parent / "paths" / checkpoint.name
        state = json.loads(file.read_text(encoding="utf-8"))
        check_path(state, row.fingerprint, previous["options"]["max_features"])
        dependencies[str(file)] = sha(file)
        dependencies[str(checkpoint)] = sha(checkpoint)
        path_rows.append(dict(arm=row.arm, protocol=row.protocol, repeat=row.repeat, fold=row.fold,
            path=str(file), sha256=sha(file), candidate_pool=39, n_steps=len(state["steps"]),
            n_trials=len(state["trials"]), all_candidates_allowed=True))
    assignments_file = Path(previous["source"]) / "outer_assignments.csv"
    for file in [reference_file, assignments_file, source / "analysis_manifest.json",
                 source / "checkpoint_provenance.csv", source / "outer_assignments.csv",
                 Path(__file__), ROOT / "summarize_icp_forward_bo_ablation.py",
                 ROOT / "test_icp_electrochemical_exclusion.py"]:
        dependencies[str(file)] = sha(file)
    for file, expected in dependencies.items():
        if sha(Path(file)) != expected:
            raise ValueError(f"Changed dependency: {file}")
    manifest = dict(source=str(source), dependencies=dependencies, versions=versions,
        allowed_transform_inputs=INPUTS, allowed_model_features=FEATURES, excluded=EXCLUDED,
        selection_policy="Verified saved fold-local searches; no new feature search",
        replay_policy="Refit outer estimators with original settings on allowlisted inputs",
        tolerance=TOLERANCE, trees=previous["search_options"]["final_trees"])
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "analysis_manifest.json"
    if saved.exists() and json.loads(saved.read_text(encoding="utf-8"))["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Changed audit dependencies; use a fresh output directory")
    save_json(saved, manifest)
    (output / "checkpoints").mkdir(exist_ok=True)
    descriptor_audit.to_csv(output / "descriptor_source_audit.csv", index=False)
    missing_reference.to_csv(output / "reference_coverage_missing.csv", index=False)
    pd.DataFrame(path_rows).to_csv(output / "forward_path_audit.csv", index=False)
    exclusions = []
    for column in EXCLUDED:
        for cohort, part in data.groupby("cohort"):
            exclusions.append(dict(field=column, cohort=cohort, n_samples=len(part),
                present_in_input=column in part, nonmissing=int(part[column].notna().sum()) if column in part else 0,
                used_in_model=False, exclusion="Strict allowlist; reference equality rejects hidden shifts"))
    pd.DataFrame(exclusions).to_csv(output / "excluded_inputs.csv", index=False)
    print(f"Verified {len(data)} descriptor rows (including reference NaNs), {len(path_rows)} selection paths "
          f"and {len(tables['settings'])} model feature lists; starting outer refits", flush=True)
    assignments = pd.read_csv(assignments_file)
    prediction_groups = dict(tuple(tables["predictions"].groupby(KEYS)))
    results, start = [], time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for result in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(replay_job)(key, data, assignments, settings, prediction_groups[key],
                manifest["trees"], output, manifest["fingerprint"])
            for key, settings in tables["settings"].groupby(KEYS)):
            results.append(result)
            if len(results) % 10 == 0:
                print(f"Exclusion replay {len(results)}/90 outer jobs; {time.perf_counter()-start:.1f}s", flush=True)
    predictions = pd.DataFrame([dict(**{k:r[k] for k in KEYS}, **p) for r in results for p in r["predictions"]])
    models = pd.DataFrame([dict(**{k:r[k] for k in KEYS}, **p) for r in results for p in r["models"]])
    keys = KEYS + ["strategy", "sample"]
    old = tables["predictions"].set_index(keys).sort_index()
    new = predictions.set_index(keys).sort_index()
    pd.testing.assert_index_equal(old.index, new.index)
    np.testing.assert_allclose(new[["y_true", "y_pred"]], old[["y_true", "y_pred"]], atol=TOLERANCE, rtol=0)
    collected = {}
    for arm in ARMS:
        settings = tables["settings"].query("arm == @arm")
        summary, repeats, folds = performance_tables(predictions.query("arm == @arm"), settings, data)
        items = dict(summary=summary, repeat_metrics=repeats, fold_metrics=folds,
                     **selection_tables(settings, FEATURES))
        for name, frame in items.items():
            collected.setdefault(name, []).append(frame.assign(arm=arm))
    combined = {name:pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
    for name, frame in combined.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    models.to_csv(output / "model_replay_audit.csv", index=False)
    verification = dict(fingerprint=manifest["fingerprint"], outer_arm_jobs=len(results), model_settings=len(models),
        unique_estimators_refitted=sum(r["unique_estimators_refitted"] for r in results),
        forward_paths=len(path_rows), n_samples=len(data), max_prediction_difference=float(np.max(np.abs(new.y_pred-old.y_pred))),
        perturbation_invariant=all(r["covariate_perturbation_invariant"] for r in results),
        unlisted_reference_elements=sorted(missing_reference.element.unique()),
        samples_with_unlisted_elements=int(missing_reference['sample'].nunique()),
        complete=True, elapsed_seconds=time.perf_counter()-start)
    save_json(output / "verification.json", verification)
    report(output, source, combined["summary"], verification)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
