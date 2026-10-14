"""Matched with/without-BO training ablation of nested forward selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from icp_bo_augmentation_cv import representation
from icp_forward_selection_revision import evaluate_subset, save_json, select_models
from icp_forward_three_tasks import PROTOCOLS, TABLES, check_split, forward_path, plan_for
from icp_identity_diagnostics import property_permutations
from icp_nested_group_cv import make_model
from icp_revision_inputs import sha
from icp_two_prediction_tasks import inner_partitions
from summarize_icp_forward_three_tasks import load_results

ROOT = Path(__file__).resolve().parent
ARMS = ["without_bo", "with_bo"]


def no_bo_inner(original, protocol, seed):
    if not original.cohort.eq("initial").all():
        raise ValueError("BO data entered the initial-only selection input")
    # There is no composition-series validation set without BO. Tune on initial
    # grouped-pair CV, then evaluate zero-BO transfer to the frozen BO test folds.
    tuning_protocol = "unseen_pair" if protocol == "known_pair_new_composition" else protocol
    for train, valid in inner_partitions(original, original.iloc[:0], tuning_protocol, False, seed):
        check_split(train, valid, tuning_protocol)
        if not train.cohort.eq("initial").all() or not valid.cohort.eq("initial").all():
            raise ValueError("BO outcome entered initial-only tuning")
        yield train, valid, tuning_protocol


def prepare_no_bo(original, protocol, seed):
    inner, memberships = [], []
    for fold, (train, valid, tuning_protocol) in enumerate(no_bo_inner(original, protocol, seed)):
        transform = representation("hybrid39").fit(train)
        frame = transform.transform(train)
        inner.append(dict(X=frame.to_numpy(), V=transform.transform(valid).to_numpy(),
            y=train.separation_coefficient.to_numpy(), z=valid.separation_coefficient.to_numpy(), columns=list(frame.columns)))
        for role, rows in [("train", train), ("validation", valid)]:
            memberships.extend(dict(inner_fold=fold, role=role, sample=r["sample"], metal_pair=r.metal_pair,
                icp_group=r.icp_group, composition_group=r.composition_group, cohort=r.cohort,
                tuning_protocol=tuning_protocol) for _, r in rows.iterrows())
    return inner, memberships


def matched_plan(key, data, assignments):
    groups, augmented = plan_for(key, data, assignments)
    test = groups["testing"]
    if key[0] != "known_pair_new_composition":
        test = test[test.cohort.eq("initial")].reset_index(drop=True)
    original = groups["original"]
    if not original.cohort.eq("initial").all():
        raise ValueError("Non-initial sample in no-BO arm")
    check_split(original, test, key[0])
    return original, augmented, test


def seed_for(key):
    protocol, repeat, fold = key
    return (2000 if protocol == "unseen_pair" else 7000) + int(repeat)*100 + int(fold)


def audit(data, assignments):
    rows, membership = [], []
    keys = assignments[["protocol", "repeat", "fold"]].drop_duplicates()
    if len(keys) != 45:
        raise ValueError("Expected 45 outer fits")
    for key in keys.itertuples(index=False, name=None):
        original, augmented, test = matched_plan(key, data, assignments)
        common = dict(zip(("protocol", "repeat", "fold"), key))
        for arm, train in [("without_bo", original), ("with_bo", augmented)]:
            rows.append(dict(**common, arm=arm, level="outer", inner_fold=-1,
                n_train_bo=int(train.cohort.eq("BO").sum()), n_validation_bo=int(test.cohort.eq("BO").sum()),
                **check_split(train, test, key[0])))
            for role, group in [("train", train), ("test", test)]:
                membership.extend(dict(**common, arm=arm, role=role, sample=r["sample"], cohort=r.cohort)
                                  for _, r in group.iterrows())
        for i, (train, valid, tuning_protocol) in enumerate(no_bo_inner(original, key[0], seed_for(key))):
            if (set(train["sample"]) | set(valid["sample"])) & set(test["sample"]):
                raise ValueError("Outer-test sample entered initial-only selection")
            rows.append(dict(**common, arm="without_bo", level="inner", inner_fold=i,
                n_train_bo=0, n_validation_bo=0, tuning_protocol=tuning_protocol,
                **check_split(train, valid, tuning_protocol)))
    return pd.DataFrame(rows), pd.DataFrame(membership)


def fit_job(key, data, assignments, search_options, options, output, fingerprint, with_result):
    protocol, repeat, fold = key
    filename = f"{protocol}_r{repeat}_f{fold}.json"
    destination = output / "checkpoints" / filename
    if destination.exists():
        result = json.loads(destination.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Stale ablation checkpoint")
        return result
    train, augmented, test = matched_plan(key, data, assignments)
    seed = seed_for(key)
    inner, memberships = prepare_no_bo(train, protocol, seed)
    columns = inner[0]["columns"]
    state = forward_path(inner, options, seed, output / "paths" / filename, fingerprint)
    tuned, all_candidates, tuning_scores = [], [], []
    steps = state["steps"] + [dict(mask=list(range(len(columns))), n_features=len(columns))]
    for step in steps:
        for name in ("RF", "ET"):
            for ci, config in enumerate(search_options["configs"]):
                rows = evaluate_subset(inner, step["mask"], name, config, search_options["search_trees"], seed)
                candidate = dict(model=name, n_features=step["n_features"], config_id=ci, config=config,
                    features=[columns[i] for i in step["mask"]], inner_r2_mean=float(np.mean([r["r2"] for r in rows])),
                    inner_r2_std=float(np.std([r["r2"] for r in rows], ddof=1)))
                (all_candidates if step["n_features"]==len(columns) else tuned).append(candidate)
                tuning_scores.extend(dict(model=name, n_features=step["n_features"], config_id=ci, **r) for r in rows)
    choices, threshold = select_models(tuned, all_candidates)
    transform = representation("hybrid39").fit(train)
    X, V = transform.transform(train), transform.transform(test)
    result = dict(protocol=protocol, repeat=int(repeat), fold=int(fold), fingerprint=fingerprint,
        predictions=[], settings=[], permutation=[], inner_assignments=memberships, tuning_scores=tuning_scores,
        path=[{k:v for k,v in s.items() if k!="inner_scores"} for s in state["steps"]], fixed_control=[])
    for strategy, choice in choices.items():
        selected = choice["features"]
        model = make_model(choice["model"], choice["config"], search_options["final_trees"], seed)
        model.fit(X[selected].to_numpy(), train.separation_coefficient)
        pred = model.predict(V[selected].to_numpy())
        result["settings"].append(dict(strategy=strategy, **{k:choice[k] for k in
            ("model", "n_features", "features", "config", "inner_r2_mean", "inner_r2_std")},
            one_se_threshold=threshold, n_train_initial=len(train), n_train_bo=0))
        result["predictions"].extend(dict(strategy=strategy, sample=r["sample"], metal_pair=r.metal_pair,
            y_true=float(r.separation_coefficient), y_pred=float(p)) for (_,r),p in zip(test.iterrows(),pred))
        if strategy in ("forward_best", "forward_one_se"):
            result["permutation"].extend(dict(strategy=strategy, **r) for r in property_permutations(model, V[selected], test, pred, seed))
    # Lock representation/model/config chosen using INITIAL data, to separate the
    # data-addition effect from the different tuning-validation tasks in task 3.
    fixed = choices["all39"]
    transform_bo = representation("hybrid39").fit(augmented)
    model = make_model(fixed["model"], fixed["config"], search_options["final_trees"], seed)
    model.fit(transform_bo.transform(augmented).to_numpy(), augmented.separation_coefficient)
    pred = model.predict(transform_bo.transform(test).to_numpy())
    result["fixed_control"] = [dict(sample=r["sample"], y_true=float(r.separation_coefficient), y_pred=float(p),
        model=fixed["model"], config=fixed["config"]) for (_,r),p in zip(test.iterrows(),pred)]
    reference = pd.DataFrame(with_result["predictions"])
    for strategy, group in pd.DataFrame(result["predictions"]).groupby("strategy"):
        old = reference[reference.strategy.eq(strategy)].set_index("sample")
        if set(old.index) != set(group["sample"]):
            raise ValueError("With/without-BO test membership mismatch")
        np.testing.assert_allclose(group.y_true, old.loc[group["sample"], "y_true"], atol=1e-12, rtol=0)
    save_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-bo", type=Path, default=ROOT / "outputs/forward_three_tasks_20261013")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/forward_bo_ablation_20261013")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    with_bo, output = args.with_bo.resolve(), args.output.resolve()
    data, _, previous = load_results(with_bo)
    source = Path(previous["source"])
    upstream = json.loads((source / "analysis_manifest.json").read_text(encoding="utf-8"))
    assignments = pd.read_csv(source / "outer_assignments.csv")
    provenance = pd.read_csv(with_bo / "checkpoint_provenance.csv")
    reused = { (r.protocol, int(r.repeat), int(r.fold)):json.loads(Path(r.checkpoint).read_text(encoding="utf-8"))
              for r in provenance.itertuples(index=False)}
    sources = {**previous["source_hashes"], str(with_bo / "analysis_manifest.json"):sha(with_bo / "analysis_manifest.json"),
               str(with_bo / "checkpoint_provenance.csv"):sha(with_bo / "checkpoint_provenance.csv")}
    sources.update({r.checkpoint:r.sha256 for r in provenance.itertuples(index=False)})
    code = dict(previous["code_hashes"])
    for file in [Path(__file__), ROOT / "summarize_icp_forward_three_tasks.py"]:
        code[str(file)] = sha(file)
    manifest = dict(source=str(source), with_bo_source=str(with_bo), source_hashes=sources, code_hashes=code,
        options=previous["options"], search_options=upstream["options"], protocols=PROTOCOLS, arms=ARMS,
        repeats=3, folds=5, inner_folds=3, versions=previous["versions"],
        known_pair_without_bo_tuning="Initial-only unordered-pair grouped CV; no BO outcomes used in selection or tuning.",
        comparison="Frozen matched test samples. With-BO fits reused; without-BO forward path and tuning are independent.",
        fixed_control="All39 with model/config chosen on initial-only inner CV, held fixed when adding eligible BO training data.")
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    file = output / "analysis_manifest.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8"))["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Changed source/code/options; use a fresh output directory")
    save_json(file, manifest)
    split_audit, memberships = audit(data, assignments)
    split_audit.to_csv(output / "split_audit.csv", index=False)
    memberships.to_csv(output / "outer_assignments.csv", index=False)
    print("AUDIT: 90 matched outer arms, 135 initial-only inner splits; zero BO tuning rows in no-BO arm", flush=True)
    if args.audit_only:
        return
    for folder in ("checkpoints", "paths"):
        (output / folder).mkdir(exist_ok=True)
    results, start = [], time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for result in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(fit_job)(key, data, assignments, upstream["options"], previous["options"], output,
                                 manifest["fingerprint"], reused[key]) for key in sorted(reused)):
            results.append(result)
            print(f"Without-BO fits {len(results)}/45 complete; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    all_results = [("with_bo", r) for r in reused.values()] + [("without_bo", r) for r in results]
    for category in TABLES:
        rows = [dict(arm=arm, **{k:r[k] for k in ("protocol", "repeat", "fold")}, **entry)
                for arm,r in all_results for entry in r[category]]
        pd.DataFrame(rows).to_csv(output / f"{category}.csv", index=False)
    pd.DataFrame([dict(**{k:r[k] for k in ("protocol", "repeat", "fold")}, **entry)
        for r in results for entry in r["fixed_control"]]).to_csv(output / "locked_all39_bo_predictions.csv", index=False)
    provenance["arm"] = "with_bo"
    added = []
    for result in results:
        key = {k:result[k] for k in ("protocol", "repeat", "fold")}
        file = output / "checkpoints" / f"{key['protocol']}_r{key['repeat']}_f{key['fold']}.json"
        added.append(dict(**key, arm="without_bo", checkpoint=str(file), sha256=sha(file),
                          reused=False, fingerprint=manifest["fingerprint"]))
    pd.concat([provenance, pd.DataFrame(added)], ignore_index=True).to_csv(output / "checkpoint_provenance.csv", index=False)
    save_json(output / "verification.json", dict(fingerprint=manifest["fingerprint"], reused_with_bo=45,
        fitted_without_bo=len(results), matched_test_sets=True, no_bo_selection_isolation=True,
        elapsed_seconds=time.perf_counter()-start))
    print("Three-task BO ablation complete", flush=True)


if __name__ == "__main__":
    main()
