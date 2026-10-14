"""Add the random task to the frozen nested-forward grouped comparisons."""

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
from icp_forward_selection_revision import PATH_CONFIG, evaluate_subset, save_json, select_models, choose_candidate
from icp_identity_diagnostics import property_permutations
from icp_nested_group_cv import make_model
from icp_revision_inputs import sha
from icp_two_prediction_tasks import inner_partitions

ROOT = Path(__file__).resolve().parent
PROTOCOLS = ["stratified_random", "unseen_pair", "known_pair_new_composition"]
TABLES = ["predictions", "settings", "permutation", "inner_assignments", "tuning_scores", "path"]


def check_split(train, test, protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
    for field in ("sample", "icp_group"):
        if train[field].isna().any() or test[field].isna().any():
            raise ValueError(f"Missing {field}")
        if set(train[field]) & set(test[field]):
            raise ValueError(f"Overlapping {field}: {protocol}")
    if train["sample"].duplicated().any() or test["sample"].duplicated().any():
        raise ValueError("Duplicate sample within a split")
    pair_overlap = set(train.metal_pair) & set(test.metal_pair)
    composition_overlap = set(train.composition_group) & set(test.composition_group)
    if protocol == "unseen_pair" and pair_overlap:
        raise ValueError("Overlapping metal_pair: unseen_pair")
    if protocol == "known_pair_new_composition" and composition_overlap:
        raise ValueError("Overlapping composition_group: known_pair_new_composition")
    bo_overlap = set(train.loc[train.cohort.eq("BO"), "composition_group"]) & set(test.composition_group)
    if bo_overlap:
        raise ValueError("Training BO overlaps a held-out composition bin")
    return dict(n_train=len(train), n_test=len(test), shared_samples=0, shared_icp=0,
        shared_pairs=len(pair_overlap), shared_composition_bins=len(composition_overlap),
        training_bo_in_test_bins=0,
        fraction_test_pair_seen=float(test.metal_pair.isin(train.metal_pair).mean()))


def plan_for(key, data, assignments):
    protocol, repeat, fold = key
    part = assignments[(assignments.protocol == protocol) & (assignments.repeat == repeat) & (assignments.fold == fold)]
    groups = {role: data.loc[part.loc[part.role.eq(role), "sample"]].reset_index(drop=True)
              for role in ("original", "augmented", "testing")}
    train = pd.concat([groups["original"], groups["augmented"]], ignore_index=True)
    check_split(train, groups["testing"], protocol)
    return groups, train


def prepared_inner(original, added, protocol, seed):
    inner, assignments = [], []
    for fold, (train, valid) in enumerate(inner_partitions(original, added, protocol, True, seed)):
        check_split(train, valid, protocol)
        transform = representation("hybrid39").fit(train)
        frame = transform.transform(train)
        inner.append(dict(X=frame.to_numpy(), V=transform.transform(valid).to_numpy(),
            y=train.separation_coefficient.to_numpy(), z=valid.separation_coefficient.to_numpy(), columns=list(frame.columns)))
        for role, group in [("train", train), ("validation", valid)]:
            assignments.extend(dict(inner_fold=fold, role=role, sample=r["sample"], metal_pair=r.metal_pair,
                icp_group=r.icp_group, composition_group=r.composition_group) for _, r in group.iterrows())
    return inner, assignments


def reference_candidates(key, source, source_manifest, columns):
    protocol, repeat, fold = key
    candidates, payloads = [], {}
    for name in ("RF", "ET"):
        file = source / "checkpoints" / f"{protocol}_r{repeat}_f{fold}_hybrid39_{name}.json"
        payload = json.loads(file.read_text(encoding="utf-8"))
        if payload["fingerprint"] != source_manifest["fingerprint"]:
            raise ValueError(f"Reference checkpoint mismatch: {file}")
        payloads[name] = payload
        candidates.extend(dict(model=name, features=columns, **r) for r in payload["candidates"]
                          if r["training"] == "with_cofe" and r["n_features"] == 39)
    return candidates, payloads


def check_reference(pred, test, payload):
    old = pd.DataFrame(payload["predictions"])
    old = old[old.training.eq("with_cofe") & old.strategy.eq("tuned_all")].set_index("sample")
    np.testing.assert_allclose(pred, old.loc[test["sample"], "y_pred"], atol=1e-10, rtol=0)


def forward_path(inner, options, seed, file, fingerprint):
    if file.exists():
        state = json.loads(file.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Stale forward path")
    else:
        state = dict(fingerprint=fingerprint, selected=[], steps=[], trials=[])
    for size in range(len(state["selected"]) + 1, options["max_features"] + 1):
        candidates = []
        for index, name in enumerate(inner[0]["columns"]):
            if index in state["selected"]:
                continue
            mask = state["selected"] + [index]
            rows = [dict(model=model, **r) for model in ("RF", "ET") for r in
                    evaluate_subset(inner, mask, model, PATH_CONFIG, options["selection_trees"], seed)]
            candidates.append(dict(n_features=size, added_feature=name, added_index=index, mask=mask,
                mean_r2=float(np.mean([r["r2"] for r in rows])), inner_scores=rows))
        winner = sorted(candidates, key=lambda r: (-r["mean_r2"], r["added_index"]))[0]
        state["selected"] = winner["mask"]
        state["steps"].append(winner)
        state["trials"].extend(candidates)
        save_json(file, state)
        if size % 4 == 0:
            print(f"{file.stem}: forward {size}/{options['max_features']}", flush=True)
    return state


def random_job(key, data, assignments, source, source_manifest, options, output, fingerprint):
    protocol, repeat, fold = key
    if protocol != "stratified_random":
        raise ValueError("Grouped fits must be reused from their verified source")
    file = f"{protocol}_r{repeat}_f{fold}.json"
    destination = output / "checkpoints" / file
    if destination.exists():
        result = json.loads(destination.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Stale random-task checkpoint")
        return result
    groups, train = plan_for(key, data, assignments)
    test = groups["testing"]
    if not test.cohort.eq("initial").all():
        raise ValueError("Frozen random task should test initial samples only")
    seed = 7000 + int(repeat) * 100 + int(fold)
    inner, memberships = prepared_inner(groups["original"], groups["augmented"], protocol, seed)
    columns = inner[0]["columns"]
    state = forward_path(inner, options, seed, output / "paths" / file, fingerprint)
    tuned, tuning_scores = [], []
    for step in state["steps"]:
        for name in ("RF", "ET"):
            for ci, config in enumerate(source_manifest["options"]["configs"]):
                rows = evaluate_subset(inner, step["mask"], name, config, source_manifest["options"]["search_trees"], seed)
                tuned.append(dict(model=name, n_features=step["n_features"], config_id=ci, config=config,
                    features=[columns[i] for i in step["mask"]], inner_r2_mean=float(np.mean([r["r2"] for r in rows])),
                    inner_r2_std=float(np.std([r["r2"] for r in rows], ddof=1))))
                tuning_scores.extend(dict(model=name, n_features=step["n_features"], config_id=ci, **r) for r in rows)
    all_candidates, payloads = reference_candidates(key, source, source_manifest, columns)
    choices, threshold = select_models(tuned, all_candidates)
    transform = representation("hybrid39").fit(train)
    X, V = transform.transform(train), transform.transform(test)
    result = dict(protocol=protocol, repeat=int(repeat), fold=int(fold), fingerprint=fingerprint,
        predictions=[], settings=[], permutation=[], inner_assignments=memberships, tuning_scores=tuning_scores,
        path=[{k:v for k,v in step.items() if k != "inner_scores"} for step in state["steps"]])
    cache = {}
    for strategy, choice in choices.items():
        selected = choice["features"]
        cache_key = (choice["model"], tuple(selected), choice["config_id"])
        if cache_key not in cache:
            model = make_model(choice["model"], choice["config"], source_manifest["options"]["final_trees"], seed)
            model.fit(X[selected].to_numpy(), train.separation_coefficient)
            cache[cache_key] = model
        model = cache[cache_key]
        pred = model.predict(V[selected].to_numpy())
        if strategy == "all39":
            check_reference(pred, test, payloads[choice["model"]])
        result["settings"].append(dict(strategy=strategy, **{k:choice[k] for k in
            ("model", "n_features", "features", "config", "inner_r2_mean", "inner_r2_std")}, one_se_threshold=threshold))
        result["predictions"].extend(dict(strategy=strategy, sample=r["sample"], metal_pair=r.metal_pair,
            y_true=float(r.separation_coefficient), y_pred=float(p)) for (_,r),p in zip(test.iterrows(),pred))
        if strategy in ("forward_best", "forward_one_se"):
            result["permutation"].extend(dict(strategy=strategy, **r) for r in
                property_permutations(model, V[selected], test, pred, seed))
    save_json(destination, result)
    return result


def verified_grouped(source, grouped):
    old = json.loads((grouped / "analysis_manifest.json").read_text(encoding="utf-8"))
    if Path(old["source"]).resolve() != source.resolve():
        raise ValueError("Grouped results use a different upstream source")
    for path, expected in {**old["source_hashes"], **old["code_hashes"]}.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Changed grouped input or code: {path}")
    results, paths = [], []
    for protocol in PROTOCOLS[1:]:
        for repeat in range(3):
            for fold in range(5):
                file = grouped / "checkpoints" / f"{protocol}_r{repeat}_f{fold}.json"
                result = json.loads(file.read_text(encoding="utf-8"))
                if result["fingerprint"] != old["fingerprint"]:
                    raise ValueError("Grouped checkpoint fingerprint mismatch")
                results.append(result)
                paths.append(file)
    return old, results, paths


def audit_plans(data, assignments):
    rows = []
    keys = assignments[["protocol", "repeat", "fold"]].drop_duplicates()
    if len(keys) != 45 or set(keys.protocol) != set(PROTOCOLS):
        raise ValueError("Expected three tasks, three repeats and five folds")
    for key in keys.itertuples(index=False, name=None):
        groups, train = plan_for(key, data, assignments)
        protocol, repeat, fold = key
        common = dict(protocol=protocol, repeat=int(repeat), fold=int(fold))
        rows.append(dict(**common, level="outer", inner_fold=-1, **check_split(train, groups["testing"], protocol)))
        seed = (2000 if protocol == "unseen_pair" else 7000) + int(repeat) * 100 + int(fold)
        for inner_fold, (a,b) in enumerate(inner_partitions(groups["original"], groups["augmented"], protocol, True, seed)):
            if (set(a["sample"]) | set(b["sample"])) & set(groups["testing"]["sample"]):
                raise ValueError("Outer-test sample entered inner selection")
            rows.append(dict(**common, level="inner", inner_fold=inner_fold, **check_split(a,b,protocol)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/cofe_inclusion_20261012")
    parser.add_argument("--grouped", type=Path, default=ROOT / "outputs/forward_selection_revision_20261013")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/forward_three_tasks_20261013")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    source, grouped, output = args.source.resolve(), args.grouped.resolve(), args.output.resolve()
    old, reused, grouped_files = verified_grouped(source, grouped)
    source_manifest = json.loads((source / "analysis_manifest.json").read_text(encoding="utf-8"))
    for name, expected in source_manifest["code_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed upstream code: {name}")
    code = {str(ROOT/name):sha(ROOT/name) for name in source_manifest["code_hashes"]}
    code.update(old["code_hashes"])
    code[str(Path(__file__))] = sha(Path(__file__))
    sources = {**old["source_hashes"], str(grouped / "analysis_manifest.json"):sha(grouped / "analysis_manifest.json")}
    sources.update({str(p):sha(p) for p in grouped_files})
    manifest = dict(source=str(source), grouped_source=str(grouped), grouped_fingerprint=old["fingerprint"],
        source_fingerprint=source_manifest["fingerprint"], source_hashes=sources, code_hashes=code,
        options=old["options"], protocols=PROTOCOLS, repeats=3, folds=5,
        design="Same fold-local forward selection; add stratified random initial-sample task; reuse 30 verified grouped fits.",
        versions=source_manifest["versions"])
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    file = output / "analysis_manifest.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8"))["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Changed source/code/options; use a fresh output directory")
    save_json(file, manifest)
    data = pd.read_csv(source / "combined_dataset.csv").set_index("sample", drop=False)
    assignments = pd.read_csv(source / "outer_assignments.csv")
    audit = audit_plans(data, assignments)
    audit.to_csv(output / "split_audit.csv", index=False)
    keys = assignments[assignments.protocol.eq("stratified_random")][["protocol", "repeat", "fold"]].drop_duplicates()
    # Replay one random-task control before starting the expensive forward search.
    key = tuple(keys.iloc[0])
    groups, train = plan_for(key, data, assignments)
    transform = representation("hybrid39").fit(train)
    X, V = transform.transform(train), transform.transform(groups["testing"])
    candidates, payloads = reference_candidates(key, source, source_manifest, list(X.columns))
    choice = choose_candidate(candidates)
    seed = 7000 + int(key[1])*100 + int(key[2])
    model = make_model(choice["model"], choice["config"], source_manifest["options"]["final_trees"], seed)
    model.fit(X.to_numpy(), train.separation_coefficient)
    check_reference(model.predict(V.to_numpy()), groups["testing"], payloads[choice["model"]])
    print("AUDIT: 45 outer and 135 inner splits verified; random all39 control replayed; 30 grouped fits reusable", flush=True)
    if args.audit_only:
        return
    for name in ("checkpoints", "paths"):
        (output / name).mkdir(exist_ok=True)
    results, start = list(reused), time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for result in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(random_job)(tuple(key), data, assignments, source, source_manifest, old["options"], output,
                                    manifest["fingerprint"]) for key in keys.to_numpy()):
            results.append(result)
            print(f"Random fits {len(results)-len(reused)}/{len(keys)} complete; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    for category in TABLES:
        rows = [dict(**{k:r[k] for k in ("protocol", "repeat", "fold")}, **row) for r in results for row in r[category]]
        pd.DataFrame(rows).to_csv(output / f"{category}.csv", index=False)
    provenance = []
    for result in results:
        protocol, repeat, fold = (result[k] for k in ("protocol", "repeat", "fold"))
        parent = output if protocol == "stratified_random" else grouped
        file = parent / "checkpoints" / f"{protocol}_r{repeat}_f{fold}.json"
        provenance.append(dict(protocol=protocol, repeat=repeat, fold=fold, checkpoint=str(file), sha256=sha(file),
                               reused=protocol != "stratified_random", fingerprint=result["fingerprint"]))
    pd.DataFrame(provenance).to_csv(output / "checkpoint_provenance.csv", index=False)
    save_json(output / "verification.json", dict(completed_folds=len(results), reused_folds=len(reused),
        fitted_random_folds=len(keys), all39_replayed=True, split_isolation_checked=True,
        fingerprint=manifest["fingerprint"], elapsed_seconds=time.perf_counter()-start))
    print("Three-task forward fitting complete", flush=True)


if __name__ == "__main__":
    main()
