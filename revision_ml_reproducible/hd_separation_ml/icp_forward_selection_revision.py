"""Historical forward replay and fold-local forward selection with corrected ICP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from icp_bo_augmentation_cv import representation
from icp_identity_diagnostics import assert_isolated, property_permutations
from icp_intrinsic_descriptor_analysis import scores
from icp_nested_group_cv import make_model
from icp_revision_inputs import sha
from icp_two_prediction_tasks import inner_partitions

ROOT = Path(__file__).resolve().parent
PROTOCOLS = ["unseen_pair", "known_pair_new_composition"]
HISTORICAL = ROOT / "outputs/06c_forward_regression_seed42_all28pool_rf_et_only"
PATH_CONFIG = dict(max_depth=8, min_samples_leaf=1, max_features="sqrt")


def save_json(path, content):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(content, indent=2), encoding="utf-8")
    temporary.replace(path)


def historical_trial(X, y, selected, candidate, folds):
    import forward_regression_seed42_all28pool_rf_et_only as legacy
    features = selected + [candidate]
    table = legacy.evaluate_cv(X, y, features, folds)
    return dict(candidate=candidate, features=features, mean=float(table.cv_r2_mean.mean()),
                rows=table.to_dict("records"))


def replay_historical(output, jobs):
    import forward_regression_seed42_all28pool_rf_et_only as legacy
    output.mkdir(parents=True, exist_ok=True)
    inputs = [legacy.FEATURE_MATRIX, legacy.CANDIDATE_FEATURES, HISTORICAL / "starting_core4_features.csv",
              HISTORICAL / "forward_regression_selected_path_summary.csv", HISTORICAL / "heldout_test_results_by_path.csv",
              Path(legacy.__file__), Path(__file__)]
    hashes = {str(p): sha(p) for p in inputs}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["hashes"] != hashes:
        raise ValueError("Historical replay source changed; use a fresh output")
    save_json(manifest_path, dict(hashes=hashes, classification_core="Archived core reused; exhaustive classification not rerun"))
    df = pd.read_csv(legacy.FEATURE_MATRIX).dropna(subset=["separation_coefficient"]).reset_index(drop=True)
    pool = pd.read_csv(legacy.CANDIDATE_FEATURES).feature.tolist()
    selected = pd.read_csv(HISTORICAL / "starting_core4_features.csv").feature.tolist()
    X, y = df[pool], df.separation_coefficient.to_numpy(float)
    folds = legacy.cv_folds(y)
    base = legacy.evaluate_cv(X, y, selected, folds)
    path = [dict(n_features=len(selected), features=list(selected), added="BASE", rows=base.to_dict("records"))]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for k in range(5, legacy.MAX_FEATURES + 1):
            checkpoint = output / f"step_{k:02d}.json"
            remaining = [c for c in pool if c not in selected]
            if checkpoint.exists():
                trials = json.loads(checkpoint.read_text(encoding="utf-8"))
                assert all(r["features"][:-1] == selected for r in trials)
            else:
                trials = Parallel(n_jobs=jobs)(delayed(historical_trial)(X, y, selected, c, folds) for c in remaining)
                save_json(checkpoint, trials)
            best = sorted(trials, key=lambda r: (-r["mean"], remaining.index(r["candidate"])))[0]
            selected = best["features"]
            path.append(dict(n_features=k, features=list(selected), added=best["candidate"], rows=best["rows"]))
            print(f"Historical replay {k}/16: {best['candidate']}; CV={best['mean']:.6f}", flush=True)
    archived = pd.read_csv(HISTORICAL / "forward_regression_selected_path_summary.csv").set_index("n_features")
    comparisons = []
    for step in path:
        old = archived.loc[step["n_features"]]
        recorded = [c.strip() for c in old.features.split(",")]
        comparisons.append(dict(n_features=step["n_features"], added=step["added"],
            features=step["features"], archived_features_match=step["features"] == recorded,
            cv_r2=float(np.mean([r["cv_r2_mean"] for r in step["rows"]])), archived_cv_r2=float(old.mean_r2_across_models)))
    pd.DataFrame(comparisons).to_csv(output / "forward_path_reproduction.csv", index=False)
    indices = np.arange(len(y))
    tr, te = train_test_split(indices, test_size=.2, random_state=42, stratify=pd.qcut(y, 5, labels=False, duplicates="drop"))
    train_test, metrics, predictions = [], [], []
    for role, ids in [("train", tr), ("test", te)]:
        train_test.extend(dict(row=int(i), role=role) for i in ids)
    pd.DataFrame(train_test).to_csv(output / "historical_split.csv", index=False)
    reference = pd.read_csv(HISTORICAL / "heldout_test_results_by_path.csv").set_index(["n_features", "model"])
    for step in path:
        for name, estimator in legacy.final_test_models().items():
            estimator.set_params(n_jobs=1)
            model = make_pipeline(SimpleImputer(strategy="median"), clone(estimator))
            model.fit(X.iloc[tr][step["features"]], y[tr])
            pred = model.predict(X.iloc[te][step["features"]])
            metric = scores(y[te], pred)
            old = reference.loc[(step["n_features"], name)]
            metrics.append(dict(n_features=step["n_features"], model=name, **metric,
                archived_r2=float(old.test_r2), r2_difference=metric["r2"] - old.test_r2))
            if step["n_features"] == 10:
                predictions.extend(dict(model=name, row=int(i), y_true=float(a), y_pred=float(b)) for i, a, b in zip(te, y[te], pred))
    pd.DataFrame(metrics).to_csv(output / "heldout_reproduction.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "ten_feature_predictions.csv", index=False)
    save_json(output / "verification.json", dict(n=len(df), n_test=len(te),
        all_forward_features_match=all(r["archived_features_match"] for r in comparisons),
        maximum_cv_difference=max(abs(r["cv_r2"]-r["archived_cv_r2"]) for r in comparisons),
        maximum_test_r2_difference=max(abs(r["r2_difference"]) for r in metrics),
        selection_before_holdout=True, classification_core_recomputed=False))
    print(pd.DataFrame(metrics).query("n_features == 10").to_string(index=False), flush=True)


def prepare_inner(original, added, protocol, seed):
    inner, assignments = [], []
    for fold, (train, valid) in enumerate(inner_partitions(original, added, protocol, True, seed)):
        assert_isolated(train, valid, protocol)
        transform = representation("hybrid39").fit(train)
        frame = transform.transform(train)
        inner.append(dict(X=frame.to_numpy(), V=transform.transform(valid).to_numpy(),
            y=train.separation_coefficient.to_numpy(), z=valid.separation_coefficient.to_numpy(), columns=list(frame.columns)))
        for role, group in [("train", train), ("validation", valid)]:
            assignments.extend(dict(inner_fold=fold, role=role, sample=r["sample"],
                metal_pair=r.metal_pair, icp_group=r.icp_group, composition_group=r.composition_group) for _, r in group.iterrows())
    assert all(r["columns"] == inner[0]["columns"] for r in inner)
    return inner, assignments


def evaluate_subset(inner, mask, model_name, config, trees, seed):
    rows = []
    for fold, data in enumerate(inner):
        model = make_model(model_name, config, trees, seed + fold)
        model.fit(data["X"][:, mask], data["y"])
        metric = scores(data["z"], model.predict(data["V"][:, mask]))
        rows.append(dict(inner_fold=fold, **metric))
    return rows


def choose_candidate(candidates):
    return sorted(candidates, key=lambda x: (-x["inner_r2_mean"], x["n_features"], x["model"], x["config_id"]))[0]


def select_models(candidates, all_candidates):
    best = choose_candidate(candidates)
    threshold = best["inner_r2_mean"] - best["inner_r2_std"] / np.sqrt(3)
    eligible = [c for c in candidates if c["inner_r2_mean"] >= threshold - 1e-12]
    smallest = min(c["n_features"] for c in eligible)
    simple = choose_candidate([c for c in eligible if c["n_features"] == smallest])
    result = dict(forward_best=best, forward_one_se=simple, all39=choose_candidate(all_candidates))
    ten = [c for c in candidates if c["n_features"] == 10]
    if ten:
        result["forward10"] = choose_candidate(ten)
    return result, threshold


def forward_job(key, data, assignments, source, source_manifest, options, output, fingerprint):
    protocol, repeat, fold = key
    filename = f"{protocol}_r{repeat}_f{fold}.json"
    destination = output / "checkpoints" / filename
    if destination.exists():
        record = json.loads(destination.read_text(encoding="utf-8"))
        if record["fingerprint"] != fingerprint:
            raise ValueError("Stale forward-selection checkpoint")
        return record
    part = assignments[(assignments.protocol == protocol) & (assignments.repeat == repeat) & (assignments.fold == fold)]
    original = data.loc[part.loc[part.role.eq("original"), "sample"]].reset_index(drop=True)
    added = data.loc[part.loc[part.role.eq("augmented"), "sample"]].reset_index(drop=True)
    test = data.loc[part.loc[part.role.eq("testing"), "sample"]].reset_index(drop=True)
    train = pd.concat([original, added], ignore_index=True)
    assert_isolated(train, test, protocol)
    if protocol == "unseen_pair":
        test = test[test.cohort.eq("initial")].reset_index(drop=True)
    seed = (2000 if protocol == "unseen_pair" else 7000) + int(repeat)*100 + int(fold)
    inner, inner_assignments = prepare_inner(original, added, protocol, seed)
    columns = inner[0]["columns"]
    partial_path = output / "paths" / filename
    if partial_path.exists():
        state = json.loads(partial_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Stale forward path")
    else:
        state = dict(fingerprint=fingerprint, selected=[], steps=[], trials=[])
    for size in range(len(state["selected"]) + 1, options["max_features"] + 1):
        candidates = []
        for index, name in enumerate(columns):
            if index in state["selected"]:
                continue
            mask = state["selected"] + [index]
            model_rows = []
            for model_name in ("RF", "ET"):
                rows = evaluate_subset(inner, mask, model_name, PATH_CONFIG, options["selection_trees"], seed)
                model_rows.extend(dict(model=model_name, **r) for r in rows)
            mean = float(np.mean([r["r2"] for r in model_rows]))
            candidates.append(dict(n_features=size, added_feature=name, added_index=index, mask=mask,
                mean_r2=mean, inner_scores=model_rows))
        winner = sorted(candidates, key=lambda r: (-r["mean_r2"], r["added_index"]))[0]
        state["selected"] = winner["mask"]
        state["steps"].append(winner)
        state["trials"].extend(candidates)
        save_json(partial_path, state)
        if size % 4 == 0:
            print(f"{protocol} r{repeat} f{fold}: forward {size}/{options['max_features']}", flush=True)
    tuned, tuning_scores = [], []
    for step in state["steps"]:
        for name in ("RF", "ET"):
            for ci, config in enumerate(source_manifest["options"]["configs"]):
                rows = evaluate_subset(inner, step["mask"], name, config, source_manifest["options"]["search_trees"], seed)
                tuned.append(dict(model=name, n_features=step["n_features"], config_id=ci, config=config,
                    features=[columns[i] for i in step["mask"]],
                    inner_r2_mean=float(np.mean([r["r2"] for r in rows])),
                    inner_r2_std=float(np.std([r["r2"] for r in rows], ddof=1))))
                tuning_scores.extend(dict(model=name, n_features=step["n_features"], config_id=ci, **r) for r in rows)
    all_candidates, payloads = [], {}
    for name in ("RF", "ET"):
        payload = json.loads((source / "checkpoints" / f"{protocol}_r{repeat}_f{fold}_hybrid39_{name}.json").read_text(encoding="utf-8"))
        if payload["fingerprint"] != source_manifest["fingerprint"]:
            raise ValueError("Source checkpoint mismatch")
        payloads[name] = payload
        for row in payload["candidates"]:
            if row["training"] == "with_cofe" and row["n_features"] == 39:
                all_candidates.append(dict(model=name, features=columns, **row))
    choices, threshold = select_models(tuned, all_candidates)
    transform = representation("hybrid39").fit(train)
    X, V = transform.transform(train), transform.transform(test)
    result = dict(protocol=protocol, repeat=int(repeat), fold=int(fold), fingerprint=fingerprint,
        predictions=[], settings=[], permutation=[], inner_assignments=inner_assignments,
        tuning_scores=tuning_scores, path=[{k:v for k,v in s.items() if k != "inner_scores"} for s in state["steps"]])
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
            old = pd.DataFrame(payloads[choice["model"]]["predictions"])
            old = old[old.training.eq("with_cofe") & old.strategy.eq("tuned_all")].set_index("sample")
            np.testing.assert_allclose(pred, old.loc[test["sample"], "y_pred"], atol=1e-10, rtol=0)
        result["settings"].append(dict(strategy=strategy, **{k: choice[k] for k in
            ("model", "n_features", "features", "config", "inner_r2_mean", "inner_r2_std")}, one_se_threshold=threshold))
        result["predictions"].extend(dict(strategy=strategy, sample=r["sample"], metal_pair=r.metal_pair,
            y_true=float(r.separation_coefficient), y_pred=float(p)) for (_, r), p in zip(test.iterrows(), pred))
        if strategy in ("forward_best", "forward_one_se"):
            result["permutation"].extend(dict(strategy=strategy, **r) for r in
                property_permutations(model, V[selected], test, pred, seed))
    save_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/cofe_inclusion_20261012")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/forward_selection_revision_20261013")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=16)
    parser.add_argument("--selection-trees", type=int, default=120)
    parser.add_argument("--historical-only", action="store_true")
    parser.add_argument("--skip-historical", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_features <= 39:
        raise ValueError("max-features must be between 1 and 39")
    output, source = args.output.resolve(), args.source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_historical:
        replay_historical(output / "historical_replay", args.jobs)
    if args.historical_only:
        return
    manifest = json.loads((source / "analysis_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["code_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed source modeling code: {name}")
    dependencies = [Path(__file__), ROOT / "icp_identity_diagnostics.py", ROOT / "icp_cofe_inclusion.py"]
    paths = [source / name for name in ("analysis_manifest.json", "combined_dataset.csv", "outer_assignments.csv")]
    paths += sorted((source / "checkpoints").glob("*hybrid39*.json"))
    options = dict(max_features=args.max_features, selection_trees=args.selection_trees)
    record = dict(source=str(source), source_fingerprint=manifest["fingerprint"], source_hashes={str(p):sha(p) for p in paths},
        code_hashes={str(p):sha(p) for p in dependencies}, options=options,
        design="Fold-local forward selection from zero features; same 3 inner folds and fixed outer assignments; no historical core imported.")
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    record["fingerprint"] = fingerprint
    manifest_path = output / "analysis_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("Sources/options changed; use a fresh output directory")
    save_json(manifest_path, record)
    for name in ("checkpoints", "paths"):
        (output / name).mkdir(exist_ok=True)
    data = pd.read_csv(source / "combined_dataset.csv").set_index("sample", drop=False)
    assignments = pd.read_csv(source / "outer_assignments.csv")
    keys = assignments.loc[assignments.protocol.isin(PROTOCOLS), ["protocol", "repeat", "fold"]].drop_duplicates()
    start, results = time.perf_counter(), []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for result in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(forward_job)(tuple(key), data, assignments, source, manifest, options, output, fingerprint) for key in keys.to_numpy()):
            results.append(result)
            print(f"Completed {len(results)}/{len(keys)} outer folds; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    for category in ("predictions", "settings", "permutation", "inner_assignments", "tuning_scores", "path"):
        rows = [dict(**{k:r[k] for k in ("protocol", "repeat", "fold")}, **row) for r in results for row in r[category]]
        pd.DataFrame(rows).to_csv(output / f"{category}.csv", index=False)
    save_json(output / "verification.json", dict(completed_folds=len(results), all39_predictions_replayed=True,
        outer_inner_group_isolation_checked=True, fingerprint=fingerprint, elapsed_seconds=time.perf_counter()-start))
    print("Forward-selection fitting complete", flush=True)


if __name__ == "__main__":
    main()
