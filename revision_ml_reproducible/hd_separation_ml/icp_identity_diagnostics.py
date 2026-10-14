"""Identity controls and explicitly sparse forests on frozen grouped outer tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from icp_bo_augmentation_cv import representation
from icp_intrinsic_descriptor_analysis import BASES, LABELS, scores
from icp_nested_group_cv import make_model, rfe_order
from icp_revision_inputs import sha
from icp_two_prediction_tasks import inner_partitions

ROOT = Path(__file__).resolve().parent
PROTOCOLS = ["unseen_pair", "known_pair_new_composition"]
ALPHAS = [.001, .01, .1, 1., 10., 100.]
CONTROLS = ["element_identity", "element_fraction", "pair_identity", "pair_quadratic"]
STRATEGIES = ["tuned_all", "nested_max", "one_se"]
N_PERMUTATIONS = 10


def canonical_fraction(data):
    first = data.metal_pair.str.split("-").str[0]
    return data.actual_frac_element1.where(data.element1.eq(first), 1 - data.actual_frac_element1)


def encode(data, kind):
    records = []
    for (_, row), fraction in zip(data.iterrows(), canonical_fraction(data)):
        if kind == "ordered_identity_forest":
            record = {"M1:" + row.element1: 1., "M2:" + row.element2: 1.}
        elif kind.startswith("element"):
            f = row.actual_frac_element1 if kind == "element_fraction" else .5
            record = {row.element1: float(f)}
            record[row.element2] = record.get(row.element2, 0.) + 1 - float(f)
        else:
            record = {row.metal_pair: 1.}
            if kind == "pair_quadratic":
                record[row.metal_pair + "_f"] = float(fraction - .5)
                record[row.metal_pair + "_f2"] = float((fraction - .5) ** 2)
        records.append(record)
    return records


def ridge_model(alpha):
    return make_pipeline(DictVectorizer(sparse=False), StandardScaler(), Ridge(alpha=alpha))


def assert_isolated(train, test, protocol):
    for field in ["sample", "icp_group", "metal_pair" if protocol == "unseen_pair" else "composition_group"]:
        if set(train[field]) & set(test[field]):
            raise ValueError(f"Overlapping {field}: {protocol}")


def tune_control(original, added, protocol, kind, seed):
    rows = []
    for fold, (train, valid) in enumerate(inner_partitions(original, added, protocol, True, seed)):
        assert_isolated(train, valid, protocol)
        X, V = encode(train, kind), encode(valid, kind)
        for alpha in ALPHAS:
            model = ridge_model(alpha).fit(X, train.separation_coefficient)
            metric = scores(valid.separation_coefficient, model.predict(V))
            rows.append(dict(control=kind, inner_fold=fold, alpha=alpha, **metric))
    table = pd.DataFrame(rows)
    ranking = table.groupby("alpha").r2.mean().sort_values(ascending=False, kind="stable")
    alpha = float(ranking.index[0])
    return alpha, rows


def identity_forest(original, added, protocol, seed, options):
    rows = []
    for fold, (train, valid) in enumerate(inner_partitions(original, added, protocol, True, seed)):
        assert_isolated(train, valid, protocol)
        encoder = DictVectorizer(sparse=False)
        X = encoder.fit_transform(encode(train, "ordered_identity_forest"))
        V = encoder.transform(encode(valid, "ordered_identity_forest"))
        for name in ("RF", "ET"):
            for ci, config in enumerate(options["configs"]):
                model = make_model(name, config, options["search_trees"], seed + fold)
                model.fit(X, train.separation_coefficient)
                rows.append(dict(control="ordered_identity_forest", model=name, config_id=ci,
                    inner_fold=fold, **scores(valid.separation_coefficient, model.predict(V))))
    means = pd.DataFrame(rows).groupby(["model", "config_id"], as_index=False).r2.mean()
    best = means.sort_values(["r2", "model", "config_id"], ascending=[False, True, True]).iloc[0]
    return str(best.model), options["configs"][int(best.config_id)], rows


def sparse_choice(payloads, k):
    candidates = [dict(model=name, **row) for name, payload in payloads.items()
                  for row in payload["candidates"] if row["training"] == "with_cofe" and row["n_features"] == k]
    return sorted(candidates, key=lambda x: (-x["inner_r2_mean"], x["model"], x["config_id"]))[0]


def permutation_orders(data, seed):
    rng = np.random.default_rng(seed)
    for _ in range(N_PERMUTATIONS):
        if data.cohort.eq("BO").all():
            order = np.arange(len(data))
            for indices in data.groupby("metal_pair").indices.values():
                order[indices] = rng.permutation(indices)
        else:
            order = rng.permutation(len(data))
        yield order


def property_permutations(model, features, test, baseline, seed):
    names = list(features.columns)
    X = features.to_numpy()
    groups = {base: [i for i, col in enumerate(names) if col in
              {base, "icp_weighted_" + base, base + "_element1", base + "_element2"}]
              for base in BASES}
    groups["all_weighted_composition"] = [i for i, c in enumerate(names) if c.startswith("icp_weighted_")]
    results = []
    y = test.separation_coefficient.to_numpy()
    # Joint property shuffles preserve raw/weighted association within a property.
    # Within BO pairs, the all-weighted shuffle is a coherent composition shuffle.
    for base, columns in groups.items():
        arrays = []
        for order in permutation_orders(test, seed):
            array = X.copy()
            if columns:
                array[:, columns] = X[order][:, columns]
            arrays.append(array)
        predictions = model.predict(np.concatenate(arrays)).reshape(N_PERMUTATIONS, len(test)) if columns else np.tile(baseline, (N_PERMUTATIONS, 1))
        for i, pred in enumerate(predictions):
            results.append(dict(property=base, permutation=i, n=len(test), n_selected=len(columns),
                sse_increase=float(np.square(y - pred).sum() - np.square(y - baseline).sum())))
    return results


def job(key, data, assignments, source, manifest, output, fingerprint):
    protocol, repeat, fold = key
    path = output / "checkpoints" / f"{protocol}_r{repeat}_f{fold}.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["fingerprint"] != fingerprint:
            raise ValueError("Stale identity diagnostics checkpoint")
        return result
    part = assignments[(assignments.protocol == protocol) & (assignments.repeat == repeat) & (assignments.fold == fold)]
    original = data.loc[part.loc[part.role.eq("original"), "sample"]].reset_index(drop=True)
    added = data.loc[part.loc[part.role.eq("augmented"), "sample"]].reset_index(drop=True)
    all_test = data.loc[part.loc[part.role.eq("testing"), "sample"]].reset_index(drop=True)
    train = pd.concat([original, added], ignore_index=True)
    assert_isolated(train, all_test, protocol)
    test = all_test[all_test.cohort.eq("initial")].copy() if protocol == "unseen_pair" else all_test.copy()
    test = test.reset_index(drop=True)
    seed = (2000 if protocol == "unseen_pair" else 7000) + int(repeat) * 100 + int(fold)
    result = dict(protocol=protocol, repeat=int(repeat), fold=int(fold), fingerprint=fingerprint,
                  predictions=[], settings=[], controls_inner=[], permutation=[], training_fit=[])

    def add_predictions(label, pred):
        result["predictions"].extend(dict(analysis=label, sample=r["sample"], metal_pair=r.metal_pair,
            y_true=float(r.separation_coefficient), y_pred=float(p)) for (_, r), p in zip(test.iterrows(), pred))

    mean = train.separation_coefficient.mean()
    pair_mean = train.groupby("metal_pair").separation_coefficient.mean()
    add_predictions("global_mean", np.full(len(test), mean))
    add_predictions("pair_mean", test.metal_pair.map(pair_mean).fillna(mean))
    controls = CONTROLS if protocol == "known_pair_new_composition" else CONTROLS[:2]
    for kind in controls:
        alpha, inner = tune_control(original, added, protocol, kind, seed)
        result["controls_inner"].extend(inner)
        model = ridge_model(alpha).fit(encode(train, kind), train.separation_coefficient)
        add_predictions(kind, model.predict(encode(test, kind)))
        result["settings"].append(dict(analysis=kind, alpha=alpha, n_features=None, selected_features=[]))

    name, config, inner = identity_forest(original, added, protocol, seed, manifest["options"])
    result["controls_inner"].extend(inner)
    encoder = DictVectorizer(sparse=False)
    identity_X = encoder.fit_transform(encode(train, "ordered_identity_forest"))
    identity_V = encoder.transform(encode(test, "ordered_identity_forest"))
    model = make_model(name, config, manifest["options"]["final_trees"], seed).fit(identity_X, train.separation_coefficient)
    add_predictions("ordered_identity_forest", model.predict(identity_V))
    result["settings"].append(dict(analysis="ordered_identity_forest", model=name, config=config,
        n_features=None, selected_features=[], n_category_indicators=int(identity_X.shape[1])))

    payloads = {name: json.loads((source / "checkpoints" / f"{protocol}_r{repeat}_f{fold}_hybrid39_{name}.json").read_text(encoding="utf-8"))
                for name in ("RF", "ET")}
    for payload in payloads.values():
        if payload["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Source fingerprint mismatch")
    transform = representation("hybrid39").fit(train)
    X, V = transform.transform(train), transform.transform(test)
    orders, fitted = {}, {}
    choices = {}
    for strategy in STRATEGIES:
        settings = [dict(model=name, **s) for name, payload in payloads.items()
                    for s in payload["settings"] if s["training"] == "with_cofe" and s["strategy"] == strategy]
        choices[strategy] = sorted(settings, key=lambda x: (-x["inner_r2_mean"], x["n_features"], x["model"]))[0]
    for k in (4, 8, 12):
        choice = sparse_choice(payloads, k)
        name = choice["model"]
        if name not in orders:
            orders[name] = rfe_order(X.to_numpy(), train.separation_coefficient.to_numpy(), name,
                                     manifest["options"]["selector_trees"], seed)[0]
        choices[f"sparse{k}"] = dict(**choice, selected_features=list(X.columns[np.sort(orders[name][:k])]))
    for label, choice in choices.items():
        selected = choice["selected_features"]
        name = choice["model"]
        cache_key = (name, tuple(selected), json.dumps(choice["config"], sort_keys=True))
        if cache_key not in fitted:
            model = make_model(name, choice["config"], manifest["options"]["final_trees"], seed)
            model.fit(X[selected].to_numpy(), train.separation_coefficient)
            fitted[cache_key] = model
        model = fitted[cache_key]
        pred = model.predict(V[selected].to_numpy())
        if label in STRATEGIES:
            saved = pd.DataFrame(payloads[name]["predictions"])
            saved = saved[saved.training.eq("with_cofe") & saved.strategy.eq(label)].set_index("sample")
            np.testing.assert_allclose(saved.loc[test["sample"], "y_pred"], pred, rtol=0, atol=1e-10)
        add_predictions(label, pred)
        result["settings"].append(dict(analysis=label, model=name, n_features=len(selected),
            selected_features=selected, config=choice["config"], inner_r2_mean=choice["inner_r2_mean"]))
        train_pred = model.predict(X[selected].to_numpy())
        mask = train.cohort.eq("BO") if protocol == "known_pair_new_composition" else train.cohort.eq("initial")
        result["training_fit"].append(dict(analysis=label, n=int(mask.sum()),
            **scores(train.loc[mask, "separation_coefficient"], train_pred[mask])))
        if label in ("sparse8", "one_se"):
            result["permutation"].extend(dict(analysis=label, **r) for r in
                property_permutations(model, V[selected], test, pred, seed))
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    return result


def descriptive(data):
    rows = []
    cohorts = {"initial": data[data.cohort.eq("initial")], "all_BO": data[data.cohort.eq("BO")], "all": data}
    for population, part in cohorts.items():
        y = part.separation_coefficient
        pair = part.groupby("metal_pair").separation_coefficient
        within = float(np.square(y - pair.transform("mean")).sum())
        total = float(np.square(y - y.mean()).sum())
        rows.append(dict(population=population, n=len(part), n_pairs=part.metal_pair.nunique(),
            singleton_pairs=int(pair.size().eq(1).sum()), observed_between_pair_fraction=1 - within / total))
    return pd.DataFrame(rows)


def aggregate(results, data, output):
    tables = {}
    for category in ("predictions", "settings", "controls_inner", "permutation", "training_fit"):
        tables[category] = pd.DataFrame([dict(**{k: result[k] for k in ("protocol", "repeat", "fold")}, **row)
                                        for result in results for row in result[category]])
        tables[category].to_csv(output / (category + ".csv"), index=False)
    p = tables["predictions"]
    baselines = p[p.analysis.eq("pair_mean")].set_index(["protocol", "repeat", "sample"]).y_pred
    metrics = []
    for (protocol, repeat, analysis), part in p.groupby(["protocol", "repeat", "analysis"]):
        if protocol == "unseen_pair":
            subsets = {"initial": part}
        else:
            subsets = {"all_BO": part, "existing_BO_84": part[~part.metal_pair.eq("Co-Fe")]}
            subsets.update({pair: group for pair, group in part.groupby("metal_pair")})
        for population, rows in subsets.items():
            assert rows["sample"].is_unique
            expected = data.loc[rows["sample"], "separation_coefficient"]
            np.testing.assert_allclose(expected, rows.y_true, atol=1e-12, rtol=0)
            base = baselines.loc[[(protocol, repeat, sample) for sample in rows["sample"]]].to_numpy()
            denominator = np.square(rows.y_true - base).sum()
            metrics.append(dict(protocol=protocol, repeat=int(repeat), analysis=analysis, population=population,
                n=len(rows), skill_over_pair_mean=float(1 - np.square(rows.y_true - rows.y_pred).sum() / denominator),
                **scores(rows.y_true, rows.y_pred)))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "repeat_metrics.csv", index=False)
    summary = metrics.groupby(["protocol", "population", "analysis"], as_index=False).agg(
        n=("n", "first"), r2=("r2", "mean"), r2_sd=("r2", "std"), rmse=("rmse", "mean"),
        skill_over_pair_mean=("skill_over_pair_mean", "mean"))
    summary.to_csv(output / "summary.csv", index=False)
    selection = []
    for _, row in tables["settings"].iterrows():
        if pd.isna(row.n_features):
            continue
        for base in BASES:
            names = [c for c in row.selected_features if c in {base + "_element1", base + "_element2", "icp_weighted_" + base}]
            selection.append(dict(protocol=row.protocol, repeat=row.repeat, fold=row.fold, analysis=row.analysis,
                property=base, any_selected=bool(names), weighted_selected="icp_weighted_" + base in names))
    stability = pd.DataFrame(selection).groupby(["protocol", "analysis", "property"], as_index=False).agg(
        selection_frequency=("any_selected", "mean"), weighted_selection_frequency=("weighted_selected", "mean"))
    stability.to_csv(output / "property_selection_stability.csv", index=False)
    permutation = tables["permutation"]
    permutation = permutation.groupby(["protocol", "repeat", "analysis", "property", "permutation"], as_index=False).agg(
        sse_increase=("sse_increase", "sum"), n=("n", "sum"))
    permutation["mse_increase"] = permutation.sse_increase / permutation.n
    denominators = {"unseen_pair": data[data.cohort.eq("initial")].separation_coefficient,
                    "known_pair_new_composition": data[data.cohort.eq("BO")].separation_coefficient}
    denominators = {k: float(np.square(y - y.mean()).sum()) for k, y in denominators.items()}
    permutation["r2_drop"] = permutation.sse_increase / permutation.protocol.map(denominators)
    permutation.to_csv(output / "permutation_repeat_metrics.csv", index=False)
    importance = permutation.groupby(["protocol", "analysis", "property"], as_index=False).agg(
        r2_drop=("r2_drop", "mean"), mse_increase=("mse_increase", "mean"))
    importance = importance.merge(stability, on=["protocol", "analysis", "property"], how="left")
    importance.to_csv(output / "heldout_importance.csv", index=False)
    desc = descriptive(data)
    desc.to_csv(output / "descriptive_pair_variation.csv", index=False)
    return summary, stability, importance, desc, tables["settings"]


def report(summary, importance, desc, settings, output):
    labels = {"global_mean": "Global mean", "pair_mean": "Pair mean", "element_identity": "Element identity",
              "ordered_identity_forest": "Metal identities (forest)",
              "element_fraction": "Element + ICP fraction", "pair_identity": "Pair identity (ridge)",
              "pair_quadratic": "Pair + ICP quadratic", "sparse4": "4 descriptors", "sparse8": "8 descriptors",
              "sparse12": "12 descriptors", "one_se": "One-SE selection", "nested_max": "Best inner selection",
              "tuned_all": "All 39 descriptors"}
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), layout="constrained")
    for ax, protocol, population, title in zip(axes, PROTOCOLS, ["initial", "all_BO"],
            ["Unseen metal pairs: initial screening (n = 554)", "Known pairs, held-out compositions: BO (n = 111)"]):
        values = summary[(summary.protocol == protocol) & (summary.population == population)].set_index("analysis")
        order = [k for k in labels if k in values.index and k != "global_mean" and not (k == "pair_mean" and protocol == "unseen_pair")]
        positions = np.arange(len(order))
        colors = ["#5f6c75" if not k.startswith("sparse") and k not in STRATEGIES else "#187f82" for k in order]
        ax.barh(positions, values.loc[order, "r2"], color=colors, height=.68)
        ax.errorbar(values.loc[order, "r2"], positions, xerr=values.loc[order, "r2_sd"], fmt="none", ecolor="#222222", capsize=2)
        ax.set_yticks(positions, [labels[k] for k in order])
        ax.invert_yaxis()
        ax.set(xlabel="Out-of-fold R-squared (mean +/- repeat SD)", title=title)
        ax.axvline(0, color="#333333", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(output / "identity_vs_descriptors.png", dpi=180)
    fig.savefig(output / "identity_vs_descriptors.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), layout="constrained")
    for ax, protocol, title in zip(axes, PROTOCOLS, ["New-pair prediction", "Within-pair composition prediction"]):
        imp = importance[(importance.protocol == protocol) & importance.analysis.eq("sparse8") &
                         importance.property.ne("all_weighted_composition")].nlargest(8, "r2_drop").sort_values("r2_drop")
        ax.barh([LABELS[b] for b in imp.property], imp.r2_drop, color="#187f82")
        ax.set(xlabel="Held-out R-squared decrease after permutation", title=title)
        ax.axvline(0, color="#333333", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "sparse8_property_importance.png", dpi=180)
    fig.savefig(output / "sparse8_property_importance.pdf")
    plt.close(fig)
    lines = ["# Metal Identity and Sparse-Descriptor Diagnostics", "",
        "This is a retrospective sensitivity analysis on the existing grouped folds, not a new external validation set.",
        "The 665-record population and Co-Fe inclusion policy are unchanged. Only protocols 2 and 3 are used.", "",
        "## Matched Out-of-Fold Results", "",
        "| Task | Model | n | R2 | Repeat SD | RMSE |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for protocol, population in zip(PROTOCOLS, ["initial", "all_BO"]):
        part = summary[(summary.protocol == protocol) & (summary.population == population)].set_index("analysis")
        for name in labels:
            if name not in part.index:
                continue
            r = part.loc[name]
            lines.append(f"| {protocol} | {labels[name]} | {int(r.n)} | {r.r2:.4f} | {r.r2_sd:.4f} | {r.rmse:.4f} |")
    lines += ["", "## What Was Controlled", "",
        "- All outer sample memberships are unchanged; shared ICP, unordered pairs (task 2), and composition bins (task 3) remain isolated.",
        "- Metal identity controls use unordered element indicators or pair indicators, without measured composition.",
        "- Composition controls use element molar fractions, or pair-specific quadratic functions of the canonical ICP fraction.",
        "- Ridge alpha is selected from six values inside the same three inner folds; encoders and scalers are fitted only on inner training data.",
        "- Forest models use the original training-only RFE and tuning candidates. For each of 4/8/12 features, RF/ET and hyperparameters are chosen by inner scores only.",
        "- Original tuned-all, nested-max and one-SE predictions are refitted and reproduced to absolute tolerance 1e-10.",
        "- Training-fit scores are diagnostic only, not validation. Smaller feature sets do not by themselves prove reduced overfitting.",
        "- Sparse8 is a prespecified diagnostic feature budget, not selected because it wins on the outer tests.", "",
        "## Metal-Pair Association", "",
        "Observed between-pair variance fractions are descriptive, not cross-validated explanatory power or causal effects.",
        "The initial/all-data fractions are particularly optimistic when many pairs have only one observation.", ""]
    for _, r in desc.iterrows():
        lines.append(f"- {r.population}: {r.observed_between_pair_fraction:.1%} between pairs; {int(r.n_pairs)} pairs, {int(r.singleton_pairs)} singletons.")
    lines += ["", "## Feature Counts and Importance", ""]
    for (protocol, strategy), group in settings[settings.analysis.isin(STRATEGIES)].groupby(["protocol", "analysis"]):
        lines.append(f"- {protocol}, {strategy}: {int(group.n_features.min())}-{int(group.n_features.max())} features; median {group.n_features.median():g}.")
    lines += ["", "The held-out importance plots use the 8-feature diagnostic models.",
        "Task 2 shuffles each property's selected raw/weighted columns jointly across held-out samples.",
        "Task 3 shuffles only within the same held-out metal pair; the all-weighted block moves together for a coherent composition test.",
        "Unselected properties receive zero importance; selection frequency is reported over all 15 outer folds.",
        "Partial property permutation can create inconsistent correlated descriptors. Interpret it as predictive reliance, not a unique mechanism.",
        "Within each fixed pair, all weighted properties are affine functions of one ICP fraction, so BO alone cannot identify a unique causal weighted property.",
        "Feature rankings are not reused to select a new model and claim accuracy on these same outer tests.", "",
        "## Reproduction", "", "```powershell",
        "python -X utf8 hd_separation_ml/icp_identity_diagnostics.py --jobs 10", "```", "",
        "For changed outcomes, first rerun the main Co-Fe comparison with the new target files; then pass --source and a fresh --output here.",
        "Input and code hashes prevent stale diagnostic checkpoints from being reused."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/cofe_inclusion_20261012")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/identity_diagnostics_20261013")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    manifest = json.loads((source / "analysis_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["code_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed modeling code: {name}")
    if sha(ROOT / "icp_cofe_inclusion.py") != manifest["source_hashes"]["code"]:
        raise ValueError("Changed Co-Fe comparison code")
    paths = [source / x for x in ["analysis_manifest.json", "combined_dataset.csv", "outer_assignments.csv"]]
    paths += sorted((source / "checkpoints").glob("*hybrid39*.json"))
    record = dict(source=str(source), source_fingerprint=manifest["fingerprint"],
        source_hashes={str(p.relative_to(source)): sha(p) for p in paths},
        code_hash=sha(Path(__file__)), source_code_hashes=manifest["code_hashes"], protocols=PROTOCOLS,
        alpha_grid=ALPHAS, sparse_sizes=[4, 8, 12], n_permutations=N_PERMUTATIONS)
    fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    record["fingerprint"] = fingerprint
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "analysis_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("Diagnostic inputs/code changed; choose a fresh output directory")
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    (output / "checkpoints").mkdir(exist_ok=True)
    data = pd.read_csv(source / "combined_dataset.csv").set_index("sample", drop=False)
    assignments = pd.read_csv(source / "outer_assignments.csv")
    assert data.index.is_unique and data.icp_group.is_unique
    keys = assignments.loc[assignments.protocol.isin(PROTOCOLS), ["protocol", "repeat", "fold"]].drop_duplicates()
    start, results = time.perf_counter(), []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for result in Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(job)(tuple(key), data, assignments, source, manifest, output, fingerprint) for key in keys.to_numpy()):
            results.append(result)
            print(f"Completed {len(results)}/{len(keys)} grouped diagnostics; {time.perf_counter()-start:.1f}s", flush=True)
    summary, stability, importance, desc, settings = aggregate(results, data, output)
    report(summary, importance, desc, settings, output)
    (output / "verification.json").write_text(json.dumps(dict(complete_jobs=len(results),
        source_predictions_replayed=True, targets_match=True, outer_inner_groups_isolated=True,
        seconds=time.perf_counter()-start, fingerprint=fingerprint), indent=2), encoding="utf-8")
    print(summary[summary.population.isin(["initial", "all_BO"])].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
