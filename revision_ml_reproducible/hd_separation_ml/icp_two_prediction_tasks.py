"""Separate unseen-chemistry prediction from known-pair composition prediction."""

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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from icp_bo_augmentation_cv import (
    ICP, OBSERVATIONS, CURATED, build_data, pair_bo, representation, sha,
)
from icp_intrinsic_descriptor_analysis import ROOT, WORKBOOK, COMPOSITION_TABLE, FEATURE_MATRIX, scores
from icp_nested_group_cv import CONFIGS, choose_best, group_folds, make_model, rfe_order
from icp_hybrid_descriptor_cv import FIXED, candidate_counts
from icp_parsimony_sensitivity import select_candidate
from icp_master import SCHEMA_VERSION

OUTPUT = ROOT / "outputs" / "icp_two_prediction_tasks_master_20261012"
PROTOCOLS = ("unseen_pair", "known_pair_new_composition")
SCENARIOS = ("raw26", "icp13", "hybrid39")
KEYS = ["protocol", "repeat", "fold", "scenario", "model"]


def known_strata(data):
    bins = data.groupby("metal_pair").separation_coefficient.transform(
        lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
    return data.metal_pair + "|q" + bins.astype(str)


def known_folds(data, splits, seed):
    held = [[] for _ in range(splits)]
    rng = np.random.default_rng(seed)
    strata = known_strata(data)
    # Split each chemistry separately so inner validation cannot become pair extrapolation.
    for j, pair in enumerate(sorted(data.metal_pair.unique())):
        idx = np.flatnonzero(data.metal_pair.eq(pair))
        part = data.iloc[idx]
        n = min(splits, part.composition_group.nunique())
        if n < 2:
            raise ValueError(f"Fewer than two independent composition groups: {pair}")
        cv = StratifiedGroupKFold(n, shuffle=True, random_state=seed + j)
        slots = rng.permutation(splits)[:n]
        for slot, (_, test) in zip(slots, cv.split(part, strata.iloc[idx], part.composition_group)):
            held[slot].extend(idx[test])
    folds = [(np.setdiff1d(np.arange(len(data)), te), np.sort(te)) for te in held]
    for tr, te in folds:
        assert not set(data.iloc[tr].composition_group) & set(data.iloc[te].composition_group)
        assert set(data.iloc[te].metal_pair) <= set(data.iloc[tr].metal_pair)
    np.testing.assert_array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(len(data)))
    return folds


def partitions(initial, bo, protocol, tr, te):
    if protocol == "stratified_random":
        original, testing = initial.iloc[tr], initial.iloc[te]
        added = bo[~bo.composition_group.isin(testing.composition_group)]
    elif protocol == "unseen_pair":
        original, target = initial.iloc[tr], initial.iloc[te]
        added, extra_test = pair_bo(original, bo), pair_bo(target, bo)
        testing = pd.concat([target, extra_test], ignore_index=True)
        assert not set(original.metal_pair) & set(testing.metal_pair)
    else:
        added, testing = bo.iloc[tr], bo.iloc[te]
        original = initial[~initial.composition_group.isin(testing.composition_group)]
        assert not set(original.composition_group) & set(testing.composition_group)
        assert not set(added.composition_group) & set(testing.composition_group)
    return tuple(x.reset_index(drop=True) for x in (original, added, testing))


def inner_partitions(original, added, protocol, augmented, seed):
    if protocol == "stratified_random":
        strata = pd.qcut(original.separation_coefficient, 5, labels=False, duplicates="drop")
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(original, strata):
            base, valid = original.iloc[tr], original.iloc[va]
            extra = added[~added.composition_group.isin(valid.composition_group)] if augmented else added.iloc[:0]
            yield pd.concat([base, extra], ignore_index=True), valid.reset_index(drop=True)
    elif protocol == "unseen_pair":
        for tr, va in group_folds(original, 3, seed):
            base, valid = original.iloc[tr], original.iloc[va]
            extra = pair_bo(base, added) if augmented else added.iloc[:0]
            training = pd.concat([base, extra], ignore_index=True)
            assert not set(training.metal_pair) & set(valid.metal_pair)
            yield training, valid.reset_index(drop=True)
    else:
        assert augmented, "The initial-only known-pair arm is fixed, without BO-label tuning."
        for tr, va in known_folds(added, 3, seed):
            valid = added.iloc[va]
            base = original[~original.composition_group.isin(valid.composition_group)]
            training = pd.concat([base, added.iloc[tr]], ignore_index=True)
            assert not set(training.composition_group) & set(valid.composition_group)
            yield training, valid.reset_index(drop=True)


def search(original, added, protocol, augmented, scenario, model_name, seed, options=None):
    options = options or {}
    configs = options.get("configs", CONFIGS)
    rows, assignments, ranks = [], [], []
    for inner_fold, (training, valid) in enumerate(inner_partitions(original, added, protocol, augmented, seed)):
        transform = representation(scenario).fit(training)
        frame = transform.transform(training)
        X, V = frame.to_numpy(), transform.transform(valid).to_numpy()
        y = training.separation_coefficient.to_numpy()
        order, ranking = rfe_order(X, y, model_name, options.get("selector_trees", 128), seed + inner_fold)
        ranks.extend(dict(inner_fold=inner_fold, feature=c, rank=int(r)) for c, r in zip(frame.columns, ranking))
        for role, data in [("train", training), ("validation", valid)]:
            assignments.extend(dict(inner_fold=inner_fold, role=role, sample=r["sample"],
                metal_pair=r.metal_pair, composition_group=r.composition_group, cohort=r.cohort)
                for _, r in data.iterrows())
        counts = options.get("feature_counts", {}).get(scenario, candidate_counts(X.shape[1]))
        for k in counts:
            mask = np.sort(order[:k])
            for ci, config in enumerate(configs):
                model = make_model(model_name, config, options.get("search_trees", 160), seed + inner_fold)
                model.fit(X[:, mask], y)
                rows.append(dict(inner_fold=inner_fold, n_features=k, config_id=ci,
                    **scores(valid.separation_coefficient, model.predict(V[:, mask]))))
    table = pd.DataFrame(rows)
    candidates = [dict(n_features=int(k), config_id=int(ci), config=configs[int(ci)],
        inner_r2_mean=float(g.r2.mean()), inner_r2_std=float(g.r2.std()),
        inner_rmse_mean=float(g.rmse.mean())) for (k, ci), g in table.groupby(["n_features", "config_id"])]
    return candidates, rows, assignments, ranks


def choices(candidates, p):
    result = {"fixed_all": dict(n_features=p, config=FIXED, inner_r2_mean=None)}
    if candidates:
        result.update(tuned_all=choose_best([c for c in candidates if c["n_features"] == p]),
                      nested_max=choose_best(candidates),
                      one_se=select_candidate(candidates, "one_se", 3)[0])
    return result


def fit_choices(training, testing, scenario, model_name, seed, candidates, options=None):
    options = options or {}
    transform = representation(scenario).fit(training)
    frame = transform.transform(training)
    columns = list(frame.columns)
    X, V = frame.to_numpy(), transform.transform(testing).to_numpy()
    y = training.separation_coefficient.to_numpy()
    order = rfe_order(X, y, model_name, options.get("selector_trees", 128), seed)[0] if candidates else np.arange(X.shape[1])
    predictions, settings, feature_rows, cache = [], [], [], {}
    for strategy, choice in choices(candidates, len(columns)).items():
        mask = np.sort(order[:choice["n_features"]])
        cache_key = (tuple(mask), json.dumps(choice["config"], sort_keys=True))
        if cache_key not in cache:
            model = make_model(model_name, choice["config"], options.get("final_trees", 600), seed).fit(X[:, mask], y)
            cache[cache_key] = model.predict(V[:, mask])
        settings.append(dict(strategy=strategy, n_features=len(mask),
            selected_features=[columns[j] for j in mask], config=choice["config"],
            inner_r2_mean=choice["inner_r2_mean"], n_train_initial=int(training.cohort.eq("initial").sum()),
            n_train_bo=int(training.cohort.eq("BO").sum())))
        feature_rows.extend(dict(strategy=strategy, feature=c, selected=j in mask)
                            for j, c in enumerate(columns))
        predictions.extend(dict(strategy=strategy, sample=row["sample"], cohort=row.cohort,
            metal_pair=row.metal_pair, shared_icp=bool(row.shared_icp),
            y_true=float(row.separation_coefficient), y_pred=float(pred))
            for (_, row), pred in zip(testing.iterrows(), cache[cache_key]))
    return predictions, settings, feature_rows


def run_job(initial, bo, protocol, tr, te, repeat, fold, scenario, model_name, fingerprint, output, options=None):
    options = options or {}
    key = dict(protocol=protocol, repeat=repeat, fold=fold, scenario=scenario, model=model_name)
    path = output / "checkpoints" / f"{protocol}_r{repeat}_f{fold}_{scenario}_{model_name}.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["fingerprint"] == fingerprint
        return result
    start = time.perf_counter()
    original, added, testing = partitions(initial, bo, protocol, tr, te)
    seed = (2000 if protocol == "unseen_pair" else 7000) + repeat * 100 + fold
    result = dict(**key, fingerprint=fingerprint, predictions=[], settings=[], candidates=[],
                  inner_scores=[], inner_assignments=[], inner_ranks=[], features=[])
    for augmented in (False, True):
        arm = "initial_plus_bo" if augmented else "initial_only"
        training = pd.concat([original, added], ignore_index=True) if augmented else original
        candidates, inner_scores, assignments, ranks = [], [], [], []
        if (scenario != "raw26" or options.get("nested_raw", False)) and (protocol != "known_pair_new_composition" or augmented):
            candidates, inner_scores, assignments, ranks = search(
                original, added, protocol, augmented, scenario, model_name, seed, options)
        pred, settings, feature_rows = fit_choices(training, testing, scenario, model_name, seed, candidates, options)
        for category, rows in [("predictions", pred), ("settings", settings), ("features", feature_rows),
                               ("candidates", candidates), ("inner_scores", inner_scores),
                               ("inner_assignments", assignments), ("inner_ranks", ranks)]:
            result[category].extend(dict(training=arm, **r) for r in rows)
    result["seconds"] = time.perf_counter() - start
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    return result


def design(initial, bo, output, options=None):
    options = options or {}
    jobs, assignments, audits, balance, nearest = [], [], [], [], []
    for protocol in options.get("protocols", PROTOCOLS):
        reference = bo if protocol == "known_pair_new_composition" else initial
        strata = (pd.qcut(initial.separation_coefficient, 5, labels=False, duplicates="drop").astype(str)
                  if protocol != "known_pair_new_composition" else known_strata(bo))
        for repeat in range(options.get("repeats", 3)):
            if protocol == "stratified_random":
                folds = list(StratifiedKFold(5, shuffle=True, random_state=42 + repeat).split(initial, strata))
            else:
                folds = (group_folds(initial, 5, 42 + repeat) if protocol == "unseen_pair"
                         else known_folds(bo, 5, 142 + repeat))
            for fold, (tr, te) in enumerate(folds):
                original, added, testing = partitions(initial, bo, protocol, tr, te)
                augmented = pd.concat([original, added], ignore_index=True)
                group = {"unseen_pair": "metal_pair", "stratified_random": "sample",
                         "known_pair_new_composition": "composition_group"}[protocol]
                assert not set(augmented[group]) & set(testing[group])
                key = dict(protocol=protocol, repeat=repeat, fold=fold)
                audits.append(dict(**key, n_train_initial=len(original), n_train_bo=len(added),
                    n_test_initial=int(testing.cohort.eq("initial").sum()),
                    n_test_bo=int(testing.cohort.eq("BO").sum()), group_overlap=0))
                for role, frame in [("train_initial", original), ("train_bo", added), ("test", testing)]:
                    assignments.extend(dict(**key, role=role, sample=r["sample"], cohort=r.cohort,
                        metal_pair=r.metal_pair, composition_group=r.composition_group)
                        for _, r in frame.iterrows())
                for role, idx in [("train", tr), ("test", te)]:
                    subset = reference.iloc[idx]
                    for st, count in strata.iloc[idx].value_counts().items():
                        balance.append(dict(**key, role=role, stratum=st, n=int(count), fraction=count / len(idx),
                            y_mean=float(subset.separation_coefficient.mean()),
                            y_min=float(subset.separation_coefficient.min()),
                            y_max=float(subset.separation_coefficient.max())))
                if protocol == "known_pair_new_composition":
                    # Quantify boundary-near neighbours; different bins do not imply a fixed gap.
                    for _, row in testing.iterrows():
                        same = added[added.metal_pair.eq(row.metal_pair)]
                        d = np.abs(same.actual_frac_element1 - row.actual_frac_element1)
                        nearest.append(dict(**key, sample=row["sample"], metal_pair=row.metal_pair,
                            min_training_bo_fraction_distance=float(d.min()),
                            within_training_fraction_range=bool(same.actual_frac_element1.min()
                                <= row.actual_frac_element1 <= same.actual_frac_element1.max())))
                    list(inner_partitions(original, added, protocol, True, 7000 + repeat * 100 + fold))
                jobs.extend((protocol, tr, te, repeat, fold, s, m) for s in options.get("scenarios", SCENARIOS) for m in ("RF", "ET"))
    for name, rows in [("outer_assignments", assignments), ("split_audit", audits),
                       ("stratum_balance", balance), ("composition_distance_audit", nearest)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    return jobs


def metric_tables(predictions):
    rows, pooled, systems = [], [], []
    group = ["protocol", "scenario", "model", "training", "strategy", "repeat"]
    for key, g in predictions.groupby(group):
        base = dict(zip(group, key))
        populations = {"BO": g[g.cohort.eq("BO")]}
        if base["protocol"] != "known_pair_new_composition":
            populations.update(initial=g[g.cohort.eq("initial")],
                initial_nonshared=g[g.cohort.eq("initial") & ~g.shared_icp])
        for population, subset in populations.items():
            if len(subset) < 2:
                continue
            assert not subset["sample"].duplicated().any()
            pooled.append(dict(**base, population=population, n=len(subset),
                               **scores(subset.y_true, subset.y_pred)))
            for fold, part in subset.groupby("fold"):
                if len(part) >= 2:
                    rows.append(dict(**base, population=population, fold=fold, n=len(part),
                        **scores(part.y_true, part.y_pred)))
        for pair, part in g[g.cohort.eq("BO")].groupby("metal_pair"):
            systems.append(dict(**base, metal_pair=pair, n=len(part), **scores(part.y_true, part.y_pred)))
    return pd.DataFrame(rows), pd.DataFrame(pooled), pd.DataFrame(systems)


def summary(table, keys):
    return table.groupby(keys, as_index=False).agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        r2_min=("r2", "min"), r2_max=("r2", "max"), rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"), mae_mean=("mae", "mean"), n_evaluations=("r2", "size"))


def aggregate(results, output):
    tables = {}
    for category in ("predictions", "settings", "candidates", "inner_scores", "inner_ranks", "features"):
        tables[category] = pd.DataFrame([dict(**{k: r[k] for k in KEYS}, **v)
            for r in results for v in r[category]])
        tables[category].to_csv(output / f"{category}.csv", index=False)
    # Store split membership once per arm/fold, without duplicating it across features/models.
    inner = pd.DataFrame([dict(**{k: r[k] for k in ("protocol", "repeat", "fold")}, **v)
                         for r in results for v in r["inner_assignments"]]).drop_duplicates()
    inner.to_csv(output / "inner_assignments.csv", index=False)
    predictions, settings = tables["predictions"], tables["settings"]
    assert not predictions.duplicated(KEYS + ["training", "strategy", "sample"]).any()
    # The automatic model family uses inner validation only, separately for each strategy.
    selection_keys = ["protocol", "repeat", "fold", "scenario", "training", "strategy"]
    selectable = settings[settings.strategy.ne("fixed_all")].copy()
    winners = selectable.sort_values(["inner_r2_mean", "n_features", "model"],
        ascending=[False, True, True]).drop_duplicates(selection_keys)
    winners.to_csv(output / "inner_selected_models.csv", index=False)
    auto = predictions.merge(winners[selection_keys + ["model"]], on=selection_keys + ["model"], validate="many_to_one")
    auto["model"] = "inner_selected_RF_ET"
    all_predictions = pd.concat([predictions, auto], ignore_index=True)
    all_predictions.to_csv(output / "predictions_with_inner_selected.csv", index=False)
    fold, pooled, systems = metric_tables(all_predictions)
    groups = ["protocol", "scenario", "model", "training", "strategy"]
    for name, table, extra in [("fold", fold, ["population"]), ("pooled_repeat", pooled, ["population"]),
                              ("system_repeat", systems, ["metal_pair"])]:
        table.to_csv(output / f"{name}_metrics.csv", index=False)
        summary(table, groups + extra).to_csv(output / f"{name}_summary.csv", index=False)
    before, after = pooled[pooled.training.eq("initial_only")], pooled[pooled.training.eq("initial_plus_bo")]
    join = [c for c in groups if c != "training"] + ["population", "repeat"]
    paired = before.merge(after, on=join, validate="one_to_one", suffixes=("_before", "_after"))
    paired["delta_r2"] = paired.r2_after - paired.r2_before
    paired["delta_rmse"] = paired.rmse_after - paired.rmse_before
    paired.to_csv(output / "paired_repeat_changes.csv", index=False)
    selection = tables["features"].groupby(groups + ["feature"], as_index=False).agg(
        selection_frequency=("selected", "mean"), selection_count=("selected", "sum"))
    selection.to_csv(output / "selection_stability.csv", index=False)
    settings.groupby(groups, as_index=False).agg(n_features_mean=("n_features", "mean"),
        n_features_min=("n_features", "min"), n_features_max=("n_features", "max")).to_csv(
            output / "feature_count_summary.csv", index=False)
    audit_results(results, inner, all_predictions, output)
    return all_predictions, pooled, systems


def audit_results(results, inner, predictions, output):
    for key, part in inner.groupby(["protocol", "repeat", "fold", "training", "inner_fold"]):
        group = {"unseen_pair": "metal_pair", "stratified_random": "sample",
                 "known_pair_new_composition": "composition_group"}[key[0]]
        a, b = part[part.role.eq("train")], part[part.role.eq("validation")]
        assert not set(a[group]) & set(b[group])
        assert not set(a["sample"]) & set(b["sample"])
    for result in results:
        table = pd.DataFrame(result["candidates"])
        for setting in result["settings"]:
            if setting["strategy"] == "fixed_all":
                continue
            candidate = table[table.training.eq(setting["training"])].to_dict("records")
            p = {"icp13": 13, "nominal13": 13, "raw26": 26, "hybrid39": 39}[result["scenario"]]
            expected = choices(candidate, p)[setting["strategy"]]
            assert expected["n_features"] == setting["n_features"] and expected["config"] == setting["config"]
    assert np.isfinite(predictions[["y_true", "y_pred"]].to_numpy()).all()
    audit = dict(completed_jobs=len(results), predictions=len(predictions),
        outer_group_overlap=0, inner_group_overlap=0, selection_verified_from_inner_only=True,
        no_outer_score_model_selection=True)
    (output / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def plot_results(predictions, pooled, systems, output):
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    labels = {"raw26": "Elemental 26", "icp13": "ICP-weighted 13", "hybrid39": "Combined 39"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), layout="constrained")
    for ax, protocol, population, title in zip(axes, PROTOCOLS, ("initial", "BO"),
            ("A  Unseen metal pairs | initial-test cohort", "B  Known pairs, held-out composition | BO test")):
        for model, offset, color in [("RF", -.11, "#237a9b"), ("ET", .11, "#a33a5c")]:
            for j, scenario in enumerate(SCENARIOS):
                part = pooled[(pooled.protocol.eq(protocol)) & pooled.population.eq(population) &
                    pooled.model.eq(model) & pooled.scenario.eq(scenario) & pooled.strategy.eq("fixed_all")]
                means = part.groupby("training").r2.mean()
                ax.plot([j + offset] * 2, [means.initial_only, means.initial_plus_bo], c=color, lw=1.4)
                ax.scatter(j + offset, means.initial_only, s=80, facecolors="white", edgecolors=color, zorder=3)
                ax.scatter(j + offset, means.initial_plus_bo, s=65, c=color, zorder=4)
                ax.text(j + offset, means.initial_plus_bo + .035, f"{means.initial_plus_bo:.2f}",
                        ha="center", fontsize=9, color=color)
        ax.set(xticks=range(3), xticklabels=[labels[s] for s in SCENARIOS], ylabel="Pooled out-of-fold R2",
               title=title, xlim=(-.45, 2.45))
        ax.axhline(0, color="0.65", lw=.8)
        ax.grid(axis="y", alpha=.2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, label=m) for m, c in [("RF", "#237a9b"), ("Extra Trees", "#a33a5c")]]
    handles += [Line2D([], [], marker="o", linestyle="none", markerfacecolor=f, color="0.3", label=l)
                for f, l in [("white", "Initial training only"), ("0.3", "Initial + BO training")]]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("BO augmentation: matched test sets, fixed model settings", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"two_tasks_matched_comparison.{ext}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), layout="constrained")
    colors = dict(zip(["Co-Fe", "Co-Mo", "Co-Ni", "Co-V", "Fe-Ni"],
                      ["#9a62ba", "#db7124", "#328abb", "#318966", "#be4974"]))
    for ax, protocol, population, title in zip(axes, PROTOCOLS, ("initial", "BO"),
            ("Unseen metal pairs", "Known pairs: new compositions")):
        part = predictions[(predictions.protocol.eq(protocol)) & predictions.cohort.eq(population) &
            predictions.scenario.eq("hybrid39") & predictions.model.eq("inner_selected_RF_ET") &
            predictions.training.eq("initial_plus_bo") & predictions.strategy.eq("nested_max")]
        averaged = part.groupby(["sample", "metal_pair"], as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        if population == "BO":
            for pair, g in averaged.groupby("metal_pair"):
                ax.scatter(g.y_true, g.y_pred, s=35, alpha=.8, label=pair, c=colors[pair])
            ax.legend(fontsize=9, loc="lower right", frameon=False)
        else:
            ax.scatter(averaged.y_true, averaged.y_pred, s=18, c="#237a9b", alpha=.55)
        metric = pooled[(pooled.protocol.eq(protocol)) & pooled.population.eq(population) &
            pooled.scenario.eq("hybrid39") & pooled.model.eq("inner_selected_RF_ET") &
            pooled.training.eq("initial_plus_bo") & pooled.strategy.eq("nested_max")]
        ax.plot([2, 12], [2, 12], "--", color="0.5", lw=1)
        ax.set(xlabel="Measured H/D separation coefficient", ylabel="Predicted H/D separation coefficient",
               title=title, xlim=(2, 12), ylim=(2, 12))
        ax.text(.04, .96, f"Nested CV R2 = {metric.r2.mean():.3f}\nRMSE = {metric.rmse.mean():.3f}",
                transform=ax.transAxes, va="top")
        ax.grid(alpha=.15)
    fig.suptitle("Combined 39: features, parameters and RF/ET chosen inside training folds", fontsize=13)
    fig.supxlabel("Points average three held-out predictions; metrics average three separate OOF evaluations.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"two_tasks_nested_predictions.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    initial, bo, inclusion = build_data()
    initial.to_csv(args.output / "initial_dataset.csv", index=False)
    bo.to_csv(args.output / "bo_dataset.csv", index=False)
    inclusion.to_csv(args.output / "bo_inclusion_audit.csv", index=False)
    files = [Path(__file__), ICP, OBSERVATIONS, CURATED, WORKBOOK, COMPOSITION_TABLE, FEATURE_MATRIX]
    files += [ROOT / (name + ".py") for name in ("icp_bo_augmentation_cv", "icp_hybrid_descriptor_cv",
        "icp_nested_group_cv", "icp_intrinsic_descriptor_analysis", "icp_parsimony_sensitivity",
        "icp_composition_weighted_sensitivity", "icp_master")]
    manifest = dict(source_hashes={str(p): sha(p) for p in files}, n_initial=len(initial), n_bo=len(bo),
        composition_schema=SCHEMA_VERSION,
        protocols=PROTOCOLS, scenarios=SCENARIOS, models=["RF", "ET"], outer_folds=5, repeats=3,
        inner_folds=3, outer_seeds=dict(unseen_pair=[42, 43, 44], known_pair_new_composition=[142, 143, 144]),
        initial_strata="HD quintiles", known_strata="metal pair x within-pair HD terciles",
        composition_bin_width=.05, final_trees=600, selector_trees=128, search_trees=160,
        configs=CONFIGS, feature_counts=dict(icp13=candidate_counts(13), hybrid39=candidate_counts(39)),
        shared_icp_bo_excluded=True, fixed_control="raw26, and initial-only known-pair model",
        inner_scoring="mean fold R2; initial validations for unseen pairs, BO validations for known pairs",
        primary_comparison="fixed settings, matched test populations before and after adding BO",
        adaptive_prediction="nested-max and one-SE; model family also selected only on inner scores",
        interpretation="retrospective, not independent external or chronological validation")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["fingerprint"] = fingerprint
    path = args.output / "analysis_manifest.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8"))["fingerprint"] == fingerprint
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    jobs = design(initial, bo, args.output)
    print(f"Audited {len(jobs)} jobs; {len(initial)} initial + {len(bo)} BO; outer and inner groups isolated.", flush=True)
    if args.audit_only:
        return
    start = time.perf_counter()
    results = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        iterator = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(run_job)(initial, bo, *job, fingerprint, args.output) for job in jobs)
        for result in iterator:
            results.append(result)
            print(f"{len(results)}/{len(jobs)} {result['protocol']} {result['scenario']} {result['model']} "
                  f"r{result['repeat']}f{result['fold']} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    predictions, pooled, systems = aggregate(results, args.output)
    plot_results(predictions, pooled, systems, args.output)
    (args.output / "runtime.json").write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        jobs=len(jobs), workers=args.jobs), indent=2), encoding="utf-8")
    print(f"Complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
