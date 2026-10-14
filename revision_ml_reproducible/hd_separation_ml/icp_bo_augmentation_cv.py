"""Retrospective, matched-test BO augmentation using measured metal fractions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import StratifiedGroupKFold

from icp_intrinsic_descriptor_analysis import BASES, ROOT, DescriptorRepresentation, load_data, scores
from icp_nested_group_cv import OUTPUT as NESTED, CONFIGS, choose_best, group_folds, make_model, rfe_order
from icp_hybrid_descriptor_cv import OUTPUT as HYBRID, FIXED, STRATEGIES, HybridRepresentation, candidate_counts
from icp_parsimony_sensitivity import select_candidate

REPO = ROOT.parent
FIGURE4 = REPO / "0-Figures" / "Figure4"
sys.path.insert(0, str(FIGURE4))
from plot_icp_actual_composition_volcano import DEFAULT_ICP, load_icp_table
from icp_master import load_master

OUTPUT = ROOT / "outputs" / "icp_bo_augmentation_master_20261012"
ICP = DEFAULT_ICP
OBSERVATIONS = REPO / "0-Figures" / "Figure1" / "data" / "figure1_bottom_this_work_from_all_data.csv"
CURATED = FIGURE4 / "curated" / "figure4_bo_complete_112.csv"
SCENARIOS = ("icp13", "hybrid39")
SHARED = {f"{pair}3": pair for pair in ("Co-Fe", "Co-Mo", "Co-Ni", "Co-V", "Fe-Ni")}
EXCLUDED_BO_SYSTEMS = {"Co-Fe"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representation(name):
    return HybridRepresentation(name) if name == "hybrid39" else DescriptorRepresentation(name)


def pair_bo(training, bo):
    return bo[bo.metal_pair.isin(training.metal_pair)].copy()


def composition_groups(frame):
    first = frame.metal_pair.str.split("-").str[0]
    f = frame.actual_frac_element1.where(frame.element1.eq(first), 1 - frame.actual_frac_element1)
    bins = np.minimum(np.floor((f.to_numpy() + 1e-12) / .05).astype(int), 19)
    return frame.metal_pair + "_bin" + pd.Series(bins, index=frame.index).astype(str)


def build_data():
    initial, elemental = load_data()
    initial["cohort"] = "initial"
    initial["shared_icp"] = initial["sample"].isin(SHARED.values())
    assert initial["sample"].is_unique
    icp = load_icp_table(ICP)
    assert icp["sample"].notna().all() and icp["sample"].is_unique
    obs = pd.read_csv(OBSERVATIONS)
    obs = obs[obs.source_type.str.startswith("bo_")][["sample", "source_type", "separation_coefficient"]]
    # The four later BO systems have a second, independent outcome export.
    curated = pd.read_csv(CURATED)
    curated["sample"] = curated.system + curated.row_id.astype(str)
    both = obs.merge(curated[["sample", "separation_coefficient"]], on="sample", validate="one_to_one")
    np.testing.assert_allclose(both.separation_coefficient_x, both.separation_coefficient_y, atol=1e-7, rtol=0)
    master, _ = load_master(ICP)
    targets = master.loc[master.cohort.eq("BO")].set_index("sample").separation_coefficient
    use_master = obs["sample"].isin(targets.index)
    obs.loc[use_master, "separation_coefficient"] = obs.loc[use_master, "sample"].map(targets)
    merged = obs.merge(icp, on="sample", how="left", validate="one_to_one", indicator=True)
    merged["status"] = np.where(merged["_merge"].eq("both"), "eligible", "no_measured_icp")
    merged.loc[merged["sample"].isin(SHARED), "status"] = "excluded_shared_icp_with_initial"
    # Reference repeats have outcomes but no independent composition measurements.
    recipe_cols = [f"{kind}{i}" for i in range(1, 5) for kind in ("substance", "volume")]
    reference_samples = set()
    for _, group in curated.groupby("system"):
        reference = group.loc[group.row_id.eq(3), recipe_cols].iloc[0]
        reference_samples.update(group.loc[group[recipe_cols].eq(reference).all(axis=1), "sample"])
    missing_reference = merged["sample"].isin(reference_samples) & merged["_merge"].eq("left_only")
    merged.loc[missing_reference, "status"] = "reference_repeat_no_individual_icp"
    excluded_system = merged["sample"].str.replace(r"\d+$", "", regex=True).isin(EXCLUDED_BO_SYSTEMS)
    merged.loc[excluded_system, "status"] = "excluded_co_fe_bo_by_user"
    for sample, old_sample in SHARED.items():
        if old_sample in EXCLUDED_BO_SYSTEMS:
            continue
        f = merged.loc[merged["sample"].eq(sample), "actual_frac1_from_icp"].item()
        old = initial.loc[initial["sample"].eq(old_sample), "actual_frac_element1"].item()
        np.testing.assert_allclose(f, old, atol=1e-8, rtol=0)
    measured = merged[merged["_merge"].eq("both")]
    np.testing.assert_allclose(measured.actual_frac1_from_icp, measured.actual_frac1_from_icp_amount, atol=1e-12)
    bo = merged[merged.status.eq("eligible")].copy()
    metals = bo["sample"].str.extract(r"^([A-Z][a-z]?)-([A-Z][a-z]?)(\d+)$")
    bo["element1"], bo["element2"] = metals[0], metals[1]
    bo["row_id"] = metals[2].astype(int)
    bo["metal_pair"] = bo[["element1", "element2"]].apply(lambda row: "-".join(sorted(row)), axis=1)
    bo["actual_frac_element1"] = bo.actual_frac1_from_icp
    bo["actual_frac_element2"] = 1 - bo.actual_frac_element1
    bo["cohort"], bo["shared_icp"] = "BO", False
    for side in ("element1", "element2"):
        for base in BASES:
            bo[f"{base}_{side}"] = bo[side].map(elemental[base])
    assert len(initial) == 554 and len(bo) == 84 and len(merged) == 148
    assert not bo.metal_pair.isin(EXCLUDED_BO_SYSTEMS).any()
    assert not merged.status.eq("no_measured_icp").any()
    assert bo.actual_frac_element1.between(0, 1).all()
    assert np.isfinite(bo.separation_coefficient).all()
    assert not set(initial["sample"]) & set(bo["sample"])
    assert set(bo.metal_pair) <= set(initial.metal_pair)
    for frame in (initial, bo):
        frame["composition_group"] = composition_groups(frame)
    return initial.reset_index(drop=True), bo.reset_index(drop=True), merged


def search_augmented(initial_train, bo_train, scenario, model_name, seed):
    p = 13 if scenario == "icp13" else 39
    rows, assignments = [], []
    for i, (tr, va) in enumerate(group_folds(initial_train, 3, seed)):
        original = initial_train.iloc[tr]
        allowed = pair_bo(original, bo_train)
        train = pd.concat([original, allowed], ignore_index=True)
        valid = initial_train.iloc[va]
        assert not set(train.metal_pair) & set(valid.metal_pair)
        assignments += [dict(inner_fold=i, role="train_bo", sample=s) for s in allowed["sample"]]
        assignments += [dict(inner_fold=i, role="validation_initial", sample=s) for s in valid["sample"]]
        transform = representation(scenario).fit(train)
        X, V = transform.transform(train).to_numpy(), transform.transform(valid).to_numpy()
        y = train.separation_coefficient.to_numpy()
        order, _ = rfe_order(X, y, model_name, 128, seed + i)
        for k in candidate_counts(p):
            mask = np.sort(order[:k])
            for ci, config in enumerate(CONFIGS):
                model = make_model(model_name, config, 160, seed + i).fit(X[:, mask], y)
                rows.append(dict(inner_fold=i, n_features=k, config_id=ci,
                                 **scores(valid.separation_coefficient, model.predict(V[:, mask]))))
    table = pd.DataFrame(rows)
    candidates = [dict(n_features=int(k), config_id=int(ci), config=CONFIGS[int(ci)],
        inner_r2_mean=float(g.r2.mean()), inner_r2_std=float(g.r2.std()),
        inner_rmse_mean=float(g.rmse.mean())) for (k, ci), g in table.groupby(["n_features", "config_id"])]
    return candidates, assignments


def choices_from_candidates(candidates, p):
    simple, _ = select_candidate(candidates, "one_se", 3)
    return {"fixed_all_features": dict(config=FIXED, n_features=p, inner_r2_mean=None),
            "tuned_all_features": choose_best([c for c in candidates if c["n_features"] == p]),
            "nested_selection": choose_best(candidates), "one_se": simple}


def eval_choices(training, testing, scenario, model_name, seed, candidates, ranks=None, balanced=False):
    transform = representation(scenario).fit(training)
    frame = transform.transform(training)
    X, V = frame.to_numpy(), transform.transform(testing).to_numpy()
    columns = list(frame.columns)
    y = training.separation_coefficient.to_numpy()
    if ranks is None:
        order, _ = rfe_order(X, y, model_name, 128, seed)
    else:
        order = np.argsort([ranks[c] for c in columns], kind="stable")
    choices = choices_from_candidates(candidates, X.shape[1])
    if balanced:
        choices["pair_weighted_fixed"] = dict(config=FIXED, n_features=X.shape[1], inner_r2_mean=None)
    predictions, settings, cache = [], [], {}
    for strategy, choice in choices.items():
        mask = np.sort(order[:choice["n_features"]])
        key = (len(mask), json.dumps(choice["config"], sort_keys=True), strategy == "pair_weighted_fixed")
        if key not in cache:
            weight = np.ones(len(training))
            if strategy == "pair_weighted_fixed":
                original_counts = training[training.cohort.eq("initial")].metal_pair.value_counts()
                bo_counts = training[training.cohort.eq("BO")].metal_pair.value_counts()
                keep = training.cohort.eq("BO")
                weight[keep] = training.loc[keep, "metal_pair"].map(original_counts).to_numpy() / training.loc[keep, "metal_pair"].map(bo_counts).to_numpy()
            model = make_model(model_name, choice["config"], 600, seed)
            if strategy == "pair_weighted_fixed":
                model.fit(X[:, mask], y, sample_weight=weight)
            else:
                model.fit(X[:, mask], y)
            cache[key] = model.predict(V[:, mask])
        pred = cache[key]
        settings.append(dict(strategy=strategy, selected_features=[columns[j] for j in mask],
            n_features=len(mask), config=choice["config"], inner_r2_mean=choice["inner_r2_mean"],
            n_train_initial=int(training.cohort.eq("initial").sum()), n_train_bo=int(training.cohort.eq("BO").sum())))
        predictions += [dict(strategy=strategy, sample=row["sample"], cohort=row.cohort,
            metal_pair=row.metal_pair, shared_icp=bool(row.shared_icp), y_true=float(row.separation_coefficient),
            y_pred=float(value)) for (_, row), value in zip(testing.iterrows(), pred)]
    return predictions, settings


def source_checkpoint(repeat, fold, scenario, model_name):
    source = NESTED if scenario == "icp13" else HYBRID
    return source / "checkpoints" / f"repeat{repeat}_fold{fold}_{scenario}_{model_name}.json"


def run_transfer(initial, bo, tr, te, repeat, fold, scenario, model_name, fingerprint, output):
    key = dict(protocol="heldout_metal_pair", repeat=repeat, fold=fold, scenario=scenario, model=model_name)
    path = output / "checkpoints" / f"transfer_r{repeat}_f{fold}_{scenario}_{model_name}.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["fingerprint"] == fingerprint
        return result
    train, test = initial.iloc[tr].reset_index(drop=True), initial.iloc[te].reset_index(drop=True)
    btrain, btest = pair_bo(train, bo), pair_bo(test, bo)
    assert len(btrain) + len(btest) == len(bo)
    testing = pd.concat([test, btest], ignore_index=True)
    augmented = pd.concat([train, btrain], ignore_index=True)
    assert not set(augmented.metal_pair) & set(testing.metal_pair)
    seed = 2000 + repeat * 100 + fold
    source = json.loads(source_checkpoint(repeat, fold, scenario, model_name).read_text(encoding="utf-8"))
    ranks = source.get("outer_ranks", source.get("outer_train_rfe_ranks"))
    baseline, before_settings = eval_choices(train, testing, scenario, model_name, seed, source["candidates"], ranks)
    # Refit the archived baseline identically before comparing new BO predictions.
    old = pd.DataFrame(source["predictions"])
    old = old[old.strategy.isin(STRATEGIES)]
    check = pd.DataFrame(baseline).merge(old, on=["strategy", "sample"], suffixes=("_new", "_old"), validate="one_to_one")
    np.testing.assert_allclose(check.y_pred_new, check.y_pred_old, atol=1e-10, rtol=0)
    candidates, inner_assignments = search_augmented(train, btrain, scenario, model_name, seed)
    after, after_settings = eval_choices(augmented, testing, scenario, model_name, seed, candidates, balanced=True)
    predictions = [dict(training="initial_only", **r) for r in baseline] + [dict(training="initial_plus_bo", **r) for r in after]
    settings = [dict(training="initial_only", **r) for r in before_settings] + [dict(training="initial_plus_bo", **r) for r in after_settings]
    result = dict(**key, fingerprint=fingerprint, predictions=predictions, settings=settings,
        candidates=candidates, inner_assignments=inner_assignments,
        training_bo_samples=list(btrain["sample"]), testing_samples=list(testing["sample"]))
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def known_pair_folds(bo, seed):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(bo, bo.metal_pair, bo.composition_group))
    for tr, te in folds:
        assert not set(bo.iloc[tr].composition_group) & set(bo.iloc[te].composition_group)
        assert set(bo.iloc[te].metal_pair) <= set(bo.iloc[tr].metal_pair)
    return folds


def run_known(initial, bo, tr, te, repeat, fold, scenario, model_name, fingerprint, output):
    key = dict(protocol="known_pair_composition_bin", repeat=repeat, fold=fold, scenario=scenario, model=model_name)
    path = output / "checkpoints" / f"known_r{repeat}_f{fold}_{scenario}_{model_name}.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["fingerprint"] == fingerprint
        return result
    test = bo.iloc[te]
    allowed = initial[~initial.composition_group.isin(test.composition_group)]
    predictions, settings = [], []
    for arm, training in [("initial_only", allowed), ("initial_plus_bo", pd.concat([allowed, bo.iloc[tr]], ignore_index=True))]:
        assert not set(training.composition_group) & set(test.composition_group)
        transform = representation(scenario).fit(training)
        X, V = transform.transform(training).to_numpy(), transform.transform(test).to_numpy()
        model = make_model(model_name, FIXED, 600, 7000 + repeat * 100 + fold).fit(X, training.separation_coefficient)
        pred = model.predict(V)
        predictions += [dict(training=arm, strategy="fixed_all_features", sample=row["sample"],
            cohort="BO", metal_pair=row.metal_pair, shared_icp=False,
            y_true=float(row.separation_coefficient), y_pred=float(value)) for (_, row), value in zip(test.iterrows(), pred)]
        settings.append(dict(training=arm, strategy="fixed_all_features", n_features=X.shape[1],
            n_train_initial=int(training.cohort.eq("initial").sum()), n_train_bo=int(training.cohort.eq("BO").sum()), config=FIXED))
    result = dict(**key, fingerprint=fingerprint, predictions=predictions, settings=settings,
        testing_groups=list(test.composition_group.unique()), training_bo_samples=list(bo.iloc[tr]["sample"]),
        testing_samples=list(test["sample"]))
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate(payloads, output):
    keys = ["protocol", "repeat", "fold", "scenario", "model"]
    predictions = pd.DataFrame([{**{k: p[k] for k in keys}, **r} for p in payloads for r in p["predictions"]])
    settings = pd.DataFrame([{**{k: p[k] for k in keys}, **r} for p in payloads for r in p["settings"]])
    groups = ["protocol", "scenario", "model", "training", "strategy"]
    assert not predictions.duplicated(groups + ["repeat", "sample"]).any()
    predictions.to_csv(output / "outer_predictions.csv", index=False)
    settings.to_csv(output / "outer_settings.csv", index=False)
    rows, pooled, systems = [], [], []
    for cohort, mask in [("initial", predictions.cohort.eq("initial")),
                         ("initial_without_shared_icp", predictions.cohort.eq("initial") & ~predictions.shared_icp),
                         ("BO", predictions.cohort.eq("BO")), ("combined", np.ones(len(predictions), bool))]:
        subset = predictions[mask]
        for key, frame in subset.groupby(groups + ["repeat", "fold"]):
            if len(frame) < 2:
                continue
            rows.append(dict(zip(groups + ["repeat", "fold"], key), evaluation=cohort, n_test=len(frame), **scores(frame.y_true, frame.y_pred)))
        for key, frame in subset.groupby(groups + ["repeat"]):
            expected = 84 if cohort == "BO" or key[0] == "known_pair_composition_bin" else {"initial":554, "initial_without_shared_icp":549, "combined":638}[cohort]
            assert len(frame) == expected
            pooled.append(dict(zip(groups + ["repeat"], key), evaluation=cohort, n_test=len(frame), **scores(frame.y_true, frame.y_pred)))
    bo_predictions = predictions[predictions.cohort.eq("BO")]
    for key, frame in bo_predictions.groupby(groups + ["repeat", "metal_pair"]):
        systems.append(dict(zip(groups + ["repeat", "metal_pair"], key), n_test=len(frame), **scores(frame.y_true, frame.y_pred)))
    metrics, pooled, systems = map(pd.DataFrame, [rows, pooled, systems])
    metrics.to_csv(output / "outer_fold_metrics.csv", index=False)
    pooled.to_csv(output / "pooled_repeat_metrics.csv", index=False)
    systems.to_csv(output / "bo_system_repeat_metrics.csv", index=False)
    for name, table, extra in [("fold_summary", metrics, ["evaluation"]), ("pooled_summary", pooled, ["evaluation"]), ("bo_system_summary", systems, ["metal_pair"])]:
        table.groupby(groups + extra, as_index=False).agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), n_scores=("r2", "size")).to_csv(output / f"{name}.csv", index=False)
    same = ["protocol", "scenario", "model", "strategy", "evaluation", "repeat", "fold"]
    paired = metrics[metrics.training.eq("initial_plus_bo")].merge(metrics[metrics.training.eq("initial_only")], on=same, suffixes=("_after", "_before"), validate="one_to_one")
    assert paired.n_test_after.eq(paired.n_test_before).all()
    paired["delta_r2"] = paired.r2_after - paired.r2_before
    paired["delta_rmse"] = paired.rmse_after - paired.rmse_before
    paired.to_csv(output / "paired_fold_differences.csv", index=False)
    paired.groupby(same[:-2], as_index=False).agg(delta_r2_mean=("delta_r2", "mean"), delta_r2_std=("delta_r2", "std"),
        positive_folds=("delta_r2", lambda s: int((s > 0).sum())), n_folds=("delta_r2", "size"), delta_rmse_mean=("delta_rmse", "mean")).to_csv(output / "paired_summary.csv", index=False)
    settings.groupby(groups, as_index=False).n_features.agg(["mean", "min", "max"]).to_csv(output / "feature_counts.csv", index=False)
    return pd.read_csv(output / "fold_summary.csv"), pd.read_csv(output / "pooled_summary.csv")


def draw(folds, pooled, output):
    plt.rcParams.update({"font.family":"Arial", "font.size":11, "pdf.fonttype":42,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    labels = [(s, m) for s in SCENARIOS for m in ("RF", "ET")]
    for ax, metric, title in zip(axes, ("r2_mean", "rmse_mean"),
                                ("Same initial-library held-out samples", "Same initial-library held-out samples")):
        sub = folds[folds.protocol.eq("heldout_metal_pair") & folds.evaluation.eq("initial") & folds.strategy.eq("tuned_all_features")]
        for i, (s, m) in enumerate(labels):
            rows = sub[(sub.scenario == s) & (sub.model == m)].set_index("training")
            ax.plot([0, 1], rows.loc[["initial_only", "initial_plus_bo"], metric], "o-", label=f"{'Weighted 13' if s=='icp13' else 'Hybrid 39'}, {m}", color=["#16867e", "#345eb0", "#c76225", "#a42e62"][i])
        ax.set_xticks([0, 1], ["Initial only", "Initial + eligible BO"])
        ax.set_xlim(-.15,1.15)
        ax.set_ylabel("Mean outer-fold R2" if metric=="r2_mean" else "Mean outer-fold RMSE")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Does adding BO improve prediction of held-out metal pairs?", fontweight="bold", fontsize=15)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"bo_augmentation_matched_initial.{ext}", dpi=210)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    for ax, protocol, title in zip(axes, ["heldout_metal_pair", "known_pair_composition_bin"],
        ["BO: entire metal pair held out", "BO: composition bin held out within known pairs"]):
        sub = pooled[pooled.protocol.eq(protocol) & pooled.evaluation.eq("BO") & pooled.strategy.eq("fixed_all_features")]
        scenarios = list(sub.scenario.unique())
        entries = [(s,m) for s in scenarios for m in ("RF","ET")]
        for offset, arm, color in [(-.16,"initial_only","#868c92"),(.16,"initial_plus_bo","#157e82")]:
            values = [sub[(sub.scenario==s)&(sub.model==m)&(sub.training==arm)].iloc[0].rmse_mean for s,m in entries]
            ax.bar(np.arange(len(entries))+offset,values,width=.30,color=color,label=arm.replace('_',' '))
        ax.set_xticks(np.arange(len(entries)), [f"{'Raw26' if s=='raw26' else 'W13' if s=='icp13' else 'H39'}\n{m}" for s,m in entries])
        ax.set_title(title,fontsize=11)
        ax.set_ylabel("BO held-out RMSE (pooled within each repeat)")
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("Transfer to new pairs and interpolation within known pairs are different tasks",fontsize=14,fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"bo_prediction_two_tasks.{ext}",dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=12)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--audit-only",action="store_true")
    args = parser.parse_args()
    initial, bo, audit = build_data()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"checkpoints").mkdir(exist_ok=True)
    initial.to_csv(args.output/"initial_dataset.csv",index=False)
    bo.to_csv(args.output/"bo_dataset.csv",index=False)
    audit.to_csv(args.output/"bo_inclusion_audit.csv",index=False)
    print(f"Initial: {len(initial)}; eligible BO: {len(bo)}; {bo.metal_pair.nunique()} BO pairs.\n{audit.status.value_counts().to_string()}",flush=True)
    if args.audit_only:
        return
    hashes = {}
    for folder, name in [(NESTED,"input_hashes"),(HYBRID,"source_hashes")]:
        source = json.loads((folder/"analysis_manifest.json").read_text(encoding="utf-8"))
        for path, expected in source[name].items():
            assert sha(REPO/path)==expected, path
            hashes[path]=expected
    inputs = [Path(__file__),ICP,OBSERVATIONS,CURATED,FIGURE4/"plot_icp_actual_composition_volcano.py",NESTED/"outer_cv_assignments.csv"]
    for repeat in range(3):
        for fold in range(5):
            inputs.extend(source_checkpoint(repeat,fold,s,m) for s in SCENARIOS for m in ("RF","ET"))
    hashes.update({str(p.relative_to(REPO)):sha(p) for p in inputs})
    policy = dict(n_initial=len(initial),n_bo=len(bo),initial_pairs=int(initial.metal_pair.nunique()),
        excluded_shared_icp=SHARED,shared_icp_confirmation="User confirmed shared measurements on 2026-09-29",
        scenarios=SCENARIOS,strategies=STRATEGIES,source_hashes=hashes,
        inner_objective="Initial validation rows only; BO from inner-training metal pairs only",
        secondary="Known-pair 0.05-wide canonical ICP fraction bins; fixed settings, no tuning or RFE",
        weighting="Primary equal row weights; fixed-model diagnostic gives each BO pair total weight equal to its initial row count",
        outer_repeats=3,outer_folds=5,inner_folds=3,selector_trees=128,search_trees=160,final_trees=600,
        caveat="Adaptive BO collection and past reuse of these data prevent an independent prospective validation claim")
    fingerprint = hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    policy.update(fingerprint=fingerprint,jobs=args.jobs)
    manifest=args.output/"analysis_manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["fingerprint"]==fingerprint
    manifest.write_text(json.dumps(policy,indent=2),encoding="utf-8")
    assignments = pd.read_csv(NESTED/"outer_cv_assignments.csv")
    transfer_tasks, known_tasks, outer_rows = [], [], []
    for repeat in range(3):
        for fold,(tr,te) in enumerate(group_folds(initial,5,42+repeat)):
            expect=assignments[(assignments['repeat']==repeat)&(assignments.fold==fold)]
            assert set(initial.iloc[te]['sample'])==set(expect['sample'])
            test=pd.concat([initial.iloc[te],pair_bo(initial.iloc[te],bo)],ignore_index=True)
            outer_rows += [dict(protocol="heldout_metal_pair",repeat=repeat,fold=fold,sample=r['sample'],cohort=r.cohort,group=r.metal_pair) for _,r in test.iterrows()]
            transfer_tasks += [(tr,te,repeat,fold,s,m) for s in SCENARIOS for m in ("RF","ET")]
        for fold,(tr,te) in enumerate(known_pair_folds(bo,142+repeat)):
            outer_rows += [dict(protocol="known_pair_composition_bin",repeat=repeat,fold=fold,sample=r['sample'],cohort=r.cohort,group=r.composition_group) for _,r in bo.iloc[te].iterrows()]
            known_tasks += [(tr,te,repeat,fold,s,m) for s in ("raw26",*SCENARIOS) for m in ("RF","ET")]
    pd.DataFrame(outer_rows).to_csv(args.output/"outer_assignments.csv",index=False)
    payloads=[]
    start=time.perf_counter()
    with parallel_config(backend="loky",inner_max_num_threads=1):
        for func,tasks in [(run_transfer,transfer_tasks),(run_known,known_tasks)]:
            print(f"Starting {func.__name__}: {len(tasks)} jobs",flush=True)
            results=Parallel(n_jobs=args.jobs,return_as="generator_unordered")(
                delayed(func)(initial,bo,*task,fingerprint,args.output) for task in tasks)
            for i,payload in enumerate(results,1):
                payloads.append(payload)
                print(f"{func.__name__} {i}/{len(tasks)} {payload['scenario']} {payload['model']} r{payload['repeat']} f{payload['fold']}; {time.perf_counter()-start:.1f}s",flush=True)
    folds,pooled=aggregate(payloads,args.output)
    draw(folds,pooled,args.output)
    display=folds[folds.protocol.eq('heldout_metal_pair')&folds.evaluation.eq('initial')&folds.strategy.eq('tuned_all_features')]
    print(display.to_string(index=False),flush=True)
    print(f"Finished in {time.perf_counter()-start:.1f}s: {args.output}",flush=True)


if __name__=="__main__":
    main()
