"""Matched-test sensitivity to including independently measured Co-Fe BO samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from icp_master import ATOMIC_MASS, calculate_measurement
from icp_revision_inputs import ROOT, REPO, sha
from icp_intrinsic_descriptor_analysis import BASES, scores
from icp_bo_augmentation_cv import composition_groups
from icp_two_prediction_tasks import known_folds, search, fit_choices, inner_partitions
from icp_revision_report import markdown_table

INPUT = ROOT / "revision_inputs" / "cofe"
BASELINE = ROOT / "outputs" / "revision_icp_20261012"
OBSERVATIONS = REPO / "0-Figures/Figure1/data/figure1_bottom_this_work_from_all_data.csv"
SCENARIOS = ["icp13", "hybrid39"]
KEYS = ["protocol", "repeat", "fold", "scenario", "model"]


def prepare(received):
    if INPUT.exists():
        raise FileExistsError("Co-Fe input folder exists; do not overwrite revised targets.")
    source = pd.read_csv(OBSERVATIONS)
    labels = source[source.source_type.eq("bo_extra_co_fe_36")][["sample", "separation_coefficient"]].copy()
    if len(labels) != 36 or not labels["sample"].is_unique:
        raise ValueError("Unexpected archived Co-Fe outcomes")
    INPUT.mkdir(parents=True)
    shutil.copy2(received, INPUT / "ICP-OES_received.xlsx")
    labels.to_csv(INPUT / "targets.csv", index=False)
    provenance = {"received_path": str(received.resolve()), "received_sha256": sha(received),
                  "target_source": str(OBSERVATIONS.relative_to(REPO)), "target_source_sha256": sha(OBSERVATIONS)}
    (INPUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def load_baseline(path):
    manifest = json.loads((path / "analysis_manifest.json").read_text(encoding="utf-8"))
    for role, source in manifest["input_paths"].items():
        if sha(source) != manifest["input_hashes"][role]:
            raise ValueError(f"Baseline input changed: {role}. Rerun the baseline first.")
    for name, expected in manifest["code_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Baseline modeling code changed: {name}")
    initial = pd.read_csv(path / "initial_dataset.csv")
    bo = pd.read_csv(path / "bo_dataset.csv")
    assignments = pd.read_csv(path / "outer_assignments.csv")
    return initial, bo, assignments, manifest


def load_cofe(initial, workbook, targets, target, properties):
    raw = pd.read_excel(workbook, sheet_name="Sheet1")
    mass_columns = list(raw.columns[1:3])
    names = raw.iloc[:, 0].astype(str).str.extract(r"^CoFe-(\d+)$")[0]
    raw = raw.loc[names.notna()].copy()
    raw["sample"] = "Co-Fe" + names.loc[raw.index]
    if raw["sample"].duplicated().any():
        raise ValueError("Duplicate Co-Fe ICP sample")
    raw["source_row"] = raw.index + 2
    raw = raw.set_index("sample")
    labels = pd.read_csv(targets)
    if not {"sample", target} <= set(labels) or labels["sample"].duplicated().any():
        raise ValueError("Co-Fe target table needs unique sample IDs and the selected target")
    if not labels["sample"].str.fullmatch(r"Co-Fe\d+").all():
        raise ValueError("Unexpected Co-Fe target sample ID")
    labels[target] = pd.to_numeric(labels[target], errors="raise")
    if np.isinf(labels[target]).any() or set(raw.index) - set(labels["sample"]):
        raise ValueError("Infinite targets or measured ICP records absent from the target table")
    elemental = pd.read_csv(properties).set_index("element")
    shared = initial.loc[initial["sample"].eq("Co-Fe")].iloc[0]
    rows = []
    for _, label in labels.iterrows():
        sample = label["sample"]
        row = dict(sample=sample, element1="Co", element2="Fe", metal_pair="Co-Fe", cohort="BO",
                   separation_coefficient=label[target], shared_icp=False, icp_group="sample:" + sample,
                   status="eligible", zero_component=False, source_row=np.nan)
        if sample not in raw.index:
            row["status"] = "no_individual_icp"
        else:
            record = raw.loc[sample]
            if not np.isfinite(record[mass_columns].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite raw ICP: {sample}")
            a, b = record[mass_columns].to_numpy(dtype=float)
            result = calculate_measurement(dict(M1="Co", M2="Fe", M1_raw=a, M2_raw=b), ATOMIC_MASS)
            row.update({"M1 raw value": a, "M2 raw value": b, "source_row": int(record.source_row),
                        "actual_frac_element1": result["molar_fraction1"],
                        "actual_frac_element2": result["molar_fraction2"],
                        "M1 mass percentage": result["mass_fraction1"],
                        "M2 mass percentage": result["mass_fraction2"], "zero_component": min(a, b) < 0})
            if result["molar_fraction1"] is None:
                row["status"] = "no_valid_icp_composition"
            if sample == "Co-Fe3":
                np.testing.assert_allclose([a, b, result["molar_fraction1"]],
                    [shared["M1 raw value"], shared["M2 raw value"], shared.actual_frac_element1], atol=1e-12, rtol=0)
                row.update(status="shared_icp_not_independent", shared_icp=True, icp_group=shared.icp_group)
        if pd.isna(row["separation_coefficient"]) and row["status"] == "eligible":
            row["status"] = "missing_target"
        for side, element in [("element1", "Co"), ("element2", "Fe")]:
            for base in BASES:
                row[f"{base}_{side}"] = elemental.loc[element, base]
        row["properties_complete"] = all(pd.notna(row[f"{b}_{s}"]) for b in BASES for s in ("element1", "element2"))
        rows.append(row)
    audit = pd.DataFrame(rows)
    eligible = audit[audit.status.eq("eligible")].copy()
    eligible["composition_group"] = composition_groups(eligible)
    return eligible.reset_index(drop=True), audit


def comparison_partitions(initial, bo, cofe, assignments):
    data = pd.concat([initial, bo], ignore_index=True).set_index("sample", drop=False)
    cofe_folds = {r: known_folds(cofe, 5, 142 + int(r)) for r in assignments.repeat.unique()}
    plans = []
    for (protocol, repeat, fold), part in assignments.groupby(["protocol", "repeat", "fold"]):
        original = data.loc[part.loc[part.role.eq("train_initial"), "sample"]].reset_index(drop=True)
        existing = data.loc[part.loc[part.role.eq("train_bo"), "sample"]].reset_index(drop=True)
        test = data.loc[part.loc[part.role.eq("test"), "sample"]].reset_index(drop=True)
        if protocol == "stratified_random":
            additional = cofe[~cofe.composition_group.isin(test.composition_group)]
        elif protocol == "unseen_pair":
            additional = cofe[cofe.metal_pair.isin(original.metal_pair)]
        else:
            tr, te = cofe_folds[repeat][int(fold)]
            additional, extra_test = cofe.iloc[tr], cofe.iloc[te]
            test = pd.concat([test, extra_test], ignore_index=True)
            # Both arms use the same purged initial rows and the same 111 BO test population.
            original = original[~original.composition_group.isin(extra_test.composition_group)].reset_index(drop=True)
        augmented = pd.concat([existing, additional], ignore_index=True)
        training = pd.concat([original, augmented], ignore_index=True)
        group = {"stratified_random": "icp_group", "unseen_pair": "metal_pair",
                 "known_pair_new_composition": "composition_group"}[protocol]
        assert not set(training[group]) & set(test[group])
        assert not set(training["sample"]) & set(test["sample"])
        assert not set(training.icp_group) & set(test.icp_group)
        seed = (2000 if protocol == "unseen_pair" else 7000) + int(repeat) * 100 + int(fold)
        for inner_train, inner_valid in inner_partitions(original, augmented, protocol, True, seed):
            assert not set(inner_train[group]) & set(inner_valid[group])
            assert not set(inner_train["sample"]) & set(test["sample"])
            assert not set(inner_valid["sample"]) & set(test["sample"])
        plans.append(dict(protocol=protocol, repeat=int(repeat), fold=int(fold), original=original,
                          existing=existing, augmented=augmented, testing=test))
    return plans


def run_comparison(plan, scenario, name, manifest, baseline, output):
    key = {k: plan[k] for k in ("protocol", "repeat", "fold")}
    key.update(scenario=scenario, model=name)
    filename = f"{key['protocol']}_r{key['repeat']}_f{key['fold']}_{scenario}_{name}.json"
    path = output / "checkpoints" / filename
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Stale Co-Fe checkpoint")
        return result
    options = manifest["options"]
    seed = (2000 if key["protocol"] == "unseen_pair" else 7000) + key["repeat"] * 100 + key["fold"]
    result = dict(**key, fingerprint=manifest["fingerprint"], predictions=[], settings=[],
                  candidates=[], inner_scores=[], inner_assignments=[], inner_ranks=[])
    for arm in ("without_cofe", "with_cofe"):
        if arm == "without_cofe" and key["protocol"] != "known_pair_new_composition":
            old = json.loads((baseline / "checkpoints" / filename).read_text(encoding="utf-8"))
            if old["fingerprint"] != manifest["baseline_fingerprint"]:
                raise ValueError("Baseline checkpoint mismatch")
            for category in ("predictions", "settings", "candidates", "inner_scores", "inner_assignments", "inner_ranks"):
                result[category].extend(dict(**{k: v for k, v in row.items() if k != "training"}, training=arm)
                    for row in old[category] if row["training"] == "initial_plus_bo")
            continue
        added = plan["existing"] if arm == "without_cofe" else plan["augmented"]
        candidates, inner_scores, assignments, ranks = search(plan["original"], added, key["protocol"],
                                                             True, scenario, name, seed, options)
        training = pd.concat([plan["original"], added], ignore_index=True)
        pred, settings, _ = fit_choices(training, plan["testing"], scenario, name, seed, candidates, options)
        for category, rows in [("predictions", pred), ("settings", settings), ("candidates", candidates),
                               ("inner_scores", inner_scores), ("inner_assignments", assignments), ("inner_ranks", ranks)]:
            result[category].extend(dict(training=arm, **row) for row in rows)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    return result


def summarize(results, initial, bo, cofe, output, manifest):
    tables = {}
    for category in ("predictions", "settings", "candidates", "inner_scores", "inner_assignments", "inner_ranks"):
        table = pd.DataFrame([dict(**{k: result[k] for k in KEYS}, **row) for result in results for row in result[category]])
        if category == "inner_assignments":
            table = table.drop(columns=["scenario", "model"]).drop_duplicates()
        table.to_csv(output / f"{category}.csv", index=False)
        tables[category] = table
    selected_keys = ["protocol", "repeat", "fold", "scenario", "training", "strategy"]
    selectable = tables["settings"][tables["settings"].strategy.ne("fixed_all")]
    winners = selectable.sort_values(["inner_r2_mean", "n_features", "model"], ascending=[False, True, True]).drop_duplicates(selected_keys)
    winners.to_csv(output / "inner_selected_models.csv", index=False)
    auto = tables["predictions"].merge(winners[selected_keys + ["model"]], on=selected_keys + ["model"], validate="many_to_one")
    auto["model"] = "inner_selected_RF_ET"
    predictions = pd.concat([tables["predictions"], auto], ignore_index=True)
    predictions.to_csv(output / "predictions_with_inner_selected.csv", index=False)
    labels = pd.concat([initial, bo, cofe]).set_index("sample").separation_coefficient
    np.testing.assert_allclose(predictions.y_true, predictions["sample"].map(labels), atol=1e-12, rtol=0)
    keys = ["protocol", "scenario", "model", "training", "strategy", "repeat"]
    metrics = []
    for key, part in predictions.groupby(keys):
        base = dict(zip(keys, key))
        if base["protocol"] == "known_pair_new_composition":
            populations = {"existing_BO_84": part[part["sample"].isin(bo["sample"])],
                           "added_CoFe_BO": part[part["sample"].isin(cofe["sample"])], "all_BO": part}
        else:
            populations = {"initial": part[part.cohort.eq("initial")]}
        for population, subset in populations.items():
            expected = len(initial) if population == "initial" else len(bo) if population == "existing_BO_84" else len(cofe) if population == "added_CoFe_BO" else len(bo) + len(cofe)
            assert len(subset) == expected and subset["sample"].is_unique
            metrics.append(dict(**base, population=population, n=len(subset), **scores(subset.y_true, subset.y_pred)))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "repeat_metrics.csv", index=False)
    join = [k for k in keys if k != "training"] + ["population"]
    paired = metrics[metrics.training.eq("without_cofe")].merge(metrics[metrics.training.eq("with_cofe")],
        on=join, validate="one_to_one", suffixes=("_before", "_after"))
    paired["delta_r2"] = paired.r2_after - paired.r2_before
    paired["delta_rmse"] = paired.rmse_after - paired.rmse_before
    paired.to_csv(output / "paired_repeat_changes.csv", index=False)
    table = paired.groupby([k for k in join if k != "repeat"], as_index=False).agg(
        n=("n_after", "first"), r2_without=("r2_before", "mean"), r2_with=("r2_after", "mean"),
        r2_with_sd=("r2_after", "std"), delta_r2=("delta_r2", "mean"),
        rmse_without=("rmse_before", "mean"), rmse_with=("rmse_after", "mean"))
    table.to_csv(output / "comparison_summary.csv", index=False)
    for key, part in tables["inner_assignments"].groupby(["protocol", "repeat", "fold", "training", "inner_fold"]):
        group = {"stratified_random": "sample", "unseen_pair": "metal_pair", "known_pair_new_composition": "composition_group"}[key[0]]
        assert not set(part.loc[part.role.eq("train"), group]) & set(part.loc[part.role.eq("validation"), group])
    display = table[table.model.eq("inner_selected_RF_ET") & table.strategy.isin(["tuned_all", "one_se"])]
    cols = ["protocol", "scenario", "strategy", "population", "n", "r2_without", "r2_with", "delta_r2", "rmse_with"]
    report = ["# Co-Fe BO Inclusion Comparison", "", f"Baseline fingerprint: `{manifest['baseline_fingerprint']}`.",
              f"{len(initial)} initial + {len(bo)} existing independent BO + {len(cofe)} independent Co-Fe BO samples.",
              "", "## Matched Test Results", "", markdown_table(display[cols]), "", "## Design", "",
              "The existing outer assignments are frozen. Random/unseen-pair baseline predictions are reused from verified checkpoints.",
              "For known-pair CV, Co-Fe composition bins are appended to those same folds. Both arms remove initial rows in any held-out bin.",
              "Both known-pair arms are refitted and tuned without test outcomes. The existing four-pair test memberships are unchanged.",
              "Existing BO, Co-Fe BO and pooled BO scores are separate: pooled R2 can change just because the response distribution changes.",
              "RF/ET selection, imputation and feature selection remain inside training folds. Tuning grid and tree budgets match the baseline.",
              "Fixed-all comparisons for each forest and all nested strategies are retained in comparison_summary.csv.",
              "Co-Fe3 shares ICP with initial Co-Fe and is excluded as an independent addition. Other unmatched ICP IDs are not imputed.",
              "The English master and its exclusion policy have not been overwritten. This is an explicit inclusion sensitivity analysis.",
              "", "## Reproduction", "", "Run icp_cofe_inclusion.py with --baseline, --workbook, --targets, --output and --jobs.",
              "For revised y, first rerun the baseline and provide matching Co-Fe targets under the same target-column name.", ""]
    (output / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    (output / "verification.json").write_text(json.dumps(dict(targets_match=True, unique_oof_samples=True,
        inner_group_overlap=0, outer_group_overlap=0, completed_jobs=len(results), fingerprint=manifest["fingerprint"]), indent=2), encoding="utf-8")
    plot_comparison(table, output)
    print(display[cols].to_string(index=False))


def plot_comparison(table, output):
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    panels = [("stratified_random", "initial", "Stratified random"),
              ("unseen_pair", "initial", "Unseen metal pairs"),
              ("known_pair_new_composition", "existing_BO_84", "Existing BO compositions"),
              ("known_pair_new_composition", "added_CoFe_BO", "Co-Fe BO compositions")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), layout="constrained")
    data = table[table.model.eq("inner_selected_RF_ET") & table.strategy.eq("tuned_all")]
    for ax, (protocol, population, title) in zip(axes, panels):
        part = data[data.protocol.eq(protocol) & data.population.eq(population)].set_index("scenario")
        for i, (scenario, label, color) in enumerate([("icp13", "ICP-weighted 13", "#24845C"), ("hybrid39", "Combined 39", "#AA405D")]):
            row = part.loc[scenario]
            ax.plot([row.r2_without, row.r2_with], [i, i], color=color, lw=2)
            ax.scatter(row.r2_without, i, s=65, facecolor="white", edgecolor=color, zorder=3)
            ax.scatter(row.r2_with, i, s=40, color=color, zorder=4)
            ax.annotate(f"{row.r2_without:.3f} to {row.r2_with:.3f}", (max(row.r2_without, row.r2_with), i),
                        xytext=(0, 12), textcoords="offset points", ha="right", fontsize=9)
        ax.set(title=title, yticks=[0, 1], yticklabels=["ICP-weighted 13", "Combined 39"], ylim=(1.5, -.5), xlabel=r"Pooled OOF $R^2$")
        ax.margins(x=.3)
        ax.axvline(0, color=".7", lw=.7)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Co-Fe BO inclusion: all-feature models tuned inside training folds", fontsize=13)
    fig.supxlabel("Open: without Co-Fe BO. Filled: with Co-Fe BO. Each comparison uses identical held-out samples.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"cofe_inclusion_comparison.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path, help="Snapshot the received ICP workbook and extract archived Co-Fe targets once")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--workbook", type=Path, default=INPUT / "ICP-OES_received.xlsx")
    parser.add_argument("--targets", type=Path, default=INPUT / "targets.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cofe_inclusion_20261012")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    if args.prepare:
        prepare(args.prepare)
    initial, bo, assignments, baseline_manifest = load_baseline(args.baseline)
    cofe, audit = load_cofe(initial, args.workbook, args.targets, baseline_manifest["target_column"],
                            baseline_manifest["input_paths"]["elemental_properties"])
    plans = comparison_partitions(initial, bo, cofe, assignments)
    options = baseline_manifest["options"].copy()
    options["scenarios"] = SCENARIOS
    frozen_files = ["initial_dataset.csv", "bo_dataset.csv", "outer_assignments.csv"]
    manifest = dict(baseline_fingerprint=baseline_manifest["fingerprint"],
        baseline_output_hashes={f: sha(args.baseline / f) for f in frozen_files},
        source_hashes={"code": sha(Path(__file__)), "received_icp": sha(args.workbook), "cofe_targets": sha(args.targets)},
        code_hashes=baseline_manifest["code_hashes"], versions=baseline_manifest["versions"], options=options,
        target_column=baseline_manifest["target_column"], n_initial=len(initial), n_existing_bo=len(bo), n_cofe=len(cofe),
        status_counts=audit.status.value_counts().to_dict(), matched_outer_tests=True)
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output / "analysis_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Inputs/code changed; use a new output directory")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit.to_csv(output / "cofe_inclusion_audit.csv", index=False)
    cofe.to_csv(output / "cofe_dataset.csv", index=False)
    pd.concat([initial, bo, cofe], ignore_index=True).to_csv(output / "combined_dataset.csv", index=False)
    membership = []
    for plan in plans:
        key = {k: plan[k] for k in ("protocol", "repeat", "fold")}
        for role in ("original", "existing", "augmented", "testing"):
            membership.extend(dict(**key, role=role, **row) for row in plan[role][["sample", "metal_pair", "composition_group", "icp_group"]].to_dict("records"))
    pd.DataFrame(membership).to_csv(output / "outer_assignments.csv", index=False)
    print(f"AUDIT: {len(initial)} initial + {len(bo)} existing BO + {len(cofe)} Co-Fe; {len(plans)*4} outer jobs", flush=True)
    if args.audit_only:
        return
    (output / "checkpoints").mkdir(exist_ok=True)
    results, start = [], time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        iterator = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
            delayed(run_comparison)(plan, scenario, model, manifest, args.baseline, output)
            for plan in plans for scenario in SCENARIOS for model in ("RF", "ET"))
        for result in iterator:
            results.append(result)
            print(f"{len(results)}/{len(plans)*4} {result['protocol']} {result['scenario']} {result['model']} "
                  f"r{result['repeat']}f{result['fold']} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    summarize(results, initial, bo, cofe, output, manifest)
    (output / "runtime.json").write_text(json.dumps(dict(seconds=time.perf_counter()-start, workers=args.jobs), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
