"""Generate reviewer-facing figures and tables from saved revision predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icp_intrinsic_descriptor_analysis import LABELS, scores

SCENARIOS = {"nominal13": "Precursor-weighted 13", "raw26": "Separate elemental 26",
             "icp13": "ICP-weighted 13", "hybrid39": "Combined 39"}
PROTOCOLS = {"stratified_random": "Stratified random", "unseen_pair": "Unseen metal pairs",
             "known_pair_new_composition": "Known pairs: new compositions"}


def grouped_metrics(frame, keys):
    rows = []
    for key, part in frame.groupby(keys):
        if len(part) >= 2:
            if part["sample"].duplicated().any():
                raise ValueError(f"Duplicated OOF sample in {key}")
            rows.append(dict(zip(keys, key), n=len(part), **scores(part.y_true, part.y_pred)))
    return pd.DataFrame(rows)


def summarize(frame, keys):
    return frame.groupby(keys, as_index=False).agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
        rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"), n=("n", "first"), repeats=("repeat", "nunique"))


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def markdown_table(frame):
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(map(fmt, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def verify_predictions(predictions, initial, bo, assignments):
    labels = pd.concat([initial, bo]).set_index("sample").separation_coefficient
    np.testing.assert_allclose(predictions.y_true, predictions["sample"].map(labels), atol=1e-12)
    key = ["protocol", "scenario", "model", "training", "strategy", "repeat"]
    for group, part in predictions.groupby(key):
        expected = set(bo["sample"]) if group[0] == "known_pair_new_composition" else set(initial["sample"])
        if group[0] == "unseen_pair":
            expected |= set(bo["sample"])
        if set(part["sample"]) != expected or part["sample"].duplicated().any():
            raise ValueError(f"Incomplete OOF coverage: {group}")
    valid = assignments[assignments.role.eq("test")][["protocol", "repeat", "fold", "sample"]]
    check = predictions.merge(valid, on=["protocol", "repeat", "fold", "sample"], how="left", indicator=True, validate="many_to_one")
    if not check._merge.eq("both").all():
        raise ValueError("Prediction assigned to a non-test sample")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    predictions = pd.read_csv(output / "predictions_with_inner_selected.csv")
    initial, bo = pd.read_csv(output / "initial_dataset.csv"), pd.read_csv(output / "bo_dataset.csv")
    verify_predictions(predictions, initial, bo, pd.read_csv(output / "outer_assignments.csv"))
    main = predictions[predictions.model.eq("inner_selected_RF_ET") & predictions.strategy.eq("one_se") &
                       predictions.training.eq("initial_plus_bo")].copy()
    main = main[(main.protocol.eq("known_pair_new_composition") & main.cohort.eq("BO")) |
                (~main.protocol.eq("known_pair_new_composition") & main.cohort.eq("initial"))]
    repeat = grouped_metrics(main, ["protocol", "scenario", "repeat"])
    table = summarize(repeat, ["protocol", "scenario"])
    repeat.to_csv(output / "primary_nested_repeat_metrics.csv", index=False)
    table.to_csv(output / "primary_nested_summary.csv", index=False)
    scenarios = [s for s in SCENARIOS if s in main.scenario.unique()]
    colors = {"nominal13": "#777777", "raw26": "#227C9D", "icp13": "#24845C", "hybrid39": "#AA405D"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
    uncertainty = table.r2_sd.fillna(0)
    lower = min(0., float((table.r2_mean - uncertainty).min()))
    upper = max(0., float((table.r2_mean + uncertainty).max()))
    span = max(upper - lower, .1)
    for ax, (protocol, title) in zip(axes, PROTOCOLS.items()):
        part = table[table.protocol.eq(protocol)].set_index("scenario")
        for j, scenario in enumerate(scenarios):
            row = part.loc[scenario]
            ax.errorbar(row.r2_mean, j, xerr=0 if pd.isna(row.r2_sd) else row.r2_sd,
                        fmt="o", capsize=4, color=colors[scenario], markersize=7)
            ax.annotate(f"{row.r2_mean:.3f}", (row.r2_mean, j), xytext=(6, 7), textcoords="offset points", fontsize=9)
        ax.set(yticks=range(len(scenarios)), yticklabels=[SCENARIOS[s] for s in scenarios],
               title=title, xlabel=r"Pooled out-of-fold $R^2$", ylim=(len(scenarios)-.5, -.5),
               xlim=(lower - .04 * span, upper + .16 * span))
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Nested validation: training-only feature selection and RF/Extra Trees selection", fontsize=13)
    fig.supxlabel("Points: mean across repeats. Bars: repeat SD. Initial + independent BO training.", fontsize=9)
    save(fig, output, "revision_validation_comparison")
    scenario = "hybrid39" if "hybrid39" in scenarios else "icp13"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), layout="constrained")
    for ax, (protocol, title) in zip(axes, PROTOCOLS.items()):
        part = main[main.protocol.eq(protocol) & main.scenario.eq(scenario)]
        averaged = part.groupby(["sample", "metal_pair"], as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        ax.scatter(averaged.y_true, averaged.y_pred, s=22, alpha=.6, c=colors[scenario], edgecolors="none")
        limits = [min(averaged.y_true.min(), averaged.y_pred.min()), max(averaged.y_true.max(), averaged.y_pred.max())]
        padding = (limits[1] - limits[0]) * .07
        limits = [limits[0]-padding, limits[1]+padding]
        ax.plot(limits, limits, linestyle="--", color=".5", lw=1)
        row = table[table.protocol.eq(protocol) & table.scenario.eq(scenario)].iloc[0]
        ax.text(.05, .95, f"$R^2$ = {row.r2_mean:.3f}\nRMSE = {row.rmse_mean:.3f}\nn = {int(row.n)}", transform=ax.transAxes, va="top")
        ax.set(xlabel=f"Measured {manifest['target_label']}", ylabel="Predicted response", title=title, xlim=limits, ylim=limits)
        ax.grid(alpha=.15)
    fig.suptitle(SCENARIOS[scenario] + ": nested one-SE models", fontsize=13)
    fig.supxlabel("Points average held-out predictions across repeats; reported metrics average separate OOF evaluations.", fontsize=9)
    save(fig, output, "revision_nested_predictions")

    baseline = pd.read_csv(output / "baseline_predictions.csv")
    baseline_rows = []
    for predictor in ("train_mean", "train_pair_mean"):
        part = baseline.rename(columns={predictor: "y_pred"})
        values = grouped_metrics(part, ["protocol", "training", "cohort", "repeat"])
        values["predictor"] = predictor
        baseline_rows.append(values)
    baseline_metrics = pd.concat(baseline_rows, ignore_index=True)
    summarize(baseline_metrics, ["protocol", "training", "cohort", "predictor"]).to_csv(output / "baseline_summary.csv", index=False)

    selected = pd.read_csv(output / "inner_selected_models.csv")
    selected = selected[selected.training.eq("initial_plus_bo") & selected.strategy.eq("one_se")]
    feature_counts = selected.groupby(["protocol", "scenario"], as_index=False).agg(
        minimum=("n_features", "min"), median=("n_features", "median"), maximum=("n_features", "max"))
    feature_counts.to_csv(output / "primary_selected_feature_counts.csv", index=False)
    changes = pd.read_csv(output / "paired_repeat_changes.csv")
    changes = changes[changes.strategy.eq("fixed_all") &
        ((changes.protocol.eq("known_pair_new_composition") & changes.population.eq("BO")) |
         (~changes.protocol.eq("known_pair_new_composition") & changes.population.eq("initial")))]
    augmentation = changes.groupby(["protocol", "scenario", "model"], as_index=False).agg(
        initial_only_r2=("r2_before", "mean"), initial_plus_bo_r2=("r2_after", "mean"),
        delta_r2=("delta_r2", "mean"), delta_rmse=("delta_rmse", "mean"))
    augmentation.to_csv(output / "matched_bo_augmentation_summary.csv", index=False)

    sensitivity_summary = None
    if (output / "sensitivity_diagnostics.csv").exists():
        sensitivity = pd.read_csv(output / "sensitivity_diagnostics.csv")
        sensitivity = sensitivity[(sensitivity.protocol.eq("unseen_pair") & sensitivity.cohort.eq("initial")) |
                                  (sensitivity.protocol.eq("known_pair_new_composition") & sensitivity.cohort.eq("BO"))]
        paired_rows = []
        for (protocol, scenario, repeat_id), part in sensitivity.groupby(["protocol", "scenario", "repeat"]):
            reference = part[part.variant.eq("primary_replay")].set_index("sample")
            for variant, changed in part[part.variant.ne("primary_replay")].groupby("variant"):
                if variant == "drop_zero_component":
                    changed = changed[~changed.zero_component]
                elif variant == "complete_property_rows":
                    changed = changed[changed.properties_complete]
                before = reference.loc[changed["sample"]]
                base_metrics, new_metrics = scores(before.y_true, before.y_pred), scores(changed.y_true, changed.y_pred)
                paired_rows.append(dict(protocol=protocol, scenario=scenario, repeat=repeat_id, variant=variant,
                    n=len(changed), r2_primary=base_metrics["r2"], r2_sensitivity=new_metrics["r2"],
                    delta_r2=new_metrics["r2"]-base_metrics["r2"], rmse_primary=base_metrics["rmse"],
                    rmse_sensitivity=new_metrics["rmse"], delta_rmse=new_metrics["rmse"]-base_metrics["rmse"]))
        paired = pd.DataFrame(paired_rows)
        paired.to_csv(output / "sensitivity_paired_repeat_metrics.csv", index=False)
        sensitivity_summary = paired.groupby(["protocol", "scenario", "variant"], as_index=False).agg(
            n=("n", "first"), primary_r2=("r2_primary", "mean"), sensitivity_r2=("r2_sensitivity", "mean"),
            delta_r2=("delta_r2", "mean"), delta_rmse=("delta_rmse", "mean"))
        sensitivity_summary.to_csv(output / "sensitivity_summary.csv", index=False)
    importance = None
    if (output / "permutation_diagnostics.csv").exists():
        permutation = pd.read_csv(output / "permutation_diagnostics.csv")
        fold_mean = permutation.groupby(["scenario", "repeat", "fold", "property"], as_index=False).r2_drop.mean()
        importance = fold_mean.groupby(["scenario", "property"], as_index=False).agg(
            mean_r2_drop=("r2_drop", "mean"), fold_sd=("r2_drop", "std"))
        importance.to_csv(output / "heldout_property_importance.csv", index=False)
        scens = list(importance.scenario.unique())
        fig, axes = plt.subplots(1, len(scens), figsize=(7*len(scens), 5.7), layout="constrained", squeeze=False)
        for ax, scenario in zip(axes[0], scens):
            part = importance[importance.scenario.eq(scenario)].sort_values("mean_r2_drop", ascending=True)
            ax.barh([LABELS.get(p, p) for p in part.property], part.mean_r2_drop, color=colors[scenario], alpha=.85)
            ax.axvline(0, color=".4", lw=.8)
            ax.set(title=SCENARIOS[scenario], xlabel=r"Held-out $R^2$ decrease after joint property permutation")
            ax.grid(axis="x", alpha=.2)
        fig.suptitle("Unseen-pair validation: predictive importance, not causal attribution", fontsize=13)
        save(fig, output, "revision_property_importance")
        shap_values = pd.read_csv(output / "shap_diagnostics.csv")
        shap_values.groupby(["scenario", "feature"], as_index=False).agg(mean_abs_shap=("mean_abs_shap", "mean"),
            selection_frequency=("selected", "mean")).to_csv(output / "heldout_shap_summary.csv", index=False)
    covariate_summary = None
    if (output / "operating_covariate_predictions.csv").exists():
        covariates = pd.read_csv(output / "operating_covariate_predictions.csv")
        values = grouped_metrics(covariates, ["scenario", "operating_covariates", "model", "repeat"])
        values.to_csv(output / "operating_covariate_repeat_metrics.csv", index=False)
        covariate_summary = summarize(values, ["scenario", "operating_covariates", "model"])
        covariate_summary.to_csv(output / "operating_covariate_summary.csv", index=False)
    lines = ["# Corrected ICP ML Revision", "", f"Target: `{manifest['target_column']}`. Input fingerprint: `{manifest['fingerprint']}`.",
        "", f"Primary analysis: {len(initial)} initial-screening samples and {len(bo)} independent BO additions.",
        "Shared ICP records remain in the input audit, but do not count as independent training samples.",
        "", "## Primary Nested Results", "", "RF/Extra Trees, feature count, and hyperparameters are chosen within training folds.",
        "The one-SE strategy favors fewer features within the inner-score tolerance. Best-inner-score and fixed-all results are also retained.",
        "R2 is pooled across the five outer test folds within each repeat, then averaged across repeats.",
        "Repeat SD measures split sensitivity, not independent experimental uncertainty.", "", markdown_table(table),
        "", "## Interpretation", "", "Random CV evaluates interpolation among represented chemistries; grouped CV evaluates unseen metal pairs.",
        "Known-pair composition CV evaluates BO compositions for already represented metal pairs. It is not new-chemistry extrapolation.",
        "Compare before/after BO using the fixed-setting matched-test results in `paired_repeat_changes.csv`; do not select a model from outer scores.",
        "The historical single-split R2 is not an error estimate for this revised nested protocol. These are different validation designs and inputs.",
        "Elemental descriptors are unshifted intrinsic properties. Current/potential covariates are a separate diagnostic, not intrinsic-property evidence."]
    lines.extend(["", "## Feature Selection", "", "Counts refer to the inner-selected RF/Extra Trees family in each outer fold.",
        "One-SE does not force a small subset: all features can be retained when smaller candidates fall outside the tolerance.",
        "The displayed representation size is the candidate pool, not a claim that every fold uses that many predictors.",
        "", markdown_table(feature_counts), "", "## BO Augmentation", "",
        "These paired comparisons use identical held-out samples and fixed model settings in both training arms.",
        "The table below shows Combined 39 (or the smoke representation); all representations are in `matched_bo_augmentation_summary.csv`.",
        "", markdown_table(augmentation[augmentation.scenario.eq("hybrid39" if "hybrid39" in scenarios else "icp13")])])
    if sensitivity_summary is not None:
        lines.extend(["", "## Sensitivity", "", "These refits retain primary inner-selected features/configuration and unchanged test outcomes.",
            "Zero-free and complete-property comparisons use matched eligible test subsets. Shared-reference averaging changes training labels only.",
            "The composition-gap check removes training samples within 2.5 mol% of any same-pair held-out composition.",
            "", markdown_table(sensitivity_summary)])
    if importance is not None:
        top = importance.sort_values("mean_r2_drop", ascending=False).groupby("scenario", sort=False).head(5)
        lines.extend(["", "## Descriptor Importance", "", "Held-out permutation jointly moves all selected representations of one property.",
            "Correlations between different properties can still distribute importance. SHAP and selection frequency are retained as complementary summaries.",
            "", markdown_table(top)])
    if covariate_summary is not None:
        lines.extend(["", "## Operating Covariates", "", "Initial-screening training only, with unseen-metal-pair tests and fixed model settings.",
            "Current and iR-corrected potential are diagnostic covariates, not evidence of an intrinsic materials mechanism.",
            "These models are not the nested intrinsic-descriptor primary analysis.", "", markdown_table(covariate_summary)])
    lines.extend(["", "## Reproduction", "", "Use `icp_revision_ml.py`, `icp_revision_diagnostics.py`, and `icp_revision_report.py`.",
        "Replace only the target CSV/column to update y. Input hashes isolate runs and prevent stale checkpoint reuse.",
        "This is retrospective internal validation; no independent external or prospective accuracy claim is made.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "report_verification.json").write_text(json.dumps(dict(all_oof_targets_match=True,
        exactly_one_prediction_per_sample_per_repeat=True, all_predictions_in_outer_test_folds=True,
        source_fingerprint=manifest["fingerprint"]), indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Report and figures saved: {output}")


if __name__ == "__main__":
    main()
