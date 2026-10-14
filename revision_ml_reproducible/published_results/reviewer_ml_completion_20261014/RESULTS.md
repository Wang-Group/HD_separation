# Remaining ML work completed: final 33-candidate workflow

Date: 2026-10-14.

## Main outcome

The outstanding calculations are complete: held-out SHAP for the reduced candidate pool and both frozen specifications; a matched precursor-versus-ICP representation comparison; and an explicit electronegativity/BO-branch analysis. No response values, existing completed runs, manuscript text or Word responses were changed. The Co-Fe recipe archive was copied into revision_inputs/cofe for traceability.

ICP weighting improves the matched full-pool prediction scores relative to nominal precursor weighting. The selected-10 models retain similar performance with fewer inputs, and work function remains the leading property-family attribution. Local intermediate-electronegativity enrichment persists, but the evidence does not support describing the whole initial dataset by one universal quadratic volcano.

## Data and evaluation

- Final curated cohort: 554 initial-screening samples plus 111 independent BO samples (Co-Fe 27; Co-Mo, Co-Ni, Co-V and Fe-Ni 21 each). Shared-ICP reference copies are not treated as independent BO observations.
- Thirty-three candidates: 11 intrinsic properties represented as M1, M2 and ICP molar-weighted values. D frequency and H/D ZPE variants were removed; H frequency retained. This is the agreed duplication sensitivity, not blanket correlation pruning.
- Three repeats of five outer folds, with three inner folds. Feature selection, imputation and hyperparameter tuning use training folds. Initial screening is the test population for Tasks 1/2; BO compositions are the test population for Task 3.
- Task 1: stratified random initial entities, with matching BO composition bins purged. Task 2: unseen unordered metal pairs. Task 3: new composition bins within known metal pairs, with overlapping initial points purged.
- The with/without-BO arms share test observations. Task 3 without BO is initial-to-BO transfer, not a second random split of initial data. Its inner tuning uses an initial-only pair-held-out proxy.
- Reported R2 values are pooled held-out predictions per repeat, then mean +/- SD across three repeats. This SD describes split variation; it is not an independent-sample confidence interval.

## 1 Updated held-out SHAP

All 270 refits reproduce the saved model predictions. Work function ranks first at the property-family level in 18/18 combinations of task, BO arm and model specification. Individual M1/M2/weighted attributions are exported separately; unselected inputs have zero attribution in an individual fitted model.

Tree SHAP uses training-tree path counts, not a held-out background. Family summaries first sum signed M1/M2/weighted contributions per sample, then take the absolute value. These are an aggregation of ordinary SHAP, not a new conditional/group-SHAP estimator. Correlated inputs can redistribute attribution, and a large attribution in a poorly performing transfer model does not establish usefulness.

[Nested 33 SHAP](shap33/nested33_shap.png) | [Frozen/nested comparison](shap33/frozen_vs_nested_family_shap.png) | [All separate feature ranks](shap33/feature_summary.csv)

### Explained model performance, with BO

| Task | Nested forward-10 / 33 | Original frozen-10 | Reselected frozen-10 / 33 |
|---|---:|---:|---:|
| Random entities | 0.556 +/- 0.009 | 0.569 +/- 0.014 | 0.573 +/- 0.013 |
| Unseen metal pairs | 0.555 +/- 0.008 | 0.578 +/- 0.006 | 0.577 +/- 0.008 |
| New compositions | 0.544 +/- 0.018 | 0.495 +/- 0.015 | 0.372 +/- 0.021 |

The frozen specifications are retrospective sensitivities selected using Task-1 information, not independent nested estimates. The reselected version has the same ten feature identities but a different order and tuned tree regularization. Its lower Task-3 score is retained and reported; do not pick a frozen specification by looking at Task-3 performance. The original frozen model never used the six deleted inputs, so this difference cannot be attributed to their direct removal.

## 2 Matched precursor-versus-ICP comparison

For each of 90 arm/task/repeat/fold combinations, the same outer and inner sample memberships, targets, imputed elemental endpoints and RF/Extra-Trees tuning grid are used. Four declared input pools are compared: raw M1/M2 properties (22); those plus nominal precursor-weighted properties (33); those plus ICP-weighted properties (33); and ICP33 plus an explicit ICP molar fraction (34). Each full pool is tuned independently inside training folds. There is no new forward search in this full-pool comparison.

Nominal BO weights reproduce the historical recipe-volume convention, volume1/(volume1+volume2). All 111 BO recipe identities and recorded response values were verified. This is a nominal precursor control, not a claim that recipe volumes equal deposited molar fractions; a molar interpretation also requires the historical equal-stock-concentration assumption. ICP weights use the corrected measured metal molar fractions.

| BO training arm | Task | Raw 22 | Precursor 33 | ICP 33 | ICP 33 + fraction |
|---|---|---:|---:|---:|---:|
| without_bo | Random entities | 0.543 +/- 0.018 | 0.534 +/- 0.011 | 0.559 +/- 0.013 | 0.559 +/- 0.009 |
| without_bo | Unseen metal pairs | 0.548 +/- 0.002 | 0.536 +/- 0.005 | 0.566 +/- 0.010 | 0.562 +/- 0.006 |
| without_bo | New compositions | -0.214 +/- 0.035 | -0.057 +/- 0.057 | 0.078 +/- 0.041 | 0.050 +/- 0.025 |
| with_bo | Random entities | 0.535 +/- 0.020 | 0.531 +/- 0.003 | 0.568 +/- 0.012 | 0.567 +/- 0.013 |
| with_bo | Unseen metal pairs | 0.526 +/- 0.011 | 0.534 +/- 0.009 | 0.566 +/- 0.008 | 0.570 +/- 0.008 |
| with_bo | New compositions | 0.186 +/- 0.048 | 0.487 +/- 0.040 | 0.552 +/- 0.031 | 0.559 +/- 0.033 |

With BO, replacing precursor weights by ICP weights increases full-pool R2 by +0.037, +0.032 and +0.065 for Tasks 1, 2 and 3. Adding fraction explicitly changes these scores by -0.001, +0.004 and +0.008: the weighted properties already carry most composition information in this model family. Raw properties alone do not distinguish compositions within a pair and perform much worse on Task 3 (0.186 vs 0.552).

A separate controlled substitution keeps each outer fold's ICP-selected ten features and tuned configuration fixed, substitutes nominal weights, and retrains. With BO, precursor -> ICP R2 is 0.533 -> 0.556, 0.537 -> 0.555, and 0.543 -> 0.544. Thus the Task-3 ICP benefit is not universal across feature sets: it is substantial in the independently tuned full-pool comparison but negligible in this conditional selected-10 comparison. This is not an independently optimized precursor forward-selection benchmark.

Without BO, composition transfer remains weak even with correct ICP weights. Composition measurements improve the representation; they do not replace training examples spanning composition space.

[Comparison figure](composition_bridge/composition_comparison.png) | [All metrics](composition_bridge/summary.csv) | [Paired changes](composition_bridge/paired_summary.csv) | [Recipes/weights](composition_bridge/composition_input_audit.csv)

## 3 Electronegativity and the BO branches

The plots use 496 initial samples with electronegativity observed under both weighting schemes, 84 independent samples from the original four BO systems, and a separate 111-point analysis adding Co-Fe. No descriptor imputation is used. The original three-descriptor trajectory panels use 226 complete initial samples and 111 BO samples.

For the four original BO systems, the ICP-weighted pooled quadratic has an interior peak at 1.861 (descriptive R2 = 0.132); including Co-Fe gives 1.856 (R2 = 0.171). These are broad, mixed-pair associations, not precise optimal compositions. Pair-specific curves are not uniformly concave and their curvature bootstrap ranges span zero. Bootstrap results are diagnostics, not p values, and within-pair BO resampling does not reproduce the dependence of adaptive BO sampling.

For the initial screening cohort, a single ICP-weighted global quadratic has no interior maximum and explains only 0.110 of the response variance in-sample. Nevertheless, empirical local means and 90th percentiles show intermediate-region enrichment. The local pattern and the global quadratic are different questions; a poor global quadratic does not disprove the local peak, nor does the local peak validate a universal mechanistic volcano.

Equal-count-bin sensitivity (6, 8, 10, 12 and 15 requested bins; at least 12 samples retained per bin) is exported separately. These are descriptive summaries of the same measured data, not cross-validated prediction or a fit to only selected high-performing points.

### Why the visual pattern and feature selection differ

| Task, with BO | M1 electronegativity | M2 electronegativity | ICP-weighted electronegativity |
|---|---:|---:|---:|
| Random entities | 1/15 | 0/15 | 0/15 |
| Unseen metal pairs | 9/15 | 0/15 | 0/15 |
| New compositions | 6/15 | 6/15 | 0/15 |

Weighted electronegativity is not selected in these forward-10 fits. A bivariate visual association does not measure its incremental predictive contribution after other correlated descriptors are available. Within a fixed pair every linearly weighted property is affine in the same ICP fraction: correlated descriptor trajectories and pair identity can explain the apparent branches without two distinct reaction mechanisms.

The extended lower-electronegativity branch is mainly Co-V; the higher-electronegativity branch is mainly Co-Mo. Co-Ni and Fe-Ni occupy the narrower central region. ICP moves the Co-Mo points substantially toward Co-rich compositions but leaves the branch geometry implied by the elemental endpoints. Co-Fe, newly included in ML, is shown separately from the original four-system comparison.

[Original four BO systems](electronegativity_branches/original_four_bo_electronegativity.png) | [All five systems](electronegativity_branches/all_five_bo_electronegativity.png) | [Pair-labelled trajectories](electronegativity_branches/bo_pair_trajectories.png) | [Shape/bootstrap values](electronegativity_branches/descriptive_shape_bootstrap.csv) | [Binning sensitivity](initial_peak_location_by_bins.csv)

## Reviewer evidence map

| Concern | Completed evidence |
|---|---|
| R1: single split / robustness / chemical leakage | Repeated nested CV across all three tasks, with/without BO, group-purged splits, feature-selection frequencies and identity/composition baselines |
| R1: electrochemical inputs and descriptor correlations | Intrinsic-only main models; matched operating-variable/potential-shift ablations; Pearson/Spearman correlations; exact-crossing eta10 policy; updated SHAP |
| R2: precursor ratios are not deposited composition | Corrected ICP master and source audit; present matched precursor-versus-ICP comparison |
| R3: electronegativity volcano versus SHAP, two BO branches | Present observed-only response summaries, unconstrained shape checks, separate M1/M2/weighted frequencies and pair-labelled trajectories |
| R3: tuning / composition information / BO coverage | Inner-fold model tuning and selection; ICP-weighted models; explicit fraction control; known-pair new-composition evaluation |

Earlier electrochemical and missing-reference controls remain valid as explicitly labelled 39-candidate sensitivities, rather than being silently renamed 33-candidate results. Main generalization evidence should use nested CV. The frozen Task-1 workflow is a model-transfer sensitivity, and interpretation should separate property association from causal active-site physics. Bulk ICP is measured composition, not measured surface-site composition or a measured alloy work function.

[39-to-33 comparison and methods](../forward33_comparison_20261013/RESULTS.md) | [Four earlier reviewer checks](../reviewer_ml_followups_20261013/RESULTS.md) | [Metal-identity controls](../identity_diagnostics_20261013/RESULTS.md)

The calculation set is ready to organize for the ML reviewer response. Remaining work is choosing the final panels/tables, integrating the Methods/response wording and packaging the reproducible code/data. Experimental issues elsewhere in the reviewer letter are outside this ML completion. No Word or manuscript changes have been applied.

## Reproduction and verification

Use the original ML environment from the repository root: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, joblib 1.4.2. SHAP's version is recorded in its run manifest. Ten process workers use one estimator thread each. Source, code, split and model hashes are checked before resuming cached work.

```powershell
$py = 'C:/Users/yibinjiang/anaconda3/python.exe'
$env:LOKY_MAX_CPU_COUNT = '10'
& $py hd_separation_ml/revision_shap33.py --jobs 10
& $py hd_separation_ml/revision_composition_bridge33.py --jobs 10
& $py hd_separation_ml/revision_electronegativity_branches.py
& $py -m pytest hd_separation_ml/test_revision_remaining.py hd_separation_ml/test_revision_followups.py hd_separation_ml/test_icp_forward_33_comparison.py hd_separation_ml/test_icp_forward_bo_ablation.py hd_separation_ml/test_icp_task1_frozen_sensitivity.py --junitxml=hd_separation_ml/outputs/reviewer_ml_completion_20261014/test_results.xml -q
& $py hd_separation_ml/summarize_revision_remaining.py
```

These follow-ups use the saved upstream CV fits/splits, not an isolated new training pipeline. If the response values change, regenerate the upstream input snapshot and nested/frozen runs in fresh output directories first. Existing fingerprints deliberately reject mixing new data/code with old fits. For portable sharing, retain the input snapshot, source recipes, training scripts, configuration/split artifacts and environment alongside these follow-up scripts; packaging has not been substituted for recalculation.

Verification: 42 focused tests passed; 588 dependency files checked; 90 matched representation jobs completed; 270 SHAP refits checked; independent repeat-metric discrepancy 3.3e-16; SHAP reconstruction error 9.2e-14. All six generated scientific figures were visually reviewed and PDF text bounds checked.
