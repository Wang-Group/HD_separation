# Four ML follow-ups for the revision

All four analyses are complete. The intrinsic ICP-composition model retains useful predictive signal without electrochemical inputs or sample-specific potential shifts. Work function remains the leading property attribution; missing-reference sensitivity is modest on matched test samples. These results support the revised predictive workflow, not a unique causal descriptor or a mechanistic proof.

## 1 Updated SHAP

Held-out SHAP was recomputed for both nested forward-10 and frozen Task-1 models across all three tasks, using the with-BO arm. All 90 refits reproduced saved predictions. Work function ranks first at the property-family level in all six model/task combinations. M1, M2 and ICP-weighted attributions are retained separately. H adsorption, standard potential, OH adsorption and vibration contributions depend on task/model.

Task-3 endpoint attributions can describe differences between metal pairs, not composition effects within a pair. Use the existing within-pair permutation results alongside SHAP. Correlated H/D/ZPE descriptors can redistribute attribution, so exact ranks must not be interpreted as uniquely identified physical causes.

[SHAP comparison](../reviewer_stage1_shap_20261013/current_feature_shap_comparison.png) | [Detailed methods and outputs](../reviewer_stage1_shap_20261013/RESULTS.md)

## 2 Correlations requested by Reviewer 1

Pearson/Spearman matrices cover all 39 descriptor variants, H/D separation, eta10, current, EIS resistance and corrected potential. Raw/unshifted and historical potential-shifted matrices, pairwise sample counts, and the exact analysis inputs are exported. There is no imputation for correlations.

For eta10, the primary analysis uses 312 exact crossings out of 554 initial samples. A secondary calculation includes four turning-point estimates, totaling 316. The 10/550 mV display bounds are never treated as measured eta10. Related catalysts are not independent observations; nominal p values are exploratory only.

| Variable versus H/D separation | Pearson r | Spearman rho | n |
|---|---:|---:|---:|
| eta10 | -0.235 | -0.225 | 312 |
| I_raw | -0.033 | -0.405 | 554 |
| Rs_from_EIS | -0.131 | -0.101 | 554 |
| E_IR_corrected_final | 0.329 | 0.378 | 554 |

[Correlation map](../reviewer_stage2_correlations_20261013_v2/descriptor_outcome_correlations.png) | [Detailed audit](../reviewer_stage2_correlations_20261013_v2/RESULTS.md)

## 3 Matched electrochemical ablations

All four cases use the same 554 initial samples, three repeats of five unseen-pair outer folds, and three inner folds. RF/Extra Trees plus six configurations each are tuned inside the training folds; 80 search trees, 400 final trees. These comparisons use all 39/42 declared inputs, not a new forward-selection search. The intrinsic baseline exactly reproduces the earlier all39/no-BO model.

| Inputs | R2 mean +/- SD | RMSE mean |
|---|---:|---:|
| Intrinsic 39 | 0.552 +/- 0.007 | 1.109 |
| Intrinsic 39 + operating | 0.585 +/- 0.007 | 1.067 |
| Potential-shifted 39 | 0.562 +/- 0.004 | 1.095 |
| Shifted 39 + operating | 0.577 +/- 0.006 | 1.077 |

Operating inputs are archived LSV current near -1 V, Rs, and corrected potential. BO operating inputs are absent from these source tables, so this comparison is initial-only. Potential shifts add E_IR + 0.098 V to work function and subtract it from PZC/standard potential, including weighted versions.

Adding operating inputs improves R2 by 0.033; shifts alone improve it by 0.011. ICP-weighted work function remains the largest individual SHAP attribution in all four all39/42 cases, but corrected potential ranks second when added to the intrinsic model, and other ranks change. Therefore the revised intrinsic interpretation does not depend entirely on electrochemical measurements, although those measurements contain additional predictive information. Do not claim they have no effect.

[Performance comparison](../reviewer_stage3_operating_20261013/operating_ablation.png) | [SHAP comparison](operating_ablation_shap.png) | [Detailed methods](../reviewer_stage3_operating_20261013/RESULTS.md)

## 4 Missing-reference sensitivity

The elemental-reference table lacks Eu/Nb entries, affecting 58 initial samples. Two restrictions were tested: excluding these rows, and retaining only rows with all selected descriptor values observed. Each restricted training refit is compared with the original predictions on exactly the same retained test samples. Features/configurations are frozen for this check; feature selection itself was not rerun after exclusions.

| Model with BO | Restriction | Task | Test n per repeat | Paired change in R2 |
|---|---|---|---|---:|
| nested_forward10 | listed_elements_only | New compositions | 111-111 | +0.007 +/- 0.007 |
| nested_forward10 | selected_properties_complete | New compositions | 110-111 | -0.003 +/- 0.007 |
| task1_frozen10 | listed_elements_only | New compositions | 111-111 | +0.007 +/- 0.006 |
| task1_frozen10 | selected_properties_complete | New compositions | 111-111 | -0.007 +/- 0.005 |
| nested_forward10 | listed_elements_only | Random entities | 496-496 | -0.007 +/- 0.002 |
| nested_forward10 | selected_properties_complete | Random entities | 243-320 | -0.023 +/- 0.007 |
| task1_frozen10 | listed_elements_only | Random entities | 496-496 | -0.003 +/- 0.002 |
| task1_frozen10 | selected_properties_complete | Random entities | 317-317 | -0.012 +/- 0.003 |
| nested_forward10 | listed_elements_only | Unseen metal pairs | 496-496 | -0.002 +/- 0.003 |
| nested_forward10 | selected_properties_complete | Unseen metal pairs | 257-276 | -0.034 +/- 0.023 |
| task1_frozen10 | listed_elements_only | Unseen metal pairs | 496-496 | -0.003 +/- 0.003 |
| task1_frozen10 | selected_properties_complete | Unseen metal pairs | 317-317 | -0.013 +/- 0.002 |

Eu/Nb exclusion changes with-BO R2 by less than 0.008 in absolute value. Complete-reference refits reduce R2 by 0.003-0.034, depending on task/model. They retain only 243-320 initial test samples per repeat for nested models (317 for frozen), versus 496 after Eu/Nb exclusion. This is a coverage limitation, not proof that all missing references are harmless. Complete-case R2 values exceeding 0.60 describe a different, smaller test population and must not replace the full-cohort headline scores.

Without-BO refits are also exported. Their Task-3 composition-transfer R2 changes more (up to +0.072 under complete-reference filtering), but the weak baseline does not become a strong composition model. Frozen-model initial-only refits still use a feature specification selected with BO-informed Task-1 data.

[Sensitivity plot](../reviewer_stage4_missing_20261013/missing_reference_sensitivity.png) | [Full matched comparisons and exclusions](../reviewer_stage4_missing_20261013/RESULTS.md)

## Use in the revision

Keep nested CV as the main generalization result and the frozen Task-1 specification as a retrospective consistency check. They use the same predefined splits; the frozen specification was chosen from outcomes later evaluated and is not an independent validation. Include the three-task with/without-BO table and identity/composition baselines from the completed earlier analyses.

Suggested response to the electrochemical-feature concern:

> We recalculated descriptors using experimentally measured ICP molar fractions and separated intrinsic elemental descriptors from electrochemical observables. In matched nested cross-validation with unseen metal pairs, the intrinsic descriptor model achieved R2 = 0.552 +/- 0.007, compared with 0.585 +/- 0.007 after adding current, solution resistance and corrected potential. Applying the historical potential-dependent descriptor shifts alone gave R2 = 0.562 +/- 0.004. Thus, electrochemical observables provide additional predictive information but are not required for the intrinsic model to retain substantial predictive performance. Updated held-out SHAP analyses identify work function as a prominent descriptor while showing that attribution to correlated features depends on the evaluation task. We report the full correlation matrices and missing-reference sensitivities and interpret these associations as predictive rather than mechanistically causal.

Manuscript updates still need to incorporate the new tables, SHAP panels, CV/selection methods, actual cohort counts and reference-missingness policy. Keep Figure 4 local-mean and local-90th-percentile maps clearly distinguished; neither is an independent causal validation. Add the four BO pair identities to both branches where requested, and retain empirical ICP-weighted properties as descriptors rather than measured alloy surface properties. These writing/figure changes have not been silently applied to the manuscript or response document.

Original SHAP reference: Lundberg SM and Lee SI (2017), A Unified Approach to Interpreting Model Predictions, Advances in Neural Information Processing Systems 30. [Publisher record](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).

## Reproduction

From the repository root, use the original Anaconda ML runtime (Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, sklearn 1.6.1, joblib 1.4.2). Stage manifests include the SHAP version where applicable and hashes of analysis code, inputs and original split/model artifacts. Ten workers, one estimator thread each. No targets or original completed runs were changed.

```powershell
$py = 'C:/Users/yibinjiang/anaconda3/python.exe'
$env:LOKY_MAX_CPU_COUNT = '10'
& $py hd_separation_ml/revision_shap_current.py --jobs 10
& $py hd_separation_ml/revision_correlations.py --output hd_separation_ml/outputs/reviewer_stage2_correlations_20261013_v2
& $py hd_separation_ml/revision_operating_ablation.py --jobs 10
& $py hd_separation_ml/revision_missing_references.py --jobs 10
& $py -m unittest discover -s hd_separation_ml -p test_revision_followups.py -v
& $py hd_separation_ml/summarize_revision_followups.py
```

Use new output directories when changing code/data; existing manifests deliberately reject changes. These scripts depend on the frozen upstream CV artifacts in this repository and are not a new standalone training pipeline. If H/D targets change, rerun the upstream nested/frozen workflows before these follow-ups. The first stage-2 folder without the v2 suffix is superseded; use v2, which correctly reads the turning-point estimate column.

Verification: 221 dependency files checked; 360 restricted/baseline test-set pairs independently verified; SHAP reconstruction error 9.2e-14; repeat metrics independently recomputed to 2.2e-16. Ten focused unit tests passed. Scientific figures were visually reviewed.
