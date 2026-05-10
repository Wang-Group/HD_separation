# RF/ExtraTrees-Only Forward Addition

This folder contains a forward-addition regression analysis that deliberately
excludes XGBoost from the feature-selection decision.

## Rationale

XGBoost showed less stable held-out behavior for the small, highly correlated
descriptor set. To reduce sensitivity to model-specific overfitting, this
workflow starts from the all-28 high-separation classification core and adds
features using only the mean 5-fold CV R2 of Random Forest and Extra Trees.

## Starting Core

The starting 4-feature core is inherited from the all-28 high-separation
classification search:

- `Frequency_H_element1`
- `Frequency_D_element1`
- `delta_ZPE_H_element1`
- `electronegativity_element1`

## Feature-Addition Path

The RF/ExtraTrees-only forward path added:

1. `work_function_element2`
2. `delta_EOH_element2`
3. `FE_ads_avg_element1`
4. `work_function_element1`
5. `E_std_element2`
6. `MM_bond_element2`
7. `pzc_element1`
8. `I_raw`
9. `E_std_element1`
10. `MH_bond_kJ_element2`
11. `MH_bond_kJ_element1`
12. `MD_bond_kJ_element1`

The mean RF/ExtraTrees 5-fold CV R2 improved from about 0.440 for the 4-feature
core to about 0.564 at 16 features. The held-out test R2 was more stable than in
the XGBoost-including workflow, with the best Random Forest test R2 around
0.63.

## Main Files

- `forward_regression_selected_path_summary.csv`: selected forward path and CV
  performance.
- `forward_regression_selected_path_by_model.csv`: model-level CV performance
  for each selected feature set.
- `forward_regression_candidate_trials_by_model.csv`: all candidate trials at
  each addition step.
- `heldout_test_results_by_path.csv`: held-out test performance for each
  selected feature set using Random Forest and Extra Trees.
- `forward_regression_rf_et_only_cv_test_curve.png`: CV and held-out test
  performance curve.

