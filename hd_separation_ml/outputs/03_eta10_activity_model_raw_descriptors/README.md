# Eta10 Activity Model Outputs

This folder contains the current manuscript-facing HER-activity machine-learning
analysis using eta10 as the target.

## Source Notebook

`../../notebooks/03_eta10_activity_model_raw_descriptors.ipynb`

## Target

`eta10_mV` is extracted from LSV curves at -10 mA cm^-2 and converted to the RHE
scale under alkaline conditions. The documented 2 cm^2 geometry exceptions use
the raw -20 mA crossing to remain comparable to -10 mA cm^-2 under the standard
1 cm^2 convention.

Samples with unreliable crossings are not modeled. Samples with `eta10 < 20 mV`
are excluded from the activity-modeling step as low-outlier cases.

## Artifact Sensitivity

The eta10 workflow was checked against LSV artifacts identified during the
Figure 1 eta10 audit. The existing filters already remove the major artifact
classes from the eta10 ML target: samples without true -10 mA cm^-2 crossings
are not modeled, and low-eta outliers (`eta10 < 20 mV`) are excluded before
training.

A sensitivity check in `sensitivity_eta10_artifacts/` compared the current
319-sample dataset against stricter variants that remove multi-crossing traces
and/or raise the eta10 lower threshold to 50 mV. With the current 9 selected
features fixed, the Random Forest model remained stable:

- Baseline current dataset: test R2 = 0.650, test RMSE = 95.9 mV.
- Single-crossing only: test R2 = 0.681, test RMSE = 88.8 mV.
- eta10 >= 50 mV: test R2 = 0.720, test RMSE = 82.3 mV.
- eta10 >= 50 mV and single-crossing only: test R2 = 0.730, test RMSE = 80.8 mV.

Random Forest feature-importance rankings were also stable relative to the
baseline (Spearman rank correlation >= 0.967 across the stricter variants).
Thus the eta10 ML conclusions are robust to the identified LSV/eta10 artifacts,
and the earlier true-crossing plus low-eta filtering successfully removes the
artifact classes most likely to bias the analysis.

## Dataset

- Valid true eta10 crossings before filtering: 323 samples.
- Excluded low-outlier samples: 4 samples.
- Final modeled dataset: 319 samples.
- Train/test split: 255 / 64 samples.

## Current Model

Only Random Forest and Extra Trees are compared in the current eta10 workflow.
XGBoost is not included because it did not add interpretive value for this
activity analysis.

Current best model:

- Random Forest.
- Test R2 = 0.6499.
- Test RMSE = 96.0 mV.

Current selected features:

- `Rs_from_EIS`
- `work_function_element1`
- `work_function_element2`
- `pzc_element2`
- `E_std_element2`
- `FE_ads_avg_element1`
- `FE_ads_avg_element2`
- `MH_bond_kJ_element2`
- `delta_ZPE_H_element1`

## Main Files

- `summary_eta10_raw_descriptor_model.json`: compact machine-readable summary.
- `eta10_raw_descriptor_model_results.csv`: model comparison table.
- `selected_features_eta10_raw_descriptors.csv`: final selected feature list.
- `eta10_raw_descriptor_feature_importance.csv`: mean absolute SHAP values used
  for Figure S9, with native and permutation importance retained only as
  diagnostic columns.
- `eta10_raw_descriptor_shap_importance.csv`: compact SHAP-only importance
  table for the final model.
- `eta10_raw_descriptor_test_predictions.csv`: held-out test predictions.
- `feature_matrix_561_with_eta10_raw_electronic_descriptors.csv`: working
  feature matrix with extracted eta10 values.
- `eta10_extracted_from_lsv_561.csv`: eta10 values extracted from LSV traces.
- `excluded_eta10_lt_20mV.csv`: low-outlier samples excluded from modeling.
- `sensitivity_eta10_artifacts/`: artifact sensitivity metrics, feature
  stability tables, and summary figure.

## Main Figures

- `figure_S9_eta10_activity_ml_summary.png`: stable SI Figure S9 export.
- `eta10_activity_model_summary_ABCD.png`: source combined A-D summary from the
  eta10 notebook.
- `eta10_raw_descriptor_prediction_residual.png`: prediction and residual
  panels.
- `eta10_raw_descriptor_feature_importance.png`: final SHAP feature-importance
  panel using averaged absolute SHAP values.
- `eta10_workfunction1_workfunction2_distribution.png`: eta10 distribution in
  raw work-function descriptor space.
- `raw_selected_features_vs_eta10_scatter_grid.png`: selected feature scatter
  grid against eta10.
