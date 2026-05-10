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
- `eta10_raw_descriptor_feature_importance.csv`: native and permutation
  importance values for the final model.
- `eta10_raw_descriptor_test_predictions.csv`: held-out test predictions.
- `feature_matrix_561_with_eta10_raw_electronic_descriptors.csv`: working
  feature matrix with extracted eta10 values.
- `eta10_extracted_from_lsv_561.csv`: eta10 values extracted from LSV traces.
- `excluded_eta10_lt_20mV.csv`: low-outlier samples excluded from modeling.

## Main Figures

- `figure_S9_eta10_activity_ml_summary.png`: stable SI Figure S9 export.
- `eta10_activity_model_summary_ABCD.png`: source combined A-D summary from the
  eta10 notebook.
- `eta10_raw_descriptor_prediction_residual.png`: prediction and residual
  panels.
- `eta10_raw_descriptor_feature_importance.png`: final feature-importance
  panel.
- `eta10_workfunction1_workfunction2_distribution.png`: eta10 distribution in
  raw work-function descriptor space.
- `raw_selected_features_vs_eta10_scatter_grid.png`: selected feature scatter
  grid against eta10.
