# H/D Separation Machine Learning

This folder contains the current machine-learning data extraction, feature
selection, model training notebooks, and checked outputs for the no-BO
initial-screening dataset.

## Current Scope

- Dataset: 561 no-BO initial-screening samples.
- Target: `separation_coefficient`.
- Electrochemical input: current and potential information extracted from the
  raw files in `data_extraction/all_data/`.
- Rs handling: Rs is estimated directly from EIS raw curves for the current
  workflow; precomputed EIS fitting tables are not required.
- Final model: 8-feature Extra Trees model with random seed 17.

## Active Notebooks

Run notebooks from this folder's parent repository root or from the notebook
folder so relative paths resolve.

1. `notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb`
   - Builds the 561-sample feature matrix.
   - Estimates Rs from EIS.
   - Performs feature selection without EIS-fitting descriptors.
   - Writes outputs to
     `outputs/feature_selection_561_rs_from_eis_no_eis_fitting/`.
2. `notebooks/02_final_model_8features_561_seed17_rs_from_eis.ipynb`
   - Trains the final fixed 8-feature model with random seed 17.
   - Writes outputs to
     `outputs/final_model_8features_561_seed17_rs_from_eis/`.

## Final Feature Set

The final model uses:

- `I_raw`
- `work_function_element2`
- `MM_bond_element2`
- `FE_ads_avg_element1`
- `delta_EOH_element1`
- `delta_EOH_element2`
- `electronegativity_element1`
- `electronegativity_element2`

Descriptor display names are saved with the model outputs where applicable,
for example in
`outputs/final_model_8features_561_seed17_rs_from_eis/feature_importance.csv`.

## Current Final Model Output

Main output folder:

`outputs/final_model_8features_561_seed17_rs_from_eis/`

Important files:

- `feature_matrix_8features.csv`: final 561-sample feature matrix.
- `selected_features.csv`: final descriptor list.
- `final_ml_I_raw_results.csv`: model metrics and tuned parameters.
- `final_prediction_I_raw.png`: predicted-vs-observed plot.
- `feature_importance.csv` and `feature_importance.png`: model feature
  importance.
- `final_shap_I_raw/`: SHAP plots for test-set and all-sample views.

Current checked Extra Trees performance:

- CV R2: 0.5661
- test R2: 0.6676
- test RMSE: 0.9758

## Data Extraction Inputs

- `data_extraction/all_data/LSV/`: LSV raw curves.
- `data_extraction/all_data/EIS/`: EIS raw curves used for Rs estimation.
- `data_extraction/all_data/CV20/`: CV raw curves retained for provenance.
- `data_extraction/all_data/Et/`: chrono/potential traces retained for
  electrochemical provenance.
- `data_extraction/separation_coefficient.xlsx`: separation-coefficient labels.

The LSV folder contains `lsv_geometry_area_metadata.csv`, which records the
initial-screening samples measured with 2 cm^2 geometry.

## Output Folders

- `outputs/data_audit/`: notebook rerun logs and data-audit helpers.
- `outputs/feature_selection_561_rs_from_eis_no_eis_fitting/`: feature
  selection results.
- `outputs/final_model_8features_561_seed17_rs_from_eis/`: final model
  results.
- `outputs/md_bond_ablation_561_seed17/`: descriptor-ablation checks involving
  M-D bond features.
