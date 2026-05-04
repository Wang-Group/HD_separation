# Machine-Learning Notebooks

This folder contains the active ML notebooks for the current 561-sample no-BO
dataset.

## Recommended Order

1. `01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb`
   - Builds the current feature matrix.
   - Estimates Rs from EIS raw files.
   - Runs feature selection without EIS-fitting descriptors.
2. `02_final_model_8features_561_seed17_rs_from_eis.ipynb`
   - Uses the selected 8-feature descriptor set.
   - Trains the final seed-17 model.
   - Generates prediction, feature-importance, and SHAP outputs.

Outputs are written under `../outputs/`.
