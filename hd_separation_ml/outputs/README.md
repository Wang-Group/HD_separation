# Machine-Learning Outputs

This folder stores generated outputs from the active ML notebooks.

## Active Output Folders

- `feature_selection_561_rs_from_eis_no_eis_fitting/`
  - Feature-selection history, selected-feature tables, comparison plots, and
    the 561-sample feature matrix used during selection.
- `final_model_8features_561_seed17_rs_from_eis/`
  - Final model outputs for the 8-feature seed-17 model.
  - Includes prediction plot, model metrics, feature importance, SHAP plots,
    and the final feature matrix.
- `md_bond_ablation_561_seed17/`
  - Additional descriptor-ablation checks involving M-D bond features.
- `data_audit/`
  - Notebook rerun logs and data-audit helper outputs.

These folders are generated analysis outputs, not raw data. Raw electrochemical
inputs are stored under `../data_extraction/all_data/`.
