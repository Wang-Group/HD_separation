# H/D Separation Machine Learning

This folder contains the corrected data extraction inputs, machine-learning
feature matrix, and the current manuscript-facing H/D-separation descriptor
workflow for the no-BO initial-screening dataset.

## Current Scope

- Dataset: 561 no-BO initial-screening samples.
- Main target: `separation_coefficient`.
- Electrochemical input: current and potential information extracted from the
  raw files in `data_extraction/all_data/`.
- Rs handling: Rs is estimated directly from high-frequency EIS real impedance;
  precomputed EIS fitting tables are not required for the current H/D
  separation model.
- Current final H/D model: 10-feature Random Forest / Extra Trees workflow with
  random seed 42.

## Current Manuscript-Facing Workflow

The current H/D-separation workflow is documented here:

`current_hd_separation_workflow/README.md`

It consists of:

1. all-28 high-separation classification feature screening;
2. RF/ExtraTrees-only forward feature addition;
3. final 10-feature model interpretation using SHAP;
4. BO overlay in descriptor space using electronegativity, work-function, and H
   adsorption descriptor families.

Run the workflow wrapper:

```powershell
.\current_hd_separation_workflow\run_current_hd_workflow.ps1
```

or run the scripts individually:

```powershell
& C:\Users\yibinjiang\anaconda3\python.exe .\bruteforce_classification_then_forward_regression_seed42_all28pool.py
& C:\Users\yibinjiang\anaconda3\python.exe .\forward_regression_seed42_all28pool_rf_et_only.py
& C:\Users\yibinjiang\anaconda3\python.exe .\plot_10feature_seed42_rf_et_shap_importance.py
```

`feature_selection_workflow_core.py` contains shared helper functions for the
classification-combination search and descriptor-family weighting.

The BO overlay step is run from `../0-Figures/Figure4/`:

```powershell
& C:\Users\yibinjiang\anaconda3\python.exe .\plot_top3_shap_descriptor_bo_overlay.py
```

## Active Inputs

- `EH_extra.xlsx`: corrected elemental descriptor table.
- `data_extraction/all_data/LSV/`: LSV raw curves.
- `data_extraction/all_data/EIS/`: EIS raw curves used for Rs estimation.
- `data_extraction/all_data/CV20/`: CV raw curves retained for provenance.
- `data_extraction/all_data/Et/`: chrono/potential traces retained for
  electrochemical provenance.
- `data_extraction/separation_coefficient.xlsx`: separation-coefficient labels.

The LSV folder contains `lsv_geometry_area_metadata.csv`, which records the
initial-screening samples measured with 2 cm^2 geometry.

## Active Feature Matrix

The current workflow starts from:

`outputs/feature_selection_561_rs_from_eis_no_eis_fitting/feature_matrix_561_no_eis_fitting.csv`

and the corresponding candidate descriptor list:

`outputs/feature_selection_561_rs_from_eis_no_eis_fitting/candidate_features.csv`

The notebook retained for feature-matrix provenance is:

`notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb`

## Current H/D Separation Outputs

Classification feature screening from all 28 candidate descriptors is
reproducible from:

`bruteforce_classification_then_forward_regression_seed42_all28pool.py`

The full 1- to 4-feature combination table is treated as a regenerable
intermediate output and is not retained in the cleaned manuscript snapshot. The
selected four-feature core used for forward addition is recorded in:

`outputs/06c_forward_regression_seed42_all28pool_rf_et_only/starting_core4_features.csv`

RF/ExtraTrees-only forward addition:

`outputs/06c_forward_regression_seed42_all28pool_rf_et_only/`

Final 10-feature model and SHAP:

`outputs/final_model_10features_561_seed42_rf_et_only/`

Main combined H/D-separation ML figure export:

`outputs/final_model_10features_561_seed42_rf_et_only/figure_S8_hd_separation_ml_summary.png`

Current final 10-feature set:

- `Frequency_H_element1`
- `Frequency_D_element1`
- `delta_ZPE_H_element1`
- `electronegativity_element1`
- `work_function_element2`
- `delta_EOH_element2`
- `FE_ads_avg_element1`
- `work_function_element1`
- `E_std_element2`
- `MM_bond_element2`

Current held-out test performance:

- Random Forest: test R2 = 0.6061, test RMSE = 1.0576.
- Extra Trees: test R2 = 0.5855, test RMSE = 1.0849.

Use SHAP as the main descriptor-importance interpretation. Tree impurity feature
importance is retained only as an internal diagnostic because it is less reliable
for correlated descriptors.

## Eta10 Activity Analysis

The HER-activity model is separate from the H/D-separation workflow:

`notebooks/03_eta10_activity_model_raw_descriptors.ipynb`

Output folder:

`outputs/03_eta10_activity_model_raw_descriptors/`

This analysis uses LSV-derived `eta10_mV`, excludes unreliable crossings and
`eta10 < 20 mV` outliers, and uses raw elemental descriptors without the
sample-specific potential offset used in older H/D-separation models.

Current eta10 dataset and model:

- valid eta10 crossings before filtering: 323 samples;
- excluded low-outlier samples: 4 samples with `eta10 < 20 mV`;
- final activity-model dataset: 319 samples;
- compared models: Random Forest and Extra Trees only;
- best model: Random Forest;
- test R2 = 0.6499, test RMSE = 96.0 mV.

Current eta10 feature set:

- `Rs_from_EIS`
- `work_function_element1`
- `work_function_element2`
- `pzc_element2`
- `E_std_element2`
- `FE_ads_avg_element1`
- `FE_ads_avg_element2`
- `MH_bond_kJ_element2`
- `delta_ZPE_H_element1`

Main combined eta10 figure export:

`outputs/03_eta10_activity_model_raw_descriptors/figure_S9_eta10_activity_ml_summary.png`

## Archived Legacy H/D ML Attempts

Earlier 7-, 8-, 9-, and 11-feature H/D-separation notebooks, grouped-feature
experiments, chemistry-constrained trials, and old helper scripts were moved to:

`backups/legacy_hd_separation_ml_before_current_workflow_20260509/`

Only the legacy notebooks, scripts, and small logs are retained for provenance.
Old generated output folders from those abandoned workflows were removed from
the cleaned manuscript snapshot.

Additional exploratory root-level scripts that are no longer part of the active
workflow were moved to:

`backups/unused_root_scripts_20260510/`
