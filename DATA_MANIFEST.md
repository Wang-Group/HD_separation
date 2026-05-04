# Data Manifest

This file summarizes the active data organization for the figure and
machine-learning analyses. It is intended as a compact map for readers who want
to reproduce the current plots without inspecting every notebook.

## Canonical Active Tables

### Figure 1 this-work table

- Path: `0-Figures/Figure1/data/figure1_bottom_this_work_from_all_data.csv`
- Rows: 709
- Scope:
  - 561 no-BO initial-screening samples.
  - 112 four-system Figure 4 BO samples.
  - 36 extra Co-Fe follow-up samples.
- Key columns:
  - `sample`
  - `plot_x_overpotential_mV`
  - `eta10_mV_vs_RHE_full_lsv`
  - `separation_coefficient`
  - `LSV_geometry_area_cm2`
  - `target_current_A_raw`
- Notes:
  - LSV values are extracted from
    `hd_separation_ml/data_extraction/all_data/LSV`.
  - Separation coefficients are aligned to
    `hd_separation_ml/data_extraction/separation_coefficient.xlsx`.
  - Multi-crossing LSV traces use the most negative crossing for conservative
    overpotential reporting.

### Machine-learning feature matrix

- Path:
  `hd_separation_ml/outputs/final_model_8features_561_seed17_rs_from_eis/feature_matrix_8features.csv`
- Rows: 561
- Scope: current no-BO initial-screening samples used by the final ML model.
- Key fields:
  - `sample`
  - `E_actual`
  - `Rs_from_EIS`
  - `I_raw`
  - `separation_coefficient`
  - selected descriptor columns.

### Figure 4 BO table

- Path: `0-Figures/Figure4/curated/figure4_bo_complete_112.csv`
- Rows: 112
- Scope:
  - `Co-Mo`: 28 rows.
  - `Co-V`: 28 rows.
  - `Co-Ni`: 28 rows.
  - `Fe-Ni`: 28 rows.
- Purpose: manuscript-facing table for Figure 4 Pareto/IoH plots.

## Figure Mapping

### Figure 1, bottom panel

- Notebook: `0-Figures/Figure1/notebooks/v4.ipynb`
- Inputs:
  - `0-Figures/Figure1/data/figure1_bottom_this_work_from_all_data.csv`
  - `0-Figures/Figure1/data/figure1_bottom_literature_benchmarks.csv`
- Output: `0-Figures/Figure1/outputs/pareto_plot_v2.png`
- Dataset scope:
  - 709 this-work points.
  - 76 literature-reference points.
  - 2 benchmark points: Ni foam and Pt sheet.
  - 787 total plotted points.

### Figure 3, top panel

- Notebook: `0-Figures/Figure3/notebooks/visualizewithcolors.ipynb`
- Derived data: `0-Figures/Figure3/data/figure3_top_from_ml_dataset.csv`
- Output: `0-Figures/Figure3/outputs/before_BO_Fe_Co_Others.png`
- Dataset scope: 561 no-BO initial-screening samples from the current ML
  feature matrix.
- LSV handling:
  - target: -10 mA cm^-2 under the appropriate geometry convention.
  - multi-crossing traces: use the most negative crossing.
  - 2 cm^2 geometry exceptions are read from
    `hd_separation_ml/data_extraction/all_data/LSV/lsv_geometry_area_metadata.csv`.

### Figure 3, ML panels

- Feature-selection notebook:
  `hd_separation_ml/notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb`
- Final-model notebook:
  `hd_separation_ml/notebooks/02_final_model_8features_561_seed17_rs_from_eis.ipynb`
- Final output folder:
  `hd_separation_ml/outputs/final_model_8features_561_seed17_rs_from_eis/`
- Dataset scope: 561 no-BO initial-screening samples.
- Final feature set:
  - `I_raw`
  - `work_function_element2`
  - `MM_bond_element2`
  - `FE_ads_avg_element1`
  - `delta_EOH_element1`
  - `delta_EOH_element2`
  - `electronegativity_element1`
  - `electronegativity_element2`

### Figure 4

- Folder: `0-Figures/Figure4`
- Main notebook:
  `0-Figures/Figure4/pareto_polygon_enclosure_code.ipynb`
- Descriptor-map notebook:
  `0-Figures/Figure4/BO_analysis -2D_symbolic.ipynb`
- Primary table:
  `0-Figures/Figure4/curated/figure4_bo_complete_112.csv`
- Descriptor table:
  `0-Figures/Figure4/curated/figure4_bo_weighted_ml_top3_descriptors.csv`
- Current Pareto/IoH output:
  `0-Figures/Figure4/eight_panels_2x4_batch1_7_lg.png`

## BO Count Note

The complete this-work Figure 1 table contains 148 digit-suffixed BO/follow-up
samples: 112 from the four Figure 4 BO systems and 36 extra Co-Fe samples.
Figure 4 uses only the four BO systems, so its plotted BO table has 112 rows.

## Automation / BO Code Provenance

The Figure 4 BO algorithm path is documented in:

- `EC_tool_sys/BO_PROVENANCE.md`
- `EC_tool_sys/manager/platform_manager_HD_Hcell.py`
- `EC_tool_sys/botorch_test/botorch_tool/active_learning.py`

The manuscript-relevant function is `generate_new_exp_MO_one_system_V2`, which
uses `lsv_overpotential_-0.01` and `separation_coefficient` as objectives with
BoTorch `qNoisyExpectedHypervolumeImprovement`.
