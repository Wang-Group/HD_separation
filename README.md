# Data and Code for Electrochemical H/D Separation

This repository contains the data tables, analysis notebooks, machine-learning
workflow, Bayesian-optimization records, and platform-control code associated
with the H/D separation study.

The repository is organized for manuscript reproducibility and later reuse. It
is not a single installable software package. Figure notebooks and ML notebooks
can be inspected and rerun from their local folders; the hardware-control code
is included for provenance and should only be run in the intended lab setup.

## Repository Map

```text
.
|-- DATA_MANIFEST.md          # current figure/data map
|-- 0-Figures/                # figure-specific data, notebooks, and outputs
|   |-- Figure1/
|   |-- Figure3/
|   `-- Figure4/
|-- hd_separation_ml/         # curated ML data extraction and model notebooks
|   |-- data_extraction/
|   |-- notebooks/
|   `-- outputs/
`-- EC_tool_sys/              # platform automation and BO provenance code
    |-- BO_PROVENANCE.md
    |-- botorch_test/
    `-- manager/
```

## Current Canonical Data

- `0-Figures/Figure1/data/figure1_bottom_this_work_from_all_data.csv`
  - Current this-work table for the Figure 1 bottom comparison panel.
  - 709 rows: 561 initial-screening samples, 112 four-system Figure 4 BO
    samples, and 36 extra Co-Fe follow-up samples.
  - LSV values are extracted from
    `hd_separation_ml/data_extraction/all_data/LSV`.
  - Figure 1 reports eta10 within a bounded plotting window of 10-550 mV.
    Exact crossings, recovered near-target Co-V eta estimates, and
    turning-point eta estimates are eligible for this-work Pareto-front
    decisions. Non-HER/background-current cases at the 10 mV boundary,
    right-censored non-reachers, and charging-current forced-cap cases are
    shown but excluded.
- `hd_separation_ml/outputs/feature_selection_561_rs_from_eis_no_eis_fitting/feature_matrix_561_no_eis_fitting.csv`
  - Current no-BO ML feature matrix used by the H/D-separation workflow.
  - 561 initial-screening samples.
- `0-Figures/Figure4/curated/figure4_bo_complete_112.csv`
  - Current Figure 4 BO table.
  - 112 rows: 28 completed experiments each for Co-Mo, Co-V, Co-Ni, and
    Fe-Ni.

See `DATA_MANIFEST.md` for a more detailed figure-by-figure map.

## Figure Reproduction

### Figure 1

- Script: `0-Figures/Figure1/plot_figure1_censored_eta10.py`
- Main inputs:
  - `0-Figures/Figure1/data/figure1_bottom_this_work_from_all_data.csv`
  - `0-Figures/Figure1/data/figure1_bottom_literature_benchmarks.csv`
  - `0-Figures/Figure1/data/figure1_this_work_eta10_censoring.csv`
- Main outputs:
  - `0-Figures/Figure1/outputs/pareto_plot_v2_censored_eta10.png`
  - `0-Figures/Figure1/outputs/pareto_plot_v2_censored_eta10_no_text.png`
- LSV audit outputs:
  - `0-Figures/Figure1/outputs/eta_below_10_lsv/`
  - `0-Figures/Figure1/outputs/right_censored_middle_lsv/`

### Figure 3

Top panel:

- Notebook: `0-Figures/Figure3/notebooks/visualizewithcolors.ipynb`
- Derived data: `0-Figures/Figure3/data/figure3_top_from_ml_dataset.csv`
- Output: `0-Figures/Figure3/outputs/before_BO_Fe_Co_Others.png`

Machine-learning panels:

- Feature-matrix notebook:
  `hd_separation_ml/notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb`
- Current script-based H/D-separation workflow:
  `hd_separation_ml/current_hd_separation_workflow/README.md`
- Final H/D-separation outputs:
  `hd_separation_ml/outputs/final_model_10features_561_seed42_rf_et_only/`
- Separate eta10 activity outputs:
  `hd_separation_ml/outputs/03_eta10_activity_model_raw_descriptors/`

### Figure 4

- Folder: `0-Figures/Figure4/`
- Pareto/IoH notebook: `0-Figures/Figure4/pareto_polygon_enclosure_code.ipynb`
- Descriptor-map SI script: `0-Figures/Figure4/assemble_figure_S15_descriptor_panels.py`
- Descriptor-overlay script: `0-Figures/Figure4/plot_top3_shap_descriptor_bo_overlay.py`
- Curated BO table: `0-Figures/Figure4/curated/figure4_bo_complete_112.csv`
- Current Pareto/IoH export:
  `0-Figures/Figure4/eight_panels_2x4_batch1_7_lg.png`
- Descriptor outputs:
  `0-Figures/Figure4/outputs/figure_S15_descriptor_space_interpretation_BO_outcomes.png`

## Machine Learning

The current ML workflow is documented in `hd_separation_ml/README.md`.

Recommended notebook order:

1. `hd_separation_ml/notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb`
   - Builds the current 561-sample no-BO feature matrix.
   - Avoids EIS-fitting descriptors; Rs is estimated directly from EIS.
2. `hd_separation_ml/notebooks/03_eta10_activity_model_raw_descriptors.ipynb`
   - Separate eta10 HER-activity model using LSV-derived eta10 values and raw
     element-level descriptors.

The current manuscript-facing H/D-separation model is script-based:

```powershell
cd hd_separation_ml
.\current_hd_separation_workflow\run_current_hd_workflow.ps1
```

Current checked final model:

- dataset: 561 no-BO initial-screening samples;
- H/D-separation descriptor set: 10 features;
- H/D-separation best model: Random Forest;
- H/D-separation test R2: 0.6061;
- H/D-separation test RMSE: 1.0576.

The eta10 HER-activity analysis is separate from the H/D-separation model:

- dataset after eta10 filtering: 319 samples;
- descriptor set: 9 features;
- compared models: Random Forest and Extra Trees;
- best model: Random Forest;
- test R2: 0.6499;
- test RMSE: 96.0 mV.
- artifact sensitivity check:
  `hd_separation_ml/eta10_artifact_sensitivity_check.py`, with outputs in
  `hd_separation_ml/outputs/03_eta10_activity_model_raw_descriptors/sensitivity_eta10_artifacts/`;
  stricter eta10 artifact filters gave stable performance and feature-importance
  rankings.

## Bayesian Optimization and Automation

The BO/platform provenance is documented in:

- `EC_tool_sys/BO_PROVENANCE.md`
- `EC_tool_sys/manager/README.md`
- `EC_tool_sys/botorch_test/README.md`

The manuscript-relevant active-learning helper is:

```python
generate_new_exp_MO_one_system_V2
```

It uses BoTorch multi-objective Bayesian optimization with
`lsv_overpotential_-0.01` and `separation_coefficient` as the objectives.

## Electrochemical Data Notes

- The active LSV, EIS, CV20, and Et files are under
  `hd_separation_ml/data_extraction/all_data/`.
- The separation-coefficient table is
  `hd_separation_ml/data_extraction/separation_coefficient.xlsx`.
- For LSV traces that cross a target current more than once, figure-generation
  notebooks use the conservative crossing: the most negative raw potential.
- For Figure 1 traces that do not reach -10 mA cm^-2, near-target endpoints
  within 2 mA cm^-2 are plotted at the measured endpoint overpotential; the
  rest are assigned to the 550 mV cap.
- Traces already beyond -10 mA cm^-2 at the scan start are plotted at 10 mV as
  non-HER/background-current cases and are excluded from this-work Pareto-front
  calculation unless a near-target Co-V estimate or turning-point eta estimate
  is explicitly recovered.
- Recovered near-target and turning-point estimates require the measured or
  selected turning-point current to be within 10 mA cm^-2 of -10 mA cm^-2 and
  the LSV shape to be interpretable. `Fe-Pt`, `Pd-Pd`, `Y-Pd`, and `Co-Pt` use
  turning-point eta estimates; `Co-Fe29`, `Co-Fe33`, `Sc-Pd`, and `W-Pd`
  remain non-HER lower-bound cases and are excluded from Pareto calculations.
- `Sc-Pb`, `Cd-Y`, `W-Pb`, and `Ce-Pb` are treated as charging-current artifacts
  and forced to the 550 mV cap.
- `hd_separation_ml/data_extraction/all_data/LSV/lsv_geometry_area_metadata.csv`
  records the initial-screening samples measured with 2 cm^2 geometry.

## Hardware Safety

`EC_tool_sys/manager/` contains hardware-facing automation code. It assumes a
specific Windows lab environment with configured COM ports, instrument drivers,
vendor software, robotic hardware, and mass-spectrometry control. Do not run
platform-manager scripts unless the hardware setup and safety procedures have
been verified.

## Contact

For questions about the data, code, or manuscript context, please contact:

- Yibin Jiang: `yibinjiang@xmu.edu.cn`
- Cheng Wang: `wangchengxmu@xmu.edu.cn`
