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
- `hd_separation_ml/outputs/final_model_8features_561_seed17_rs_from_eis/feature_matrix_8features.csv`
  - Current no-BO ML feature matrix.
  - 561 initial-screening samples.
- `0-Figures/Figure4/curated/figure4_bo_complete_112.csv`
  - Current Figure 4 BO table.
  - 112 rows: 28 completed experiments each for Co-Mo, Co-V, Co-Ni, and
    Fe-Ni.

See `DATA_MANIFEST.md` for a more detailed figure-by-figure map.

## Figure Reproduction

### Figure 1

- Notebook: `0-Figures/Figure1/notebooks/v4.ipynb`
- Main inputs:
  - `0-Figures/Figure1/data/figure1_bottom_this_work_from_all_data.csv`
  - `0-Figures/Figure1/data/figure1_bottom_literature_benchmarks.csv`
- Output: `0-Figures/Figure1/outputs/pareto_plot_v2.png`

### Figure 3

Top panel:

- Notebook: `0-Figures/Figure3/notebooks/visualizewithcolors.ipynb`
- Derived data: `0-Figures/Figure3/data/figure3_top_from_ml_dataset.csv`
- Output: `0-Figures/Figure3/outputs/before_BO_Fe_Co_Others.png`

Machine-learning panels:

- Final notebook:
  `hd_separation_ml/notebooks/02_final_model_8features_561_seed17_rs_from_eis.ipynb`
- Final outputs:
  `hd_separation_ml/outputs/final_model_8features_561_seed17_rs_from_eis/`

### Figure 4

- Folder: `0-Figures/Figure4/`
- Pareto/IoH notebook: `0-Figures/Figure4/pareto_polygon_enclosure_code.ipynb`
- Descriptor-map notebook: `0-Figures/Figure4/BO_analysis -2D_symbolic.ipynb`
- Curated BO table: `0-Figures/Figure4/curated/figure4_bo_complete_112.csv`
- Current Pareto/IoH export:
  `0-Figures/Figure4/eight_panels_2x4_batch1_7_lg.png`
- Descriptor outputs: `0-Figures/Figure4/outputs/`

## Machine Learning

The current ML workflow is documented in `hd_separation_ml/README.md`.

Recommended notebook order:

1. `hd_separation_ml/notebooks/01_feature_selection_561_rs_from_eis_no_eis_fitting.ipynb`
   - Feature selection on the current 561-sample no-BO dataset.
   - Avoids EIS-fitting descriptors; Rs is estimated directly from EIS.
2. `hd_separation_ml/notebooks/02_final_model_8features_561_seed17_rs_from_eis.ipynb`
   - Final 8-feature model using random seed 17.

Current checked final model:

- dataset: 561 no-BO initial-screening samples;
- descriptor set: 8 features;
- best model: Extra Trees;
- test R2: 0.6676;
- test RMSE: 0.9758.

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
