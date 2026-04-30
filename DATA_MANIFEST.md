# Data Manifest

This file summarizes the current data organization for manuscript figures and
machine-learning analysis. It is intended to make the figure/data relationship
explicit without requiring readers to inspect every notebook.

## Canonical Tables

### Full corrected sample table

- Path: `all_samples_708_corrected.csv`
- Rows: 708
- Key columns: `sample`, `E_actual (V)`, `Rs (Ohm)`, `Potential (V)`,
  `separation_coefficient`
- Interpretation:
  - 560 no-BO initial-screening samples have names without a trailing digit.
  - 148 digit-suffixed BO/follow-up samples are present:
    - `Co-Fe`: 36
    - `Co-Mo`: 28
    - `Co-Ni`: 28
    - `Co-V`: 28
    - `Fe-Ni`: 28
- This is the repo-level corrected table for Figure 1 and the active Figure 3
  top-panel notebook.

### Machine-learning label table

- Path: `hd_separation_ml/sample_overpotentials_new_corrected_verison.csv`
- Rows: 707
- Interpretation:
  - 560 no-BO initial-screening rows.
  - 147 digit-suffixed BO/follow-up rows.
- This table is used by the ML notebooks together with EIS/LSV/descriptor data.

### Final no-BO ML snapshot

- Path: `hd_separation_ml/datasets/final_no_BO_560.csv`
- Rows: 560
- Purpose: stable snapshot of the corrected no-BO dataset used by the final
  fixed 12-feature ML model.

## Figure Mapping

### Figure 1, bottom panel

- Notebook: `0-Figures/Figure1/notebooks/v4.ipynb`
- Inputs:
  - `0-Figures/Figure1/data/all_data_corrected.xlsx` for literature and
    benchmark points.
  - `all_samples_708_corrected.csv` for all `This work` points.
- Output: `0-Figures/Figure1/outputs/pareto_plot_v2.png`
- Dataset scope:
  - 708 `This work` points from the corrected repo-level table.
  - 76 literature-reference points.
  - 2 benchmark points: `Ni foam` and `Pt sheet`.
  - 786 total plotted points.
- Note: `0-Figures/Figure1/data/figure1_bottom_plot_points.csv` is retained as
  an audit copy from the earlier workbook-derived extraction and may not reflect
  the current 708-point rebuild unless regenerated.

### Figure 3, top panel

- Notebook: `0-Figures/Figure3/notebooks/visualizewithcolors.ipynb`
- Main input: `all_samples_708_corrected.csv`
- Output: `0-Figures/Figure3/outputs/before_BO_Fe_Co_Others.png`
- Dataset scope: initial screening only.
- Filtering rule:
  - split `sample` by `-`;
  - keep rows where the second component contains only letters;
  - remove digit-suffixed BO/follow-up rows.
- Current count:
  - 708 rows before filtering.
  - 560 initial-screening rows after filtering.

### Figure 3, ML panels

- Final notebook: `hd_separation_ml/notebooks/03_final_model_fixed12_560.ipynb`
- Final output folder: `hd_separation_ml/outputs/final_model_fixed12_560/`
- Dataset scope: corrected no-BO initial-screening ML matrix.
- Current final ML count: 560 samples.
- Feature-selection provenance:
  - The 12-feature descriptor set is retained from the legacy 558-sample
    feature-selection workflow.
  - The fixed 12-feature model was then evaluated on the corrected 560-sample
    dataset.
- Figure-local copies are stored in:
  - `0-Figures/Figure3/outputs/ml/`
  - `0-Figures/Figure3/data/ml/`

### Figure 4

- Folder: `0-Figures/Figure4/1-data`
- Dataset scope: Bayesian-optimization and BO-derived descriptor/visualization
  data.
- Main workbooks:
  - `BO_data.xlsx`
  - `weighted_BO.xlsx`
  - `EH_extra.xlsx`
- Completed Figure 4 BO data:
  - `0-Figures/Figure4/1-data/curated/figure4_bo_complete_112.csv`
  - 112 completed datapoints from four 28-point systems: `Co-Mo`, `Co-Ni`,
    `Co-V`, and `Fe-Ni`.
- Main notebooks:
  - `BO.ipynb`
  - `BO_analysis -2D_symbolic.ipynb`
  - `pareto_polygon_enclosure_code.ipynb`
- Generated outputs are in `0-Figures/Figure4/1-data/outputs/`.
- Raw TIFF images are in `0-Figures/Figure4/1-data/raw_images/`.
- Superseded Figure 4 notebook drafts are in
  `0-Figures/Figure4/1-data/archive/legacy_notebooks/`.

## BO Count Note

The repo-level corrected table contains 148 digit-suffixed samples. Figure 4
uses only the four BO systems `Co-Mo`, `Co-Ni`, `Co-V`, and `Fe-Ni`, for 112
completed datapoints. The additional `Co-Fe` follow-up samples are included in
`all_samples_708_corrected.csv` but are not part of Figure 4.

## Automation / BO Code Provenance

The Figure 4 BO algorithm path is documented in:

- `EC_tool_sys/BO_PROVENANCE.md`
- `EC_tool_sys/manager/platform_manager_HD_Hcell.py`
- `EC_tool_sys/botorch_test/botorch_tool/active_learning.py`

The active-learning function used for the qNEHVI workflow is
`generate_new_exp_MO_one_system_V2`, which uses two objectives
(`lsv_overpotential_-0.01` and `separation_coefficient`) and BoTorch
`qNoisyExpectedHypervolumeImprovement`.
