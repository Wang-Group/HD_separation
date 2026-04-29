# Data Manifest

This file summarizes the current data organization for the manuscript figures and
machine-learning analysis. It is intended to make the figure/data relationship
explicit without moving files or changing notebook paths.

## Canonical Tables

### All screened and BO samples

- Path: `hd_separation_ml/sample_overpotentials_new_corrected_verison.csv`
- Rows: 706
- Columns: `sample`, `E_actual (V)`, `Rs (Ohm)`, `Potential (V)`, `separation_coefficient`
- Interpretation:
  - 559 initial-screening samples have names without a trailing digit.
  - 147 digit-suffixed samples are treated as Bayesian-optimization/follow-up samples by the ML notebook.

### Figure-ready all-data comparison table

- Path: `0-Figures/Figure1/data/all_data_corrected.xlsx`
- Used by: `0-Figures/Figure1/notebooks/v4.ipynb`
- Sheet: `Sheet1`
- Extracted by the notebook as:
  - 706 "This work" points
  - 76 literature-reference points
  - 2 benchmark points: `Ni foam` and `Pt sheet`
  - 784 total plotted points
- This is the current source for the bottom comparison/Pareto panel of Figure 1.
- Long-format extracted copy:
  - `0-Figures/Figure1/data/figure1_bottom_plot_points.csv`
  - This contains the exact plotted points in a flat table.

### Initial-screening figure table

- Path: `0-Figures/Figure3/data/sample_overpotentials_new_corrected_verison.csv`
- Used by: `0-Figures/Figure3/notebooks/visualizewithcolors.ipynb`
- Rows: 706 before filtering
- The notebook filters to 559 initial-screening samples by keeping only rows where the second part of the sample name is letters only.
- The first five columns match `hd_separation_ml/sample_overpotentials_new_corrected_verison.csv` for the same 559 rows.

## Figure Mapping

### Figure 1, bottom panel

- Notebook: `0-Figures/Figure1/notebooks/v4.ipynb`
- Main input: `0-Figures/Figure1/data/all_data_corrected.xlsx`
- Output: `0-Figures/Figure1/outputs/pareto_plot_v2.png`
- Dataset scope: all this-work data plus literature/reference data.
- Current count from notebook output: 706 this-work points, 76 literature-reference points, and 2 benchmark points.

### Figure 3, top panel

- Notebook: `0-Figures/Figure3/notebooks/visualizewithcolors.ipynb`
- Main input: `0-Figures/Figure3/data/sample_overpotentials_new_corrected_verison.csv`
- Output: `0-Figures/Figure3/outputs/before_BO_Fe_Co_Others.png`
- Dataset scope: initial screening only.
- Filtering rule: remove rows whose second sample-name component contains digits.
- Current plotted count: 559.

### Figure 3, middle panel

- Notebook: `hd_separation_ml/0-noble_no_BO_seed42.ipynb`
- Main inputs:
  - `hd_separation_ml/sample_overpotentials_new_corrected_verison.csv`
  - `hd_separation_ml/independent_fitting_results.csv`
  - `hd_separation_ml/EH_extra.xlsx`
  - `hd_separation_ml/data_extraction/all_data/LSV/*.txt`
- Dataset scope: initial-screening samples, then merged with EIS and LSV features.
- Count:
  - 706 total rows in source table
  - 147 digit-suffixed rows removed
  - 559 initial-screening rows remain
  - 558 rows enter ML after EIS/LSV merge because `Mg-Mg` is missing valid LSV current data
- Outputs are written to `hd_separation_ml/0-noble_no_BO_seed42/`.
- Figure-local copies of the current ML outputs are stored in
  `0-Figures/Figure3/outputs/ml/` and `0-Figures/Figure3/data/ml/`.

### Figure 3, bottom panel

- Dataset scope: no-BO initial-screening ML outputs.
- Conceptual dataset: 559 initial-screening rows after removing 147 BO/follow-up
  samples.
- Actual trained model matrix: 558 rows, because `Mg-Mg` is lost during EIS/LSV
  merge.
- Figure-local current outputs:
  - `0-Figures/Figure3/outputs/ml/section3_interpretable_ml_composite.png`
  - `0-Figures/Figure3/outputs/ml/final_prediction_I_raw.png`
  - `0-Figures/Figure3/outputs/ml/shap_bar.png`
  - `0-Figures/Figure3/outputs/ml/shap_swarm.png`

### Figure 4

- Folder: `0-Figures/Figure4/1-data`
- Dataset scope: Bayesian-optimization and BO-derived descriptor/visualization data.
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

The ML exclusion rule identifies 147 digit-suffixed samples:

- `Co-Fe`: 36 samples, indices 0-35
- `Co-Mo`: 28 samples, indices 0-27
- `Co-Ni`: 27 samples, indices 0-27 with `Co-Ni4` missing
- `Co-V`: 28 samples, indices 0-27
- `Fe-Ni`: 28 samples, indices 0-27

This is not the same as the completed Figure 4 BO dataset. Figure 4 contains
complete BO files for `Co-Mo`, `Co-Ni`, `Co-V`, and `Fe-Ni`, with 28 completed
datapoints each, for 112 completed Figure 4 datapoints. The digit-suffixed ML
table additionally contains `Co-Fe` follow-up samples and does not align exactly
with the Figure 4 source workbook.

Before final submission, the manuscript wording should be aligned with the
dataset actually used for Figure 4 and the ML exclusion rule.
