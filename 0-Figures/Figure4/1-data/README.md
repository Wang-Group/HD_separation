# Figure 4 BO Data Notes

This folder is the current home for Figure 4 Bayesian-optimization and
BO-derived figure data. Active notebooks and source tables remain in this root
folder so existing relative notebook paths keep working. Generated plots, raw
TIFFs, curated manuscript tables, and older draft notebooks are separated into
subfolders.

## Main Files

- `BO_data.xlsx`
  - Sheets: `all_merged`, `initial_sampling_Co_Mo`, `initial_sampling_Co_V`,
    `initial_sampling_Co_Ni`, `initial_sampling_Fe_Ni`
  - The four system sheets each contain 32 rows.
  - In each system sheet, rows `0-27` have completed separation-coefficient
    values, giving 28 completed datapoints per system.
  - Rows `28-31` are present in each sheet but lack completed
    separation-coefficient values.
  - `all_merged` currently contains 88 rows and should not be used as the
    authoritative count for the completed BO figure dataset.
- `curated/figure4_bo_complete_112.csv`
  - Flat 112-row manuscript-facing table extracted from the four completed
    28-point system sheets in `BO_data.xlsx`.
- `curated/figure4_bo_complete_summary.csv`
  - Per-system count summary for the completed and incomplete rows.
- `weighted_BO.xlsx`
  - One sheet with 128 rows.
  - Adds weighted descriptor columns, including weighted electronegativity,
    weighted H adsorption/free-energy descriptors, and weighted M-D bond energy.
- `EH_extra.xlsx`
  - Descriptor source used for element properties and adsorption energetics.
- `initial_sampling_*.json`
  - JSON source files for the four BO systems used by the notebooks.

## Active Notebooks

- `BO.ipynb`
- `BO_analysis -2D_symbolic.ipynb`
- `pareto_polygon_enclosure_code.ipynb`

Older draft notebooks were moved to `archive/legacy_notebooks/`.

## Subfolders

- `outputs/`
  - Generated PNG/SVG plots, legends, contours, joint scatter plots, and
    initialization summaries.
- `curated/`
  - Manuscript-facing checked tables extracted from the workbook.
- `raw_images/`
  - Raw TIFF images for `CoMo`, `CoNi`, `CoV`, and `FeNi`.
- `archive/legacy_notebooks/`
  - Superseded notebook drafts kept for traceability.
- `archive/system/`
  - System-generated files such as `Thumbs.db`.

## Current BO Systems Present Here

The explicit BO workbook/JSON/image files in this folder are organized around:

- `Co-Mo`
- `Co-Ni`
- `Co-V`
- `Fe-Ni`

The completed Figure 4 BO dataset is therefore:

- `Co-Mo`: 28 completed datapoints
- `Co-Ni`: 28 completed datapoints
- `Co-V`: 28 completed datapoints
- `Fe-Ni`: 28 completed datapoints
- Total: 112 completed datapoints

## Cross-check with ML Digit-suffixed Samples

The ML table at `hd_separation_ml/sample_overpotentials_new_corrected_verison.csv`
contains 147 digit-suffixed samples that are excluded from the no-BO model:

- `Co-Fe`: 36
- `Co-Mo`: 28
- `Co-Ni`: 27
- `Co-V`: 28
- `Fe-Ni`: 28

This means the ML digit-suffixed set and the current Figure 4 folder are related
but not identical. In particular, the ML table includes `Co-Fe` follow-up samples,
while this Figure 4 folder contains the complete four-system BO figure dataset:
`Co-Mo`, `Co-Ni`, `Co-V`, and `Fe-Ni`, 28 completed datapoints each.

Before finalizing the manuscript, align the Figure 4 data description with the
specific systems and row counts actually plotted.
