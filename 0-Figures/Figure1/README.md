# Figure 1

This folder now separates the active Figure 1 files from raw materials and old
drafts.

## Active Files

- `data/all_data_corrected.xlsx`
  - Source workbook for the bottom comparison/Pareto panel.
  - `Sheet1` stores x/y pairs as adjacent columns.
  - The first row repeats semantic headers such as `Potential (V)`,
    `Overpotential`, and `Separation Coefficient`.
- `data/figure1_bottom_plot_points.csv`
  - Long-format extraction of the exact plotted points from
    `all_data_corrected.xlsx`.
  - Columns: `source_type`, `source_label`, `reference_label`, `x_raw`,
    `plot_x_overpotential_mV`, `separation_coefficient`.
  - For `This work`, `x_raw` is the original potential in V and
    `plot_x_overpotential_mV` is `-1000 * x_raw`.
  - For literature/benchmark rows, `x_raw` is already overpotential in mV.
- `notebooks/v4.ipynb`
  - Active notebook for the bottom panel.
  - Patched to read `../data/all_data_corrected.xlsx`.
  - Patched to save current outputs to `../outputs/`.
- `outputs/pareto_plot_v2.png`
  - Current English output.
- `outputs/pareto_plot_v2——chinese.png`
  - Current Chinese-labeled output.

## Data Count

The current `v4.ipynb` output reports:

- 706 "This work" points
- 76 literature-reference points
- 2 benchmark points: `Ni foam` and `Pt sheet`
- 784 total plotted points

So the Figure 1 bottom panel includes all this-work data in the comparison
workbook, not only the 559 initial-screening samples.

## Raw Materials

- `raw_materials/`
  - Assembled figure files, icons, PPTX, SVGs, TIFFs, and other visual assets
    used during Figure 1 assembly.

## Archive

- `archive/notebooks/`
  - Older notebook versions: `v1.ipynb`, `v2.ipynb`, `v3.ipynb`.
- `archive/outputs/`
  - Older Pareto/output variants.
- `archive/dta/`
  - Older data bundle and draft outputs that previously lived in `0-data/dta`.
- `archive/data.rar`
  - Compressed old data bundle.
