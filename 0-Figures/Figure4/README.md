# Figure 4: Bayesian Optimization and Descriptor Projections

This folder contains the data, notebooks, scripts, and exported panels used for
the Bayesian optimization (BO) analysis in Figure 4.

The current Figure 4 workflow has two parts:

- Pareto-front and improvement-of-hypervolume analysis for four BO systems.
- Descriptor-space projections comparing BO samples with the initial-screening
  dataset.

Run notebooks and scripts from this folder unless otherwise noted.

## Main Data Sources

### BO Performance Table

`curated/figure4_bo_complete_112.csv`

Authoritative BO table for Figure 4A/B.

- Contains 112 completed BO experiments.
- Systems: `Co-Mo`, `Co-V`, `Co-Ni`, and `Fe-Ni`.
- Count: 28 experiments per system, corresponding to seven batches with four
  experiments per batch.
- Stores the curated separation coefficient and LSV-derived potential values
  used for Figure 4.
- For non-monotonic LSV traces, the table follows the conservative/platform
  convention used in the manuscript figure generation.

### BO Weighted Descriptor Table

`curated/figure4_bo_weighted_ml_top3_descriptors.csv`

Descriptor table used for Figure 4C-style descriptor projections.

- Each BO sample is represented by metal-fraction-weighted descriptors.
- The metal fractions are calculated from the two metal precursor volumes.
- Descriptor columns include:
  - `weighted_electronegativity`
  - `weighted_FE_ads_avg`
  - `weighted_delta_EOH`
  - `separation_coefficient`

### Initial-Screening Descriptor Source

`../../hd_separation_ml/outputs/feature_selection_561_rs_from_eis_no_eis_fitting/feature_matrix_561_no_eis_fitting.csv`

Used by `plot_figure4C_descriptor_joint_marginals.py` to rebuild the
initial-screening descriptor background.

For initial bimetallic screening, the descriptor weighting assumes the nominal
metal ratio of 6:1. For same-metal samples, the first metal receives weight 1.0
and the second receives weight 0.0.

## Figure 4A/B: BO Pareto and Hypervolume

Notebook:

`pareto_polygon_enclosure_code.ipynb`

Primary output:

`eight_panels_2x4_batch1_7_lg.png`

This notebook draws the four-system BO panel:

- Top row: Pareto-front expansion for each BO system.
- Bottom row: improvement of hypervolume (IoH, %) across batches.
- Batches are numbered 1-7.
- IoH is calculated relative to batch 1.
- The x-axis potential is plotted as RHE-converted potential at
  -10 mA cm^-2 in mV, following the current manuscript figure style.

## Figure 4C: Descriptor Projection Panels

Script:

`plot_figure4C_descriptor_joint_marginals.py`

Run:

```powershell
& C:\Users\yibinjiang\anaconda3\python.exe .\plot_figure4C_descriptor_joint_marginals.py
```

The script generates three descriptor-projection variants from the same curated
data.

### 1. Local BO-Region Contour

Output:

`outputs/figure4C_local_BO_descriptor_joint_marginals_initial_contour_same_bins_circles_old_panel_aspect_masked_strict055.png`

This is the recommended Figure 4C candidate.

- Shows the descriptor region near the BO-sampled compositions.
- Initial-screening samples are shown as faint open circles.
- BO samples are shown as filled circles.
- BO sample color indicates measured H/D separation coefficient.
- The background contour is generated only from initial-screening samples.
- The contour uses the same discrete separation-coefficient bins as the old
  Figure 4C style: `2-4`, `4-6`, `6-8`, `8-10`, and `10-12`.
- The contour is masked by distance to initial-screening points using a
  normalized nearest-neighbor threshold of `0.055`; this prevents the plot from
  visually over-interpreting unsupported descriptor regions.

Interpretation: this panel is a local descriptor-space comparison. It shows that
the BO samples lie near a descriptor region enriched in higher H/D separation
coefficients, but the contour itself is not a BO-trained prediction surface.

### 2. Global Descriptor-Space Contour

Output:

`outputs/figure4C_global_descriptor_joint_marginals_initial_contour_same_bins_circles_masked_strict055.png`

This version uses the full descriptor range of the initial-screening and BO
datasets.

Use it as a supporting/SI-style view to show where the BO region sits within
the broader initial-screening descriptor space.

### 3. Global Points-Only Projection

Output:

`outputs/figure4C_global_descriptor_joint_marginals_points_only_initial_edgecolor_bo_filled.png`

This version removes contour filling.

- Initial-screening samples are open circles.
- The edge color of each initial-screening circle indicates its measured H/D
  separation coefficient.
- BO samples are filled circles.

Use this version when a purely data-point-based representation is preferred.
It avoids interpolation assumptions entirely.

## Figure S14: Descriptor-Space Supporting Summary

Script:

`assemble_figure_S14_descriptor_panels.py`

Run:

```powershell
& C:\Users\yibinjiang\anaconda3\python.exe .\assemble_figure_S14_descriptor_panels.py
```

Output:

`outputs/figure_S14_four_panel_BO_descriptor_analysis_rows.png`

This script reuses the same Figure 4C plotting code and curated descriptor
tables to assemble the four-row supporting-information summary:

- global descriptor landscape;
- global point distribution;
- local BO-region descriptor landscape;
- local BO-region point distribution.

## Legacy Descriptor Panels

Notebook:

`BO_analysis -2D_symbolic.ipynb`

Outputs:

- `outputs/joint_weighted_EN_vs_FEadsH_sepcoef_stacked_marginals.png`
- `outputs/joint_weighted_EN_vs_deltaEOH_sepcoef_stacked_marginals.png`
- `outputs/joint_weighted_deltaEOH_vs_FEadsH_sepcoef_stacked_marginals.png`

These are the older BO-only weighted-descriptor joint marginal plots. They are
retained for traceability and comparison with the newer Figure 4C projection
script.

## Legacy/Intermediate Inputs

The following files are retained because they document the earlier plotting and
BO data-processing path:

- `initial_sampling_Co_Mo.json`
- `initial_sampling_Co_V.json`
- `initial_sampling_Co_Ni.json`
- `initial_sampling_Fe_Ni.json`
- `BO_data.xlsx`
- `weighted_BO.xlsx`
- `EH_extra.xlsx`
- `BO.ipynb`

The current manuscript-facing BO analysis should use the curated CSV tables in
`curated/` rather than these legacy/intermediate inputs.

## Notes on Interpretation

- The BO Pareto/hypervolume panels are based on measured BO results.
- The Figure 4C contour background is reconstructed from the initial-screening
  dataset and should be described as a descriptor-space guide, not as a formal
  predictive model.
- The BO points overlaid in Figure 4C are measured BO samples with weighted
  descriptors calculated from the actual BO metal precursor ratios.
