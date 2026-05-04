# Figure 4 Data

This folder keeps the source files needed to reproduce the current Figure 4
Bayesian-optimization panels and the weighted-descriptor projection plots.

## Active Notebooks

- `pareto_polygon_enclosure_code.ipynb`
  - Draws the four-system Pareto/hypervolume BO panel from
    `curated/figure4_bo_complete_112.csv`.
  - Uses batches 1-7 only, 28 completed experiments per system.
  - Uses RHE-converted potential at -10 mA/cm^2 in mV for the x-axis.
  - Calculates IoH as the hypervolume improvement relative to batch 1.
- `BO_analysis -2D_symbolic.ipynb`
  - Draws the weighted-descriptor panels.

Run notebooks from this folder so the relative paths resolve.

## Primary Data

- `curated/figure4_bo_complete_112.csv`
  - Manuscript-facing BO table for Figure 4.
  - Contains 112 completed BO datapoints: 28 each for `Co-Mo`, `Co-V`,
    `Co-Ni`, and `Fe-Ni`.
  - Derived from the platform BO JSON files. These values follow the platform
    convention for non-monotonic LSV traces, where the conservative/worst
    crossing is retained.
  - Separation coefficients are aligned to
    `../../hd_separation_ml/data_extraction/separation_coefficient.xlsx`.
- `curated/figure4_bo_weighted_ml_top3_descriptors.csv`
  - Weighted descriptor table used by the descriptor projection plots.

## Current Figure Exports

- `eight_panels_2x4_batch1_7_lg.png`
  - Current Pareto/hypervolume panel export.
- `outputs/joint_weighted_EN_vs_FEadsH_sepcoef_stacked_marginals.png`
- `outputs/joint_weighted_EN_vs_deltaEOH_sepcoef_stacked_marginals.png`
- `outputs/joint_weighted_deltaEOH_vs_FEadsH_sepcoef_stacked_marginals.png`
  - Current weighted-descriptor panel exports.

## Legacy Inputs

The four `initial_sampling_*.json` files, `BO_data.xlsx`, `weighted_BO.xlsx`,
and `EH_extra.xlsx` are retained as legacy/intermediate inputs. The current
Pareto/hypervolume figure should be regenerated from
`curated/figure4_bo_complete_112.csv`.
