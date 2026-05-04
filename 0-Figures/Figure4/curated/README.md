# Figure 4 Curated Tables

This folder contains only the manuscript-facing tables needed to reproduce the
current Figure 4 panels.

## Current Trusted Table

- `figure4_bo_complete_112.csv`
  - Current authoritative Figure 4 BO table.
  - Count: 112 completed BO datapoints, with 28 each for `Co-Mo`, `Co-V`,
    `Co-Ni`, and `Fe-Ni`.
  - Derived from the platform BO JSON files.
  - Uses the conservative/platform value for non-monotonic LSV traces.
  - Separation coefficients are aligned to the current
    `hd_separation_ml/data_extraction/separation_coefficient.xlsx`.

## Descriptor Tables

- `figure4_bo_weighted_ml_top3_descriptors.csv`
  - Weighted descriptor values used for the Figure 4 descriptor projection
    plots.
