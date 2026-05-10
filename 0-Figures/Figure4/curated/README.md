# Figure 4 Curated Table

This folder keeps the manuscript-facing BO performance table used for
Figure 4A/B.

## Current Trusted Table

- `figure4_bo_complete_112.csv`
  - Current authoritative BO table.
  - Count: 112 completed BO datapoints, with 28 each for `Co-Mo`, `Co-V`,
    `Co-Ni`, and `Fe-Ni`.
  - Derived from the platform BO JSON files.
  - Uses the conservative/platform value for non-monotonic LSV traces.
  - Separation coefficients are aligned to the current
    `hd_separation_ml/data_extraction/separation_coefficient.xlsx`.

Weighted descriptor projection tables are generated from current sources by
`../plot_top3_shap_descriptor_bo_overlay.py` and written to `../outputs/`.
