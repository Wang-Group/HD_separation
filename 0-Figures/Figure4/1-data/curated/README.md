# Figure 4 Curated Tables

These files are checked, manuscript-facing extracts from `../BO_data.xlsx`.

- `figure4_bo_complete_112.csv`
  - 112 completed Bayesian-optimization datapoints.
  - Source sheets: `initial_sampling_Co_Mo`, `initial_sampling_Co_V`,
    `initial_sampling_Co_Ni`, and `initial_sampling_Fe_Ni`.
  - Rule: keep rows where `separation_coefficient` is present.
  - Count: 28 completed rows per system.
- `figure4_bo_complete_summary.csv`
  - Per-system count summary, including the four incomplete rows present in
    each 32-row source sheet.
- `figure4_validation_summary.csv`
  - Cross-validation summary for the regenerated BO panels.
- `figure4_final_pareto_points.csv`
  - Final Pareto-front samples and coordinates for each BO system.

Use these tables for manuscript row counts and Figure 4 data auditing. Use
`../BO_data.xlsx` as the authoritative workbook source.

Note: cross-validation updated two stale/low-precision values: `Fe-Ni6`
separation coefficient from `3.16` to `3.950333367`, and `Co-Mo7` from `10.5`
to `10.39096773`. These now match the repo-level
`all_samples_708_corrected.csv`.
