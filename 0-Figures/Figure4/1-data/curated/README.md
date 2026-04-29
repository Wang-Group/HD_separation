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

Use these tables for manuscript row counts and Figure 4 data auditing. Use
`../BO_data.xlsx` as the authoritative workbook source.
