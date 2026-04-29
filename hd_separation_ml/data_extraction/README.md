# Data Extraction Notes

This folder contains intermediate tables used to build the labeled ML table.

## `all_substances.json`

`all_substances.json` stores one record per catalyst entry with:

- `substance1`
- `substance2`
- `ir_value`
- `lsv_overpotential_-0.01`
- `EIS_V`

Most `ir_value` values are from the original extracted robot/experiment record.
`EIS_V` is calculated as:

```text
EIS_V = lsv_overpotential_-0.01 - ir_value * 0.01
```

## Known Gap for Later ML Assembly

`Co-Ni4` is absent from the current `all_substances.json` even though its raw
data and separation coefficient are present:

- `all_data/EIS/Co-Ni4.txt`
- `all_data/LSV/Co-Ni4.txt`
- `all_data/Et/Co-Ni4.txt`
- `all_data/CV20/Co-Ni4.txt`
- `separation_coefficient.xlsx`

Do not edit `all_substances.json` directly for this case. Keep it as the
historical intermediate extraction file. When building a complete 709-sample ML
table later, add the missing `Co-Ni4` values in the table-generation step.

The values to use for `Co-Ni4` are:

```text
substance1 = Co
substance2 = Ni4
ir_value = 3.159323533079328
lsv_overpotential_-0.01 = -1.1367623108
EIS_V = -1.1683555461307933
```

The `ir_value` for `Co-Ni4` should be inferred from the EIS fitting result
`independent_fitting_results.csv` as the fitted `R_s`, not from the original
robot-extracted `all_substances.json` record. The LSV value should come from the
Figure 4 BO source table `0-Figures/Figure4/1-data/BO_data.xlsx`
(`initial_sampling_Co_Ni`, `row_id = 4`).
