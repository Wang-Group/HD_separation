# Bond-energy descriptor units

| Input | Unit |
|---|---|
| `MM_bond` | kcal/mol |
| `MH_bond_kJ` | kJ/mol |
| `MD_bond_kJ` (excluded from predictors) | kJ/mol |

The same units apply to the M1, M2 and composition-weighted representations.
Values in `inputs/reference/elemental_properties.csv` retain the source units.

`inputs/reference/elemental_properties_source.xlsx`, Sheet1, gives this construction:

```text
MH_kcal = 0.5 * (MM_bond + 102.4) + 23.06 * (electronegativity - 2.2)^2
MH_bond_kJ = 4.184 * MH_kcal
```

For Co, cells F6 and G6 contain 1.88 and 17; H6 evaluates the first equation
and I6 converts it to kJ/mol. The result is 259.664663296 kJ/mol.
Reconstruction across all 28 available elements agrees with the packaged M-H
column within 6e-14 kJ/mol. M-H is therefore derived from M-M and
electronegativity in this input table, not an independent surface measurement.
H adsorption free energy (`FE_ads_avg`, eV) is a separate property.

The unit clarification does not rescale the stored inputs or alter fitted results.
