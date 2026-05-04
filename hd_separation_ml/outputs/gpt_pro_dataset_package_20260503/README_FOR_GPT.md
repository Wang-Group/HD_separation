# GPT-Pro Compact Dataset Package for H/D Separation Electrochemical Data

This package consolidates the repository data into a small number of tables so GPT-Pro does not need to inspect hundreds of small raw text files.

## Scope

- Total samples in `sample_metadata_709.csv`: 709
- Initial-screening samples: 561
- Four-system BO samples: 112
- Co-Fe follow-up/extra samples: 36

## Experimental design summary

Initial screening:
- Sample IDs without numeric suffix, e.g. `Co-V`, are initial-screening samples.
- Metal 1, metal 2, and Na3Cit were mixed at a molar ratio of 6:1:20/3.
- Electrodeposition was performed at -300 mA cm^-2 for 20 min.

Four-system Bayesian optimization:
- Raw file/sample IDs use names like `Co-V5`; `display_sample_id` also provides a readable form like `Co-V-5`.
- BO design variables are in `bo_4system_conditions_112.csv`.
- Search space: two 0.15 M metal-salt stock solution volumes, 1-19 mL each; 1.0 M sodium citrate, 0-18 mL; water added to total 20 mL.
- 28 experiments per system in 7 rounds, 4 experiments per round.
- `bo_index_zero_based` is the numeric suffix used in raw filenames. `bo_round = floor(index/4)+1`.

## Electrochemical conversion notes

- LSV potentials in raw files are converted to an RHE-equivalent overpotential using:
  `eta_mV_RHE_equiv = -1000 * (E_V_raw + 0.098 + 0.0592 * 14)`.
- Current density is area-corrected using `geometry_area_cm2`.
- Most LSV files store current in A. Some files store current density directly, such as `i(A/cm2)`; this is recorded in `current_column_interpretation`.
- Geometry-area exceptions are included in `sample_metadata_709.csv` and `lsv_raw_long.csv`; known 2 cm^2 LSV cases include Ag-Ce, Co-V, and Mo-Mg.

## Why Et/CV20 are summaries, not full raw tables

The full Et raw directory is about 259 MB and CV20 is about 77 MB before consolidation. To make this package practical for GPT-Pro, point-by-point raw tables are included for LSV and EIS, while Et and CV20 are included as compact summaries.

## Suggested GPT-Pro analysis question

Use the LSV/EIS raw tables and the separation coefficients to test whether literature-inspired high-current HER motifs such as Co-Mo, Mo-Co, Mo-Ni, Co-Ni, Fe-Ni, etc. are supported in this dataset. Distinguish low-current activity (`eta10`) from high-current behavior (`eta50`, `eta100`, and current density at fixed overpotentials such as 200 or 300 mV). Do not assume that BO numeric suffixes are replicate numbers; they are optimization row indices.

## EIS impedance units

Some raw EIS files report impedance in ohm, while some exported files report area-normalized impedance in ohm.cm2. The column `impedance_unit` in `eis_raw_long.csv` and `eis_summary_rs.csv` records this distinction.
