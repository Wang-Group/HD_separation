# Data Curation Log

This note records data-cleaning decisions for the no-BO machine-learning
dataset. It is intended as a reproducibility log for the repository, not as
manuscript text.

## Current No-BO Dataset

The current no-BO ML input table is:

- `sample_overpotentials_new_corrected_verison.csv`
- 707 total samples
- 147 digit-suffixed BO/follow-up samples
- 560 no-BO initial-screening samples

The no-BO master list from `data_extraction/separation_coefficient.xlsx`
contains 561 initial-screening samples. The one sample not included in the
current Et-dependent ML label table is `Ag-Ce`, because no valid Et trace is
available.

## Recovered Samples

Earlier no-BO ML runs used 558 samples after merging the label table with the
electrochemical features. `Mg-Mg` and `Mo-Mg` were not included because their
electrochemical test data were incomplete at that time.

The missing electrochemical data for `Mg-Mg` and `Mo-Mg` were added later. This
updated the no-BO workflow from 558 to 560 samples.

## Ag-Ce Handling

`Ag-Ce` remains excluded from Et-dependent ML because the Et raw file contains
no valid data rows. Its LSV trace also does not reach 50 mA, so no Et value was
inferred from LSV.

Available `Ag-Ce` LSV/EIS-derived information should be treated only as
LSV/EIS-derived descriptors. It should not be used as a substitute for the
1100-1200 s Et potential.

## Feature-Selection Note

The previously selected 12-feature descriptor set is:

```text
I_raw
work_function_element2
MM_bond_element1
MM_bond_element2
E_std_element1
E_std_element2
FE_ads_avg_element1
delta_EOH_element2
electronegativity_element1
electronegativity_element2
MH_bond_kJ_element2
MD_bond_kJ_element1
```

This 12-feature set was retained for final modeling because it is chemically
interpretable and remained predictive after data correction.

Notebook outputs support this decision:

```text
Legacy 558-sample feature-selection workflow:
  Selected fixed 12-feature descriptor set

Legacy 558-sample dataset, fixed 12-feature result:
  Best model: Extra Trees
  Test R2:   0.6691
  Test RMSE: 0.9238

Corrected 560-sample dataset, fixed 12-feature set:
  Best model: Extra Trees
  Test R2:   0.6726
  Test RMSE: 0.9109
```

The active repository therefore keeps three notebook entry points:

- `notebooks/01_feature_selection_558.ipynb`
- `notebooks/02_final_model_fixed12_558.ipynb`
- `notebooks/03_final_model_fixed12_560.ipynb`

An alternative greedy feature-selection run on the corrected 560-sample dataset
is retained only as an archived reference under
`archive/feature_selection_560_alternative/`.
