# H/D Separation Machine Learning

This directory contains the no-BO machine-learning workflows used to model the
H/D separation coefficient from electrochemical measurements and metal
descriptors.

The repository is organized around three reproducible notebook runs:

1. Legacy 558-sample feature selection.
2. Legacy 558-sample fixed-feature model result.
3. Corrected 560-sample fixed-feature final model result.

The final manuscript-facing model is the corrected 560-sample run.

## What To Run

Run the notebooks in this order:

| Step | Notebook | Purpose | Output folder |
| --- | --- | --- | --- |
| 1 | `notebooks/01_feature_selection_558.ipynb` | Reproduce the legacy 558-sample feature-selection workflow. | `outputs/feature_selection_558/` |
| 2 | `notebooks/02_final_model_fixed12_558.ipynb` | Reproduce the 558-sample result using the fixed 12-feature descriptor set. | `outputs/final_model_fixed12_558/` |
| 3 | `notebooks/03_final_model_fixed12_560.ipynb` | Train/evaluate the final corrected 560-sample model using the same fixed 12-feature descriptor set. | `outputs/final_model_fixed12_560/` |

Do not run the archived root-level notebooks unless you are intentionally
inspecting historical intermediate files.

## Current Result Summary

The compact run summary is saved at:

```text
outputs/model_run_summary.csv
```

Current checked results:

| Run | Samples | Feature set | Best model | Test R2 | Test RMSE |
| --- | ---: | --- | --- | ---: | ---: |
| 558 feature-selection reference | 558 | fixed 12-feature set | Extra Trees | 0.6691 | 0.9238 |
| 558 fixed-feature result | 558 | fixed 12-feature set | Extra Trees | 0.6691 | 0.9238 |
| 560 final corrected result | 560 | fixed 12-feature set | Extra Trees | 0.6726 | 0.9109 |

## Final Descriptor Set

The final model uses this fixed 12-feature descriptor set:

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

This set combines electrochemical response, metal electronic properties,
metal-metal bonding, and H/D adsorption/bonding descriptors.

## Active Inputs

The active notebooks load these root-level inputs:

```text
sample_overpotentials_new_corrected_verison.csv
independent_fitting_results.csv
EH_extra.xlsx
data_extraction/all_data/LSV/*.txt
```

The final corrected no-BO dataset snapshot is also saved as:

```text
datasets/final_no_BO_560.csv
```

The full corrected label table contains 707 total rows:

- 560 no-BO initial-screening rows.
- 147 digit-suffixed BO/follow-up rows.

The earlier 558-sample workflow did not include `Mg-Mg` and `Mo-Mg` because
their electrochemical test data were incomplete at that time. These data were
added later, giving the corrected 560-sample dataset.


## Directory Guide

```text
hd_separation_ml/
|-- notebooks/       stable notebook entry points
|-- outputs/         separated outputs for each notebook workflow
|-- datasets/        stable CSV snapshots
|-- data_extraction/ raw and intermediate extracted data
|-- archive/         historical notebooks, old outputs, backups, and legacy data
|-- DATA_README.md   active data notes
`-- DATA_CURATION_LOG.md
```

Important archive locations:

- `archive/legacy_558_reference/`
  - Self-contained legacy data snapshot used by the two 558-sample notebooks.
- `archive/root_legacy_files/`
  - Old root-level notebooks, old root output folder, backup CSVs, and previous
    descriptor-table artifacts moved out of the active workspace.
- `archive/feature_selection_560_alternative/`
  - Earlier 560-sample greedy feature-selection alternative retained for
    reference, not used as the final model.

## Reproducibility Notes

- The two 558-sample notebooks automatically switch into
  `archive/legacy_558_reference/` and use its local input snapshot.
- The 560-sample notebook uses the current corrected root-level input files.
- Notebook outputs are separated by workflow so rerunning one notebook does not
  overwrite the outputs from another.
- Data recovery and feature-selection decisions are documented in
  `DATA_CURATION_LOG.md`.

## Contact

For questions about the machine-learning data or workflow, please contact:

- Yibin Jiang: `yibinjiang@xmu.edu.cn`
