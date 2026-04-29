# Machine-learning Data Notes

This folder contains the metric table, descriptor sources, EIS fitting outputs,
raw LSV files, and notebooks used for the H/D separation coefficient model.
Auxiliary or older files have been moved to `archive/`.

## Active Scope

The manuscript-facing ML analysis in this folder is the no-BO initial-screening
model. The BO-inclusive ML notebook, `0-noble_BO_seed42.ipynb`, is intentionally
not part of the active folder contents because it is not used for the manuscript
Figure 3 ML panel.

## No-BO Model

- Notebook: `0-noble_no_BO_seed42.ipynb`
- Output folder: `0-noble_no_BO_seed42/`

### Inputs

- `sample_overpotentials_new_corrected_verison.csv`
  - 706 total rows
  - 559 initial-screening rows after removing digit-suffixed samples
  - 147 digit-suffixed BO/follow-up rows
- `independent_fitting_results.csv`
  - EIS fitting-derived features.
- `EH_extra.xlsx`
  - Element descriptors and adsorption/free-energy values.
- `data_extraction/all_data/LSV/*.txt`
  - Raw LSV files used to extract signed current at the target potential.

### Filtering and Merge

The notebook removes BO/follow-up samples using:

```python
if sample[-1].isdigit():
    noble_metal_samples.append(sample)
```

Count progression:

- 706 total rows in the source table
- 147 digit-suffixed rows removed
- 559 no-BO rows remain
- 558 rows enter the final ML matrix after EIS/LSV merge

The one no-BO sample that drops during the EIS/LSV merge is `Mg-Mg`, because it
is missing from the valid LSV-current set.

### Model Outputs

The executed notebook outputs show:

- Train/test split: 446 train, 112 test
- Best executed model in the final no-BO run: Extra Trees
- Test R2: 0.6691
- Test RMSE: 0.9238

Important selected descriptors in the final executed run include:

- `FE_ads_avg_element1`
- `electronegativity_element1`
- `electronegativity_element2`
- `I_raw`

These support the statement that H adsorption energetics and electronegativity
are important descriptors, while directional claims about "intermediate" or
"moderately high" values should be supported with separate bin/correlation
analysis.
