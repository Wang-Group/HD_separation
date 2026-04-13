# Supplementary Information: ML For H/D Separation Coefficient

This repository contains data extraction, feature construction, and machine-learning scripts/notebooks for modeling the **H/D separation coefficient** (`separation_coefficient`) from electrochemistry measurements and element-level descriptors.

The **final figure / analysis entrypoint** is:

- `0-noble_no_BO_seed42_plot.ipynb`

The notebook writes its main outputs to:

- `0-noble_no_BO_seed42/`

## 1. Repository Layout (What Each Folder/File Is For)

Top-level:

- `0-noble_no_BO_seed42_plot.ipynb`: Final notebook used to generate the selected-feature model plots and SHAP importance.
- `0-noble_no_BO_seed42.ipynb`: Closely related notebook (analysis without the final plotting-focused edits).
- `0-noble_BO_seed42.ipynb`: Variant using Bayesian optimization (BO) for tuning; not used for the final figure.
- `EIS_data_corrected.ipynb`: Small consistency check comparing EIS fitting result files.
- `0-old/`: Legacy exploratory notebooks (factor analysis / PCA variants, different stratification settings, etc.).
- `0-noble_no_BO_seed42/`: Output directory created by the final notebook (results CSV, SHAP plots, prediction plot).

Data and extraction:

- `data_extraction/`: Raw data archive and extraction notebook/spreadsheets.
  - `data_extraction/data_extraction.ipynb`: Collects/organizes raw data into `all_data/` (contains machine-specific absolute paths).
  - `data_extraction/all_data/LSV/*.txt`: Raw LSV curves (potential/current vs time).
  - `data_extraction/all_data/EIS/*.txt`: Raw EIS impedance spectra files.
  - `data_extraction/Et_potential_means_1100_1200.xlsx`: Per-sample mean potential values.
  - `data_extraction/separation_coefficient.xlsx`: Per-sample separation coefficients.
  - `data_extraction/all_substances.json`: Per-pair IR/Rs values (and some LSV/EIS metadata).
- `extract_overpotential_separation_rs.py`: Script to combine potential + separation coefficient + Rs into a per-sample CSV.
- `sample_overpotentials_new.csv`: Earlier extracted sample table; the final notebook uses the corrected file `sample_overpotentials_new_corrected_verison.csv`.

EIS fitting:

- `independent_fitting_manual_reorganized.py`: Fits each EIS spectrum independently and writes `independent_fitting_results.csv`.
- `improved_eis_data_loader.py`: Robust loading for EIS text/JSON/CSV formats.
- `starting_point_generator.py`: Generates diverse optimization starting points for EIS fitting.
- `independent_fitting_results.csv`: EIS fitting summary per sample (A/B/C/C_dl/R_s/roughness/W_sigma, etc.).

Element-property ML (auxiliary):

- `1-ML_for_features/pzc_ml_pipeline.py`: Builds ML models for element-level targets (e.g., `delta_EOH` and PZC) from literature tables.
- `1-ML_for_features/*.xlsx`, `1-ML_for_features/*.png`: Outputs from the auxiliary element-property modeling pipeline.
- `EH_extra_filled_with_predictions.xlsx`: An alternative descriptor workbook with filled/predicted entries (not used by the final notebook, which reads `EH_extra.xlsx`).

## 2. Data Sources Used By The Final Notebook

The final notebook builds a sample-level dataset by joining:

1. **Target & operating-point table**
   - `sample_overpotentials_new_corrected_verison.csv`
   - Columns include `sample`, `E_actual (V)`, `Rs (Ohm)`, and `separation_coefficient`.

2. **EIS fitting summary**
   - `independent_fitting_results.csv`
   - Columns include `A`, `B`, `C`, `C_dl`, `R_s`, `roughness_factor`, `W_sigma`, etc.

3. **LSV curve extraction**
   - `data_extraction/all_data/LSV/*.txt`
   - Used to extract a single sign-preserving current value (`I_raw`) at a fixed potential.

4. **Element-level descriptors**
   - `EH_extra.xlsx`
   - `Sheet1`: work function, PZC, electronegativity, bond energies, frequencies, etc.
   - `Sheet2`: adsorption and free energy of adsorption values on multiple facets (used to compute an element-wise mean).

## 3. Dataset Assembly And Filtering (As Implemented)

### 3.1 Sample parsing

Samples are strings like `Ag-Al` (sometimes with trailing digits, e.g. `Co-Mo0` in some datasets). The notebook:

- strips trailing digits: `re.sub(r"\d+$", "", sample)`
- splits by `-` to obtain `element1` and `element2`

### 3.2 Filtering rule used in the final notebook

The final notebook removes samples whose names end with digits (treated as "Bayesian/synthetic" labels in the notebook comments).

On the current repo snapshot, this removes:

- 147 rows from `sample_overpotentials_new_corrected_verison.csv` (706 total -> 559 remaining)

### 3.3 Join logic and final sample count

The final dataset is formed by **inner-joining on `sample`** across:

- `sample_overpotentials_new_corrected_verison.csv`
- `independent_fitting_results.csv`
- extracted `I_raw` from LSV files

On the current repo snapshot, after filtering and joining:

- `N = 558` samples remain (no NaNs after the notebook's imputation steps)

## 4. Feature Construction

The notebook starts from three feature groups:

### 4.1 LSV feature: `I_raw`

`I_raw` is extracted from each LSV file as:

- read `Potential(V)` and `Current(A)` from `data_extraction/all_data/LSV/<sample>.txt`
- interpolate current onto a fixed potential grid:
  - `np.arange(-0.92, -1.51, -0.02)`
- select the value at the configured `lsv_potential`
  - final notebook uses `lsv_potential = -1.0`
- convert `A -> mA` by multiplying by `1000`
- **preserve the sign** (positive and negative currents are kept)

This is a key design choice explicitly noted in the notebook: sign-preserving current helps distinguish normal HER-like behavior (typically negative current) from abnormal behavior (e.g., dissolution-related positive currents).

### 4.2 EIS-derived electrochemical features (considered during feature selection)

Using EIS fit results (`A`, `B`, `C`, `C_dl`, `roughness_factor`, etc.), the notebook constructs:

- `roughness_factor_limited = clip(roughness_factor, upper=200)`
- normalization:
  - `A_normalized = A / roughness_factor_limited`
  - `B_normalized = B / roughness_factor_limited`
- derived circuit quantities:
  - `Rct = 1 / A_normalized`
  - `Cp = -(A_normalized**2) / B_normalized`
  - `Rp = -1 / (A_normalized**2 * C / B_normalized + A_normalized)`
- IR-corrected potential proxy:
  - `E_IR_corrected = E_actual (V) + Rs (Ohm) * 0.05 * 0.9`
- Tafel slope proxy (with clipping):
  - `tafel_slope = ln(10) * j * (Rct + Rp) / roughness_factor_limited * 1000`
  - constants used: `j = 0.01`, `ln(10) = 2.303`
  - `tafel_slope_limited = clip(tafel_slope, 0, 500)`

These features are then log-transformed (with sign preservation for quantities that can be negative):

- `Rct_log = log1p(abs(Rct)) * sign(Rct)`
- `Rp_log = log1p(abs(Rp)) * sign(Rp)`
- `Cp_log = log1p(abs(Cp)) * sign(Cp)`
- `C_dl_log = log1p(C_dl)`
- `W_sigma_log = log1p(W_sigma)`

Note: The final selected-feature model (Section 6) ultimately does not keep these EIS-derived features, but they are part of the candidate feature pool used in backward elimination.

### 4.3 Element-level descriptors (Sheet1 + Sheet2 -> sample-level columns)

The final notebook defines the per-element descriptor set:

```
metal_feature_cols = [
  "work_function", "pzc", "MM_bond", "E_std", "FE_ads_avg",
  "delta_EOH", "electronegativity", "MH_bond_kJ", "MD_bond_kJ",
  "Frequency_OH", "Frequency_H", "Frequency_D", "delta_ZPE_H",
]
```

Where:

- `work_function`, `pzc`, `electronegativity`, `MM_bond`, `MH_bond_kJ`, `MD_bond_kJ`, `Frequency_*`, `delta_ZPE_H`, `delta_EOH` are taken from `EH_extra.xlsx` (`Sheet1`)
- `FE_ads_avg` is computed from `EH_extra.xlsx` (`Sheet2`) as the element-wise mean of `Free Energy of Adsorption (eV)` across facets
- `E_std` is provided via a hard-coded dictionary of standard potentials in the notebook

For each sample, each descriptor is mapped to *both* elements:

- `<descriptor>_element1`
- `<descriptor>_element2`

Missing descriptor values are filled using the **global median** computed across both element1 and element2 columns for that descriptor.

### 4.4 Potential-aligned shifting of some descriptors

After dataset assembly, the notebook applies a per-sample offset:

- `offset = E_IR_corrected_final + 0.098`

Then modifies:

- for any feature column containing `pzc` or `E_std`:
  - `feature = feature - offset`
- for any feature column containing `work_function`:
  - `feature = feature + offset`

This transformation is applied *before* the final train/test split and model fit in the notebook.

## 5. Train/Test Split And Feature Selection

### 5.1 Split strategy

The notebook uses an 80/20 split:

- `random_state = 42`
- to approximate stratification for a continuous target, it bins the target:
  - `y_bins = qcut(y, q=10)`
  - `train_test_split(..., stratify=y_bins)`

On the current repo snapshot:

- Train: 446 samples
- Test: 112 samples

### 5.2 Backward elimination (manual)

The notebook includes a manual backward elimination loop:

- estimator: `RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, min_samples_split=2, random_state=42, n_jobs=-1)`
- score: mean 5-fold CV `r2` on the training set
- iteratively removes the single feature whose removal yields the best CV score improvement

It saves:

- `0-noble_no_BO_seed42/backward_elimination_I_raw.png`
- `0-noble_no_BO_seed42/backward_elimination_I_raw_history.csv`

## 6. Final Selected Features (Used For The Final Figure)

In `0-noble_no_BO_seed42_plot.ipynb`, the final selected feature list is hard-coded as:

```
[
  "I_raw",
  "work_function_element2",
  "MM_bond_element1",
  "MM_bond_element2",
  "E_std_element1",
  "E_std_element2",
  "FE_ads_avg_element1",
  "delta_EOH_element2",
  "electronegativity_element1",
  "electronegativity_element2",
  "MH_bond_kJ_element2",
  "MD_bond_kJ_element1",
]
```

The model target is:

- `y = separation_coefficient`

## 7. Model Training, Tuning, And Reported Metrics

### 7.1 Candidate models and grid search

The final notebook trains and tunes (5-fold CV, scoring=`r2`) the following regressors:

- Random Forest (`sklearn.ensemble.RandomForestRegressor`)
  - `n_estimators`: [100, 200]
  - `max_depth`: [10, 15, None]
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [1, 2]
- Extra Trees (`sklearn.ensemble.ExtraTreesRegressor`)
  - same grid as Random Forest
- XGBoost (`xgboost.XGBRegressor`)
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [3, 5, 7]
  - `learning_rate`: [0.05, 0.1]
  - `subsample`: [0.8, 1.0]

The notebook chooses the "best model" as the one with the **highest test-set R2** (as implemented in `train_models_with_selected_features`).

### 7.2 Metrics file written by the notebook

The grid search + train/test evaluation results are saved to:

- `0-noble_no_BO_seed42/final_ml_I_raw_results.csv`

On the current repo snapshot, the file contains:

| Model | Best CV R2 | Test R2 | Test RMSE |
| --- | ---:| ---:| ---:|
| Extra Trees | 0.5340 | 0.6691 | 0.9238 |
| XGBoost | 0.5522 | 0.6643 | 0.9304 |
| Random Forest | 0.5434 | 0.6574 | 0.9399 |

## 8. Final Plots And Outputs

The final notebook writes:

- Prediction scatter + residual plot:
  - `0-noble_no_BO_seed42/final_prediction_I_raw.png`
- SHAP summary plots:
  - `0-noble_no_BO_seed42/final_shap_I_raw/shap_swarm.png`
  - `0-noble_no_BO_seed42/final_shap_I_raw/shap_bar.png`

Feature labels in the plots are mapped to more readable names by `map_feature_names(...)` in the notebook (e.g., `E_std` -> `E^0`, `pzc` -> `Potential of Zero Charge`, etc.).

## 9. Reproducibility Notes

### 9.1 Python dependencies

The notebooks/scripts use:

- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`
- `xgboost`
- `shap`
- `openpyxl` (for reading/writing `.xlsx` via `pandas`)

### 9.2 Paths

Some helper functions inside notebooks use hard-coded absolute paths for `lsv_data_dir`. If running on a different machine, update these to a relative path:

- `data_extraction/all_data/LSV`

### 9.3 Randomness control

The main train/test split and most models use `random_state=42`, but note:

- XGBoost and scikit-learn tree ensembles can still exhibit minor nondeterminism depending on threading and library versions.

### 9.4 Selection bias note (test-set model selection)

In the final notebook implementation, the best model is selected using the test-set R2. For strict reporting, prefer selecting the model using CV on the training set only (or nested CV), reserving the test set purely for the final, single evaluation.

## 10. Reproducing The Final Figure

1. Install dependencies (example):
   - `python -m pip install numpy pandas scikit-learn matplotlib xgboost shap openpyxl`
2. Open and run:
   - `0-noble_no_BO_seed42_plot.ipynb`
3. If running on a different machine, update any hard-coded `lsv_data_dir` path to point to:
   - `data_extraction/all_data/LSV`
4. Verify outputs exist under:
   - `0-noble_no_BO_seed42/`
