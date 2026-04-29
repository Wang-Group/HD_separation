# Figure 3

This folder contains the active Figure 3 panel data and figure-ready outputs.
The runnable machine-learning workflow remains in `hd_separation_ml/` because
that notebook depends on the ML directory's descriptor, EIS, and LSV files.

## Top Panel: Initial Screening Scatter

- Notebook: `notebooks/visualizewithcolors.ipynb`
- Main input: `data/sample_overpotentials_new_corrected_verison.csv`
- Output: `outputs/before_BO_Fe_Co_Others.png`

The notebook is patched to read from `../data/` and save to `../outputs/`.

### Filtering Rule

The notebook:

1. Splits `sample` into first and second components using `-`.
2. Keeps rows where the second component is letters only: `[A-Za-z]+`.
3. Removes digit-suffixed BO/follow-up rows such as `Co-Fe0`.

Current count:

- 706 rows before filtering
- 559 rows after filtering

## CSV Variants

- `data/sample_overpotentials_new.csv`
  - 706 rows, clean five-column metric table.
- `data/sample_overpotentials_new_corrected_verison.csv`
  - Active notebook input.
  - 706 rows; first five columns match the canonical ML CSV for the same rows.
  - Includes extra unnamed columns in this figure-local copy.
- `data/sample_overpotentials_new_corrected_v2.csv`
  - Larger figure-local variant with many extra columns.
  - Not active unless the notebook path is manually switched.

## Middle/Bottom Panels: Machine Learning

- Notebook: `../../hd_separation_ml/0-noble_no_BO_seed42.ipynb`
- Output folder: `../../hd_separation_ml/0-noble_no_BO_seed42/`
- Figure-local copies:
  - `outputs/ml/section3_interpretable_ml_composite.png`
  - `outputs/ml/final_prediction_I_raw.png`
  - `outputs/ml/shap_bar.png`
  - `outputs/ml/shap_bar_fixed_for_composite.png`
  - `outputs/ml/shap_swarm.png`
  - `data/ml/final_ml_I_raw_results.csv`
- Data flow:
  - 706 source rows
  - 147 digit-suffixed rows removed
  - 559 no-BO initial-screening rows remain
  - 558 rows enter ML after EIS/LSV merge because `Mg-Mg` lacks valid LSV data

So the ML panels are based on the 559 initial-screening set conceptually, but the
actual trained ML model uses 558 merged samples.
