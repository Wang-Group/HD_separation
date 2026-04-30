# Figure 3

This folder contains the active Figure 3 panel data and figure-ready outputs.
The runnable machine-learning workflow remains in `hd_separation_ml/` because
that notebook depends on the ML directory's descriptor, EIS, and LSV files.

## Top Panel: Initial Screening Scatter

- Notebook: `notebooks/visualizewithcolors.ipynb`
- Main input: `../../all_samples_708_corrected.csv`
- Output: `outputs/before_BO_Fe_Co_Others.png`

The notebook is patched to read the repo-level corrected 708-sample table and
save to `../outputs/`.

### Filtering Rule

The notebook:

1. Splits `sample` into first and second components using `-`.
2. Keeps rows where the second component is letters only: `[A-Za-z]+`.
3. Removes digit-suffixed BO/follow-up rows such as `Co-Fe0`.

Current count:

- 708 rows before filtering
- 560 rows after filtering

## CSV Variants

- `data/sample_overpotentials_new.csv`
  - 706 rows, clean five-column metric table.
- `data/sample_overpotentials_new_corrected_verison.csv`
  - Historical figure-local input retained for comparison.
  - 706 rows; this file predates recovery of the final two no-BO points.
  - Includes extra unnamed columns in this figure-local copy.
- `data/sample_overpotentials_new_corrected_v2.csv`
  - Larger figure-local variant with many extra columns.
  - Not active unless the notebook path is manually switched.

## Middle/Bottom Panels: Machine Learning

- Notebook: `../../hd_separation_ml/notebooks/03_final_model_fixed12_560.ipynb`
- Output folder: `../../hd_separation_ml/outputs/final_model_fixed12_560/`
- Figure-local copies:
  - `outputs/ml/section3_interpretable_ml_composite.png`
  - `outputs/ml/final_prediction_I_raw.png`
  - `outputs/ml/shap_bar.png`
  - `outputs/ml/shap_bar_fixed_for_composite.png`
  - `outputs/ml/shap_swarm.png`
  - `data/ml/final_ml_I_raw_results.csv`
- Data flow:
  - 708 repo-level source rows
  - 148 digit-suffixed BO/follow-up rows removed
  - 560 no-BO initial-screening rows remain
  - 560 rows enter the final corrected fixed-feature ML run

So the current ML panels should be based on the corrected 560-sample no-BO
dataset. The earlier 558-sample workflow is retained only as a legacy feature
selection reference in `hd_separation_ml/archive/legacy_558_reference/`.
