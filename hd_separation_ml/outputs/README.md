# Model Outputs

This folder separates outputs from different no-BO ML workflows so that runs do
not overwrite each other.

## feature_selection_558

Reference feature-selection outputs for the legacy 558-sample matrix.

Key files:

- `feature_selection_558_model_summary.csv`

The corresponding notebook is:

```text
notebooks/01_feature_selection_558.ipynb
```

## final_model_fixed12_558

Legacy model outputs for the 558-sample matrix using the fixed 12-feature
descriptor set.

The corresponding notebook is:

```text
notebooks/02_final_model_fixed12_558.ipynb
```

## final_model_fixed12_560

Final model outputs for the corrected 560-sample dataset using the fixed
12-feature descriptor set.

Key files:

- `final_ml_I_raw_results.csv`
- `final_prediction_I_raw.png`
- `final_shap_I_raw/`
- `section3_interpretable_ml_composite.png`

The corresponding notebook is:

```text
notebooks/03_final_model_fixed12_560.ipynb
```

## model_run_summary.csv

Compact comparison of the legacy reference run, the corrected 560-sample
feature-selection run, and the corrected 560-sample fixed-12-feature final run.
