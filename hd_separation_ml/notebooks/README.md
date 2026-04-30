# Notebooks

Use these notebooks as the stable machine-learning entry points.

## 01_feature_selection_558.ipynb

Legacy feature-selection workflow on the 558-sample no-BO matrix.

This notebook writes to:

```text
outputs/feature_selection_558/
```

It uses the self-contained legacy input snapshot in
`archive/legacy_558_reference/`.

## 02_final_model_fixed12_558.ipynb

Legacy result workflow using the fixed 12-feature descriptor set on the
558-sample no-BO matrix.

This notebook writes to:

```text
outputs/final_model_fixed12_558/
```

It uses the same legacy input snapshot as `01_feature_selection_558.ipynb`.

## 03_final_model_fixed12_560.ipynb

Final model workflow using the fixed 12-feature descriptor set on the corrected
560-sample no-BO dataset.

This notebook writes to:

```text
outputs/final_model_fixed12_560/
```

This is the recommended manuscript-facing ML entry point.

## Running Notes

The notebooks include a setup cell that changes the working directory to the
`hd_separation_ml` folder before loading data. This keeps relative paths stable
whether the notebook is opened from this folder or from the repository root.
