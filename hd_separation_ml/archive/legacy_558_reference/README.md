# Legacy 558-Sample Reference

This folder preserves the earlier no-BO ML reference run that produced the
fixed 12-feature descriptor set.

It is retained for reproducibility and historical comparison. It is not the
current manuscript-facing final model.

## Contents

- `0-noble_no_BO_seed42_legacy558.ipynb`
  - Legacy feature-selection/modeling notebook.
  - Patched to use the local files in this folder instead of the old absolute
    LSV path.

- `sample_overpotentials_new_corrected_verison.csv`
  - Legacy 706-row label table used by the notebook.

- `independent_fitting_results.csv`
  - EIS fitting table used by the notebook.

- `EH_extra.xlsx`
  - Element descriptor workbook used by the notebook.

- `data_extraction/all_data/LSV/`
  - Legacy LSV snapshot.
  - This snapshot does not contain `Mg-Mg.txt`, reproducing the earlier
    558-sample merged ML matrix.

- `outputs/0-noble_no_BO_seed42/`
  - Preserved legacy output files.

## Expected Reference Result

The preserved notebook output shows:

```text
Total samples: 706
After removing digit-suffixed BO/follow-up samples: 559
After EIS/LSV merge and NaN removal: 558
Best model with fixed 12-feature set: Extra Trees
Test R2: 0.6691
Test RMSE: 0.9238
```

