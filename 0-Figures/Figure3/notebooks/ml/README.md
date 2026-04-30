# Figure 3 ML Panel Pointer

The runnable machine-learning notebook remains here:

`../../../hd_separation_ml/notebooks/03_final_model_fixed12_560.ipynb`

It depends on files in `hd_separation_ml`, including descriptor tables, EIS fitting
results, and raw LSV files. For that reason, the notebook itself is not duplicated
or moved into `0-Figures/Figure3`.

Current Figure 3 copies of the final ML outputs are stored in:

- `../outputs/ml/section3_interpretable_ml_composite.png`
- `../outputs/ml/final_prediction_I_raw.png`
- `../outputs/ml/shap_bar.png`
- `../outputs/ml/shap_bar_fixed_for_composite.png`
- `../outputs/ml/shap_swarm.png`
- `../data/ml/final_ml_I_raw_results.csv`

Data count:

- 707 source rows in `hd_separation_ml/sample_overpotentials_new_corrected_verison.csv`
- 147 digit-suffixed BO/follow-up rows removed
- 560 initial-screening rows remain
- 560 rows enter the final corrected fixed-feature ML model

So the current ML panel should be based on the corrected 560-sample no-BO
model. The earlier 558-sample workflow is retained in
`../../../hd_separation_ml/archive/legacy_558_reference/` as feature-selection
provenance.
