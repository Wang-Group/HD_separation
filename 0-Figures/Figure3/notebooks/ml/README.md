# Figure 3 ML Panel Pointer

The runnable machine-learning notebook remains here:

`../../../hd_separation_ml/0-noble_no_BO_seed42.ipynb`

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

- 706 source rows in `hd_separation_ml/sample_overpotentials_new_corrected_verison.csv`
- 147 digit-suffixed BO/follow-up rows removed
- 559 initial-screening rows remain
- 558 rows enter ML after EIS/LSV merge because `Mg-Mg` lacks valid LSV data

So the ML panel is based on the 559 initial-screening set conceptually, but the
actual trained model uses 558 merged samples.
