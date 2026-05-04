# Figure 3

This folder keeps only the figure-local material needed for the Figure 3 top
panel. The machine-learning workflow and outputs are maintained in
`../../hd_separation_ml/`, because those notebooks depend on the curated ML raw
data, descriptor tables, EIS files, and LSV files.

## Top Panel

- Notebook: `notebooks/visualizewithcolors.ipynb`
- Derived data: `data/figure3_top_from_ml_dataset.csv`
- Output: `outputs/before_BO_Fe_Co_Others.png`

The notebook rebuilds the top-panel scatter from the current 561-sample no-BO
machine-learning dataset:

`../../hd_separation_ml/outputs/final_model_8features_561_seed17_rs_from_eis/feature_matrix_8features.csv`

For each sample, the plotted eta10 value is re-extracted from:

`../../hd_separation_ml/data_extraction/all_data/LSV/`

If an LSV trace crosses the target current more than once, the notebook uses
the worst crossing, i.e. the most negative raw potential / largest
overpotential, for conservative reporting.

The conversion uses:

`eta10_mV = -1000 * (E_raw + 0.098 + 0.0592 * 14)`

Samples listed in `LSV/lsv_geometry_area_metadata.csv` are corrected for their
2 cm^2 LSV geometry by using the raw current corresponding to -10 mA cm^-2.

To keep the scatter plot consistent with the original initial-screening LSV
window, eta10 values are plotted within the common LSV lower-potential limit of
-1.50 V. The data CSV keeps both the plotted common-window eta10 and the
full-LSV eta10 for provenance.

## ML Panels

Do not duplicate machine-learning notebooks or copied ML output images here.
Use the current ML outputs in:

`../../hd_separation_ml/outputs/final_model_8features_561_seed17_rs_from_eis/`

The active final model uses the current 561-sample no-BO dataset and the
8-feature descriptor set documented in `hd_separation_ml`.
