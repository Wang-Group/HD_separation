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

## EIS Validation

- Script: `eis_hd_separation_validation.py`
- Derived data: `data/eis_hd_validation/`
- Output figures: `outputs/eis_hd_validation/`

The script rebuilds EIS frequency-domain descriptors from the current raw files
in:

`../../hd_separation_ml/data_extraction/all_data/EIS/`

and aligns them to:

`../../hd_separation_ml/data_extraction/separation_coefficient.xlsx`

It reports only the Figure 3 initial-screening 561-sample no-BO set. BO and
follow-up spectra are intentionally excluded from the analysis so the EIS-HD
relationship is not biased by Bayesian-optimization sampling. The current
raw-frequency validation supports two main trends: lower total EIS area/integral
is associated with higher HD separation, and a larger low-frequency `|Z|` area
fraction is associated with lower HD separation.

## DRT Validation

- Script: `eis_drt_hd_separation_validation.py`
- Derived data: `data/eis_pydrttools_validation/`
- Output figures: `outputs/eis_pydrttools_validation/`

This script runs the official `pyDRTtools.simple_run` solver in the `DRT` conda
environment:

`C:\Users\yibinjiang\anaconda3\envs\DRT\python.exe eis_drt_hd_separation_validation.py`

The current batch uses the pyDRTtools piecewise-linear RBF option, combined
real/imaginary impedance fitting, inductance enabled, and a fixed lambda of
`1e-3`. The Figure 3 analysis is restricted to the initial-screening no-BO
spectra. The recovered `gamma(tau)` distributions are interpolated to a common
tau grid before integration.

In the official pyDRTtools validation, total DRT area is negatively correlated
with HD separation. Absolute long-tau/low-frequency DRT area is also negative,
but the normalized long-tau fraction remains weak and should not be overstated.

### GCV Sensitivity

- Script: `eis_drt_gcv_sensitivity_analysis.py`
- Derived data: `data/eis_pydrttools_pwl_gcv_validation/`
- Output figures: `outputs/eis_pydrttools_pwl_gcv_validation/`

This reruns the initial-screening no-BO pyDRTtools batch with `cv_type="GCV"`
while keeping the PWL RBF basis. The GCV-selected lambda values change some
individual fits, but the main correlations are essentially unchanged from the
fixed lambda run: total DRT area remains negatively correlated with HD
separation, and low-frequency/long-tau normalized fraction remains weak.

## DRT Post-Fit Content Analysis

- Script: `eis_drt_postfit_content_analysis.py`
- Derived data: `data/eis_pydrttools_validation/drt_postfit_*.csv`
- Output figures: `outputs/eis_pydrttools_validation/figure3_drt_postfit_*.png`

This follow-up checks which DRT descriptors remain associated with HD
separation after controlling for total DRT area. It also saves top/bottom HD
quantile contrasts, tau-resolved partial correlations, and a compact
cross-validated predictive comparison between total-area-only and multi-DRT
descriptors.

## EIS/DRT Figure Panels

The current Figure 3 EIS/DRT visualization uses the fitted Nyquist panel as the
left panel and the DRT descriptor map as the right panel. These panels are based
only on the initial-screening no-BO dataset.

### Current Selected Outputs

Use the no-grid, good-fit-background version for the current left EIS panel:

`outputs/eis_pydrttools_validation/figure3AB_eis_drt_top_row_goodfit_background_initial_20260505_211231.png`

Vector version:

`outputs/eis_pydrttools_validation/figure3AB_eis_drt_top_row_goodfit_background_initial_20260505_211231.svg`

Standalone EIS panel with sample labels:

`outputs/eis_pydrttools_validation/figure3B_eis_top_bottom_goodfit_background_initial_20260505_211231.png`

Standalone DRT map from the base panel script:

`outputs/eis_pydrttools_validation/figure3A_drt_2d_map_initial.png`

### Scripts

Base EIS/DRT panels:

`figure3_eis_drt_ab_panels.py`

Good-fit EIS background alternate:

`figure3_eis_goodfit_background_alt.py`

Run with the DRT conda environment:

```powershell
& "C:\Users\yibinjiang\anaconda3\envs\DRT\python.exe" "C:\Users\yibinjiang\Documents\GitHub\HD_separation\0-Figures\Figure3\figure3_eis_drt_ab_panels.py"
& "C:\Users\yibinjiang\anaconda3\envs\DRT\python.exe" "C:\Users\yibinjiang\Documents\GitHub\HD_separation\0-Figures\Figure3\figure3_eis_goodfit_background_alt.py"
```

The alternate script writes timestamped files and refuses to overwrite existing
outputs.

### Nyquist Panel Design

The EIS panel plots Nyquist curves as `Z' - Rs` versus `-Z''`, after subtracting
the fitted high-frequency real intercept. Frequencies above `1000 Hz` are not
shown, so the plotted region emphasizes the mid/low-frequency impedance
features most visible in the catalyst comparison.

The current left-panel style is:

- Arial bold font.
- No grid lines.
- Background layer: fitted EIS lines only, no raw points.
- Background color: HD separation coefficient, using the same red-blue color
  scale as the DRT map.
- Background alpha: `0.80`.
- Background linewidth: `0.52`.
- Background z-order: bottom layer.
- Foreground layer: selected high- and low-HD examples, with fitted lines and
  experimental points.

The background layer is loaded from:

`data/eis_pydrttools_validation/figure3B_background_good_fit_eis_fits.csv`

For the current no-grid output, the background contains 141 fitted EIS curves.
Only the 10 foreground samples are removed from the background to avoid drawing
the same curves twice. Samples such as `Mn-Y`, `Co-V`, `In-La`, and `Al-Al` are
not manually excluded by the alternate script, although they are not present in
the current precomputed fitted-background table.

The background sample list for the current no-grid output is saved at:

`data/eis_pydrttools_validation/figure3B_goodfit_background_samples_20260505_211231.csv`

### Foreground EIS Examples

The selected foreground examples are saved at:

`data/eis_pydrttools_validation/figure3AB_top_bottom_samples.csv`

Current foreground samples:

| Group | Sample | Separation coefficient | Fit relative RMSE |
| --- | --- | ---: | ---: |
| High HD | In-Ni | 8.59 | 0.072 |
| High HD | Fe-Ga | 9.20 | 0.084 |
| High HD | Fe-Ni | 9.00 | 0.086 |
| High HD | Co-Mo | 9.80 | 0.120 |
| High HD | Co-Ga | 8.00 | 0.097 |
| Low HD | Eu-Cd | 2.61 | 0.026 |
| Low HD | Y-La | 2.61 | 0.086 |
| Low HD | In-Cu | 2.59 | 0.065 |
| Low HD | Mo-In | 2.60 | 0.117 |
| Low HD | Eu-In | 2.45 | 0.076 |

### DRT Map

The right DRT panel plots:

- x-axis: `Total DRT Area (Ohm)`.
- y-axis: `DRT Area at 0.8-20 ms (Ohm)`.
- color: HD separation coefficient.
- both axes: log scale.
- colorbar: horizontal, placed above the DRT map.

The total DRT area is the integrated relaxation contribution excluding `Rs`.
Across the initial-screening dataset, total DRT area is negatively correlated
with HD separation:

`Spearman rho = -0.42, p = 2.25e-25`

The tau-resolved DRT area at `0.8-20 ms` also shows a weaker negative
correlation:

`Spearman rho = -0.28, p = 3.28e-11`

This weaker tau-window trend is largely coupled to total DRT area and should be
described as an additional kinetic comparison rather than an independent
mechanistic axis.
