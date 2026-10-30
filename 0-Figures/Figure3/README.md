# Figure 3 Panel Sources

**Assembled figure:** [SVG](../manuscript/Figure3.svg),
[PNG](../manuscript/Figure3.png), [PDF](../manuscript/Figure3.pdf).

The H/D prediction/SHAP panel and global descriptor maps come from
[ml_revision](../../ml_revision/README.md), using three-repeat random-stratified
validation without BO and training-only correlation-aware selection.
The complete [panel source map](../manuscript/README.md) identifies those assets.

## EIS and DRT

This directory retains the EIS/DRT analysis and panel dependencies:

| Purpose | Script or data |
| --- | --- |
| Raw frequency-domain descriptors | `eis_hd_separation_validation.py` |
| pyDRTtools fits and integrated DRT descriptors | `eis_drt_hd_separation_validation.py` |
| GCV and post-fit checks | `eis_drt_gcv_sensitivity_analysis.py`, `eis_drt_postfit_content_analysis.py` |
| Foreground EIS curves and DRT map | `figure3_eis_drt_ab_panels.py` |
| Good-fit EIS background layer | `figure3_eis_goodfit_background_alt.py` |
| DRT heatmap | `figure3_drt_fitting_heatmap.py` |
| Explicit EIS cohort identities | `data/eis_initial_samples.csv` |
| Numerical results and fitted spectra | `data/eis_hd_validation/`, `data/eis_pydrttools_validation/` |

The EIS/DRT cohort contains 561 initial-screening acquisitions. Its sample list
matches both saved EIS and DRT tables exactly. This is a characterization cohort,
not the 560-sample material-descriptor ML cohort. No BO spectra enter this analysis.
Raw EIS and H/D labels are in
[data_extraction](../../hd_separation_ml/data_extraction/README.md).

The EIS/DRT source composition is
[`figure3AB_eis_drt_top_row_goodfit_background_initial_20260505_211231.svg`](outputs/eis_pydrttools_validation/figure3AB_eis_drt_top_row_goodfit_background_initial_20260505_211231.svg).
It is a panel source, not the complete manuscript Figure 3.
The fitted background and foreground membership tables are retained under
`data/eis_pydrttools_validation/`.

## Checks and Regeneration

From this directory, check the saved cohort and raw-file availability without fitting:

```powershell
python -m unittest test_eis_sources.py
```

EIS/DRT refitting additionally requires `pyDRTtools` and `cvxopt`; it is separate
from the ML environment and is not run by ML verification. The DRT script
records the fixed regularization and solver settings. Plot scripts use the
saved numerical tables; the good-fit-background script writes timestamped
panel exports. Complete manuscript layout edits are made in the assembled SVG.
