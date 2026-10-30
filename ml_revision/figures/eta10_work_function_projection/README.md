# Work-function projection of eta10 targets

Reproduce from `ml_revision`:

```powershell
python plot_eta10_work_function_projection.py
```

The four panels use the saved original-only and averaged crossing-only targets
from `results/eta10_target_comparison/target_audit.csv`. Panels A/B use observed
Metal 1 and Metal 2 work functions as coordinates; panels C/D show the two targets
against ICP-weighted work function. All panels share a 10-550 mV color scale.
The weighted value is `f_M1 * WF_M1 + (1 - f_M1) * WF_M2`, where `f_M1` is the
ICP-derived metal molar fraction, matching the material-descriptor workflow.
No model is fitted; this is a descriptive projection, not a prediction surface.

Of the 298 matched ML samples, 281 have both work functions tabulated. The 17
samples with missing coordinates are omitted from all four panels, not from the ML
comparison. Their identities are exported in `missing_work_function.csv`.
All 298 samples, coordinates, ICP fractions, weighted work functions, raw/capped
targets and the plotting flag are in
`sample_coordinates_and_targets.csv`.

The 281 plotted samples occupy 176 distinct M1/M2 coordinate pairs. Coincident points
overlap without jitter, with identical sample order in the matched panels. Neither
interpolation nor coordinate imputation is used. BO-reference measurements are
excluded from target averaging, as in the finalized ML comparison.

Exports: PNG, editable-text SVG and PDF. `manifest.json` records counts, settings,
input/code hashes and output hashes. The figure uses Arial and native mathematical
subscripts. Names are independent of SI figure numbering.
