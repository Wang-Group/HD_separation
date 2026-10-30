# Nearest-point curve review and approval

All 35 plotted points were approved on 2026-10-25 as nearest-point estimates.
`approval.json` records the user's decision and the reviewed-points CSV hash.
The images and PDF are preserved from before approval; their "unapproved"
labels describe that earlier review stage, not the current adopted status.

Open report.html for the indexed 35-curve review, or the 35-page PDF.
Each curve has a full recorded range and a low-current view with its candidate marked.
Dashed and dotted lines show -10 and -20 mA/cm2. All figures use Arial.

The points are estimates, not measured eta10 or automatically identified HER onsets.
Potential and actual current density are recorded in separate estimate columns.
Their eta10 and scan-limit fields remain blank; they are not pooled into eta10 mean/SD.
Raw currents use the extraction manifest normalization, including flagged 1 cm2 assumptions.
No smoothing, additional iR correction, target clipping, ML fitting or Figure 4 changes were applied.

Reproduce the pre-approval review plots from revision_figure with
`python plot_unapproved_estimates.py`; it uses the frozen pre-approval classification.
Current adopted records are in `../../eta10_curves.csv` and `../../eta10_samples.csv`.
