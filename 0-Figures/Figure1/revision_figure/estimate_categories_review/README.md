# Estimate category plots

Current snapshot: 7 nearest-point estimates, 64 rising-point estimates, and 25 no-accepted-result curves.
Each category has a full-range/low-current PDF (one curve per page) and a low-current overview PNG.
All plots use Arial and retain the extraction area normalization and potential convention.

Nearest-point estimates are recorded points globally closest to -10 mA/cm2 when neither a crossing nor eligible rise exists.
Rising-point estimates are automatically detected raw points associated with a later cathodic rise; they need not be closest to -10.
When multiple rising candidates qualify, select the one with the largest overpotential.
Both require -20 < selected j < -10 mA/cm2; neither is measured eta10 or proof of HER onset.
The explicit Co-Ce repeat4 noise exception retains its crossing and is not included in these estimate PDFs.
Gray circles on rising-point curves show the largest-overpotential crossing for comparison, not an adopted eta10.
For no-accepted-result curves, gray crosses are diagnostic closest points only. Their |j| is at least 20 mA/cm2.
Overview axis ranges vary; the low-current window expands when necessary to show the diagnostic point.

Run `python plot_estimate_categories.py` from revision_figure. Only this output directory is written.
No classifications, raw TXT files, ML outputs, or Figure 4/BO records are changed.
