# Eta10 repeatability: descriptive review

Distribution of sample standard deviations of accepted crossing-based eta10 values for 349 sample groups with at least two measurements. SD was calculated with the n-1 denominator from uncapped per-curve values; each sample group contributes one SD. Median SD: 24.01 mV; interquartile range: 11.08-40.85 mV; mean SD: 31.32 mV. All retained high-SD groups are included. The left panel uses 10-mV histogram bins; the right panel shows the empirical cumulative distribution.

Scope: all catalysts in the current Figure 1 extraction, not only the ICP-eligible ML cohort. Measurements with matching audited precursor recipes are pooled under ordinary catalyst names, each counted once.

75 single-crossing groups have undefined SD and are excluded, not assigned zero. 248 groups have no accepted eta10 crossing.

Rising-point and nearest-point estimates, scan-limit bounds and no-result placeholders are excluded from numerical eta10 SDs. Samples with mixed statuses contribute their accepted crossings only; excluded-status counts are retained in the sample table.

The 10-550 mV ML cap and graphical boundary clipping are not applied to experimental means or SDs. Geometric area normalization follows the verified extraction, including 2 cm2 curves.

These are across-retained-measurement SDs. Acquisition conditions can differ (including scan rate), so this is not a claim of identical-condition analytical precision. SD estimates with n=2 are less stable than those from larger replicate counts. No curves were removed to narrow this distribution.

Extraction uses the common automatic rising-point rule with one explicitly recorded Co-Ce repeat4 noise exception. This script does not fit models or change Figure 4.

Run `python plot_eta10_repeatability.py` from revision_figure.
