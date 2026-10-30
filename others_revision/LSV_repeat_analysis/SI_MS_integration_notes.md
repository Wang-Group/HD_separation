# Proposed SI and Manuscript Integration

Use the finalized extraction records and pooled statistics in `data/` and
`tables/`. These notes do not edit the working SI or manuscript.

## Supporting Information

Place the repeatability section after the LSV measurement and processing
methods. Use the SD histogram/ECDF as Figure Sx and the replicate counts as
Table Sx; assign final numbers during integration.

- Retain 1,583 curves and 677 source labels, pooled into 672 catalysts using the
  five audited same-recipe series.
- Record 1,105 measured crossings, 64 rising estimates, seven nearest estimates,
  382 scan limits and 25 curves with no accepted result separately.
- Crossing-only means are available for 424 catalysts: 349 have SDs from
  1,030 crossings, and 75 have one crossing. The other 248 have no measured eta10.
- Median SD is 24.01 mV, mean SD 31.32 mV and the 95th percentile 85.14 mV.
  221/349 (63.32%) have SD <=30 mV and 286/349 (81.95%) have SD <=50 mV.
  Eleven catalysts exceed 100 mV; none were removed to narrow the distribution.
- Use the counts in `tables/replicate_counts.csv`, not an assumed number of
  triplicates. A curve count does not establish independent electrode preparation.

Describe the shared automatic rule: terminal-point scan-limit gate; latest
eligible rising point with -20 < j < -10 mA/cm2; otherwise largest-overpotential
crossing; otherwise eligible nearest point; otherwise unassigned. Retain the
single Co-Ce repeat4 noise exception. Detector settings are in
`../../0-Figures/Figure1/revision_figure/lsv_rules.py`.

Report linear crossing interpolation, the 2 cm2 normalization, the configured
+0.9268 V potential conversion and no additional iR correction. Explain the
inherited 1 cm2 convention for unconfirmed current-only exports and retain
limitations from differing scan rates and acquisition conditions. Confirm the
potential-reference description against the experimental Methods.

Keep estimate potential and actual current density separate from measured
eta10. Estimates, scan limits and unassigned curves never enter eta10 means or
SDs. Use uncapped across-measurement sample SD (n-1), not SEM or an ML noise ceiling.

## Manuscript

Figure 1B retains initial-submission H/D values and uses the same uncapped LSV
aggregation as the reproducibility overview. Only the overview uses retained
H/D repeat means and vertical SDs. Same-recipe reference measurements remain
pooled for these experimental plots.

Proposed caption addition:

> For catalysts with at least two accepted target-current crossings, horizontal
> error bars show the sample standard deviation of eta10 across original and
> repeat LSV measurements. Sample-specific counts and extraction rules are
> provided in the Supporting Information. Estimate and boundary markers are
> distinct from crossing-derived eta10 measurements.

Do not assign SD=0 for a singleton. Preserve the paired LSV/EIS characterization
features elsewhere in the manuscript. The original-only versus averaged eta10
ML comparison uses its own reference-excluded crossing targets and the 10-550 mV
cap; experimental Figure 1B/repeatability means and SDs are not ML targets.

## Integration Checks

Use the LSV repeatability, H/D repeatability and reproducibility overview assets when integrating these findings.
Use Arial and proper superscripts/subscripts when integrating the SI.
Keep sample counts distinct from curve counts and ML-validation repeats.
Independent verification, scale-up and temperature/current experiments are
handled separately and are outside this repository package.

