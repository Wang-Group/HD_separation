# Figure 1B: reviewed LSV extraction and plotting

Last documentation review: 2026-10-24.

48 original curves were restored with explicit user approval, without duplicating
Ag-Ce or Mo-Mg. The dataset now contains 1,583 curves. See `restoration_20261024`
for the previous manifest, extraction, workbook and figure, restoration records,
and before/after sample comparisons. The finalized ML input snapshot was NOT
updated and ML was NOT rerun; it still corresponds to the earlier extraction.

This directory contains per-curve extraction, per-sample statistics, and the
standalone Figure 1B. It is a reviewable analysis snapshot, not a finalized ML
training dataset. Raw measurements must remain unchanged. No curves are removed
by this documentation update.

Input: `../../../hd_separation_ml/data_extraction/all_data/LSV_repeat`.
Each TXT is treated as a separate measurement and identified by its manifest
record and SHA256. Source files are never modified.

## Rules

- Convert current to geometric current density using the manifest area.
  Confirmed area overrides are treated as raw current, including the 2 cm2
  archive measurements. Already normalized density without an area override
  is converted from A/cm2 to mA/cm2 without another area division.
- Current exports without a confirmed area use a flagged 1 cm2 assumption.
- Use `eta_mV = -1000 * (exported_E_V + 0.9268)`. The offset is a configured
  measurement convention, not a newly verified electrode calibration.
  No additional iR correction is applied.
- The target is signed current density `j = -10 mA/cm2`.
- If the final recorded current density is greater than -10 mA/cm2, record
  the maximum scanned overpotential as a scan limit, not measured eta10.
  This applies even if an earlier transient crossed the target.
- Otherwise find all raw-point crossings by linear interpolation and select
  the largest overpotential. An exact-target plateau contributes one crossing
  at its largest overpotential. No smoothing is used.
- Curves entirely more cathodic than -10 mA/cm2 with no crossing receive no
  eta10. Scan reversal is flagged, but does not automatically reject a curve.
- The 20 explicitly approved rising points are selected by SHA256 and raw TXT
  line in `approved_rising_points.csv`. Their potential and current density are
  recalculated from raw data into separate rising-point columns; eta10 is blank
  for these curves. `largest_crossing_mV` is diagnostic, not their adopted result.
  A geometric total-current crossing does not independently establish that
  all the measured current is HER current.
- Sample means and sample SD (ddof=1) use eta10 crossings only. Scan limits
  and unassigned results are not averaged as eta10. SD is blank for n < 2.
  Rising-point potential and current-density means and SDs are reported
  separately, never pooled with eta10. No plot-axis clipping is applied.

## Outputs

- `restoration_20261024/before/eta10_extraction.xlsx`: archived previous workbook;
  not the current extraction. Current results are the CSV files below.
- `eta10_curves.csv`: full traceable per-curve extraction.
- `eta10_samples.csv`: separate eta10 and rising-point sample summaries.
- `approved_rising_points.csv`: 20 explicit raw-line selections and verification values.
- `extraction_summary.json`: counts and verification status.
- `tmp_comparison/`: optional frozen afternoon baseline and comparison.
  These files do not influence extraction. Differences involving rising
  points are metric changes, not necessarily numerical errors.

## Reproduction

Run these commands from this directory:

```powershell
python -m unittest test_extract_eta10.py
python extract_eta10.py
python plot_figure1b.py
```

For comparison, append `--compare-afternoon PATH_TO_LSV_METRIC_RECORDS.csv`.
The extractor uses only the Python standard library and writes CSV/JSON.
The Excel workbook is a formatted view of those outputs; these commands do not
regenerate it. If extraction changes, refresh and verify the workbook separately
before treating it as current. Plotting requires numpy, pandas and matplotlib.
Re-running the scripts overwrites their generated outputs, so preserve a dated
snapshot before changing rules. The optional comparison folder is not required
and is not present in this snapshot.

## Current extraction snapshot

The counts below come from `extraction_summary.json`, not from plotted markers.

| Per-curve status | Count | Numerical treatment |
| --- | ---: | --- |
| `eta10_crossing` | 1,106 | Included in sample eta10 mean and sample SD |
| `rising_point` | 20 | Potential and current density reported separately |
| `scan_limit` | 382 | Recorded as a bound, not averaged as eta10 |
| `above_target_no_crossing` | 75 | No numerical eta10 assigned |
| Total | 1,583 | 677 sample groups |

Source hashes were verified in the saved extraction. No additional iR correction
was applied and raw files were not modified.

## Figure 1B conventions

- `plot_figure1b.py` combines `eta10_samples.csv` with
  `figure1b_inputs/original_HD.csv` and `figure1b_inputs/literature.csv`.
- H/D values are the original values. Newly added H/D repeats and H/D error bars
  are not included in this figure.
- Horizontal error bars are sample SD, not SEM. Samples with only one accepted
  crossing have a mean but no SD. Repeats are aggregated within sample groups.
- Sample-level display precedence is accepted eta10, then an approved rising
  point, then the furthest recorded scan limit, then a no-crossing placeholder.
  Consequently, the number of plotted placeholders is not the number of
  no-crossing raw curves.
- The saved plot contains 430 eta10 groups, 2 rising-point groups, 240 scan-limit
  groups and 5 left-boundary placeholders: 677 groups in total.
- No-crossing placeholders are open left triangles at 10 mV. This position is
  only a plotting convention, not a measured eta10, a proven HER onset, or proof
  of exceptional HER activity. Their numerical eta10 and SD remain blank.
- Display coordinates and uncertainty endpoints are clipped to 10-550 mV;
  the x axis spans 0-560 mV. Numerical means and SD are not clipped.
- The combined frontier uses unclipped eta10 means and eligible literature
  points. Rising points, bounds, placeholders and our pure-metal samples are
  excluded from frontier calculation. Same-recipe reference groups are treated
  as ordinary sample groups, not duplicated as independent BO observations.
- The canvas is 16 x 8 inches with Times New Roman text and mathematical
  superscripts/subscripts. Catalyst labels are omitted for later Inkscape
  annotation; SVG text remains editable.
- `figure1b_outputs/` contains SVG, PDF, PNG, plotted sample values, frontier
  values, caption and `provenance.json` with input hashes and plotting policies.

## Pending review: the curves referred to as "74"

The user requested that we retain a reminder to revisit and potentially exclude
"74 curves" later. **The earlier extraction identified 72 curves with
status `above_target_no_crossing`; restoration adds three, giving 75 now.** The intended 74-curve set has not
been identified. Do not silently substitute a different set or remove two more
curves to obtain that total. The 20 approved rising-point curves form a separate
category; the two rising-point-only plotted sample groups are not two additional
no-crossing curves.

These 75 curves already contribute nothing to numerical eta10 means or SD.
They remain in the raw dataset and traceability table. Five sample groups have
only the no-crossing display fallback after the other metrics are considered.
Removing those markers from Figure 1B is a plotting change, distinct from
excluding raw measurements from future analyses.

To inspect the current set without changing any files:

```powershell
Import-Csv .\eta10_curves.csv |
    Where-Object { $_.status -eq 'above_target_no_crossing' } |
    Select-Object sample,file,curve_id,sha256,source_relative_path
```

When returning to this issue:

1. Reconcile the intended 74-curve list against the current 75 records using
   `curve_id`, SHA256 and source path. Do not match only by sample name.
2. Review current units, electrode area, scan coverage and possible non-HER
   current. A curve always beyond the target does not measure a target crossing.
3. Record each explicit decision and reason in a separate exclusion manifest.
   Distinguish exclusion from numerical analysis, exclusion from plotting, and
   an unresolved measurement. Preserve all original TXT files.
4. Rebuild sample summaries and the figure only after the decision is approved.
   Check which samples lose their last usable measurement or plotting fallback.
5. Compare before/after counts, mean, SD, metric type and frontier membership.
   Refresh the Excel, caption and provenance together. Do not infer eta10 from
   the 10 mV placeholder or from a rising-point potential.

## Downstream work, not executed here

The finalized ML workflows retain their earlier input snapshot. This restoration
updates only Figure 1B and its extraction, not ML or the response letter.
Use one sample-level row for eta10 modelling; keep repeat count and
experimental SD alongside the mean. Do not treat repeats as independent train
and test rows, and do not train ordinary eta10 regression on bounds, rising
points or display placeholders.

The coordinated scope includes the SI Figure S9 eta10 model and work-function
map, eta10 correlations in the response letter, and activity performance plots.
The old model's blanket exclusion below 20 mV needs reconciliation with the
reviewed valid low-overpotential crossings. LSV-current features used in other
models require their own extraction audit; replacing eta10 does not update them.
Keep original H/D targets unless a separate change is approved. Preserve the
historical observations used for actual BO decisions when evaluating repeats
retrospectively. Review changes before any GitHub commit or push.
