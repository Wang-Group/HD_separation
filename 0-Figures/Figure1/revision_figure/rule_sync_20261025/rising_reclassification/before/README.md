# Figure 1B: reviewed LSV extraction and plotting

Last documentation review: 2026-10-25.

48 original curves were restored with explicit user approval, without duplicating
Ag-Ce or Mo-Mg. The dataset now contains 1,583 curves. See `restoration_20261024`
for the previous manifest, extraction, workbook and figure, restoration records,
and before/after sample comparisons. The 2026-10-25 rule synchronization updates
extraction and classifications only. Figures, ML inputs/results, and the response
letter are unchanged in this run. Figure 4 and the recorded BO data are untouched.

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
- Approved point estimates require explicit raw-curve review AND a selected
  current density strictly between -20 and -10 mA/cm2, before rounding.
  Passing the current threshold alone does not approve an estimate.
- Preserve the 20 selections in `approved_rising_points.csv`. The additional
  register carries forward 13 reviewed original-curve estimates from the June
  Figure 1 classification (four turning points and nine nearest points), plus
  two Co-V11 repeat selections. Original Co-V11 occurs in both decision sources
  but is included once. The subsequent 35-curve review was explicitly approved
  on 2026-10-25. The total is 24 rising and 46 nearest-point estimates.
- All estimates are tied to curve ID, SHA256 and a raw TXT line. A nearest-point
  estimate must be globally closest to -10 mA/cm2 on a curve without a crossing;
  a reviewed rising point need not be the nearest point. Neither can override
  the final-current scan-limit rule. Their potential and actual current density
  are recorded separately; eta10 stays blank.
  `largest_crossing_mV` is diagnostic, not their adopted result.
  A geometric total-current crossing does not independently establish that
  all the measured current is HER current.
- Sample means and sample SD (ddof=1) use eta10 crossings only. Scan limits
  and unassigned results are not averaged as eta10. SD is blank for n < 2.
  Rising-point and nearest-point means and SDs are reported separately from
  each other and never pooled with eta10. No plot-axis clipping is applied.

## Outputs

- `restoration_20261024/before/eta10_extraction.xlsx`: archived previous workbook;
  not the current extraction. Current results are the CSV files below.
- `eta10_curves.csv`: full traceable per-curve extraction.
- `eta10_samples.csv`: separate eta10, rising-point and nearest-point summaries.
- `approved_rising_points.csv`: 20 explicit raw-line selections and verification values.
- `approved_additional_estimates.csv`: 50 verified raw-line decisions: 15 carried
  forward and 35 newly approved, with evidence-file paths and hashes.
- `extraction_summary.json`: counts and verification status.
- `rule_sync_20261025/`: counts, changed classifications, the 18 reviewed scan
  limits, verification results, and the pre-sync extraction snapshot.
- `rule_sync_20261025/nearest_point_review/approval.json`: explicit approval of
  the 35 plotted points, tied to the reviewed CSV by SHA256.
- `tmp_comparison/`: optional frozen afternoon baseline and comparison.
  These files do not influence extraction. Differences involving rising
  points are metric changes, not necessarily numerical errors.

### Record separation

`status` defines the adopted result. Only `eta10_crossing` populates `eta10_mV`.
`rising_point` populates `rising_point_potential_mV` and `rising_point_j_mA_cm2`;
`nearest_point_estimate` populates `nearest_point_potential_mV` and
`nearest_point_j_mA_cm2`. The `approved_estimate_*` columns are common estimate
fields, not eta10. Only `scan_limit` populates `scan_limit_mV`, accompanied by
`scan_limit_reason`. No accepted result means all adopted metric fields are blank.
Diagnostic nearest/crossing columns are not adopted results. Sample summaries
keep separate counts and statistics for each metric; scan limits are counted,
not averaged. Zero is a valid potential and is never used to mean missing.

## Reproduction

Run these commands from this directory:

```powershell
python extract_eta10.py
python -m unittest test_extract_eta10.py
python sync_eta10_rules.py
```

The checked-in approval registers are sufficient to rerun extraction. To rebuild
the additional register from its recorded evidence, run
`python sync_eta10_rules.py --prepare-approvals` first. This verifies original
TXT hashes and the exact selected points; it does not infer new approvals.
The final audit command compares with the fixed pre-sync snapshot.

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
| `rising_point` | 24 | Potential and current density reported separately |
| `nearest_point_estimate` | 46 | Potential and current density reported separately |
| `scan_limit` | 382 | Recorded as a bound, not averaged as eta10 |
| `above_target_no_crossing` | 25 | No accepted crossing or approved estimate |
| Total | 1,583 | 677 sample groups |

| Origin of curve file | Accepted eta10 | Approved estimates | Scan limits | No accepted result | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original LSV directory | 423 | 18 | 256 | 10 | 707 |
| Additional LSV/error-bar/archive files | 683 | 52 | 126 | 15 | 876 |
| Total | 1,106 | 70 | 382 | 25 | 1,583 |

The 876 additional files comprise 660 LSV_error_bar, 73 S, and 143 archive-168
curves. Ag-Ce and Mo-Mg remain represented by their retained archive versions;
the two duplicate original files are not added again. BO-reference raw files
are counted according to their source, not as an extra origin category.

430 existing sample labels have at least one accepted eta10; 353 have at least
two and a defined sample SD. 42 labels have approved estimates, of which six
(Co-Fe29, Co-Fe33, Co-V11, Eu-Pd, V-Pd and W-Pd) have no accepted eta10.
These coverage counts overlap.
Sample grouping is unchanged; this run does not merge recipe-equivalent labels.

All 1,106 accepted values, all 382 scan limits, and accepted-eta10 sample means
and SDs are unchanged. The first update carried forward 15 previously unassigned
estimates; the subsequent approval adds the 35 reviewed nearest points. The
remaining 25 no-crossing curves fail the strict current-density window.
The 18 reviewed original curves remain scan limits at their maximum
scanned potential, and the four known original charging-artifact curves remain
excluded from accepted eta10. No bounds are replaced numerically by 550 mV.

Source hashes were verified in the saved extraction. No additional iR correction
was applied and raw files were not modified.

## Saved Figure 1B conventions (not refreshed in this run)

The saved figure predates the additional estimate register. Its plotting code
does not yet incorporate the new nearest-point summaries. Review that display
policy before regenerating Figure 1B; the saved figure must not be presented as
an updated rendering of this extraction. Accepted eta10 mean/SD inputs did not
change, but estimate-only marker placement may change in a later figure update.

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
status `above_target_no_crossing`; restoration increased that to 75.** The intended 74-curve set has not
been identified. Do not silently substitute a different set or remove two more
curves to obtain that total. The 20 approved rising-point curves form a separate
category; the two rising-point-only plotted sample groups are not two additional
no-crossing curves.

The current 2026-10-25 classification has 25 curves in this category after 15
reviewed estimates were carried forward and 35 plotted nearest points approved.
Both these 25 and all approved estimates
contribute nothing to numerical eta10 means or SD. All raw curves remain in the
dataset. The saved pre-sync figure still has five no-crossing display fallbacks.
Removing those markers from Figure 1B is a plotting change, distinct from
excluding raw measurements from future analyses.

To inspect the current set without changing any files:

```powershell
Import-Csv .\eta10_curves.csv |
    Where-Object { $_.status -eq 'above_target_no_crossing' } |
    Select-Object sample,file,curve_id,sha256,source_relative_path
```

When returning to this issue:

1. Reconcile the intended 74-curve list against the dated snapshots using
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

The finalized ML workflows retain their saved input snapshots. The rule-sync
run updates only extraction, approval provenance, tests and documentation,
not any figures, ML workflows, or the response letter.
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
