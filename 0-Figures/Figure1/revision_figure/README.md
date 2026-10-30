# LSV Extraction and Activity Statistics

The same rules apply to all 1,583 initial and repeated LSV curves in
`../../../hd_separation_ml/data_extraction/all_data/LSV_repeat/`. The manifest
identifies each raw TXT by curve ID, SHA256, sample and electrode-area metadata.
Extraction never modifies raw files.

## Extraction Rules

The implementation is `lsv_rules.py`, called by `extract_eta10.py`.

1. Convert current to geometric current density using the manifest area.
   Confirmed area overrides are treated as raw current, including the 2 cm2
   archive measurements. Already normalized density without an override is
   converted from A/cm2 to mA/cm2 without another area division. Unconfirmed
   current exports use a flagged 1 cm2 assumption.
2. Use `eta_mV = -1000 * (exported_E_V + 0.9268)`. No additional iR correction
   is applied. The conversion convention is documented in the code.
3. If the final recorded density is greater than -10 mA/cm2, record the maximum
   scanned overpotential as a scan limit, including curves with earlier crossings.
4. Otherwise, detect rising candidates on a 1 mV grid with a five-point median
   filter, 0.5 mA/cm2 prominence, 10 mV separation and at least 2 mA/cm2 subsequent
   magnitude increase within up to 100 mV. Require more than 30 mV of subsequent
   scan and at least 5 mV preceding the candidate. Select the largest-overpotential
   candidate with raw density `-20 < j < -10 mA/cm2`, after the last crossing when
   crossings exist. Filtering is for detection; reported coordinates are raw points.
5. Without an eligible rising candidate, use the largest-overpotential crossing
   by linear interpolation of raw points. Exact-target plateaus contribute their
   largest-overpotential point. Scan reversal alone does not reject a curve.
6. Without a rising candidate or crossing, use the raw point closest to the
   target only if `-20 < j < -10 mA/cm2`; otherwise leave the result unassigned.

`lsv_noise_exceptions.json` records one curve-specific Co-Ce noise exception:
retain the 135.77 mV crossing instead of the automatic later candidate. Identity,
file hash and raw coordinates are checked; the exception cannot override a scan limit.

## Records and Counts

| Status | Curves | Adopted quantity |
| --- | ---: | --- |
| `eta10_crossing` | 1,105 | `eta10_mV` |
| `rising_point` | 64 | Rising-point potential and actual current density |
| `nearest_point_estimate` | 7 | Nearest-point potential and actual current density |
| `scan_limit` | 382 | Measurement boundary and reason |
| `above_target_no_crossing` | 25 | No accepted crossing or eligible estimate |

`eta10_curves.csv` preserves per-curve values and raw-line references;
`eta10_samples.csv` contains separate crossing, rising-point and nearest-point
statistics. Counts are checked in `extraction_summary.json`. Estimates and scan
limits never populate numerical eta10. Diagnostic crossings are not adopted
crossings when `status` indicates an estimate or limit.

Sample means and sample SDs (`ddof=1`) use crossings only. SD is blank for fewer
than two crossings. Means and SDs are not capped. `lsv_plot_data.py` pools audited
recipe-equivalent labels once; the evidence is in
`../../../others_revision/reproducibility_overview/recipe_evidence.json`.
After pooling, 424 catalysts have a crossing mean and 349 have a crossing SD.

## Reproduction and Tests

From this directory, using the scientific dependencies in `ml_revision/requirements.txt`:

```powershell
python extract_eta10.py
python plot_eta10_repeatability.py
python -m unittest test_extract_eta10.py test_automatic_rising.py test_plot_data.py
```

The first two commands regenerate their CSV/JSON/figure outputs; the tests check
the saved data without changing the extraction. `plot_estimate_categories.py`
produces the separately labelled estimate and unassigned-curve illustrations.
The CSVs are the numerical source, not a separately formatted workbook.

The automatic extractor does not use the approval registers. The registers,
`rule_sync_20261025/` and `automatic_rising_review_20261025/` retain test fixtures
for raw-point and classification regression checks, not alternate published results.

## Plotting and ML

The activity/separation panel in the assembled manuscript uses experimental
mean and SD for both axes; see [manuscript panel sources](../../manuscript/README.md).
Its data include the audited same-recipe reference measurements. Display clipping
does not change the numerical means or SDs. Estimates, limits and unassigned
results remain distinct from eta10 and do not enter the measured-catalyst frontier.

`plot_figure1b.py` and `figure1b_outputs/` provide the companion comparison with
screening H/D coefficients and horizontal LSV SD only. They do not regenerate
the manuscript's two-axis repeatability-mean panel. The shared crossing means
and SDs come from the same per-curve extraction.

The joint experimental summaries are in
`../../../others_revision/reproducibility_overview/`. Publication ML uses
`../../../ml_revision/`: BO-reference measurements are excluded from its averaged
targets, and the crossing mean is then capped to 10-550 mV. Paired electrochemical
descriptors retain their acquisition-matched measurements. Neither estimates nor
scan-limit placeholders are activity-model targets.
