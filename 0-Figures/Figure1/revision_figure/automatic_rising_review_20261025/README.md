# Automatic rising-point rule: comparison only

Subsequent decision (2026-10-25): adopt this proposal except for the Co-Ce
repeat4 noise-driven selection. The active extraction and five actual changes
are documented in `../automatic_rising_adoption_20261025/`. This folder preserves
the original six-change proposal for comparison; it is not the current result.

Date: 2026-10-25. This is a proposed deterministic alternative, not an adopted
extraction. It applies the same rule to all 1,583 original and additional curves.
The current extraction, approval registers, raw TXT files, publication figures,
Figure 4/BO, ML inputs/results and response letter remain unchanged.

## Rule

Keep the existing normalization and potential convention. Confirmed area
overrides divide raw current by geometric area; already exported current density
is not divided twice. Unconfirmed current-export areas retain the flagged 1 cm2
assumption. No additional iR correction or target capping is introduced.

```text
If the final j > -10 mA/cm2:
    Record the maximum scanned overpotential and its current as a scan limit.
Otherwise:
    Detect rising candidates using the fixed shape criteria below.
    Keep candidates with -20 < raw j < -10 mA/cm2.
    If crossings exist, candidates must be later than the largest crossing.
    If eligible candidates exist:
        Select the largest-overpotential candidate as a rising-point estimate.
    Else if a -10 crossing exists:
        Select the largest-overpotential interpolated crossing as eta10.
    Else:
        Find the raw point closest to -10, breaking ties by largest potential.
        If -20 < raw j < -10: retain it as a nearest-point estimate.
        Otherwise: no accepted result.
```

No approval list, sample name, origin label or previous selected value is used
to choose a proposed point. The baseline is read only to calculate differences.

### Fixed shape criteria

The detector settings are retained from the existing diagnostic detector, not
tuned to match the earlier selections:

1. Require an overall increasing-overpotential scan, at least 20 mV span, and
   no individual reverse step exceeding 1 mV. Curves failing this shape check
   can still use the crossing/nearest fallback; they are not dropped.
2. Interpolate `q = -j` onto a 1 mV grid. Apply a five-point median filter for
   detection only. Repeated potentials retain the first recorded value.
3. Find local minima of q (local maxima of signed j) with prominence at least
   0.5 mA/cm2 and at least 10 mV between detected minima.
4. Require at least 5 mV preceding the candidate and more than 30 mV after it.
5. Require at least 2 mA/cm2 subsequent increase in q. For candidate grid index
   k and e = min(k + 100, final index), calculate
   `median(q[max(k + 10, e - 10):e + 1]) - q[k]`.
6. Map each detected candidate to the nearest raw potential. Check the strict
   -20 < raw j < -10 window before rounding; report raw potential/current,
   not smoothed or interpolated estimate coordinates.
7. With multiple eligible candidates, select the largest overpotential. For a
   curve with crossings, also require it to be later than the largest crossing.

These are operational shape criteria, not a proof of HER onset or a method for
chemically separating HER from non-HER current. Narrow fluctuations can pass
them. No claim of validated onset detection is made.

## Changes

| Category | Current | Automatic proposal |
| --- | ---: | ---: |
| Accepted eta10 crossings | 1,106 | 1,104 |
| Rising-point estimates | 63 | 65 |
| Nearest-point estimates | 7 | 7 |
| Scan limits | 382 | 382 |
| No accepted result | 25 | 25 |

Six curves change, all sourced from the original LSV directory. All 876
additional curves retain the same metric, potential and current density.
In total, 1,577 curves retain their selected metric and values.

| File | Current type | Current potential (mV) | Proposed type | Proposed potential (mV) | Proposed j (mA/cm2) |
| --- | --- | ---: | --- | ---: | ---: |
| repeat4_Co-Ce.txt | eta10 crossing | 135.77 | Rising estimate | 273.75 | -19.37 |
| repeat6_Co-Pt.txt | Rising estimate | 43.31 | Rising estimate | 39.30 | -12.99 |
| repeat6_Fe-Pt.txt | Rising estimate | 37.71 | Rising estimate | 33.72 | -13.20 |
| repeat5_Ga-Cd.txt | eta10 crossing | 322.87 | Rising estimate | 489.76 | -16.55 |
| repeat3_Pd-Pd.txt | Rising estimate | 69.67 | Rising estimate | 67.26 | -15.15 |
| repeat5_Y-Pd.txt | Rising estimate | 55.27 | Rising estimate | 53.67 | -15.47 |

The four small positional changes use the detector's raw-point mapping instead
of the previously recorded raw-line choice. The two category changes are more
important: a crossing is replaced by a later estimate, not a revised eta10.

### Sample-statistic consequences

- Co-Ce: accepted crossings decrease from two to one. The eta10 mean changes
  from 123.54 to 111.32 mV; its former 17.29 mV SD becomes undefined (n = 1).
  The proposed 273.75 mV estimate is recorded separately, not averaged into it.
- Ga-Cd: its only accepted crossing is replaced by an estimate. The numerical
  eta10 mean becomes missing; four scan limits and the estimate remain separate.
- Co-Pt, Fe-Pt, Pd-Pd and Y-Pd: only rising-point summary statistics change.
  Their accepted eta10 counts, means and SDs do not change.
- Sample labels with accepted eta10 decrease from 430 to 429; labels with an
  eta10 SD decrease from 353 to 352.

### Review concern

Co-Ce's proposed point appears as a short fluctuation on a generally descending
trace, rather than an unambiguous broad second rise. Its filtered prominence is
2.02 mA/cm2. Ga-Cd has a shallow later maximum with prominence 0.63 mA/cm2. Both
meet the current numerical criteria. The PDF shows them explicitly; neither is
adopted here. Any stricter shape criterion should be defined and applied to the
entire dataset, not introduced as a catalyst-specific exception.

## Files and reproduction

- `changed_curves_review.pdf`: six pages, full curve and selection detail; Arial.
- `changed_curves_overview.png`: six-panel overview of current/proposed points.
- `changed_curves.csv`: all six changes, raw-line provenance and both densities.
- `curve_comparison.csv`: comparison for every curve.
- `proposed_curves.csv`: separate metric columns for the automatic proposal.
- `changed_samples.csv`: all affected sample-level statistics.
- `proposed_samples.csv`: separate eta10/rising/nearest summaries for all labels.
- `all_detected_candidates.csv`: candidates, eligibility, prominence, recovery
  and selected flags; candidates suppressed by the terminal gate are not detected.
- `comparison_summary.json`: counts, fixed parameters, dependency versions and
  protected-file/directory hashes.
- `before/`: frozen current extraction used for comparison.

Run from the parent `revision_figure` directory:

```powershell
python -m unittest test_automatic_rising.py test_extract_eta10.py
python compare_automatic_rising.py --plots
```

The comparison now imports the unchanged detector and automatic selection from
the shared `lsv_rules.py`. It does not read approval records or apply the adopted
Co-Ce exception. It imports raw parsing from `extract_eta10.py` without running
the production extraction, and compares against the frozen `before/` snapshot.
The comparison is deterministic and does not use random seeds or fit a model.
The script verifies raw SHA256 hashes and checks that protected current files,
ML, raw-data and publication-figure directories are unchanged.

No ML was rerun, no manuscript/response text was changed, and nothing was committed.
