# Rising-point reclassification

Date: 2026-10-25. The user confirmed the strict interval
`-20 < j < -10 mA/cm2` and instructed that multiple eligible rising candidates
be resolved by selecting the largest overpotential.

## Scope and result

The fixed `before/` snapshot contains 46 nearest-point estimates and 24
already-approved rising-point estimates. Applying the established turning-point
detector to the 46 curves reassigns 39 to rising-point estimates; seven remain
nearest-point estimates. The 24 previously adopted rising points have no later
eligible candidate under this detector and remain unchanged.

Current totals: 1,106 accepted eta10 crossings, 63 rising-point estimates,
seven nearest-point estimates, 382 scan limits and 25 no-accepted-result curves.
The 70 approved estimates remain separate from measured eta10. No estimate or
scan limit is included in accepted-eta10 means or sample SDs.

## Detection and selection

The detector retains the earlier diagnostic settings, rather than introducing
a new HER-onset definition:

- Require a predominantly increasing-overpotential scan, at least 20 mV span,
  and no individual reversal exceeding 1 mV.
- Interpolate current magnitude onto a 1 mV potential grid; apply a five-point
  median filter for candidate detection only.
- Detect local minima of current magnitude with at least 0.5 mA/cm2 prominence
  and 10 mV separation, followed by at least 2 mA/cm2 recovery within the
  defined following window (up to 100 mV).
- Map each candidate to the nearest recorded potential. Evaluate the strict
  current-density cutoff on that raw point, before rounding.
- Choose the largest raw overpotential among eligible candidates. Do not choose
  an ineligible later point or the candidate merely closest to -10 mA/cm2.

All numeric settings and edge-window handling are in
`../../reclassify_nearest_as_rising.py`. Detection smoothing does not modify raw
TXT data, adopted potential/current values, or accepted eta10 interpolation.
These are descriptive, approved estimates, not independently established HER
onsets or measurements at exactly -10 mA/cm2.

## User-highlighted examples

| Sample | File | Adopted rising potential (mV) | Current density (mA/cm2) |
| --- | --- | ---: | ---: |
| Co-Mo23 | repeat3_Co-Mo23.txt | 116.06 | -19.13 |
| Co-Mo26 | repeat3_Co-Mo26.txt | 100.06 | -14.11 |
| Sc-Pd | repeat3_Sc-Pd.txt | 65.70 | -14.57 |
| Y-Pd | repeat2_Y-Pd.txt | 90.07 | -15.16 |
| Y-Pd | repeat3_Y-Pd.txt | 112.08 | -13.87 |
| Y-Pd | repeat4_Y-Pd.txt | 102.07 | -13.97 |

Y-Pd repeat3 also has an eligible candidate near 10.06 mV; repeat4 has one near
6.06 mV. Their later candidates above are adopted, as requested.

## Reproduction and evidence

Run from `revision_figure`:

```powershell
python reclassify_nearest_as_rising.py
python sync_eta10_rules.py --prepare-approvals
python extract_eta10.py
python -m unittest test_extract_eta10.py
python sync_eta10_rules.py
python reclassify_nearest_as_rising.py --verify
python plot_estimate_categories.py
```

Candidate detection and tests require numpy/scipy. Plotting requires matplotlib
and Arial. Extraction itself uses the Python standard library.

- `approval.json`: user rule and frozen baseline hash.
- `all_detected_candidates.csv`: detected points, eligibility and selection flags.
- `nearest_46_audit.csv`: old and new point values for every reviewed curve.
- `approved_reassignments.csv`: 39 raw-line decisions with TXT hashes.
- `candidate_summary.json`: detector settings and counts.
- `verification.json`: exactly 39 changed curve records and 1,544 identical
  records, unchanged accepted eta10 statistics, and protected-directory hashes.
- `before/`: preserved pre-reclassification records and review figures.

The updated PDFs and overview PNGs are in `../../estimate_categories_review/`.
Figure 1B, Figure 4/BO, raw data and ML inputs/results were not changed. No ML
was rerun and nothing was committed.
