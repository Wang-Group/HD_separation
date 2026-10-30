# Unified LSV extraction review

Date: 2026-10-25. This update classifies the original and additional LSV files
with the same rules. It does not change raw data, sample grouping, Figure 4/BO,
ML inputs/results, or the response letter. No figures were regenerated.

## Curve counts

| Classification | Original LSV files | Additional files | Total |
| --- | ---: | ---: | ---: |
| Accepted eta10 crossings | 423 | 683 | 1,106 |
| Approved rising-point estimates | 8 | 16 | 24 |
| Approved nearest-point estimates | 10 | 36 | 46 |
| Scan limits | 256 | 126 | 382 |
| No accepted crossing or approved estimate | 10 | 15 | 25 |
| Total | 707 | 876 | 1,583 |

These are counts of curve files, not catalysts. The additional files comprise
660 LSV_error_bar curves, 73 S curves and 143 archive-168 curves. The duplicate
original Ag-Ce and Mo-Mg files remain omitted; their archive versions are retained.
Reference files are counted by source and are not duplicated as another category.

## Rules and changes

- Normalize current before extraction, retaining the documented area overrides,
  exported-density treatment and flagged 1 cm2 assumptions in the manifest.
- The final-current scan-limit rule has priority. If final j > -10 mA/cm2,
  record the largest scanned overpotential as a bound, even with earlier crossings.
- Otherwise use the largest-overpotential -10 mA/cm2 crossing by linear
  interpolation, except for an explicitly approved rising-point selection.
- An approved estimate must satisfy -20 < selected j < -10 mA/cm2 before
  rounding. This restriction applies to the selected point, not the entire curve.
- A nearest estimate must be the recorded point closest to -10 on a curve
  without a crossing. A rising estimate is a separately reviewed raw point.
- Explicit curve review is required: the numerical threshold alone does not
  approve a curve. Estimates retain their potential AND actual current density,
  with blank eta10. Neither estimates nor scan limits enter eta10 mean/SD.

The 70 approved estimates comprise the 20 existing rising selections, 13 verified
original-curve decisions carried forward from the June Figure 1 review, and two
additional Co-V11 repeat selections, and the 35 newly approved nearest points.
The original Co-V11 is already among the
13 June records and is not counted twice. The 13 original decisions contain nine
nearest points and four turning points. All selected raw lines and source hashes
were verified, including byte identity with the corresponding original LSV files.

The initial synchronization changed 15 classifications. The subsequent approval
changes exactly the 35 plotted candidates, for 50 changes from the fixed pre-sync
snapshot, all from unassigned no-crossing to an approved estimate.
All 1,106 accepted eta10 values, all 382 scan limits, the 20 prior rising
selections, and accepted-eta10 sample means/SDs remain numerically unchanged.
The 18 user-reviewed original curves remain scan limits. The known original
Sc-Pb, Cd-Y, W-Pb and Ce-Pb charging-artifact curves are not accepted eta10.

All 35 plotted candidates were explicitly approved as nearest-point estimates.
Their potential and actual current density remain separate from eta10. The
remaining 25 no-crossing curves fail the numerical window; none qualify but
lack approval. `unapproved_nearest_candidates.csv` is retained only as the
pre-approval review input, not as a current outstanding-candidate list.

## Sample coverage

- 677 existing sample labels are retained; no recipe regrouping was performed.
- 430 have at least one accepted eta10; 353 have at least two and an SD.
- 42 have at least one approved estimate. This overlaps the eta10 coverage.
- Six have estimates but no accepted eta10: Co-Fe29, Co-Fe33, Co-V11, Eu-Pd,
  V-Pd and W-Pd.
- Co-V11 has three approved nearest estimates and one excluded curve whose
  nearest magnitude is 20.127298 mA/cm2, above the strict cutoff.
- 315 labels have at least one scan-limit curve; this also overlaps other counts.

## Files and verification

- `counts.csv`: curve counts by original/additional source.
- `classification_changes.csv`: 50 changed records relative to the pre-sync snapshot.
- `reviewed_18_scan_limits.csv`: the 18 preserved original-curve boundary decisions.
- `unapproved_nearest_candidates.csv`: frozen 35-curve input from before approval.
- `remaining_unapproved_nearest_candidates.csv`: current list; header only (zero).
- `nearest_point_review/approval.json`: user approval and reviewed-points CSV hash.
- `nearest_point_review/before_approval/`: extraction and counts before this approval.
- `nearest_point_review/approval_verification.json`: checks for this 35-curve update.
- `verification.json`: numerical preservation checks.
- `scope_verification.json`: before/after directory content hashes.
- `before/`: fixed extraction/code/documentation snapshot for the comparison.
- `../eta10_curves.csv` and `../eta10_samples.csv`: complete updated results.
- `../approved_additional_estimates.csv`: 50 additional selections and evidence hashes.

Unit/integration tests and the before/after audit verified all
accepted crossings and scan boundaries and checked that estimates were never
pooled into eta10. Raw-data and Figure 4/ML directory content hashes were unchanged.
The saved Figure 1B still predates this synchronization; its nearest-estimate
display requires a separate update. No ML was rerun and nothing was committed.
