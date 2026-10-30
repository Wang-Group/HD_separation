# Reproducibility Overview

Updated 25 October 2026 using the current automatic LSV extraction.
Figure 1B, the LSV SD distribution and this overview share the same
`lsv_plot_data.py` aggregation. No special Co-V11 selection override remains.

## Figure and counts

- 358 catalysts: 97 with both SDs, 255 with potential SD only, six with H/D SD only.
- 349 catalysts have a measured-eta10 SD. Three additional catalysts have an
  estimate-potential SD: Co-V11, Eu-Pd and W-Pd.
- All 103 retained H/D SDs are preserved without new mass-spectrometry processing.
- Open diamonds and labels identify estimates, not measured eta10.
- Missing error bars mean fewer than two numerical values, not zero variability.
- Means and sample SDs are uncapped. This is not SEM or proof of identical
  acquisition conditions or independent electrode preparation.

The union includes all catalysts with an eta10 SD, an estimate-potential SD,
or an H/D SD. Crossings take precedence at the sample level. Only when no
crossing is available are adopted rising/nearest potentials averaged together,
and then only as estimates. Scan limits and unassigned results are not averaged.

## Audited grouping

Matching initial-screening and reference recipes are pooled for Co-Fe, Co-Mo,
Co-Ni, Co-V and Fe-Ni. Each recorded recipe contains 15 volume units of the
first metal precursor, 2.5 of the second and 2.5 Na3Cit, with no added water
in the BO record. `recipe_evidence.json` retains JSON source hashes and
index-specific checks, and byte-matches all 37 packaged reference LSV files
to the raw archive. No duplicate raw-curve hash occurs within a pooled series.
H/D measurements are pooled by individual measurement ID, not by averaging
precomputed means or SDs. All five series use ordinary catalyst names.

The recipe audit establishes recorded ingredients and volumes, not a new
verification of stock concentrations or all synthesis conditions. The archive
curves retain their manifest identities and confirmed 2 cm2 areas. The current
Co-V export is not byte-identical to the initial batch 11-Co-1/index 6 file;
its existing manifest identity is retained, without adding another raw curve.

Co-V11 has two rising estimates and one nearest estimate under the uniform
rule, giving 16.44 +/- 13.18 mV; its fourth curve has no accepted result.
Its measured-eta10 fields remain blank.

## Difference from Figure 1B

Both figures have identical numerical LSV means/SDs before Figure 1B's
display-only clipping. This overview uses retained H/D repeat means and SDs.
Figure 1B uses initial-submission H/D values and no vertical error bars.
Both use the same recipe aliases, without duplicate catalyst entries.

## Reproduction

```bash
python others_revision/HD_repeat_analysis/analyze.py
python others_revision/HD_repeat_analysis/plot_distribution.py
python others_revision/LSV_repeat_analysis/plot_and_tables.py
python others_revision/reproducibility_overview/build_overview.py
python -m unittest discover -s others_revision/reproducibility_overview -p "test_*.py"
```

Run from the repository root with NumPy, pandas, Matplotlib and Arial.
All current inputs are local. No raw extraction, model fitting, or document
editing occurs.

- `overview_points.csv`: means, SDs, counts, metric categories and missing-SD flags.
- `summary.json`: counts, policies and source hashes.
- `reproducibility_overview.{pdf,png,svg}`: current figures.
- `recipe_evidence.json`, `audit_recipes.py`: same-recipe identity evidence.
  Rechecking external source files requires `--source-root`; plotting does not.
- `CoFe_curve_audit.csv`, `CoFe_HD_audit.csv`: individual-value checks.
- `test_overview.py`: pooling, current automatic cutoff and Figure 1B agreement.

## Response reproducibility overview

Reproducibility overview for catalysts with at least one experimental SD.
Horizontal and vertical bars show sample SDs of potential and H/D separation
coefficient, respectively. Symbols distinguish both SDs, potential SD only,
and H/D SD only. Measurements with matching documented recipes are pooled
once. Open diamonds identify the three
estimate-only potential means, not measured eta10. All other horizontal means
use accepted crossings. Missing SDs are not assigned zero. Means and error bars
are not display-clipped. Counts on the two axes do not imply paired experiments.
