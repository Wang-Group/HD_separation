# Reviewed Raw LSV Curves

Local review selection, updated 2026-10-24. Not automatically approved for publication.

This directory contains 1583 byte-preserved TXT files across 677
sample/same-recipe groups: 670 original curves,
733 supplementary repeats, 143 historical archive curves,
and 37 historical same-recipe reference measurements.
On 2026-10-24, the user explicitly restored 48 previously excluded original
curves, overriding those historical UI decisions. Other excluded repeat curves
remain omitted. Ag-Ce and Mo-Mg retain only the numerically identical 168-archive
versions; their original copies were not added again. Co-Fe32 retains its
reference_Co-Fe grouping. Restoration records and the previous manifest are in
`0-Figures/Figure1/revision_figure/restoration_20261024` at repository root.
The 68 all-above-target curves excluded from Figure 1 ARE included here.
Figure-specific selection does not determine raw-data inclusion and must be
applied separately downstream, without removing these raw measurements.

## Organization

- `repeat/<sample>/`: retained original, supplementary and archive measurements.
- `reference/<system>/`: existing identifiers for same-recipe BO measurements.
  These are ordinary sample groups in the current figure, not a separate
  performance class. Preserve their mapping to avoid duplicate counting.
- `manifest.csv`: relative source paths, curve identities, units, area evidence,
  and SHA256 checksums. No local absolute paths are included.

Existing file numbers are preserved and may have gaps after exclusions.
Repeat numbering does not by itself establish independent-electrode counts.
The folder includes retained original curves; do not blindly concatenate it
with the neighboring historical `LSV` directory and count those files twice.

## Units and Area

All TXT bytes, values and acquisition headers are unchanged.
The 144 files marked `-2cm2` retain that confirmed area annotation;
the plotting-only division by two has NOT been applied to the raw files.
Check the exported unit before area normalization: an already normalized
current density must not be divided by area again. Absence of the suffix
does not independently establish a 1 cm2 electrode area.

No eta10 extraction, current scaling, fitting or machine learning is performed
by this export. The original raw inputs are not modified.
