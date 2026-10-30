# Other Revision Analyses

Assets are named by analysis content, independently of manuscript, SI and response-letter numbering.

This package reproduces the adopted NMR quantification and mass-spectrometry
control figures and tables, plus the LSV/H-D repeatability analyses.
Each analysis contains its required inputs, scripts and scientific outputs.

The package does not modify raw files, H/D labels, LSV selections, ML fits,
the manuscript, or the response document when run.

## Reproduce

Tested with Python 3.13.5 and the versions in `requirements.txt`. Install Arial
on the plotting machine to reproduce the requested typography. Numerical
results do not depend on the font. Run from the repository root:

```bash
python -m pip install -r others_revision/requirements.txt
python others_revision/run.py
python others_revision/run.py --verify-only
```

No network drive or other worktree is required. The runner verifies raw-input
hashes, executes the analysis/export scripts, checks the NMR numerical
references and MS source-table consistency, and prints completion status to
the console. It does not save execution logs or create an execution archive.
`--verify-only` checks the existing results without regenerating figures.

[ASSET_INDEX.json](ASSET_INDEX.json) links content names to figure/table files,
source results, reproduction commands and checksums. Run
`python others_revision/index_publication_assets.py --check` to validate it without
running analyses; omit `--check` to refresh it after an intentional export.

## Files and Destinations

| Analysis | Code | Results |
|---|---|---|
| NMR calibration, enrichment and spectra | `nmr/run_analysis.py`, `nmr/plot_response.py`, `nmr/export_tables.py` | `nmr/results/figures`, `nmr/results/tables`, numerical CSV/JSON files |
| Hiden hydrogen control | `mass_spectrometry/analyze.py`, `mass_spectrometry/plot_response.py` | `mass_spectrometry/outputs/interference_sensitivity/hiden_hydrogen_control.*` |
| H3 interference sensitivity | `mass_spectrometry/interference_sensitivity.py` | `mass_spectrometry/outputs/interference_sensitivity/h3_separation_sensitivity.*` and CSVs |
| ML figures and tables | Separate finalized three-repeat package | `../ml_revision/FIGURE_INDEX.json` and `../ml_revision/REPORTING_MAP.md` |
| Figure 1B and LSV repeatability | Separate LSV extraction/plotting workflow | `../0-Figures/Figure1/revision_figure/` |
| LSV repeatability | `LSV_repeat_analysis/plot_and_tables.py` | `LSV_repeat_analysis/figures/lsv_repeatability.*` |
| H/D repeatability | `HD_repeat_analysis/analyze.py`, `HD_repeat_analysis/plot_distribution.py` | `HD_repeat_analysis/figures/HD_repeatability_distribution.*` |
| Reproducibility overview | `reproducibility_overview/build_overview.py` | `reproducibility_overview/reproducibility_overview.*` |

The table above defines this package's reproduction scope; it is not a
complete source-data archive for every figure in the response letter.

Document figure, table and citation numbering is maintained outside this package.

## LSV Repeatability

`LSV_repeat_analysis/` contains the Arial SD-distribution figure,
extraction-table inputs, reproduction scripts and proposed SI/MS
integration notes. It reuses the existing Figure 1 extraction without changing
raw curves, extraction rules or ML results.
This analysis is not
run by the NMR/MS runner above. See its README for reproduction.

## H/D Repeatability

`HD_repeat_analysis/` reviews the adopted H/D measurement-level coefficients,
with an Arial SD histogram and cumulative distribution, source-linked tables and a proposed
SI paragraph. Technical MS injections are kept distinct from repeat
measurements. Its standalone scripts do not update Figure 1B, ML labels or
working documents, and are not part of the NMR/MS runner above.

## Combined Reproducibility Overview

`reproducibility_overview/` contains the combined eta10-versus-H/D scatter plot
for catalysts with either SD, its complete numerical point table and explicit
missing-coordinate flags. It uses the two repeatability folders above without
changing raw measurements. Run its standalone `build_overview.py` to reproduce
the overview in Arial. It uses the same recipe-based pooling and LSV means/SDs as Figure
1B. The three estimate-only means are labelled separately from eta10.
See its README for counts, recipe checks, reproduction order and evidence.

## NMR Definition and Limits

There are 35 spectrum roles but 29 unique FIDs: 16 standard roles, seven product
roles, seven feed roles and five comparator roles. Stage 2 reuses the Stage-1
product spectrum because that material was transferred directly without
dilution; it is not an independent feed measurement. Stages 3-7 have separately
prepared feeds. Directory-to-stage assignments and internal-standard groups
are explicit in `nmr/sample_mapping.json`.

The adopted `saved_1r` method integrates the instrument-saved real spectra.
Peak centers are determined independently of phase from Fourier-transformed
FID magnitudes. Integration uses +/-0.25 ppm windows, interpolated endpoints,
local linear baselines and signed trapezoids. The low-range calibration fits
the source-water background and response coefficient; the high-range fit uses
that background with its own coefficient. Unknown concentration is water/DMSO
integral ratio divided by the appropriate coefficient, with no additional
blank subtraction or 0.95 factor. The stock value is 99.86 atom% D under the
stated volume-mixing approximation.

The outputs retain ACME and integration-window/standard-exclusion checks
separately as robustness diagnostics. They are not substituted for the adopted
saved-phase concentrations, response figures or tables. Superseded recipes
and prior manuscript concentration tables are not used by this package.

Uncertainty uses 50,000 shared sequential Student-t calibration draws, seed
20261023, with the reading component drawn once per unique FID. These are
conditional model intervals, not experimental replicate SDs. Stock purity,
preparation, systematic phase and volume uncertainties are not covered.
Stage-4/5 feed acquisitions contain ADC-overflow records. The 50% standard
preparation remains a recorded sensitivity limitation. Stage-7 intervals are
not clipped at 100%; the endpoint is approximately 99.8 atom% D.

The enrichment factor is product atom% D / feed atom% D. It is not the
gas/liquid H/D separation coefficient used for the screening ML target.
Stagewise enrichment inventory's volume-based apparent retention is an inventory proxy, not measured
recovered-water yield. Independently formulated stages do not establish a
closed or physically connected seven-stage isotope balance.

## Mass Spectrometry Definition and Limits

All 40 detected Hiden control plateaus are retained. They are not 40
independent experiments. The conditional H3 sensitivity uses the rounded
Hiden H2/HD calibration equations supplied in the experimental figure,
10 atom% D in liquid, negligible gas-phase D2, and an additional residual
m/z=3 contribution beyond the calibration. It is not a correction applied
to catalyst labels. Its control spread is not a confidence interval.

See [the MS README](mass_spectrometry/README.md) for the exact input, commands,
calibration coefficients and figure outputs. This folder does not reproduce
independent instrument-calibration fits or the instrument comparison.

## Figure Typography

After exporting figures, run `python index_publication_assets.py`. This applies
the shared numeric-superscript correction in `../0-Figures/figure_typography.py`,
renders PNG/PDF companions from editable SVG, and refreshes asset hashes.
Only excessive superscript positions are changed; scientific data and figure
dimensions are preserved. Audit without writing with
`python ../0-Figures/figure_typography.py .`.
