# Acquisition Data and Extraction

## Data

- `all_data/LSV/`: acquisition LSV curves and electrode-area metadata.
- [all_data/LSV_repeat/](all_data/LSV_repeat/README.md): initial and repeated
  LSV files with sample identities, areas and file hashes.
- `all_data/EIS/`: raw EIS spectra.
- `all_data/CV20/`: raw cyclic-voltammetry curves.
- `all_data/Et/`: potential/time traces.
- `separation_coefficient.xlsx`: H/D measurement labels.
- `data_extraction.ipynb` and audit scripts: acquisition processing and provenance.

## Area Normalization

`all_data/LSV/lsv_geometry_area_metadata.csv` identifies 2 cm2 electrodes in
the acquisition set. Geometric density is `j = I / A`; a 2 cm2 electrode reaches
-10 mA/cm2 at -20 mA total current. The repeated-LSV manifest records the area
used for each file. Already-normalized exports are not divided by area twice.

## Publication Analyses

Use [LSV extraction](../../0-Figures/Figure1/revision_figure/README.md) for the
consistent crossing/estimate/scan-limit rules and experimental statistics.
Use [ml_revision](../../ml_revision/README.md) for reported predictive results;
it packages its target tables and acquisition-matched LSV/EIS/Et inputs.
The paired covariates are not replaced by repeatability averages.

Raw exports and their audit records remain unchanged by figure reorganization.
