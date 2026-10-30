# Electrochemical Data and Extraction

**For the reported machine-learning results, use [ml_revision](../ml_revision/README.md).**
That package contains the correlation-aware 30-descriptor workflow, three-repeat
nested validation, saved folds and seeds, predictions, SHAP and no-fit checks.

This directory contains acquisition data and extraction provenance:

| Content | Location |
| --- | --- |
| Raw LSV, EIS, CV and potential traces | [data_extraction/all_data](data_extraction/all_data/) |
| Initial and repeated LSV files, areas and hashes | [LSV_repeat](data_extraction/all_data/LSV_repeat/README.md) |
| H/D measurement labels | `data_extraction/separation_coefficient.xlsx` |
| Elemental-property source workbook | `EH_extra.xlsx` |
| Acquisition extraction and raw-data audit scripts | [data_extraction](data_extraction/README.md) |

The [LSV analysis](../0-Figures/Figure1/revision_figure/README.md) provides
crossing values, separately identified estimates/limits and per-catalyst
statistics. The [repeatability package](../others_revision/README.md) provides
experimental means and SDs.

Publication ML reads its packaged inputs under `ml_revision/inputs/`, including
acquisition-matched electrochemical covariates and their raw-source checks.
Use that package's commands rather than the acquisition notebook to reproduce
the reported validation results.

The complete [manuscript figures](../0-Figures/manuscript/README.md) have a
single set of editable SVG masters and matching PNG/PDF exports.
