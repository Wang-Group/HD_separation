# Root Legacy Files

This folder stores files that previously lived in the `hd_separation_ml` root
but are no longer active entry points.

They were moved here to keep the root directory focused on the current
reproducible ML workflow.

## Contents

- `0-noble_no_BO_seed42.ipynb`
  - Older root-level working notebook.
  - Replaced by stable notebook copies in `../../notebooks/`.

- `0-noble_no_BO_seed42_plot.ipynb`
  - Older root-level fixed-feature/plotting notebook.
  - Replaced by stable notebook copies in `../../notebooks/`.

- `0-noble_no_BO_seed42/`
  - Older shared output folder.
  - Its contents were ambiguous because multiple notebook runs wrote to the
    same folder.
  - Current outputs are separated under `../../outputs/`.

- `sample_overpotentials_new.csv`
  - Older uncorrected/previous label table.

- `sample_overpotentials_new_corrected_verison.before_add_Mo-Mg_20260429_200233.csv`
  - Backup before `Mo-Mg` was appended to the corrected label table.

- `SECTION3_descriptor_table.xlsx`
- `SECTION3_descriptor_table_with_names.xlsx`
  - Previous descriptor-table artifacts retained for reference.

