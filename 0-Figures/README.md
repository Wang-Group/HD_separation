# 0-Figures Folder

The figure folders are organized by manuscript figure and by file role.

## Layout

- `Figure1/`
  - Figure 1 source workbook, active notebook, outputs, raw visual materials,
    and archived drafts.
- `Figure3/`
  - Initial-screening scatter data and notebook for the top Figure 3 panel.
  - The middle ML panel is documented here but remains in `hd_separation_ml/`
    because the ML notebook depends on that directory's local data files.
- `Figure4/`
  - Bayesian-optimization source files, notebooks, generated outputs, and raw
    images.
  - Active notebooks and their input workbooks/JSON files intentionally remain
    together in `Figure4/1-data/` because those notebooks use root-relative
    paths. Checked manuscript-facing extracts are in `Figure4/1-data/curated/`.

## Convention

Within each figure folder:

- `data/` stores source tables used by active notebooks.
- `notebooks/` stores active figure-generation notebooks.
- `outputs/` stores generated figure panels or plot exports.
- `raw_materials/` or `raw_images/` stores visual source material.
- `archive/` stores older drafts or files retained for traceability.

Figure 4 is the one deliberate exception: active notebooks stay next to their
source `xlsx`/`json` files to preserve the existing runnable paths.
