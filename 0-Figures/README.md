# 0-Figures

This folder contains the active figure-generation material used by the current
manuscript figures.

## Layout

- `Figure1/`
  - Figure 1 bottom-panel data, notebook, and output.
  - Uses explicit CSV inputs for this-work and literature/benchmark points.
- `Figure3/`
  - Figure 3 top-panel data, notebook, and output.
  - ML panels are generated from `../hd_separation_ml/` rather than duplicated
    here.
- `Figure4/`
  - Bayesian-optimization Pareto/IoH notebook, descriptor-map notebook,
    curated BO tables, and final output images.

## Folder Convention

For Figure 1 and Figure 3:

- `data/` stores active CSV tables.
- `notebooks/` stores active figure-generation notebooks.
- `outputs/` stores generated plot exports.

Figure 4 keeps its active notebooks, workbooks, and JSON files directly in
`Figure4/` because the notebooks use those local relative paths. Checked
manuscript-facing tables are in `Figure4/curated/`.
