# Figure 4: Bayesian Optimization and Descriptor Projections

This folder contains the data, scripts, notebooks, and exported panels used for
the Bayesian optimization (BO) analysis in Figure 4 and the corresponding SI
descriptor-space figure.

## Trusted Data

### BO Performance Table

`curated/figure4_bo_complete_112.csv`

This is the authoritative BO table for Figure 4A/B.

- 112 completed BO experiments.
- Four systems: `Co-Mo`, `Co-V`, `Co-Ni`, and `Fe-Ni`.
- 28 experiments per system, corresponding to seven batches with four
  experiments per batch.
- Stores curated H/D separation coefficients and LSV-derived potentials used
  for the Pareto-front and hypervolume plots.

### Primary-Screening Descriptor Source

`../../hd_separation_ml/outputs/feature_selection_561_rs_from_eis_no_eis_fitting/feature_matrix_561_no_eis_fitting.csv`

This table is generated from the current machine-learning workflow and the
current descriptor table in `../../hd_separation_ml/EH_extra.xlsx`.

For initial bimetallic screening, weighted descriptors assume the nominal
metal ratio of 6:1. For same-metal samples, the first metal receives weight
1.0 and the second receives weight 0.0.

## Figure 4A/B: Pareto Front and IoH

Notebook:

`pareto_polygon_enclosure_code.ipynb`

Primary output:

`eight_panels_2x4_batch1_7_lg.png`

This notebook draws:

- Pareto-front expansion for the four BO systems.
- Improvement of hypervolume (IoH, %) across batches.
- Batches 1-7 only; IoH is calculated relative to batch 1.
- RHE-converted potential at -10 mA cm^-2 in mV.

## Figure 4C / Figure S15 Descriptor Maps

Main SI assembly script:

`assemble_figure_S15_descriptor_panels.py`

Helper:

`descriptor_joint_panel_helpers.py`

Run:

```powershell
& C:\Users\yibinjiang\anaconda3\python.exe .\assemble_figure_S15_descriptor_panels.py
```

Output:

`outputs/figure_S15_descriptor_space_interpretation_BO_outcomes.png`

The current descriptor-space analysis uses the three descriptor dimensions
that match the current machine-learning interpretation:

- weighted electronegativity;
- weighted H adsorption free energy;
- weighted work function.

The BO weighted descriptor table is generated on the fly by
`plot_top3_shap_descriptor_bo_overlay.py` from the current ML feature matrix,
the current elemental descriptor table, and the curated BO table. The figure
does not use the removed legacy OH-weighted descriptor table.

## Top-SHAP Descriptor Overlay

Script:

`plot_top3_shap_descriptor_bo_overlay.py`

Run:

```powershell
& C:\Users\yibinjiang\anaconda3\python.exe .\plot_top3_shap_descriptor_bo_overlay.py
```

Outputs:

- `outputs/top3_shap_weighted_descriptor_bo_overlay.png`
- `outputs/top3_shap_metal1_metal2_descriptor_bo_overlay.png`
- `outputs/top3_shap_initial_descriptor_projection_table.csv`
- `outputs/top3_shap_bo_descriptor_projection_table.csv`

The weighted plot uses metal-salt-ratio-weighted electronegativity, H
adsorption free energy, and work function. This is the recommended
descriptor-space view for BO composition trends because BO samples vary
continuously in precursor ratio.

Samples with missing raw work-function values in `hd_separation_ml/EH_extra.xlsx`
are omitted only from projections that use weighted work function; they remain
available in projections that use the other descriptor dimensions.

## Notes

- The BO Pareto/hypervolume panels are based on measured BO results.
- The descriptor-map contours are reconstructed from the primary-screening
  dataset and should be described as descriptor-space guides, not formal
  predictive surfaces.
- Legacy OH-based Figure 4C/S14 inputs and outputs were removed after the OH
  descriptor table was corrected, to avoid accidental reuse of stale
  `weighted_delta_EOH` values.
