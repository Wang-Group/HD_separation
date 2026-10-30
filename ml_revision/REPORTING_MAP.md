# Reporting assets

| Content | Asset | Reproduction |
|---|---|---|
| H/D nested validation and representation comparisons | `tables/performance_full_precision.csv` | `publication.py` |
| Main H/D SHAP, random validation without BO | `figures/prediction/hd_shap_random.*` | `publication.py` |
| H/D validation-arm SHAP and selection frequency | `figures/prediction/hd_validation_shap.*`, `hd_selection_frequency.*` | `publication.py` |
| All input attributions and selection frequencies | `tables/feature_importance_and_frequency.csv` | `publication.py` |
| Paired electrochemical configurations | `figures/prediction/electrochemical_shap.*` and performance tables | `publication.py` |
| Initial-LSV versus averaged activity prediction | `figures/prediction/eta10_*` | `publication.py` |
| Observed descriptor/outcome correlations | `descriptive/observed_correlations/descriptor_outcome_correlations.*` | `observed_correlations.py` |
| Work-function versus activity projections | `figures/eta10_work_function_projection/` | `plot_eta10_work_function_projection.py` |
| Nominal versus measured composition | `figures/precursor_vs_icp_composition/` | `plot_composition.py` |
| Work function / H adsorption / electronegativity maps | `figures/weighted_descriptor_maps/` | `plot_weighted_trio_letter.py` |
| Six-property pairwise atlas | `figures/si_candidates/six_weighted_descriptor_pairs/` | `plot_six_weighted_descriptor_pairs.py` |

The SI table rows in `tables/si_table_rows.json` are generated from the same
full-precision records. They include selected and all-input benchmarks.
Figures use Arial and editable SVG text. `figure_typography.py` normalizes
excessive superscript rise and regenerates the matching PDF/PNG files.
