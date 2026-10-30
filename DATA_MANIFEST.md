# Publication Data Map

Paths below are relative to the repository root. Numerical source tables, not
document panel numbers, define the analysis inputs.

## Predictive Models

| Data | Location | Use |
| --- | --- | --- |
| Sample identities, recipes, elemental properties and H/D targets | `ml_revision/inputs/reference/` | H/D models and descriptor construction |
| ICP workbooks and composition mapping | `ml_revision/inputs/icp/`, `ml_revision/reference_icp/` | Measured composition weights; shared mapping for recipe-equivalent references |
| Paired raw LSV/EIS/Et and extracted covariates | `ml_revision/inputs/electrochemical/`, `ml_revision/inputs/paired/` | Acquisition-matched electrochemical comparison |
| Per-curve activity records and excluded BO references | `ml_revision/inputs/lsv_crossings/` | Crossing-only initial and averaged activity targets |
| Outer and inner memberships | `ml_revision/inputs/fold_locks/` | Matched validation comparisons |
| Target inclusion and values | `ml_revision/results/eta10_target_audit.csv` | 298 matched activity-model samples |
| Held-out predictions, metrics and fitting records | `ml_revision/results/` | Reported three-repeat model results |
| Performance and feature-attribution tables | `ml_revision/tables/` | Full-precision source data for reporting |

H/D validation uses 560 initial-screening samples. The with-BO training arm adds
111 independent BO observations; the 37 reference measurements are excluded
from every ML workflow. The activity comparison also excludes BO-reference
measurements when averaging targets. Both activity targets use the same samples
and folds, with the mean capped to 10-550 mV after aggregation.

The material feature space contains 30 descriptors. Paired electrochemical
configurations add three measurements; the explicit-composition comparison adds
one fraction. Feature selection is correlation-aware and confined to training
data. Details and checks are in [ml_revision/README.md](ml_revision/README.md).

## Experimental Repeatability

| Data | Location |
| --- | --- |
| Raw LSV files, file hashes and area metadata | `hd_separation_ml/data_extraction/all_data/LSV_repeat/` |
| Per-curve extraction | `0-Figures/Figure1/revision_figure/eta10_curves.csv` |
| Separate crossing/estimate statistics | `0-Figures/Figure1/revision_figure/eta10_samples.csv` |
| LSV SD distribution and figures | `others_revision/LSV_repeat_analysis/` |
| H/D replicate data, SD distribution and figures | `others_revision/HD_repeat_analysis/` |
| Recipe pooling evidence and joint mean/SD table | `others_revision/reproducibility_overview/` |

There are 1,583 LSV files: 1,105 accepted crossings, 64 rising-point estimates,
seven nearest-point estimates, 382 scan limits and 25 unassigned no-crossing
curves. These are curve counts, not ML sample counts. Crossings alone enter
experimental eta10 means and SDs; estimates and limits retain separate fields.
Recipe-equivalent reference measurements are retained for experimental
repeatability, unlike the ML cohorts. See the [extraction rules](0-Figures/Figure1/revision_figure/README.md).

## NMR and Mass Spectrometry

The `input_manifest.json` files in `others_revision/nmr/` and
`others_revision/mass_spectrometry/`, together with `others_revision/ASSET_INDEX.json`,
record packaged dependencies and publication assets. The package includes NMR
calibration/enrichment analyses and numerical MS calibration/interference data.
Use [others_revision/README.md](others_revision/README.md) for the available
calculations and verification commands. Other experimental characterizations
are not implied to be reproduced by this package.

## Figures and BO Records

- [Assembled manuscript figures](0-Figures/manuscript/README.md) are editable SVGs
  with PNG/PDF companions. Panel-level analysis sources are listed there.
- [ML reporting map](ml_revision/REPORTING_MAP.md) links SHAP, performance,
  correlations, composition plots and descriptor maps to their plotting scripts.
- The full six-property SI atlas is retained in
  `ml_revision/figures/si_candidates/six_weighted_descriptor_pairs/`.
- `0-Figures/Figure4/curated/figure4_bo_complete_112.csv` contains the recorded
  four-system BO performance data: 28 observations each for Co-Mo, Co-V, Co-Ni
  and Fe-Ni. These records are not replaced by repeatability means.
- [BO provenance](EC_tool_sys/BO_PROVENANCE.md) documents the optimization code
  and its recorded objectives.

Experimental replicate SD, validation-repeat SD and conditional NMR intervals
are different uncertainty quantities and remain separately identified.
