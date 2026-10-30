# H/D separation and HER prediction

This package contains the inputs, code, saved nested-validation records, tables
and figures for the predictive analyses. `reproduce.py` is the main entry point.
No external results directory is required.

## Analysis design

- H/D validation: 560 initial-screening samples, three repeats of five outer
  folds with three inner folds, using target-stratified random and unseen-metal-pair
  protocols. Outer split seeds are 42, 43 and 44.
- BO augmentation: 111 independent BO records are eligible for training only.
  The 37 BO reference measurements are excluded from every ML workflow and from
  averaged activity targets. They remain available for experimental repeatability
  and shared recipe-level ICP assignments.
- Input pool: 30 material descriptors, comprising ten properties with M1, M2
  and composition-weighted representations. M-D bond energy, D vibrational
  frequency and the H/D zero-point-energy difference are excluded as correlated
  isotope-derived quantities. M-H bond energy, H frequency, electronegativity,
  standard potential and weighted properties are retained.
  M-M bond energy is in kcal/mol; M-H bond energy is in kJ/mol. The source
  formula and conversion are documented in [DESCRIPTOR_UNITS.md](DESCRIPTOR_UNITS.md).
- Selection: training-only imputation, complete-linkage groups of absolute
  Spearman correlation at 0.90, 20 seeded 80/20 training resamples for ranking
  groups and representatives, and threefold inner selection of model size and
  RF/ET hyperparameters. The highest mean inner R2 determines model size;
  ties favor fewer inputs. This is not a fixed top-ten model. Grouping does not
  guarantee that every pair of representatives has correlation below 0.90.
- Candidate sizes are 1, 2, 3, 5, 8, 10, 12, 16 and 20, plus the largest group
  count common to the inner folds. Ranking uses 120 trees, tuning 80 trees and
  final fits 400 trees. Every fit and resampling seed is recorded.
- Precursor/ICP weighting and explicit ICP M1 fraction are matched-fold
  comparisons. The extra fraction gives 31 candidates.
- Electrochemical configurations use acquisition-matched LSV, EIS and Et,
  not replicate substitutions. The current near -1 V is current density in
  mA/cm2. Measured configurations have 33 candidates. The potential-conditioning
  offset is corrected operating potential + 0.098 V; it is added to work function
  and subtracted from PZC and standard-potential representations.
- Activity prediction compares the initial LSV crossing with the accepted-crossing
  mean for the same 298 catalysts and identical folds. Only the 30 material
  descriptors are predictors. Average first, then cap targets to 10-550 mV.
  Rising/nearest-point estimates and scan limits do not enter regression targets.

Same-recipe, shared-ICP, composition-group and unseen-pair eligibility are checked
at outer and inner boundaries and inside selector resampling. The settings and
memberships in `results/` are the scientific records, not just plotted summaries.

## Reproduction

Use Python 3.13 with the pinned versions in `requirements.txt`. Install Arial
for consistent plots and Inkscape for SVG superscript normalization and export.
Run commands from this directory:

```bash
python -m pip install -r requirements.txt
python reproduce.py --check-only --jobs 6
python reproduce.py --plots-only
python reproduce.py --jobs 6 --atlas
```

The first command after installation verifies saved records without fitting.
`--plots-only` rebuilds figures/tables without fitting predictive models. The
full command resumes valid checkpoints, rebuilds raw paired inputs, checks
deterministic final refits, and exports reporting assets. To refit from scratch,
copy the package and move that copy's `results/` outside its package before
running the full command. Completed training checkpoints are deliberately
reused unless removed from the working copy.

The six-property atlas uses the repository's shared
`../0-Figures/Figure4/descriptor_joint_panel_helpers.py`. All scientific ML
inputs and raw paired electrochemical exports are packaged under `inputs/`.
The accepted-crossing input snapshot is traceable to the repository LSV data
and its extraction workflow in `../0-Figures/Figure1/revision_figure/`.

## Outputs

- `results/`: predictions, SHAP, per-repeat metrics, tuning records, selected
  groups, fit settings, outer/inner memberships and completed validation.
- `tables/`: full-precision reporting tables, all-candidate SHAP and selection
  frequencies. SD is across three pooled repeat-level scores.
- `figures/prediction/`: SHAP bars, signed sample-mean beeswarms and prediction
  insets. Unselected inputs contribute zero; the leading ten are displayed.
- `descriptive/observed_correlations/`: uncapped initial-LSV and paired-input
  correlations, using pairwise complete observations without imputation.
- `figures/weighted_descriptor_maps/`: work function, H adsorption free energy
  and electronegativity, using measured outcomes and initial-only backgrounds.
- `figures/si_candidates/six_weighted_descriptor_pairs/`: complete six-property,
  15-pair atlas with precursor/ICP weights and global/local views.
- `reference_icp/`: recipe-audited shared ICP assignments for reference samples;
  these mappings do not make those references eligible ML observations.

See `REPORTING_MAP.md` for content-based asset paths. Document numbering is not
encoded in file names. Experimental replicate SD, validation-repeat SD and
conditional calibration intervals are distinct quantities.

`results/manifest.json` fingerprints the completed fitting code and inputs.
Its configuration retains unused compatibility keys (`forward_steps` and
`selected_features`); the active selector uses `selection_config.json` and the
inner-best candidate-size rule above. These keys do not fix model size at ten.
