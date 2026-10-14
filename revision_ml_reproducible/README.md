# ICP-Informed H/D Separation ML: Reproducible Revision Package

The main workflow uses **33 candidates -> nested forward selection of 10 features**.
The original 39 candidates are a redundancy sensitivity, not a second primary model.
Contact: **Cheng Wang, wangchengxmu@xmu.edu.cn**.

This directory can be moved or shared by itself. It does not read the parent
checkout, Downloads, Desktop, network drives, or the original absolute run paths.
No network access is required after installing the Python environment.

## Contents and Path Convention

This is a small repository-shaped snapshot. An input at `hd_separation_ml/...`
or `0-Figures/...` in the original repository has **exactly the same relative
path from this folder's root**. The files are copies, not symlinks.

The package-local `.gitattributes` disables Git line-ending conversion so that
cloning on Windows or another platform preserves the checksum-verified bytes.

```text
revision_ml_reproducible/
  run.py                       Supported entry point
  workflow/                    Portable orchestration and package tests
  config.json                  Candidate pool, seeds/settings, environment
  requirements.txt
  environment.yml
  hd_separation_ml/
    revision_inputs/           Separate targets, metadata, properties, Co-Fe ICP
    EH_extra.xlsx              Elemental-property source workbook
    *.py                       Unchanged numerical helper modules
  0-Figures/Figure4/
    data/                      Archived ICP source files
    curated/                   Original BO recipes and responses
  outputs/icp_simple_20261012/
    ICP_raw_and_percentages_EN.xlsx
  reference/                   Verified data, ordered splits, settings, predictions
  published_results/           Archived results, figures and interpretation notes
  validation/                  Package validation evidence
  runs/                        Regenerable new outputs; ignored by Git
  source_inventory.json        Original relative paths and byte-level hashes
  package_manifest.json        Checksums of the distributable folder
```

`outputs/.../ICP_raw_and_percentages_EN.xlsx` is the approved English analysis
workbook containing raw ICP values and derived percentages. The received ICP
source snapshots are retained under their original `0-Figures/Figure4/data/...`
paths. The workflow checks the workbook percentages against the raw values and
approved atomic masses every time; it does not trust percentages blindly.

Raw electrochemical LSV/EIS curves are not inputs to this intrinsic-descriptor
workflow. Separate eta10, DRT and experimental-analysis pipelines are outside
this package. Original file locations inside provenance text are historical
metadata, not files the portable runner opens.

## Environment

Numerical replay was validated using Python **3.13.5** and the pinned packages.
Create a separate environment rather than changing another project's environment:

```shell
conda env create -f environment.yml
conda activate hd-separation-revision
```

Alternatively, with Python 3.13.5 already installed:

```shell
python -m pip install -r requirements.txt
```

The commands below are written from this folder. `run.py` also works from another
working directory when invoked by absolute path. All relative `--source`,
`--output` and `--targets` arguments are resolved against this folder, not the
shell's working directory. Fonts can differ across operating systems; Arial is
used when available. Exact scientific replay enforces the original core runtime.

## Quick Verification and Refit

```shell
python run.py audit --regenerate-splits
python run.py test
python run.py refit --pool 33 --jobs 10
```

The audit rebuilds **554 initial + 111 independent BO** rows from the workbooks,
checks reference properties and shared-ICP exclusions, and verifies ordered outer
and inner memberships. The 111 BO points comprise Co-Fe (27) and Co-Mo, Co-Ni,
Co-V, Fe-Ni (21 each).

The refit trains fresh forests using the saved fold-local feature choices and
hyperparameters, checks predictions against the archived results, and recomputes
held-out SHAP. It is **not a rerun of feature selection**. It covers both BO arms,
three tasks, three repeats and five outer folds, plus both frozen specifications.
Use `--no-shap` for a faster predictions-only check. Outputs go to `runs/refit_33/`.

Expected selected-10 R2 with BO, mean +/- repeat SD:

| Evaluation | R2 |
|---|---:|
| Random initial entities | 0.556 +/- 0.009 |
| Unseen unordered metal pairs | 0.555 +/- 0.008 |
| New compositions in known pairs | 0.544 +/- 0.018 |

## Full Feature Selection from Scratch

```shell
python run.py train --pool 33 --jobs 10
```

This recalculates the forward path (1-16 features), then tunes RF/Extra Trees
inside three inner folds. It reports forward-10, inner-best, one-SE and all-33
models. The 90 outer jobs use 120 selection trees, 80 tuning trees and 400 final
trees, matching the completed analysis. No archived selection scores are reused.
The full search is substantially slower than refitting. Jobs and path/tuning
steps are checkpointed; rerun the same command to resume unchanged work.

For a short plumbing test only:

```shell
python run.py train --smoke --jobs 1 --output runs/smoke
```

Smoke results use two path steps and small forests; never use them as scientific
results. `--limit N` runs only the first N outer jobs and is also marked partial.

The original 39-candidate sensitivity uses the same implementation:

```shell
python run.py refit --pool 39 --jobs 10
python run.py train --pool 39 --jobs 10
```

## Frozen Model and Composition Controls

```shell
python run.py freeze --source reference/pool33 --jobs 10
python run.py composition --source reference/pool33 --jobs 10
python run.py figures
```

`freeze` reconstructs the Task-1 consensus ten features and retunes the global
RF/Extra-Trees specification using Task-1 information only, then refits it across
the three tasks. This remains a **retrospective sensitivity**, not an independent
nested estimate. The original and reselected frozen specifications are both
included in `refit`; their Task-3 differences are retained rather than hidden.

`composition` compares raw22, nominal precursor33, ICP33 and ICP33 plus explicit
fraction, using matched splits and independent inner tuning. It also runs a
fixed selected-10 weighting substitution. Nominal weights use the historical
recipe-volume convention, not an assumption of equal deposited composition.

`figures` regenerates the observed electronegativity comparison and pair-labelled
BO trajectories without imputation. Means, 90th percentiles and unconstrained
quadratic fits are distinguished. These are descriptive plots, not CV metrics.

After a fresh search, replace `reference/pool33` with `runs/train_33` in the
`freeze` and `composition` commands to use its newly selected models.

## Changing the Response Values Later

Preserve the approved snapshot. Create a separate target table:

```shell
python run.py targets-template --output runs/new_y_template
```

`runs/new_y_template/targets.csv` contains exactly the included 665 sample IDs.
Update its `separation_coefficient` values while retaining unique IDs and finite
values. Other files must not be used to inject revised y into cached runs.

```shell
python run.py train --pool 33 --targets runs/new_y_template/targets.csv --output runs/new_y_nested --jobs 10
python run.py freeze --targets runs/new_y_template/targets.csv --source runs/new_y_nested --output runs/new_y_frozen --jobs 10
python run.py composition --targets runs/new_y_template/targets.csv --source runs/new_y_nested --output runs/new_y_composition --jobs 10
python run.py figures --targets runs/new_y_template/targets.csv --output runs/new_y_figures
```

Changing y automatically regenerates stratified/grouped splits and disables
comparison against the old prediction values. Fresh feature selection and tuning
are required. Changes to cohort membership or ICP/property source data require
a newly reviewed input snapshot, not silently editing the sealed package.

## Interpretation and Reference Results

- 33 candidates = 11 property families x M1/M2/ICP-weighted representations.
- D frequency and H/D-ZPE variants are removed; H frequency is retained.
- Ten features are selected separately within each outer fold, not globally.
- Imputation uses training-only elemental medians before composition weighting.
- Task 1 is a stratified interpolation benchmark; Task 2 isolates metal pairs;
  Task 3 holds out 5 mol% composition bins. Shared ICP observations are not split
  across train/test, and matching BO composition bins are purged.
- All main inputs are intrinsic properties and measured ICP composition. Current,
  eta10, resistance and sample-specific potential shifts are excluded.
- Repeat SD is split variability, not an independent-sample confidence interval.
- ICP is bulk composition, not a measurement of active-site composition or alloy
  surface properties. SHAP is predictive attribution, not causal identification.

See [the latest results note](published_results/reviewer_ml_completion_20261014/RESULTS.md)
and [the 39-to-33 sensitivity](published_results/forward33_comparison_20261013/RESULTS.md).
Earlier operating-variable, missing-reference and identity controls are archived
as supporting evidence. Their historical scripts are retained as dependencies,
but use this package's `run.py` commands rather than invoking legacy helper CLIs
that expect the original sequence of run folders.

New outputs belong in `runs/`, which is ignored by Git. The input snapshot,
reference results, portable code, archived figures and validation evidence remain
versionable. Hashes reject stale checkpoints after code/data/settings changes.
See [VALIDATION.md](VALIDATION.md) for what was actually rerun during packaging.
