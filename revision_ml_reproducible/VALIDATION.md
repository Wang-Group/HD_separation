# Packaging Validation

Validated on 2026-10-14 using the pinned Windows ML environment.

- 51 tests passed, including nine package-specific tests.
- Rebuilt all 665 included samples from packaged ICP workbooks, metadata and property tables.
- Regenerated outer assignments exactly and verified ordered memberships for 90 outer arms and 270 inner partitions.
- Refit all 90 primary task/arm/fold combinations: selected-10, all-33, original frozen-10 and reselected frozen-10, with held-out SHAP for the three ten-feature specifications.
- Recomputed Task-1 consensus/global tuning and all 90 reselected frozen-model evaluations.
- Recomputed one complete 33-candidate outer job from scratch: all 16 forward steps, all declared hyperparameter candidates and all four final strategies.
- Refit one 39-candidate sensitivity job and recalculated one complete six-scenario composition comparison.
- Regenerated all three electronegativity/BO trajectory figures with the full 1,000-draw shape bootstrap.
- Maximum primary/frozen/scratch replay prediction discrepancy: 1.78e-15.
- Six-scenario composition replay discrepancy: 0.
- Copied the distributable package into a temporary directory outside this checkout, with spaces in its path. Used isolated Python from another working directory and an audit hook prohibiting reads from the original repository. The full input/split audit and one fresh four-model refit passed.

## Scope

The complete 90-job nested feature-selection search was not rerun during packaging. Its verified original outputs are included, and the portable scratch command is supplied. One full scratch job, all 90 primary refit jobs, and full frozen retuning/refits were executed to validate the numerical transfer. Partial checks remain explicitly marked partial in their machine-readable records.

All five re-rendered PNGs were checked for nonblank pixels and compared with archived images. See validation/summary.json for the comparison results and validation/tests.xml for individual tests.

Source inputs/helpers are byte-identical copies recorded in source_inventory.json. package_manifest.json seals the distributable files; runs/ is intentionally excluded. Historical report text is retained verbatim, including old local command examples. Use the portable commands in README.md instead.
