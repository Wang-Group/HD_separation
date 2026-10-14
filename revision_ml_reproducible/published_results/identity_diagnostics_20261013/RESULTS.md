# Metal Identity and Sparse-Descriptor Diagnostics

This is a retrospective sensitivity analysis on the existing grouped folds, not a new external validation set.
The 665-record population and Co-Fe inclusion policy are unchanged. Only protocols 2 and 3 are used.

## Matched Out-of-Fold Results

| Task | Model | n | R2 | Repeat SD | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| unseen_pair | Global mean | 554 | -0.1558 | 0.0139 | 1.7803 |
| unseen_pair | Pair mean | 554 | -0.1558 | 0.0139 | 1.7803 |
| unseen_pair | Element identity | 554 | 0.3748 | 0.0041 | 1.3093 |
| unseen_pair | Metal identities (forest) | 554 | 0.4960 | 0.0030 | 1.1756 |
| unseen_pair | Element + ICP fraction | 554 | 0.3123 | 0.0230 | 1.3732 |
| unseen_pair | 4 descriptors | 554 | 0.4748 | 0.0108 | 1.2001 |
| unseen_pair | 8 descriptors | 554 | 0.4749 | 0.0049 | 1.2000 |
| unseen_pair | 12 descriptors | 554 | 0.4922 | 0.0076 | 1.1800 |
| unseen_pair | One-SE selection | 554 | 0.5311 | 0.0136 | 1.1339 |
| unseen_pair | Best inner selection | 554 | 0.5561 | 0.0091 | 1.1033 |
| unseen_pair | All 39 descriptors | 554 | 0.5561 | 0.0091 | 1.1033 |
| known_pair_new_composition | Global mean | 111 | -4.4140 | 0.0114 | 3.5910 |
| known_pair_new_composition | Pair mean | 111 | 0.1907 | 0.0447 | 1.3880 |
| known_pair_new_composition | Element identity | 111 | -0.2356 | 0.0173 | 1.7155 |
| known_pair_new_composition | Metal identities (forest) | 111 | 0.1990 | 0.0372 | 1.3810 |
| known_pair_new_composition | Element + ICP fraction | 111 | -0.0093 | 0.0097 | 1.5505 |
| known_pair_new_composition | Pair identity (ridge) | 111 | 0.1903 | 0.0444 | 1.3884 |
| known_pair_new_composition | Pair + ICP quadratic | 111 | 0.5844 | 0.0090 | 0.9949 |
| known_pair_new_composition | 4 descriptors | 111 | 0.3929 | 0.0724 | 1.2011 |
| known_pair_new_composition | 8 descriptors | 111 | 0.5148 | 0.0279 | 1.0748 |
| known_pair_new_composition | 12 descriptors | 111 | 0.5366 | 0.0521 | 1.0494 |
| known_pair_new_composition | One-SE selection | 111 | 0.5088 | 0.0706 | 1.0797 |
| known_pair_new_composition | Best inner selection | 111 | 0.5607 | 0.0305 | 1.0225 |
| known_pair_new_composition | All 39 descriptors | 111 | 0.5473 | 0.0541 | 1.0372 |

## What Was Controlled

- All outer sample memberships are unchanged; shared ICP, unordered pairs (task 2), and composition bins (task 3) remain isolated.
- Metal identity controls use unordered element indicators or pair indicators, without measured composition.
- Composition controls use element molar fractions, or pair-specific quadratic functions of the canonical ICP fraction.
- Ridge alpha is selected from six values inside the same three inner folds; encoders and scalers are fitted only on inner training data.
- Forest models use the original training-only RFE and tuning candidates. For each of 4/8/12 features, RF/ET and hyperparameters are chosen by inner scores only.
- Original tuned-all, nested-max and one-SE predictions are refitted and reproduced to absolute tolerance 1e-10.
- Training-fit scores are diagnostic only, not validation. Smaller feature sets do not by themselves prove reduced overfitting.
- Sparse8 is a prespecified diagnostic feature budget, not selected because it wins on the outer tests.

## Metal-Pair Association

Observed between-pair variance fractions are descriptive, not cross-validated explanatory power or causal effects.
The initial/all-data fractions are particularly optimistic when many pairs have only one observation.

- initial: 78.9% between pairs; 389 pairs, 224 singletons.
- all_BO: 37.8% between pairs; 5 pairs, 0 singletons.
- all: 84.0% between pairs; 389 pairs, 224 singletons.

## Feature Counts and Importance

- known_pair_new_composition, nested_max: 12-39 features; median 39.
- known_pair_new_composition, one_se: 39-39 features; median 39.
- known_pair_new_composition, tuned_all: 39-39 features; median 39.
- unseen_pair, nested_max: 39-39 features; median 39.
- unseen_pair, one_se: 4-39 features; median 39.
- unseen_pair, tuned_all: 39-39 features; median 39.

The held-out importance plots use the 8-feature diagnostic models.
Task 2 shuffles each property's selected raw/weighted columns jointly across held-out samples.
Task 3 shuffles only within the same held-out metal pair; the all-weighted block moves together for a coherent composition test.
Unselected properties receive zero importance; selection frequency is reported over all 15 outer folds.
Partial property permutation can create inconsistent correlated descriptors. Interpret it as predictive reliance, not a unique mechanism.
Within each fixed pair, all weighted properties are affine functions of one ICP fraction, so BO alone cannot identify a unique causal weighted property.
Feature rankings are not reused to select a new model and claim accuracy on these same outer tests.

## Reproduction

```powershell
python -X utf8 hd_separation_ml/icp_identity_diagnostics.py --jobs 10
```

For changed outcomes, first rerun the main Co-Fe comparison with the new target files; then pass --source and a fresh --output here.
Input and code hashes prevent stale diagnostic checkpoints from being reused.
