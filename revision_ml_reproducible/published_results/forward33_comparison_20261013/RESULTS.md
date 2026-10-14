# Original 39 versus reduced 33 descriptors

This comparison removes D vibration and H/D ZPE in M1, M2 and ICP-weighted form, retaining H vibration as the representative of their shared information. The remaining 33 inputs are unchanged. It is a fixed descriptor-family ablation, not general correlation-threshold pruning.

## Main findings

With BO training, nested forward-10 R2 changes as follows: Random entities 0.555 to 0.556; Unseen metal pairs 0.554 to 0.555; New compositions 0.526 to 0.544. The reduced pool retains predictive performance and the main selection patterns, with a modest composition-task improvement. The three repeats do not establish a statistically significant improvement.

Without BO training, the composition task remains weak despite redundancy removal. Full-pool models improve modestly in all six task/arm comparisons, whereas the one-SE results are mixed. Redundancy removal is justified as a simpler candidate definition, not a universal performance gain.

The separate frozen re-selection retains 10/10 feature identities but changes the feature order and tuned model settings. Its with-BO composition R2 changes from 0.495 to 0.372. This is tuning/transfer sensitivity, not evidence that deleting D frequency and ZPE removes needed inputs from the original frozen model: neither was in that model.

## Matched design

The same 665-sample snapshot, outer/inner memberships, 3 repeats x 5 outer folds, 3 inner folds, RF/Extra Trees grid and seeds are used. Random and unseen-pair tests contain 554 initial samples per repeat; new-composition tests contain 111 independent BO samples. Both with-BO and without-BO training arms are retained. In the no-BO composition task, inner tuning uses the existing initial-only unseen-pair proxy; no BO outcomes enter its selection.

Forward selection is repeated in the 33-input pool through 16 steps. Candidate ordering, tie-breaking, 120-tree selection models, 80-tree tuning models and 400-tree final models match the original workflow. Full-pool, forward-10, inner-best and one-SE variants are compared. Exact cached CV evaluations are reused only for identical ordered input subsets, memberships, configurations and seeds. Changed subsets are fitted anew; the first cached candidate is numerically replayed in every job.

New forward candidate evaluations: 6495; reused exact evaluations: 30225. Each evaluation contains two models x three inner folds. All 90 outer training-arm jobs completed. 169 unchanged final model specifications also reproduce their original predictions to 1e-10.

## Performance

Mean R2 +/- SD across three pooled out-of-fold repeats. Differences are paired 33-minus-39; these are not independent-replicate confidence intervals.

| Training | Task | Strategy | 39 candidates | 33 candidates | Paired change |
|---|---|---|---:|---:|---:|
| with_bo | Random entities | Full pool | 0.550 +/- 0.001 | 0.568 +/- 0.012 | +0.018 |
| with_bo | Random entities | Forward 10 | 0.555 +/- 0.018 | 0.556 +/- 0.009 | +0.001 |
| with_bo | Random entities | Inner-best | 0.553 +/- 0.018 | 0.555 +/- 0.013 | +0.002 |
| with_bo | Random entities | One-SE | 0.529 +/- 0.020 | 0.527 +/- 0.023 | -0.002 |
| with_bo | Random entities | Frozen re-selection | 0.569 +/- 0.014 | 0.573 +/- 0.013 | +0.003 |
| with_bo | Unseen metal pairs | Full pool | 0.556 +/- 0.009 | 0.566 +/- 0.008 | +0.010 |
| with_bo | Unseen metal pairs | Forward 10 | 0.554 +/- 0.006 | 0.555 +/- 0.008 | +0.001 |
| with_bo | Unseen metal pairs | Inner-best | 0.554 +/- 0.011 | 0.555 +/- 0.008 | +0.002 |
| with_bo | Unseen metal pairs | One-SE | 0.525 +/- 0.008 | 0.525 +/- 0.011 | -0.000 |
| with_bo | Unseen metal pairs | Frozen re-selection | 0.578 +/- 0.006 | 0.577 +/- 0.008 | -0.002 |
| with_bo | New compositions | Full pool | 0.547 +/- 0.054 | 0.552 +/- 0.031 | +0.004 |
| with_bo | New compositions | Forward 10 | 0.526 +/- 0.026 | 0.544 +/- 0.018 | +0.018 |
| with_bo | New compositions | Inner-best | 0.510 +/- 0.020 | 0.523 +/- 0.031 | +0.012 |
| with_bo | New compositions | One-SE | 0.480 +/- 0.034 | 0.477 +/- 0.029 | -0.003 |
| with_bo | New compositions | Frozen re-selection | 0.495 +/- 0.015 | 0.372 +/- 0.021 | -0.122 |
| without_bo | Random entities | Full pool | 0.551 +/- 0.017 | 0.559 +/- 0.013 | +0.008 |
| without_bo | Random entities | Forward 10 | 0.543 +/- 0.010 | 0.545 +/- 0.009 | +0.002 |
| without_bo | Random entities | Inner-best | 0.546 +/- 0.012 | 0.551 +/- 0.004 | +0.004 |
| without_bo | Random entities | One-SE | 0.530 +/- 0.007 | 0.533 +/- 0.009 | +0.003 |
| without_bo | Random entities | Frozen re-selection | 0.563 +/- 0.011 | 0.570 +/- 0.012 | +0.007 |
| without_bo | Unseen metal pairs | Full pool | 0.552 +/- 0.007 | 0.566 +/- 0.010 | +0.014 |
| without_bo | Unseen metal pairs | Forward 10 | 0.549 +/- 0.010 | 0.549 +/- 0.007 | -0.000 |
| without_bo | Unseen metal pairs | Inner-best | 0.551 +/- 0.008 | 0.554 +/- 0.004 | +0.003 |
| without_bo | Unseen metal pairs | One-SE | 0.514 +/- 0.005 | 0.512 +/- 0.016 | -0.003 |
| without_bo | Unseen metal pairs | Frozen re-selection | 0.575 +/- 0.007 | 0.577 +/- 0.009 | +0.002 |
| without_bo | New compositions | Full pool | 0.052 +/- 0.018 | 0.078 +/- 0.041 | +0.025 |
| without_bo | New compositions | Forward 10 | -0.065 +/- 0.054 | -0.016 +/- 0.070 | +0.050 |
| without_bo | New compositions | Inner-best | -0.005 +/- 0.022 | 0.009 +/- 0.025 | +0.014 |
| without_bo | New compositions | One-SE | -0.053 +/- 0.058 | -0.070 +/- 0.044 | -0.017 |
| without_bo | New compositions | Frozen re-selection | -0.000 +/- 0.015 | -0.002 +/- 0.019 | -0.002 |

[Performance figure](performance_39_vs_33.png)

## Selection stability

Selection-frequency files distinguish M1, M2 and weighted variants and include both BO training arms, all tasks, and all three subset-selection rules. Counts are out of 15 outer fits. Removed features are explicitly marked as unavailable in pool 33, not interpreted as rejected by selection. A shared H/D vibration family row permits a like-for-like family comparison.

| With-BO forward-10 | Task | 39 pool family count | 33 pool family count |
|---|---|---:|---:|
| work_function | Random entities | 15/15 | 15/15 |
| work_function | Unseen metal pairs | 15/15 | 15/15 |
| work_function | New compositions | 15/15 | 15/15 |
| E_std | Random entities | 14/15 | 14/15 |
| E_std | Unseen metal pairs | 13/15 | 13/15 |
| E_std | New compositions | 15/15 | 14/15 |
| FE_ads_avg | Random entities | 15/15 | 15/15 |
| FE_ads_avg | Unseen metal pairs | 13/15 | 13/15 |
| FE_ads_avg | New compositions | 1/15 | 1/15 |
| delta_EOH | Random entities | 14/15 | 14/15 |
| delta_EOH | Unseen metal pairs | 10/15 | 11/15 |
| delta_EOH | New compositions | 15/15 | 15/15 |
| electronegativity | Random entities | 1/15 | 1/15 |
| electronegativity | Unseen metal pairs | 9/15 | 9/15 |
| electronegativity | New compositions | 10/15 | 11/15 |
| shared_H_D_vibration_information | Random entities | 12/15 | 12/15 |
| shared_H_D_vibration_information | Unseen metal pairs | 10/15 | 10/15 |
| shared_H_D_vibration_information | New compositions | 8/15 | 8/15 |

[M1/M2/weighted frequency comparison](feature_frequency_39_vs_33.png)

## Frozen Task 1 sensitivity

A new consensus of 10 features is selected from the reduced-pool Task-1 with-BO forward-10 fits. The same consensus-frequency and mean-forward-rank rules and the same global Task-1 inner-CV model grid are used. This single specification is then refitted on all three tasks and both training arms. Like the original frozen model, it is retrospective and not independent validation. Its initial-only refit arm is still BO-informed at specification selection.

Original model: ET, configuration {'max_depth': None, 'min_samples_leaf': 3, 'max_features': 1.0}. Reduced-pool model: ET, configuration {'max_depth': 8, 'min_samples_leaf': 1, 'max_features': 1.0}.

The two leading configurations in the reduced-pool Task-1 tuning differ by only 0.000203 mean inner R2. Randomized tree fitting can depend on input-column order; the two exchanged feature positions change the exact random realization. The near-tied tuning winner is therefore not strong evidence of a better transferable specification. Do not choose between these settings using Task-3 test scores.

Feature identity overlap is 10/10; table row order is the model input order, not a change of feature identity.

| Original frozen feature | Reduced-pool frozen feature |
|---|---|
| FE_ads_avg_element1 | FE_ads_avg_element1 |
| icp_weighted_work_function | icp_weighted_work_function |
| work_function_element1 | work_function_element1 |
| work_function_element2 | work_function_element2 |
| Frequency_H_element1 | Frequency_H_element1 |
| E_std_element2 | delta_EOH_element2 |
| delta_EOH_element2 | E_std_element2 |
| E_std_element1 | E_std_element1 |
| delta_EOH_element1 | delta_EOH_element1 |
| MM_bond_element2 | MM_bond_element2 |

The original frozen feature list already contains none of the six removed descriptors. Simply applying the 33-column restriction to that already-frozen specification would leave its predictions unchanged. Differences in the newly selected frozen model arise from repeating consensus selection/tuning, not from deleting an input it previously used.

## Scope

Other correlations remain, including relations between H/D bond descriptors and the geometry of weighted properties within a fixed metal pair. Removing duplicate vibration descriptors does not establish causal feature importance. Any additional pruning threshold would be a separate training-fold-only comparison. Original completed 39-pool runs, data and manuscript figures are unchanged.

## Reproduction

From the repository root, using the original ML environment:

```powershell
$py = 'C:/Users/yibinjiang/anaconda3/python.exe'
$env:LOKY_MAX_CPU_COUNT = '10'
& $py hd_separation_ml/icp_forward_33_comparison.py --jobs 10
& $py hd_separation_ml/icp_frozen33_sensitivity.py --jobs 10
& $py hd_separation_ml/summarize_icp_33_comparison.py
& $py -m unittest discover -s hd_separation_ml -p test_icp_forward_33_comparison.py -v
```

The source CV artifacts are required. Code/data hashes and manifests prevent silently combining changed inputs with cached fits. Individual path steps, tuning steps and final models are checkpointed for resumption. For new response values, regenerate the upstream 39-pool workflow before this matched comparison.
