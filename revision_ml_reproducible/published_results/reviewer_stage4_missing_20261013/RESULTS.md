# Missing-reference sensitivity

Unlisted reference elements: Eu, Nb. Affected samples: 58.

360 refits: two restrictions, two current model specifications, two BO training arms, three tasks, three repeats, five folds. Original feature identities and hyperparameters are held fixed; this tests refitting/coverage sensitivity, not selection stability after changing the dataset.

Each restricted refit is compared against the original model's predictions on exactly the same retained test samples. Different test populations must not be compared directly. Full per-fold exclusions and sample counts are exported. Complete-selected-property eligibility can vary by fold because nested features vary.

The frozen model's features/config were chosen with Task-1 outcomes, including BO, so its results remain retrospective, including the initial-only refit arm. No missing Eu/Nb references were invented. Known zero component fractions need no reference value for that absent component in a weighted property.

| With BO | Restriction | Task | n per repeat | Paired R2 change |
|---|---|---|---|---:|
| nested_forward10 | listed_elements_only | New compositions | 111-111 | +0.007 +/- 0.007 |
| nested_forward10 | selected_properties_complete | New compositions | 110-111 | -0.003 +/- 0.007 |
| task1_frozen10 | listed_elements_only | New compositions | 111-111 | +0.007 +/- 0.006 |
| task1_frozen10 | selected_properties_complete | New compositions | 111-111 | -0.007 +/- 0.005 |
| nested_forward10 | listed_elements_only | Random entities | 496-496 | -0.007 +/- 0.002 |
| nested_forward10 | selected_properties_complete | Random entities | 243-320 | -0.023 +/- 0.007 |
| task1_frozen10 | listed_elements_only | Random entities | 496-496 | -0.003 +/- 0.002 |
| task1_frozen10 | selected_properties_complete | Random entities | 317-317 | -0.012 +/- 0.003 |
| nested_forward10 | listed_elements_only | Unseen metal pairs | 496-496 | -0.002 +/- 0.003 |
| nested_forward10 | selected_properties_complete | Unseen metal pairs | 257-276 | -0.034 +/- 0.023 |
| task1_frozen10 | listed_elements_only | Unseen metal pairs | 496-496 | -0.003 +/- 0.003 |
| task1_frozen10 | selected_properties_complete | Unseen metal pairs | 317-317 | -0.013 +/- 0.002 |
