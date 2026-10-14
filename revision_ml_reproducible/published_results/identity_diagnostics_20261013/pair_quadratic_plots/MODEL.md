# Pair-Specific Quadratic Model

For metal pair p and measured ICP molar fraction x of Co (Co-containing pairs) or Fe (Fe-Ni):

`predicted H/D separation = a_p + b_p * (x - 0.5) + c_p * (x - 0.5)^2`

Each pair has its own intercept, slope and curvature. Coefficients are fitted jointly using pair indicators
and pair-by-composition interactions, feature standardization and ridge regularization.
No elemental physicochemical descriptors are inputs, and curvature is not constrained to be negative.
This is a known-pair composition model, not a model for transfer to an unseen metal pair.

## Validation and Visualization

The quoted R2 is 0.584420 +/- 0.009022 (SD across 3 repeats), pooled over 111 BO samples within each repeat.
It is not the average of the five per-pair R2 values, and it is not the fit score of the bold curves.
Both between-pair and within-pair variation contribute to the pooled score.
Outer CV holds out 5 mol% composition bins within each pair; initial-screening rows in those bins are also removed.
Three-fold inner CV selects ridge alpha from 0.001, 0.01, 0.1, 1, 10 and 100, fitting encoders/scaling on training data only.
All 15 outer models and 333 saved predictions were reproduced.

Bold curves are descriptive refits on all 665 eligible samples after inner grouped-CV alpha selection (alpha = 10).
The plots show the 111 BO points and 10 initial-screening points belonging to these five pairs.
Faint curves are the 15 outer-fold fits, restricted to their training composition ranges. They are not confidence intervals.
All full-data curves are restricted to observed composition ranges. No extrapolated optimum is claimed.

## Full-Data Coefficients and Held-Out Scores

Coefficients below use the centered fraction (x - 0.5); x is a fraction from 0 to 1, not a percentage from 0 to 100.

| Pair | a | b | c | BO n | BO CV R2 | BO CV RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Co-Fe | 9.11234 | -0.25817 | -0.09139 | 27 | -0.368 | 0.575 |
| Co-Mo | 5.59069 | 4.06198 | 14.97962 | 21 | 0.324 | 1.213 |
| Co-Ni | 7.82432 | 4.07665 | -7.44569 | 21 | 0.435 | 1.001 |
| Co-V | 5.34617 | 2.39652 | 12.40939 | 21 | 0.176 | 1.131 |
| Fe-Ni | 9.60061 | 5.14865 | -9.83616 | 21 | 0.477 | 1.023 |

Negative curvature: Co-Fe, Co-Ni, Fe-Ni. Positive curvature: Co-Mo, Co-V.
Negative curvature alone does not establish an interior maximum in the measured composition range.
Negative per-pair held-out R2: Co-Fe. A high pooled score must not be interpreted as equally good predictions for every pair.
These are descriptive composition-response fits, not proof of a single volcano relationship or a mechanistic assignment.

## Reproduction

```powershell
python -X utf8 hd_separation_ml/plot_icp_pair_quadratics.py
```

Use --diagnostics and --output to plot a rerun with updated targets. Input/code hashes and prediction replay are checked before plotting.
