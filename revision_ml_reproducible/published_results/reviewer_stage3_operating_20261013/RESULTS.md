# Matched operating-variable and potential-shift ablation

Initial-only evaluation: all 554 samples have all three operating covariates. BO covariates are unavailable in these archived inputs and were not fabricated. Unseen unordered metal pairs are isolated in both outer and inner CV.

Twelve model/config candidates were tuned separately in each outer fold using the same three inner splits. All 39 (or 42) declared features are used. This controls the information sources, not a new forward-selection search. Original selected-feature results remain unchanged.

| Inputs | R2 mean +/- SD | RMSE |
|---|---:|---:|
| Intrinsic 39 | 0.552 +/- 0.007 | 1.109 |
| Intrinsic 39 + operating | 0.585 +/- 0.007 | 1.067 |
| Potential-shifted 39 | 0.562 +/- 0.004 | 1.095 |
| Shifted 39 + operating | 0.577 +/- 0.006 | 1.077 |

The intrinsic baseline reproduces all archived all39/no-BO predictions. Potential-shifted descriptors explicitly contain a measured operating potential; they are not intrinsic elemental properties. SHAP tables are provided separately for each case, retaining M1/M2/weighted distinctions. Correlated attributions are not causal effects.

The corrected potential is E_actual + 0.9*0.05*Rs. Offset = corrected potential + 0.098 V. Add the offset to work function; subtract it from PZC and standard potential, including the weighted versions. Unknown intrinsic references are imputed from training data before applying this shift.
