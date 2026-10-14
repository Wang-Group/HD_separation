# Reviewer correlation audit

Initial cohort: 554. Exact eta10: 312. Including four turning estimates: 316.

All source joins are one-to-one; H/D targets agree exactly. Duplicate historical current/Rs columns agree. Historical potential shifts reproduce archived values.

Correlations are descriptive, not evidence of causality or leakage. Pairwise n varies with missing elemental references and eta10 censoring. Nominal p and exploratory BH q assume independent rows; related metal pairs violate that assumption, so they must not be used as confirmatory significance claims.

I_raw is the archived signed, area-normalized LSV current near -1 V. Its historical extraction is retained, not reinterpreted as the -50 mA measurement current. E_IR = E_actual + 0.9*0.05*Rs. Offset = E_IR + 0.098 V. Potential-shifted means are retrospective diagnostics, not intrinsic properties.

| Variable | Pearson r | Spearman rho | n |
|---|---:|---:|---:|
| eta10 | -0.235 | -0.225 | 312 |
| I_raw | -0.033 | -0.405 | 554 |
| Rs_from_EIS | -0.131 | -0.101 | 554 |
| E_IR_corrected_final | 0.329 | 0.378 | 554 |
