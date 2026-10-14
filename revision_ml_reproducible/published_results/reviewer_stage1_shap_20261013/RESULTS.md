# Stage 1: Updated SHAP

90 outer-model refits: two model specifications, three tasks, three repeats and five folds. Only the with-BO training arm is interpreted here. All predictions were reproduced to 1e-10 and SHAP additivity checked to 1e-6.

Tree-path-dependent TreeSHAP uses the fitted trees' training path counts; no held-out background data are supplied. The frozen-model explanations remain retrospective, while the nested models retain their original fold-local selection.

Feature summaries pool sums of absolute SHAP across test samples within each repeat, then average the three repeats. Unselected features are zero in the overall summary. Property-group attribution is abs(sum of signed SHAP for M1/M2/weighted versions), not a new conditional/group SHAP calculation. Individual values and fold sums are retained.

H/D/ZPE redundancy and other correlations can redistribute attribution. SHAP explains model predictions, not causal effects. For Task 3, endpoint-property attributions may explain differences BETWEEN pairs, not variation of composition WITHIN a pair. The existing within-pair permutation results provide complementary evidence on composition dependence.

The distribution plot uses repeat 1 to avoid triple plotting observations. Rank-based feature colors include training-imputed values. No fitted SHAP ranking was used to reselect or retune these models.

Primary SHAP citation: Lundberg and Lee (2017), A Unified Approach to Interpreting Model Predictions, NeurIPS 30.

Reproduce: `python hd_separation_ml/revision_shap_current.py --jobs 10`.
