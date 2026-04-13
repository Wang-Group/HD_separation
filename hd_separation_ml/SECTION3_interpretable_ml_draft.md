# Section 3 Draft: Interpretable Machine Learning on Bimetallic Catalysts

This document is split into two parts:

- `Manuscript text`: paste-ready SI prose with no repository or file references.
- `Editor notes`: non-manuscript notes for figure insertion and layout.

## Manuscript text

### Overview

To identify the physicochemical factors governing H/D separation across the initially screened bimetallic catalysts, we performed an interpretable machine-learning analysis using the complete primary screening dataset after preprocessing and quality filtering. The aim of this analysis was twofold: first, to determine whether the H/D separation coefficient could be predicted from a compact set of electrochemical and element-level descriptors; second, to extract the dominant descriptor trends associated with high separation performance. In this way, the machine-learning analysis provided a data-driven mechanistic framework connecting the large-scale screening campaign to the descriptor-based interpretation used later for Bayesian optimization.

### Dataset

The machine-learning dataset was assembled by combining the initial screening results with electrochemical impedance fitting outputs, linear sweep voltammetry information, and element-level descriptor data. Standard data-cleaning and completeness filtering were applied before merging the electrochemical and descriptor information, yielding a final dataset of 558 samples with complete inputs for machine learning.

The prediction target was the H/D separation coefficient. For model development, the dataset was divided into training and test subsets using an 80/20 split. Because the target is continuous, quantile binning was used to approximate stratification during the split, resulting in 446 training samples and 112 test samples.

### Feature Set

The candidate feature pool consisted of three groups. The first group was an LSV-derived current descriptor, defined as the interpolated current at -1.00 V and denoted as I_raw. The current was converted to mA and its original sign was retained. This sign-preserving treatment was important because negative current values correspond to normal HER-like cathodic behavior, whereas positive current values can indicate abnormal behavior such as metal dissolution.

The second group consisted of EIS-derived electrochemical descriptors. These quantities were constructed from the fitted impedance parameters and included charge-transfer- and circuit-related terms such as Rct, Cp, Rp, C_dl, W_sigma, and an EIS-based Tafel-slope proxy. Before these quantities were calculated, the roughness factor was clipped to an upper limit of 200 in order to suppress extreme values. The quantities Rct, Rp, and Cp, which can adopt either sign, were then transformed using a sign-preserving natural-logarithm treatment, defined as the natural logarithm of (1 + |x|) multiplied by the sign of x, while C_dl and W_sigma were transformed using the natural logarithm of (1 + x).

The third group comprised element-resolved descriptors describing the two metals in each bimetallic catalyst. The full list of candidate descriptors and their physical meaning is summarized in Table SX. For each sample, the elemental descriptors were assigned separately to metal 1 and metal 2 according to the order of the two elements in the catalyst label. Missing descriptor values were imputed using the global median across the corresponding metal-1 and metal-2 columns.

Some elemental descriptors were further converted into potential-referenced features before model training. Specifically, a sample-specific offset was defined as E_IR_corrected_final + 0.098, where E_IR_corrected_final is the iR-corrected operating-potential proxy. The PZC- and standard-potential-related descriptors were shifted by subtracting this offset, whereas the work-function-related descriptors were shifted in the opposite direction by adding the same offset. These transformations should be interpreted as feature-engineering steps designed to align intrinsic descriptor values with the operating electrochemical potential of each sample, rather than as direct raw material constants. After preprocessing, the full candidate pool contained 34 features in total, including 8 electrochemical descriptors and 26 element-resolved descriptor columns.

### Feature Selection

Feature selection was carried out by manual backward elimination using a tree-based regressor and the mean 5-fold cross-validated R^2 score on the training set as the selection criterion. Starting from the full 34-feature candidate pool, one feature was removed at each iteration, and the subset giving the most favorable cross-validated performance was retained. This procedure yielded a final 12-feature model comprising the sign-preserving current descriptor, work function of metal 2, M-M bond energies of metals 1 and 2, standard reduction potentials of metals 1 and 2, H adsorption energy of metal 1, OH adsorption energy of metal 2, electronegativities of metals 1 and 2, M-H bond energy of metal 2, and M-D bond energy of metal 1.

### Interpretation

Three regression models were compared for the final selected feature set: Random Forest, Extra Trees, and XGBoost. Among these candidates, the Extra Trees regressor provided the best held-out performance and was therefore used for the final interpretation. On the test set, the selected model achieved R^2 = 0.6691 and RMSE = 0.9238, indicating that a relatively compact set of electrochemical and element-level descriptors could capture a substantial fraction of the variation in H/D separation across the screened catalyst space.

SHAP analysis was then used to interpret the final model. The dominant contributions clustered around adsorption thermodynamics, electronic structure, isotope-specific bonding, and the electrochemical current response. In particular, the most influential features included H adsorption energy, electronegativity, M-D bond energy, work function, M-M bond energy, and the sign-preserving I_raw descriptor. Higher H adsorption energy, higher electronegativity, higher work function, and stronger M-M bonding were generally associated with larger predicted separation coefficients. In contrast, stronger M-D bonding tended to shift the model output downward, indicating that relatively weaker M-D binding favored higher H/D separation. The current descriptor also remained mechanistically informative: more negative current values generally contributed positively to the predicted separation coefficient, whereas positive current values contributed negatively, consistent with the distinction between normal HER-like behavior and abnormal anodic or dissolution-related behavior.

Taken together, these results indicate that the H/D separation coefficient in the initially screened bimetallic catalyst space is governed primarily by a limited set of adsorption, electronic, and isotope-bonding descriptors rather than by a diffuse combination of unrelated variables. In the following section, these same governing trends can be recast into weighted descriptor maps to interpret the experimental trajectories explored during Bayesian optimization.

## Suggested Table

**Table SX.** Candidate descriptors used in the interpretable machine-learning analysis and their physical meaning.

| Group | Descriptor | Physical meaning |
| --- | --- | --- |
| LSV-derived | Signed current at -1.00 V (I_raw) | Electrochemical current response at a fixed cathodic bias; reflects HER-like activity and helps distinguish normal cathodic behavior from abnormal positive-current behavior. |
| EIS-derived | Charge-transfer resistance [sign-preserving log transform; Rct_log] | Resistance associated with interfacial electron-transfer kinetics during the electrochemical reaction. |
| EIS-derived | Effective pseudocapacitance [sign-preserving log transform; Cp_log] | Capacitive response associated with the interfacial reaction process, derived from the fitted impedance parameters. |
| EIS-derived | Polarization resistance [sign-preserving log transform; Rp_log] | Effective resistive contribution associated with overall interfacial polarization under operating bias. |
| EIS-derived | Double-layer capacitance [log-transformed; C_dl_log] | Interfacial charging capacity, often used as a proxy for electrochemically accessible surface area. |
| EIS-derived | iR-corrected operating-potential proxy (E_IR_corrected_final) | Effective reaction potential after accounting for ohmic voltage drop. |
| EIS-derived | Tafel-slope proxy (tafel_slope) | Kinetic descriptor reflecting how sensitively overpotential changes with current during HER. |
| EIS-derived | Warburg coefficient [log-transformed; W_sigma_log] | Descriptor of diffusion- or mass-transport-related impedance contributions. |
| Element-level | Work function (work_function_element1 / work_function_element2) | Energy required to remove an electron from the material surface to vacuum; reflects surface electronic structure. |
| Element-level | Potential of zero charge (PZC) (pzc_element1 / pzc_element2) | Potential at which the net surface charge is zero. |
| Element-level | Metal-metal bond energy (MM_bond_element1 / MM_bond_element2) | Strength of bonding between metal atoms, related to structural cohesion and electronic environment. |
| Element-level | Standard reduction potential (E_std_element1 / E_std_element2) | Thermodynamic tendency of the corresponding metal redox couple to accept electrons. |
| Element-level | H adsorption free energy (FE_ads_avg_element1 / FE_ads_avg_element2) | Thermodynamic strength of hydrogen adsorption on the catalyst surface. |
| Element-level | OH adsorption energy (delta_EOH_element1 / delta_EOH_element2) | Thermodynamic strength of hydroxyl adsorption on the catalyst surface. |
| Element-level | Electronegativity (electronegativity_element1 / electronegativity_element2) | Tendency of an atom to attract electron density. |
| Element-level | M-H bond energy (MH_bond_kJ_element1 / MH_bond_kJ_element2) | Strength of bonding between the metal and hydrogen. |
| Element-level | M-D bond energy (MD_bond_kJ_element1 / MD_bond_kJ_element2) | Strength of bonding between the metal and deuterium. |
| Element-level | OH vibrational frequency (Frequency_OH_element1 / Frequency_OH_element2) | Vibrational characteristic of the adsorbed OH species, related to bond stiffness. |
| Element-level | H vibrational frequency (Frequency_H_element1 / Frequency_H_element2) | Vibrational characteristic of adsorbed hydrogen, related to bond stiffness. |
| Element-level | D vibrational frequency (Frequency_D_element1 / Frequency_D_element2) | Vibrational characteristic of adsorbed deuterium, related to bond stiffness. |
| Element-level | Zero-point-energy difference between H and D (delta_ZPE_H_element1 / delta_ZPE_H_element2) | Isotope-dependent quantum energy difference relevant to H/D discrimination. |

Element-level descriptors were assigned separately to the two metallic components of each bimetallic catalyst.

## Suggested Figure Caption

**Figure S8.** Interpretable machine-learning results for the final 12-feature model trained on the initial bimetallic screening dataset. **(A)** Held-out test-set prediction and residual plots for the Extra Trees regressor, showing a test performance of R^2 = 0.6691 and RMSE = 0.9238. **(B)** SHAP swarm plot showing the direction and magnitude of each retained feature contribution to the model output. **(C)** Mean absolute SHAP values ranking the retained features by overall importance. The dominant descriptors cluster around H adsorption energy, electronegativity, M-D bond energy, work function, M-M bond energy, and the sign-preserving LSV current.

## Editor notes

These notes are not intended for the SI text itself.

- Recommended composite figure asset: `0-noble_no_BO_seed42/section3_interpretable_ml_composite.png`
- If this figure is inserted after current `Figure S7`, shift all existing figure numbers from `Figure S8` onward by `+1`.
- If the layout is too tight, the prediction plot and SHAP plots can also be inserted as two separate figures instead of the composite panel.
