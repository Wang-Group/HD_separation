# Shared ICP for BO reference recipes

`assignments.csv` is the consistent reference-composition mapping for all five
audited BO series. It includes the five initial samples and 37 BO reference
records, linked to five measured ICP source records.

| Series | BO reference indices | ICP source |
|---|---|---|
| Co-Fe | 3, 7, 8, 12, 16, 20, 24, 28, 32 | Co-Fe3 |
| Co-Mo | 3, 7, 8, 12, 16, 20, 24 | Co-Mo3 |
| Co-Ni | 3, 7, 8, 12, 16, 20, 24 | Co-Ni3 |
| Co-V | 3, 7, 8, 12, 16, 20, 24 | Co-V3 |
| Fe-Ni | 3, 7, 8, 12, 16, 20, 24 | Fe-Ni3 |

## Assignment rule

Use complete recorded precursor identities and amounts, including citrate and
water, not a naming pattern or a metal ratio alone. The initial recipe evidence
is in `recipe_groups.json`; BO recipes are in the packaged reference inputs.
All matched recipes use 15 parts M1, 2.5 parts M2 and 2.5 parts citrate. The
initial records have no fourth ingredient; the BO references specify zero water.
Different citrate/water amounts remain different recipes.

The initial samples already point to the corresponding index-3 ICP source.
The eight Co-Fe references without separate ICP rows now use that same source.
The mapping distinguishes `measured_source` from `shared_recipe_reference`.
It does not claim new ICP measurements or prove identical deposited composition
across preparations. Recipe-level sharing assumes the reference measurement is
representative; stock concentrations and preparation conditions are not newly
verified. Existing conflicting measurements cause an error, not an overwrite.

Raw source workbooks and recorded experimental outcomes are unchanged. The
mapping includes source sample, workbook row, reported source record, molar
fractions, recipe group and ICP group. Fractions are on a 0-1 scale.

## Model eligibility

Shared composition metadata and independent ML rows are different questions.
The five initial representatives remain included; all 37 BO reference records
remain represented by those initial rows rather than added as training rows.
The fitting cohort stays 560 initial plus 111 eligible BO records. Saved fitting
audits, manifests, fold assignments, predictions, SHAP and figures are immutable.

For current reference metadata, use `load_reference_metadata()` from
`reference_icp.py`. It returns the identical fitting matrix plus an enriched
record audit, with the Co-Fe fractions filled and all 37 BO references labelled
`shared_reference_represented_by_initial`. Source workbooks are not rewritten.
An audit's absence of an individual measurement does not imply the absence of
an adopted shared reference assignment; consult this mapping for that distinction.

The adopted policy excludes all 37 reference experiments as additional ML rows
in every workflow, including BO-augmented training. They remain in experimental
repeatability analyses and in this composition mapping. Their LSV measurements
are also excluded from all five averaged ML targets under the adopted policy.
The revised target review is in `../audits/eta10_reference_exclusion_20261025`;
Figure 1B/repeatability retain their separate pooling. Shared ICP does not make independently performed
H/D or LSV measurements identical, and does not authorize changing them.

## Reproduce

From `ml_revision`:

```text
python reference_icp.py
python reference_icp.py --check
python -m unittest test_reference_icp -v
```

The build command writes only the assignment table and verification report here.
`--check` writes nothing and fits no models. The ordinary package verifier also
checks these assignments. No external drive is needed to reproduce the mapping.
