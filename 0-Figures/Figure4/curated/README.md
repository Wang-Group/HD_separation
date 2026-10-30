# Recorded BO Performance Data

`figure4_bo_complete_112.csv` contains 112 completed experiments: 28 each for
Co-Mo, Co-V, Co-Ni and Fe-Ni. It supplies the batch-performance panels in
[Figure 4](../../manuscript/Figure4.svg).

The table records platform BO objectives derived from the BO JSON files, with
H/D coefficients aligned to `hd_separation_ml/data_extraction/separation_coefficient.xlsx`.
These objectives are kept separate from experimental repeatability means.

The [BO plotting notebook](../pareto_polygon_enclosure_code.ipynb) reads this
table directly. Composition-weighted descriptor projections are reproduced by
[ml_revision](../../../ml_revision/README.md), not by this performance table alone.
