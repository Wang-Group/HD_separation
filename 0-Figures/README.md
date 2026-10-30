# Manuscript Figures and Panel Sources

The assembled manuscript figures are in **[manuscript/](manuscript/README.md)**.
Use those SVG masters and matching exports, not an individual panel output.

| Figure | Content | Editable master | Preview | PDF |
| --- | --- | --- | --- | --- |
| 1 | Automated experimental platform | [SVG](manuscript/Figure1.svg) | [PNG](manuscript/Figure1.png) | [PDF](manuscript/Figure1.pdf) |
| 3 | EIS/DRT, H/D SHAP and initial-only global descriptor maps | [SVG](manuscript/Figure3.svg) | [PNG](manuscript/Figure3.png) | [PDF](manuscript/Figure3.pdf) |
| 4 | BO performance, descriptor overlays and activity/separation mean and SD | [SVG](manuscript/Figure4.svg) | [PNG](manuscript/Figure4.png) | [PDF](manuscript/Figure4.pdf) |

## Reproduction Sources

- [LSV extraction](Figure1/revision_figure/README.md): per-curve activity records,
  statistics and regression tests. The manuscript comparison is Figure 4D.
- [EIS and DRT](Figure3/README.md): raw-spectrum analysis, fitted curves and panel sources.
- [BO performance](Figure4/README.md): recorded four-system objectives and plotting notebook.
- [Machine learning](../ml_revision/README.md): correlation-aware validation,
  held-out SHAP, descriptor maps and the six-property SI atlas.
- [Repeatability and analytical calibration](../others_revision/README.md):
  numerical source data and reproduction scripts.

`figure_typography.py` and `test_figure_typography.py` provide shared
superscript/subscript export handling and checks. The complete compositions
are editable SVG assemblies; panel scripts do not overwrite them automatically.
