# Figure 1

This folder contains only the active files needed to regenerate the Figure 1
bottom comparison/Pareto panel.

## Active Files

- `data/figure1_bottom_literature_benchmarks.csv`
  - Literature and benchmark comparison points extracted from the legacy
    workbook into an explicit active CSV.
  - Columns: `source_type`, `source_label`, `reference_label`,
    `plot_x_overpotential_mV`, `separation_coefficient`.
- `data/figure1_bottom_this_work_from_all_data.csv`
  - Full this-work extraction table from the current trusted electrochemical
    raw-data folder.
  - Derived from `hd_separation_ml/data_extraction/all_data/LSV` and
    `hd_separation_ml/data_extraction/separation_coefficient.xlsx`.
  - Includes geometry-area handling for Ag-Ce, Co-V, and Mo-Mg.
  - For traces that cross the target current multiple times, uses the worst
    crossing, i.e. the most negative raw potential / largest overpotential.
- `data/figure1_bottom_plot_points.csv`
  - Long-format audit table for all plotted points.
  - Columns: `source_type`, `source_label`, `reference_label`, `x_raw`,
    `plot_x_overpotential_mV`, `separation_coefficient`.
- `data/figure1_bottom_this_work_worst_crossing_updates.csv`
  - Audit table listing this-work points whose plotted overpotential changed
    by more than 0.5 mV when the conservative worst-crossing rule was applied.
- `notebooks/v4.ipynb`
  - Active notebook for the bottom panel.
  - Reads `../data/figure1_bottom_literature_benchmarks.csv` and
    `../data/figure1_bottom_this_work_from_all_data.csv`.
  - Saves the current output to `../outputs/pareto_plot_v2.png`.
- `plot_this_work_pareto_front_only.py`
  - Recomputes the Pareto front from only the 709 this-work points, where
    lower overpotential and higher separation coefficient are preferred.
  - Saves `outputs/this_work_pareto_front_only.png` and
    `outputs/this_work_pareto_front_points.csv`.
- `outputs/pareto_plot_v2.png`
  - Current output PNG.

## Data Count

The active notebook rebuilds the this-work series from the trusted
`hd_separation_ml/data_extraction` data:

- 709 "This work" points
- 561 initial-screening points
- 112 four-system BO points
- 36 extra Co-Fe BO points
- 76 literature-reference points
- 2 benchmark points: `Ni foam` and `Pt sheet`
- 787 total plotted points

So the Figure 1 bottom panel includes all this-work data in the comparison
workflow, not only the 561 initial-screening samples.
