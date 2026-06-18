# Figure 1

This folder contains only the active files needed to regenerate the Figure 1
bottom comparison/Pareto panel.

## Active Files

- `data/figure1_bottom_literature_benchmarks.csv`
  - Literature and benchmark comparison points extracted from the legacy
    workbook into an explicit active CSV.
  - Columns: `source_type`, `source_label`, `reference_label`,
    `plot_x_overpotential_mV`, `separation_coefficient`,
    `measurement_condition`.
- `data/figure1_bottom_this_work_from_all_data.csv`
  - Full this-work extraction table from the current trusted electrochemical
    raw-data folder.
  - Derived from `hd_separation_ml/data_extraction/all_data/LSV` and
    `hd_separation_ml/data_extraction/separation_coefficient.xlsx`.
  - Includes geometry-area handling for Ag-Ce, Co-V, and Mo-Mg.
  - For traces that cross the target current multiple times, uses the worst
    crossing, i.e. the most negative raw potential / largest overpotential.
  - For traces whose current at the most negative point still does not reach
    -10 mA cm^-2, keeps the measured endpoint overpotential only when the
    trace is within 2 mA cm^-2 of the target current; otherwise assigns the
    Figure 1 cap of 550 mV as the plotted worst-case overpotential.
  - Points already active at the scan start are generally plotted at the 10 mV
    boundary as samples with non-HER/background current and are excluded from
    Pareto front calculations.
  - A small Co-V subset with a clear near-target LSV maximum is recovered using
    the measured point closest to -10 mA cm^-2. `Co-V8`, `Co-V11`, `Co-V20`,
    and `Co-V22` are retained as borderline recovered cases.
  - Red/near-target and purple/turning-point estimates are accepted only when
    the measured/turning-point current is within 10 mA cm^-2 of
    -10 mA cm^-2 and the LSV shape is interpretable. `Fe-Pt`, `Pd-Pd`,
    `Y-Pd`, and `Co-Pt` are treated as turning-point eta estimates.
    `Co-Fe29`, `Co-Fe33`, `Sc-Pd`, and `W-Pd` remain non-HER lower-bound
    cases and are excluded from Pareto calculations.
  - `Sc-Pb`, `Cd-Y`, `W-Pb`, and `Ce-Pb` are treated as charging-current
    artifacts that never provide a valid HER eta10 and are forced to the
    550 mV cap.
  - Plotted eta10 values are bounded to the Figure 1 reporting window:
    10-550 mV.
- `plot_figure1_censored_eta10.py`
  - Current script for the bottom Pareto panel with eta10 censoring,
    literature-condition filtering, and exclusion of non-HER current points
    from the Pareto front.
  - Saves `outputs/pareto_plot_v2_censored_eta10.png`,
    `outputs/pareto_plot_v2_censored_eta10_no_text.png`, and the corresponding
    PDF/CSV audit outputs.
- `data/figure1_bottom_plot_points.csv`
  - Long-format audit table for all plotted points.
  - Columns: `source_type`, `source_label`, `reference_label`, `x_raw`,
    `plot_x_overpotential_mV`, `separation_coefficient`.
- `data/figure1_this_work_eta10_censoring.csv`
  - Current audit/classification table for exact eta10, right-censored
    non-reaching traces, recovered near-target Co-V eta estimates, samples
    with non-HER/background current, turning-point eta estimates, and
    charging-current forced-cap cases.
- `plot_eta_below_10_lsv.py`
  - Draws LSVs for the 30 samples initially caught by the eta/bound < 10 mV
    audit, including near-target recovered Co-V samples and charging-current
    forced-cap cases.
  - Saves outputs under `outputs/eta_below_10_lsv/`.
- `plot_right_censored_middle_lsv.py`
  - Draws LSVs for right-censored samples that are close to -10 mA cm^-2 and
    therefore kept at their measured endpoint overpotential instead of being
    assigned to the 550 mV cap.
  - Saves outputs under `outputs/right_censored_middle_lsv/`.
- `outputs/pareto_plot_v2_censored_eta10.png`
  - Current annotated Figure 1 bottom-panel output.
- `outputs/pareto_plot_v2_censored_eta10_no_text.png`
  - Current no-text Figure 1 bottom-panel output.

## Data Count

The active script rebuilds the this-work series from the trusted
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
