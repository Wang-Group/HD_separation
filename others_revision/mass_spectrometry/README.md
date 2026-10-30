# Mass Spectrometry Controls

This folder contains the Hiden calibration and H2-only control data, the
matched mass-3 response comparison, and the conditional H3 sensitivity calculation.
The nine-fit summaries and Linglu/FeNi instrument-comparison source data are
maintained separately by Wang Xue and are not part of this repository package.

## Input

`raw/mass_calibration/hiden_data/file147.csv` is the unmodified Hiden export.
The file contains acquisition metadata followed by time (ms), m/z = 2 and
m/z = 3 signals (mbar). `input_manifest.json` records its SHA-256 hash.

`inputs/hiden_calibration_controls.xlsx` contains 21 H2 and 21 HD calibration
readings, 40 H2-only control plateaus, local backgrounds, Excel regression
formulas, matched-response calculations and four editable scatter charts.
The calibration readings are the numerical values verified from `two_ref.ipynb`;
they are not digitized from an image. Seven levels occur in three acquisition
windows. A synthetic origin is not included in either regression. The workbook
is self-contained and does not require the acquisition file or notebook to
recalculate its fits and charts.

The conditional sensitivity script retains the following rounded coefficients:

| Channel | Slope | Intercept |
|---|---:|---:|
| H2 | 2.06e-7 | -1.31e-9 |
| HD | 6.16e-8 | 9.51e-12 |

The calibration convention is signal = slope * concentration (vol%) + intercept.
These constants are explicit in `interference_sensitivity.py` and its output
`manifest.json`. The workbook instead recomputes the full-precision fits from
the 42 readings. Its matched-response sheet uses alpha = 10 and 10 atom% liquid
D, giving H2/HD = 44.5 when gaseous D2 is neglected. Both control channels use
local-background subtraction; the HD fitted intercept is retained. This is a
signal comparison, not a correction applied to the catalyst coefficients.

## Reproduce

After installing `../requirements.txt`, run these commands from the repository
root, in order:

```bash
python others_revision/mass_spectrometry/analyze.py
python others_revision/mass_spectrometry/interference_sensitivity.py
python others_revision/mass_spectrometry/plot_response.py
python others_revision/mass_spectrometry/plot_mz3_response.py
```

Alternatively, `python others_revision/run.py` reproduces both NMR and MS.

Open the workbook in Excel to recalculate its fits and editable charts. Run
`python others_revision/mass_spectrometry/verify_workbook.py` to independently
check the saved formulas against the numerical inputs, without changing any
results. This check is also included in `python others_revision/run.py --verify-only`.

## Outputs

- `outputs/mz3_response/hd_calibration_h2_control.{png,pdf,svg}`: HD calibration
  and H2-only m/z = 3 response, including the measured points and linear fits.
- `outputs/mz3_response/matched_mz3_responses.{png,pdf,svg}`: matched response
  curves for alpha = 10 and 10 atom% liquid D. The full HD intercept is retained.
- `inputs/mz3_response/`: the plotted numerical tables and full-precision fit
  parameters. `verify_mz3_response.py` checks them against the workbook without
  changing results. The synthetic origin is display-only, not a calibration reading.
- `outputs/hiden_pulse_windows.csv`: 40 plateau windows, signals and backgrounds.
- `outputs/interference_sensitivity/hiden_hydrogen_control.{png,pdf,svg}`: hydrogen control traces.
- `outputs/interference_sensitivity/h3_separation_sensitivity.{png,pdf,svg}`:
  conditional interference sensitivity.
- `matched_controls.csv`, `sensitivity_by_control.csv`, `summary.csv` and
  `manifest.json` in that same subfolder: source data and assumptions for the sensitivity analysis.

The control plot shows 1.2-2.5 h of the trace. The background line is the median of local
pre-injection m/z = 3 backgrounds across all 40 selected control plateaus.
The sensitivity calculation uses all 40 plateaus between 1.2 and 14 h, not just the
four shown in the control view. Plateaus are not independent experimental replicates.

## Interpretation

The sensitivity analysis assumes 10 atom% D in the liquid, negligible gas-phase D2, and an additional
m/z = 3 contribution beyond the calibration. The primary calculation uses
local-background-subtracted control signals; the raw-signal alternative is
retained in the source tables. Transfer of control/calibration response to
sample measurements is an assumption. Calibration can already include some
H3 contribution, so this calculation must not be applied as a correction to
the catalyst labels. The plotted control range is descriptive, not a confidence
interval or experimental replicate SD. No H/D or ML targets are modified.
