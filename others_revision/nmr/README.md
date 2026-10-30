# NMR Data and Analysis

This folder contains the NMR inputs, processing code and numerical outputs for
calibration, stage-product spectra, concentrations and enrichment. See the [package README](../README.md)
for the adopted method, uncertainty definition and unresolved experimental
limitations.

## Contents

- `inputs/`: frozen instrument exports, including FIDs and saved real spectra.
- `input_manifest.json`: input paths and SHA-256 hashes.
- `sample_mapping.json`: spectrum roles, stage assignments and calibration groups.
- `config.json`: processing settings, random seed and uncertainty settings.
- `run_analysis.py`, `phase_validation.py`: integration, calibration and checks.
- `plot_response.py`: Arial figure exports in PNG, PDF and SVG.
- `plot_calibration_spectra.py`: combined low- and high-response calibration spectra
  for the SI, using saved real spectra and checking each peak area against the
  adopted integrals. This is a plotting-only export, not a calibration refit.
- `export_tables.py`: response-table CSV exports.
- `results/`: concentrations, integrals, processing records, uncertainty draws,
  figures and tables.
- `validation/`: numerical references used for reproduction checks.

## Reproduce

Run the complete non-ML package from the repository root:

```bash
python others_revision/run.py
python others_revision/run.py --verify-only
```

For NMR alone, after installing `others_revision/requirements.txt`:

```bash
python others_revision/nmr/run_analysis.py
python others_revision/nmr/export_tables.py
python others_revision/nmr/plot_response.py
python others_revision/nmr/plot_calibration_spectra.py
```

Raw inputs are not edited. Saved-phase results are the adopted values; alternative
phase/window checks remain separate. These NMR enrichment factors are not the
screening H/D separation-coefficient labels used by the ML workflows.
