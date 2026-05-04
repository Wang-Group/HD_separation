# Data Extraction Inputs

This folder contains the trusted electrochemical and separation-coefficient
inputs used by the current machine-learning and figure-generation workflows.

## Active Data

- `all_data/LSV/`: LSV raw curves.
- `all_data/EIS/`: EIS raw curves.
- `all_data/CV20/`: CV raw curves retained for provenance.
- `all_data/Et/`: chrono/potential traces retained for provenance.
- `separation_coefficient.xlsx`: H/D separation-coefficient labels.

## LSV Geometry Metadata

`all_data/LSV/lsv_geometry_area_metadata.csv` records initial-screening samples
measured with 2 cm^2 geometry. For those samples, eta10 under the standard
1 cm^2 convention is extracted from the raw -20 mA point.

## Current Usage

- Figure 1 bottom panel rebuilds this-work overpotentials from `all_data/LSV/`
  and labels from `separation_coefficient.xlsx`.
- Figure 3 top panel extracts eta10 from `all_data/LSV/` for the current ML
  sample set.
- ML notebooks estimate Rs directly from `all_data/EIS/`.

`backups/` contains local backup copies created during data correction. The
active workflows do not read from that folder.
