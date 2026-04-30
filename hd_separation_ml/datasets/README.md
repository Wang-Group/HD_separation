# Dataset Snapshots

This folder contains stable CSV snapshots used by the no-BO machine-learning
workflows.

## Current Dataset

- `all_samples_707_corrected.csv`
  - Full corrected label table.
  - 707 total rows.
  - Includes 560 no-BO initial-screening rows and 147 digit-suffixed
    BO/follow-up rows.

- `final_no_BO_560.csv`
  - Final no-BO initial-screening dataset.
  - Created by removing digit-suffixed samples from
    `all_samples_707_corrected.csv`.
  - Includes `Mg-Mg` and `Mo-Mg` after their missing electrochemical test data
    were added.
  - Excludes `Ag-Ce` because no valid Et trace is available.

- `sample_overpotentials_new_corrected_verison.csv`
  - A copy of the current root-level label table, preserved here as a stable
    dataset snapshot.

## Legacy Reference

- `legacy_all_samples_706_before_Mo-Mg.csv`
  - Label table before the missing `Mo-Mg` electrochemical data were added.

- `legacy_no_BO_559_label_before_Mo-Mg.csv`
  - No-BO subset of the legacy 706-row label table.
  - The earlier executable ML matrix had 558 samples because the electrochemical
    test data for `Mg-Mg` and `Mo-Mg` were incomplete at that time.
