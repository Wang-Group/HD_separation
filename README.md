# HD Separation Repository

This repository combines two related but distinct codebases:

- `EC_tool_sys/`: a Windows-based electrochemical lab automation stack for synthesis, testing, robotic handling, and active-learning-assisted experiment execution.
- `hd_separation_ml/`: data extraction, EIS fitting, descriptor engineering, and machine-learning analysis for H/D separation coefficient modeling.

At a high level, the repo captures both sides of the workflow:

1. Run or coordinate electrochemical experiments on the hardware platform.
2. Extract and clean electrochemistry data.
3. Fit EIS-derived parameters and assemble sample-level datasets.
4. Train or analyze ML models for H/D separation behavior.

## Top-Level Structure

```text
.
|-- README.md
|-- .gitignore
|-- EC_tool_sys/
|   |-- README.md
|   |-- tool.py
|   |-- botorch_test/
|   `-- manager/
`-- hd_separation_ml/
    |-- README.md
    |-- SUPPLEMENTARY_INFORMATION.md
    |-- data_extraction/
    |-- 1-ML_for_features/
    |-- 0-noble_no_BO_seed42/
    |-- extract_overpotential_separation_rs.py
    |-- improved_eis_data_loader.py
    |-- independent_fitting_manual_reorganized.py
    `-- starting_point_generator.py
```

## `EC_tool_sys/`

`EC_tool_sys` is the lab automation side of the repository. It contains instrument control, pump and valve drivers, motion control, workflow managers, and a BoTorch-based experiment suggestion package.

### Main folders

- `manager/`: core hardware orchestration code.
- `manager/configs/`: machine-specific JSON configuration files for pumps, controllers, instruments, and robotic hardware.
- `manager/pumps/`: syringe pump, switch valve, and peristaltic pump drivers.
- `manager/robotic_arm/`: MG400 and motion-controller helpers.
- `manager/instrument/`: Corrtest and MASsoft-related instrument wrappers.
- `manager/bopai_controller/`: motion/output controller integration.
- `botorch_test/`: active learning and Bayesian optimization utilities, plus notebooks and test assets.
- `tool.py`: general electrochemistry / signal-processing helpers.

### Important entry points

- `EC_tool_sys/manager/platform_manager_HD_Hcell_one_cell_v3_BO.py`
- `EC_tool_sys/manager/platform_manager_HD_Hcell_two_cell.py`
- `EC_tool_sys/botorch_test/botorch_tool/active_learning.py`

### Notes

- This code is tightly coupled to a specific Windows lab environment.
- Config files include machine-specific COM ports, IP addresses, and vendor software paths.
- Some imports appear to expect the package name `EC_tool`, while this checkout is named `EC_tool_sys`.

## `hd_separation_ml/`

`hd_separation_ml` is the analysis and modeling side of the repository. It contains raw-data extraction assets, EIS fitting scripts, descriptor tables, notebooks for final figures, and supporting ML pipelines.

### Main folders

- `data_extraction/`: raw data archive, extraction notebook, summary spreadsheets, and assembled metadata.
- `1-ML_for_features/`: auxiliary ML pipeline for element-level descriptors such as PZC and related outputs.
- `0-noble_no_BO_seed42/`: result artifacts from the main selected-feature no-BO workflow.

### Main scripts

- `hd_separation_ml/extract_overpotential_separation_rs.py`: assembles overpotential, separation coefficient, and Rs values into a sample-level table.
- `hd_separation_ml/improved_eis_data_loader.py`: loads EIS data from TXT, CSV, or JSON formats.
- `hd_separation_ml/starting_point_generator.py`: creates diverse starting points for EIS fitting.
- `hd_separation_ml/independent_fitting_manual_reorganized.py`: fits EIS spectra sample-by-sample and writes fitting summaries.

### Main notebooks and docs

- `hd_separation_ml/0-noble_no_BO_seed42_plot.ipynb`: final plotting / analysis notebook referenced by the supplementary notes.
- `hd_separation_ml/SUPPLEMENTARY_INFORMATION.md`: detailed explanation of dataset assembly, feature engineering, and the final ML workflow.
- `hd_separation_ml/README.md`: currently a placeholder GitLab template rather than project-specific documentation.

## Data And Artifacts

The repository includes a large amount of experiment data and generated outputs:

- raw LSV / EIS text files under `hd_separation_ml/data_extraction/all_data/`
- CSV, XLSX, PNG, and notebook artifacts in `hd_separation_ml/`
- experiment tables, notebooks, and serialized optimization outputs in `EC_tool_sys/botorch_test/test/`

This means the repo is better understood as a working research snapshot than as a minimal clean software package.

## Getting Started

There is no single root-level environment file. A practical way to approach the repository is:

1. Start by reading the subproject you care about:
   - `EC_tool_sys/README.md`
   - `hd_separation_ml/SUPPLEMENTARY_INFORMATION.md`
2. If you are working on hardware automation, validate all configs under `EC_tool_sys/manager/configs/` before running anything.
3. If you are working on the ML side, begin with:
   - `hd_separation_ml/0-noble_no_BO_seed42_plot.ipynb`
   - `hd_separation_ml/extract_overpotential_separation_rs.py`
   - `hd_separation_ml/independent_fitting_manual_reorganized.py`

## Caution

- `EC_tool_sys/` can drive real hardware and should be treated as non-sandboxed lab-control code.
- The repository contains machine-specific paths, ports, and device assumptions.
- Several subfolders contain generated artifacts, backup/conflict files, and experimental notebooks.
- This README describes the current structure only; it does not attempt to standardize the package layout.
