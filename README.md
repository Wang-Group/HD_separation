# Data and Code for H/D Separation by Water Electrolysis

This repository contains the data, analysis notebooks, machine-learning
workflows, Bayesian-optimization code, and hardware-control provenance used for
the accompanying H/D separation manuscript.

The repository is organized for manuscript review and reproducibility rather
than as a standalone installable software package. Most figure/data analyses can
be inspected from notebooks and checked output files. The hardware-control code
is included for provenance and should not be run casually because it can control
real pumps, potentiostats, mass spectrometry, and robotic hardware.

## Quick Map

```text
.
|-- all_samples_708_corrected.csv        # corrected full sample-level table
|-- DATA_MANIFEST.md                     # figure/data mapping
|-- 0-Figures/                           # figure notebooks, data, outputs
|   |-- Figure1/
|   |-- Figure3/
|   `-- Figure4/
|-- hd_separation_ml/                    # ML data curation and model notebooks
|   |-- datasets/
|   |-- notebooks/
|   |-- outputs/
|   |-- data_extraction/
|   `-- archive/
`-- EC_tool_sys/                         # automation and BO provenance
    |-- BO_PROVENANCE.md
    |-- botorch_test/
    `-- manager/
```

## Canonical Data Tables

- `all_samples_708_corrected.csv`
  - Full corrected sample-level table used for manuscript-level figure updates.
  - 708 rows total: 560 initial-screening samples and 148 digit-suffixed
    BO/follow-up samples.
- `hd_separation_ml/datasets/final_no_BO_560.csv`
  - Corrected no-BO initial-screening dataset used by the final fixed-feature ML
    model.
- `0-Figures/Figure4/1-data/curated/figure4_bo_complete_112.csv`
  - Curated Figure 4 BO table: four systems, 28 completed datapoints each,
    112 rows total.

See `DATA_MANIFEST.md` for the detailed figure/data mapping.

## Reproducing Manuscript Figures

### Figure 1

- Notebook: `0-Figures/Figure1/notebooks/v4.ipynb`
- Main data:
  - `all_samples_708_corrected.csv`
  - `0-Figures/Figure1/data/all_data_corrected.xlsx`
- Output: `0-Figures/Figure1/outputs/pareto_plot_v2.png`

### Figure 3

Top panel:

- Notebook: `0-Figures/Figure3/notebooks/visualizewithcolors.ipynb`
- Main data: `all_samples_708_corrected.csv`
- Output: `0-Figures/Figure3/outputs/before_BO_Fe_Co_Others.png`

Machine-learning panels:

- Final notebook: `hd_separation_ml/notebooks/03_final_model_fixed12_560.ipynb`
- Final outputs: `hd_separation_ml/outputs/final_model_fixed12_560/`
- Figure-local copies: `0-Figures/Figure3/outputs/ml/`

The legacy 558-sample feature-selection workflow is retained in
`hd_separation_ml/archive/legacy_558_reference/` and is documented in
`hd_separation_ml/DATA_CURATION_LOG.md`.

### Figure 4

- Main folder: `0-Figures/Figure4/1-data/`
- BO/Pareto notebook: `0-Figures/Figure4/1-data/pareto_polygon_enclosure_code.ipynb`
- Descriptor-map notebook: `0-Figures/Figure4/1-data/BO_analysis -2D_symbolic.ipynb`
- Curated BO table: `0-Figures/Figure4/1-data/curated/figure4_bo_complete_112.csv`
- Outputs: `0-Figures/Figure4/1-data/outputs/`

## Machine-Learning Workflow

The ML workflow is documented in `hd_separation_ml/README.md`.

The intended notebook order is:

1. `hd_separation_ml/notebooks/01_feature_selection_558.ipynb`
   - Reproduces the legacy 558-sample feature-selection reference.
2. `hd_separation_ml/notebooks/02_final_model_fixed12_558.ipynb`
   - Reproduces the fixed 12-feature model on the 558-sample reference matrix.
3. `hd_separation_ml/notebooks/03_final_model_fixed12_560.ipynb`
   - Runs the final corrected 560-sample fixed-feature model.

Current checked summary:

- final corrected model: 560 samples;
- fixed 12-feature descriptor set;
- Extra Trees regressor;
- test R2 = 0.6726;
- test RMSE = 0.9109.

## Bayesian Optimization and Automation Code

The BO workflow is documented in:

- `EC_tool_sys/BO_PROVENANCE.md`
- `EC_tool_sys/manager/platform_manager_HD_Hcell.py`
- `EC_tool_sys/botorch_test/botorch_tool/active_learning.py`

The relevant function is:

```python
generate_new_exp_MO_one_system_V2
```

It uses:

- recipe variables: metal 1 volume, metal 2 volume, sodium citrate volume;
- objectives: HER activity from `lsv_overpotential_-0.01` and
  `separation_coefficient`;
- acquisition function: BoTorch `qNoisyExpectedHypervolumeImprovement`.

The hardware manager writes completed experimental results back to the
experiment table, calls the BO helper after each completed batch, appends new
candidate recipes, and then continues with the appended rows. The runtime loop
uses a preset experiment count rather than hypervolume-based stopping.

## Hardware Code Warning

`EC_tool_sys/manager/` contains hardware-facing automation code. It assumes a
specific Windows lab environment with configured COM ports, instrument drivers,
vendor software, robotic hardware, and mass-spectrometry control. Do not run
these scripts unless the hardware setup and safety procedures have been
verified.

## Archive Folders

Archive folders preserve historical notebooks, old outputs, backup files, and
legacy data snapshots. They are retained for traceability but are not the
recommended entry points for reproducing the current manuscript figures.

## Recommended Reading Order

For manuscript review:

1. `DATA_MANIFEST.md`
2. `hd_separation_ml/README.md`
3. `EC_tool_sys/BO_PROVENANCE.md`
4. figure-specific READMEs under `0-Figures/`

## Contact

For questions about the data, code, or manuscript context, please contact:

- Yibin Jiang: `yibinjiang@xmu.edu.cn`
