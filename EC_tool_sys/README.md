# EC_tool_sys

`Ec_tool_sys` is a Windows-based lab automation codebase for electrochemical catalyst synthesis, screening, mass-spectrometry-assisted evaluation, and Bayesian optimization.

The repository combines two main pieces:

- A hardware orchestration layer for pumps, valves, motion control, robotic handling, potentiostat control, and mass spectrometer automation.
- A BoTorch-based active learning package used to propose new catalyst experiments from previous results.

This is not a generic plug-and-play Python package. It is a lab-specific control stack with machine-specific COM ports, IP addresses, vendor software dependencies, and real hardware side effects.

## What the repo does

At a high level, the workflow implemented here is:

1. Prepare catalyst precursor recipes with pumps and valves.
2. Run electrodeposition or electrochemical synthesis on multiple channels.
3. Move samples with a robotic arm and slide rail between storage, deposition, cleaning, drying, and test positions.
4. Perform catalytic testing in one-cell or two-cell H-cell setups.
5. Collect mass spectrometry data during testing.
6. Feed results back into an experiment table and optionally generate the next experiments with BoTorch.

## Repository layout

```text
.
|-- README.md
|-- tool.py
|-- manager/
|   |-- README.md
|   |-- platform_manager_HD_Hcell.py
|   |-- platform_manager_HD_Hcell_two_cell.py
|   |-- bopai_controller/
|   |-- configs/
|   |-- instrument/
|   |-- pumps/
|   |-- robotic_arm/
|   |-- runze_ascii/
|   `-- runze_driver/
`-- botorch_test/
    |-- setup.py
    |-- botorch_tool/
    `-- test/
```

## Main components

### `manager/`

The `manager` package contains the lab automation layer:

- `README.md`: concise map of the platform-manager files and active-learning code path.
- `platform_manager_HD_Hcell.py`: H-cell workflow manager that imports and calls `generate_new_exp_MO_one_system_V2`.
- `platform_manager_HD_Hcell_two_cell.py`: two-cell workflow manager.
- `pumps/`: liquid handling logic built on syringe pumps, switch valves, and peristaltic pump drivers.
- `robotic_arm/`: MG400 robot and slide-rail motion helpers.
- `instrument/`: wrappers around electrochemical workstation and Hiden MASsoft 10 control.
- `bopai_controller/`: motion / output control for the Bopai controller.
- `configs/`: JSON configuration files for pumps, instruments, robotic arm, and Bopai controller.
- `runze_driver/` and `runze_ascii/`: low-level communication helpers for pump hardware.

### `botorch_test/`

`botorch_test` is a separate BoTorch-focused package used for experiment suggestion and benchmarking:

- `botorch_tool/active_learning.py`: candidate generation logic. The qNEHVI function is `generate_new_exp_MO_one_system_V2`, which performs two-objective optimization over metal 1, metal 2, and sodium citrate volumes. Older development variants such as `generate_new_exp_SO_one_system_V3` are retained in the same file.
- `botorch_tool/tools.py`: GP model initialization and acquisition optimization helpers.
- `test/`: notebooks, datasets, and serialized experiment results used during model development.

### `tool.py`

Top-level signal processing helpers for tasks such as:

- reading GC-MS CSV output,
- low-pass filtering,
- step detection,
- threshold-based anomaly detection,
- time-window-based peak integration.

## Hardware and software assumptions

This code was written for a specific experimental platform and assumes access to:

- Windows with COM port access.
- Runze or compatible syringe pumps / switch valves.
- A Bopai controller.
- A Dobot MG400 robotic arm plus slide rail / motion controller.
- Corrtest electrochemical workstation channels.
- Hiden MASsoft 10 for mass spectrometry.

It also assumes vendor software and drivers are already installed and reachable from the configured machine.

## Python and package requirements

The repo does not ship a pinned root environment file, but imports indicate the following Python packages are needed:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `pyserial`
- `pywinauto`
- `pywin32`
- `pymodbus`
- `telemetrix`
- `torch`
- `botorch`
- `gpytorch`
- `jupyter`

For a fresh environment on Windows PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scipy matplotlib pyserial pywinauto pywin32 pymodbus telemetrix torch botorch gpytorch jupyter
pip install -e .\botorch_test
```

Notes:

- `botorch_test` has its own `setup.py`, so `pip install -e .\botorch_test` is the intended way to make `botorch_tool` importable.
- `pymodbus` imports in this repo use the older module layout, so you may need a 2.x-compatible release.
- Exact versions should be validated against your instrument drivers and vendor software.

## Important package-name caveat

Most source files import modules as `EC_tool.*`, for example:

```python
from EC_tool.manager.pumps import PumpOperations
```

This checkout is currently named `Ec_tool_sys`, so those imports will not work unchanged in a fresh Python environment unless you do one of the following:

- rename the repository folder to `EC_tool`,
- refactor imports to match the current folder / package name,
- package the project explicitly under the `EC_tool` name.

There are also imports that reference modules not present in this snapshot, such as `EC_tool.algorithm.data_analysis`. Plan to reconcile those before treating this repo as a clean standalone package.

## Configuration

The main configuration files live under `manager/configs/`.

| File | Purpose |
|---|---|
| `manager/configs/pumps/config.json` | Pump COM ports, slave addresses, and channel mapping for reagents / cells |
| `manager/configs/bopai/bopai.json` | Bopai serial settings, axis mapping, output mapping, motion parameters |
| `manager/configs/robotic_arm/config.json` | Robot IP, Modbus port, and slide-rail controller parameters |
| `manager/configs/instruments/config.json` | MASsoft executable path and instrument connection settings |
| `manager/configs/robotic_arm/relay.json` | Optional relay control parameters |

Current checked-in configs are machine-specific. For example, the repo currently contains hard-coded values such as:

- pump COM ports like `COM5`, `COM9`, and `COM11`,
- gas pump ports like `COM19` to `COM22`,
- robot / controller IPs in the `192.168.192.x` range,
- MASsoft installed under `C:\Program Files (x86)\Hiden Analytical\MASsoft10\`.

Update these before running on another workstation.

## Running the platform managers

There is no single polished CLI entry point. The main orchestration objects are the hardware-facing `PlatformManager` classes:

- `manager/platform_manager_HD_Hcell.py`
- `manager/platform_manager_HD_Hcell_two_cell.py`

`platform_manager_HD_Hcell.py` imports `generate_new_exp_MO_one_system_V2` from
`botorch_tool.active_learning`. The snippet below is an orientation example
only:

```python
import logging
from manager.platform_manager_HD_Hcell import PlatformManager

logger = logging.getLogger("ec_tool")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

pm = PlatformManager(
    config_folder="manager/configs",
    exp_path="path/to/experiment_table.json",
    mass_save_path="path/to/mass_output.csv",
    logger=logger,
    active_learning=True,
)

pm.initialize_devices(refill_cat_cell_flag=False)
pm.run_exp_in_parallel(terminate_num=[12, 12])
```

Other managers may take different constructor signatures and may use different
active-learning helpers.

Treat the snippet above as an orientation example, not a guaranteed drop-in script. Before running it, you will likely need to:

- fix the `EC_tool.*` imports,
- verify the experiment JSON schema,
- start vendor software,
- confirm all COM ports and networked devices are reachable,
- make sure output directories already exist.

## Experiment table format

The platform managers expect an experiment table stored as JSON and loaded with `pandas.read_json(...)`.

Columns referenced in the code include fields such as:

- `substance1`, `substance2`, `substance3`
- `volume1`, `volume2`, `volume3`
- `syn_completed`
- `storage_location`
- `in_storage`
- `cat_completed`
- `ir_value`
- `mass_time_start`
- `mass_time_end`
- `separation_coefficient`

Example experiment tables and generated datasets can be found under:

- `botorch_test/test/EC_HD/`

These files are useful as references when reconstructing the expected schema.

## Active learning workflow

The BoTorch package in `botorch_test/` is used to suggest follow-up experiments from existing results.

Key points:

- The active-learning code path is documented in `BO_PROVENANCE.md`.
- `manager/platform_manager_HD_Hcell.py` calls `generate_new_exp_MO_one_system_V2(...)` from `botorch_test/botorch_tool/active_learning.py`.
- `generate_new_exp_MO_one_system_V2(...)` builds new candidates for a specific bimetallic reactant combination.
- The optimization code uses Gaussian process models from BoTorch / GPyTorch.
- The two objectives are HER activity from `lsv_overpotential_-0.01` and the H/D separation coefficient.
- Candidate generation uses BoTorch `qNoisyExpectedHypervolumeImprovement` and is constrained to the composition simplex by inequality constraints.
- The notebooks in `botorch_test/test/` appear to be the primary experimentation and validation surface for this part of the repo.

If you only want the optimization utilities, you can often work within `botorch_test/` independently of the full hardware stack.

### Platform feedback loop

When started from a seeded experiment table, `platform_manager_HD_Hcell.py`
writes synthesis results and catalysis objectives back to the same JSON table,
calls the qNEHVI candidate generator after each batch of four catalysis results,
appends three new candidates plus one reference row, and then the main loop
picks up the appended rows for synthesis/testing.

The current runtime loop uses a preset `terminate_num` count. Hypervolume-based
stopping is not enabled in the code path.

## Known caveats in this snapshot

This repository appears to be a working lab snapshot rather than a cleaned release. In particular:

- Several imports point to modules that are not included here.
- Some package `__init__` files reference files that are not present in the current tree.
- The repo includes conflict / backup files such as `*_Conflict.py` and `*_Conflict.json`.
- MASsoft automation uses UI automation plus hard-coded screen coordinates, so changes in resolution, window layout, scaling, or MASsoft version may break it.
- Corrtest channel addresses and serial numbers are hard-coded in `manager/instrument/corrtest_eight_channel.py`.
- There is no dry-run or simulator mode in the main workflow scripts.

## Safety

This code can actuate real pumps, valves, motion systems, and robotic hardware.

Before running anything on a new machine or after editing control code:

- verify every COM port and IP address,
- confirm valve and reagent channel mappings,
- test motions without samples first,
- keep emergency stop procedures available,
- review hard-coded robot coil addresses and motion commands,
- make sure any downstream pumps or waste paths are physically safe.

Do not run the platform managers unattended until you have validated the entire hardware path on your specific setup.

## Suggested next cleanup steps

If you plan to maintain this repo long-term, the most valuable improvements would be:

1. Rename or package the project consistently as `EC_tool`.
2. Add a real root `pyproject.toml` or `requirements.txt`.
3. Move all machine-specific values out of source files and into config.
4. Separate archived notebooks / results from runtime code.
5. Add a small launcher script or CLI for the supported workflows.
6. Document the missing `algorithm` module or restore it to the repo.

## Correspondence

For questions about the platform or research context, please contact:

- Yibin Jiang: `yibinjiang@xmu.edu.cn`
- Cheng Wang: `wangchengxmu@xmu.edu.cn`

## Current status

This README is meant to make the current snapshot understandable and runnable with lab context. It does not mean the repository is fully portable yet. Treat the code as an internal platform control stack that likely needs local adaptation before first use.
