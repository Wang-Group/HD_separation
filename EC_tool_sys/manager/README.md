# Manager Code Map

This folder contains hardware-facing platform managers and documents the
active-learning helper used by the H/D separation platform.

## Active-Learning Code Path

For the H-cell active-learning workflow, the relevant files are:

- `platform_manager_HD_Hcell.py`
- `../botorch_test/botorch_tool/active_learning.py`

The call chain is:

```text
platform_manager_HD_Hcell.py
  imports generate_new_exp_MO_one_system_V2
    from botorch_tool.active_learning

experiment results are written to exp_data
  -> generate_new_exp_MO_one_system_V2(exp_data, exp_index)
  -> three new BO candidates + one reference experiment
```

`generate_new_exp_MO_one_system_V2` in
`../botorch_test/botorch_tool/active_learning.py` is the multi-objective
qNEHVI path. It uses `lsv_overpotential_-0.01` and
`separation_coefficient` as the two objectives.

## Platform Feedback Loop

`platform_manager_HD_Hcell.py` closes the loop through the shared JSON
experiment table:

- synthesis consumes rows where `syn_completed` is not `True`;
- completed synthesis writes `syn_completed`, `storage_location`, and
  `in_storage`;
- catalysis consumes rows where `in_storage` is `True`;
- completed catalysis writes LSV, EIS/IR, mass-spectrometry timing, and
  `separation_coefficient`;
- after each completed batch of four catalysis rows, the manager calls
  `generate_new_exp_MO_one_system_V2`;
- the generated three candidates plus one reference row are appended with empty
  status fields, so the main loop can synthesize and test them next.

The current loop is closed for a seeded experiment table and a preset run
length. It is not a blank-slate planner, and hypervolume-based stopping is not
enabled in the runtime loop.

## Other Platform Managers

- `platform_manager_HD_Hcell_one_cell_v3_BO.py`: later/alternate one-cell
  workflow. It imports a single-objective helper and should not be used as
  the main qNEHVI active-learning workflow.
- `platform_manager_HD_Hcell_two_cell.py`: two-cell workflow manager.

## Safety Note

These scripts are hardware-facing. Running a platform manager on the lab
computer may control pumps, valves, potentiostats, mass spectrometry, and
robot motion. Treat examples and imports as potentially hardware-active
unless you have isolated the code from instrument drivers.
