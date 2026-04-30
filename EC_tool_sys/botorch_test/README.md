# BoTorch Utilities

This folder contains the importable BoTorch helper package and exploratory
notebooks used during development of the active-learning workflow.

## Main Package

- `botorch_tool/active_learning.py`
  - Canonical active-learning helper used by the H/D separation platform.
  - Contains the qNEHVI `generate_new_exp_MO_one_system_V2` function.
  - Also retains older exploratory helpers, including single-objective
    variants, for traceability.
- `botorch_tool/tools.py`
  - Gaussian-process initialization and acquisition-function helper routines.
- `setup.py`
  - Allows local editable installation with `pip install -e .`.

## Active-Learning Code Path

The H-cell platform manager imports:

```python
from botorch_tool.active_learning import generate_new_exp_MO_one_system_V2
```

`generate_new_exp_MO_one_system_V2` uses:

- three recipe variables: metal 1 volume, metal 2 volume, sodium citrate volume;
- two objectives: shifted HER activity from `lsv_overpotential_-0.01` and
  `separation_coefficient`;
- BoTorch `qNoisyExpectedHypervolumeImprovement` as the acquisition function.

See `../BO_PROVENANCE.md` for the manuscript-facing provenance note.

## Test And Development Notebooks

The `test/` folder contains exploratory notebooks and benchmark experiments.
These are useful for development history but should not be treated as the
authoritative Figure 4 workflow unless a notebook is explicitly cited.
