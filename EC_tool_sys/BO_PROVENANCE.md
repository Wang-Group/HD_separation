# Bayesian Optimization Code Provenance

This note records the code path that supports the Bayesian-optimization
description in the manuscript.

## Entry Point

The H/D separation BO workflow is implemented through:

- `manager/platform_manager_HD_Hcell.py`
- `botorch_test/botorch_tool/active_learning.py`

The platform manager imports the BoTorch package helper:

```python
from botorch_tool.active_learning import generate_new_exp_MO_one_system_V2
```

This makes `botorch_test/botorch_tool/active_learning.py` the canonical
source for the qNEHVI BO function. The previous package copy only
contained the older single-objective helper; it has been replaced with the
full active-learning helper that contains `generate_new_exp_MO_one_system_V2`.

After experimental results have been written back to the experiment table,
the platform manager calls `generate_new_exp_MO_one_system_V2(...)`.

## Platform Workflow

The current platform workflow in `manager/platform_manager_HD_Hcell.py` is:

1. `run_exp_in_parallel(...)` repeatedly checks the experiment table and starts
   synthesis when at least four unsynthesized rows are available.
2. `run_synthesis(...)` prepares four recipes, electrodeposits catalysts, moves
   them into storage, and writes `syn_completed`, `storage_location`, and
   `in_storage` back to the JSON experiment table.
3. `run_catalysis(...)` selects the next stored catalyst, performs H/D testing,
   LSV processing, mass-spectrometry processing, and writes
   `lsv_overpotential_*`, `ir_value`, `mass_time_start`, `mass_time_end`, and
   `separation_coefficient` back to the same table.
4. After every fourth catalysis result (`cat_index % 4 == 3`), the manager calls
   `generate_new_exp_MO_one_system_V2(...)`.
5. The active-learning function returns three qNEHVI candidates plus one
   reference recipe. These rows are appended to the experiment table with empty
   completion/status fields.
6. On the next loop pass, the newly appended rows satisfy the unsynthesized-row
   condition and are picked up by `run_synthesis(...)`.

With a seeded experiment table and connected hardware, this forms the
experiment-measurement-BO update cycle used by the platform. The runtime loop
stops by a preset experiment count passed to
`run_exp_in_parallel(terminate_num=...)`; hypervolume-based autonomous stopping
is sketched in comments but is not active in the current code.

## Objective Definition

The qNEHVI function is:

```python
generate_new_exp_MO_one_system_V2(exp_data, exp_index)
```

It uses three normalized composition variables:

- `volume1`: metal 1 stock-solution volume, mapped from 1 to 19 mL
- `volume2`: metal 2 stock-solution volume, mapped from 1 to 19 mL
- `volume3`: sodium citrate solution volume, mapped from 0 to 18 mL

The water volume is calculated as:

```python
volume4 = 20 - volume1 - volume2 - volume3
```

The inequality constraint keeps the recipe inside the 20 mL composition
simplex.

The function optimizes two objectives:

```python
train_y = torch.tensor(np.concatenate([y_mean + 1.2, y_sc_mean], axis=1))
```

- `y_mean + 1.2`: HER activity from `lsv_overpotential_-0.01`, shifted so
  that the acquisition function maximizes a positive activity-like value.
- `y_sc_mean`: H/D separation coefficient.

## Acquisition Function

The acquisition function is BoTorch qNEHVI:

```python
acq_func = qNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=[0, 0],
    X_baseline=train_x,
    prune_baseline=True,
    sampler=sampler,
)
```

The optimizer requests `q=3` new candidates. The platform then appends one
reference experiment, giving four experiments per BO batch.

## Relation to Figure 4

The Figure 4 curated dataset contains four BO systems:

- Co-Mo
- Co-V
- Co-Ni
- Fe-Ni

Each system contains 28 experiments, corresponding to seven batches of four
experiments. The first two batches are uniformly sampled, and the later
batches are selected by the BO workflow above.

Hypervolume improvement in the manuscript/figure is evaluated relative to
batch 1. The pre-BO primary-screening reference values may be discussed as
context, but they are not part of the current Figure 4 hypervolume baseline.

## Do Not Confuse With Older Helpers

`botorch_test/botorch_tool/active_learning.py` also contains older or
exploratory helpers, including single-objective functions such as
`generate_new_exp_SO_one_system_V3`. Those functions use
`qLogNoisyExpectedImprovement` and are not the Figure 4 multi-objective
qNEHVI path unless a specific platform manager or notebook is shown to call
them.
