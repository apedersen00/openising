# Data

## `lagrange_runs.csv.gz`

Standard Lagrange solver using event-based solver.
Linear cluster size. Re-run from best-so-far.

```
RUNS = {
    'P': [0.05, 0.1, 0.2, 0.5, 1.0],
    'n': [0.05, 0.1, 0.2, 0.5, 1.0]
}

_N_ITER = 50
INIT_SIZES = [1, 10, 20, 40, 60, 80]
END_SIZE = 1
SEEDS = 200
```

## `lagrange_runs_nowinner.csv.gz`

Standard Lagrange solver using event-based solver.
Linear cluster size. Re-run from latest sigmas i.e. no energy calculation per iteration.

```
RUNS = {
    'P': [0.05, 0.1, 0.2, 0.5, 1.0],
    'n': [0.05, 0.1, 0.2, 0.5, 1.0]
}

_N_ITER = 50
INIT_SIZES = [1, 10, 20, 40, 60, 80]
END_SIZE = 1
SEEDS = 200
```