import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # OpenIsing Speed

    This notebook aims to explore the speed of OpenIsing and to speed-up the solvers for faster DSE.

    ## Imports
    """)
    return


@app.cell
def _():
    import yaml
    import logging
    import tempfile

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from ising import api

    return api, logging, pd, tempfile, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(api, logging, tempfile, yaml):
    ## Base config file.
    ## Can be overwritten later.

    CONF_NAME = 'example.yaml'
    CONF_PATH = '../ising/inputs/config'

    with open(f'{CONF_PATH}/{CONF_NAME}') as _f:
        base_config = yaml.safe_load(_f)

    # ==================

    def run(problem_type: str, config: dict):
        """
        Call OpenIsing.

        The framework can only take a yaml config path. To enable editing config in a notebook,
        the passed config dict will be dumped to a temporary file and passed as the argument.
        """
        with tempfile.NamedTemporaryFile('w', suffix='.yaml') as f:
            yaml.safe_dump(config, f)
            f.flush()
            ans, _ = api.get_hamiltonian_energy(
                problem_type=problem_type, config_path=f.name, logging_level=logging.WARNING
            )
            return ans

    return (run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmarks
    """)
    return


@app.cell
def _():
    BENCHMARKS = {
        # 'Maxcut': ['G/G1.txt', 'G/K2000.txt'],
        # 'TSP'   : ['TSP/burma14.tsp', 'TSP/bayg29.tsp'],
        # 'QKP'   : ['Knapsack/jeu_100_25_1.txt', 'Knapsack/jeu_300_25_1.txt']
        'Maxcut'  : ['G/K2000.txt']
    }

    # SOLVERS = ['bSB', 'dSB', 'SA', 'inSituSA', 'SCA']
    SOLVERS = ['SA']
    NB_RUNS = 10

    # 0-4 which of the five num iter counts from the paper to use
    NUM_ITER = range(5)
    return BENCHMARKS, NB_RUNS, NUM_ITER, SOLVERS


@app.cell
def _(BENCHMARKS, NB_RUNS, NUM_ITER, SOLVERS, pd, run, yaml):
    _rows = []
    for _problem, _files in BENCHMARKS.items():
        # Load optimal config from OpenIsing paper
        with open(f'../Paper_OpenIsing/Pareto_curve_OPS/config/pareto_curves_{_problem}.yaml') as _f:
            _opt_base_conf = yaml.safe_load(_f)

        # Modify base config
        for _EFFORT in NUM_ITER:
            _cfg = _opt_base_conf | {
                'solvers': SOLVERS,
                'nb_runs': NB_RUNS,
                'gen_logfile': False
            }

            # Set solver number of iterations
            for _key, _val in _opt_base_conf.items():
                if _key.startswith('num_iterations_'):
                    _cfg[_key] = _val[_EFFORT]

            for _file in _files:
                _ans = run(_problem, _cfg | {'benchmark': f'./ising/benchmarks/{_file}'})
                for _s in SOLVERS:
                    for _i, (_e, _t) in enumerate(zip(_ans.energies[_s], _ans.computation_time[_s])):
                        _rows.append(
                            {
                                'problem'   : _problem,
                                'benchmark' : _ans.benchmark,
                                'effort'    : _EFFORT,
                                'solver'    : _s,
                                'run'       : _i,
                                'energy'    : _e,
                                'best_found': _ans.best_found,
                                'time_s'    : _t,
                                'ops'       : _ans.operation_count[_s],
                            }
                        )

    bench_results = pd.DataFrame(_rows)
    return


if __name__ == "__main__":
    app.run()
