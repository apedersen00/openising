import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiments

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

    return api, logging, pd, plt, tempfile, yaml


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

    return base_config, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _(base_config):
    # [Maxcut, TSP, ATSP, MIMO, MPPI]
    problem_type = 'Maxcut'

    # Overwrite base config
    config = base_config | {
        'benchmark'         : './ising/benchmarks/G/G1.txt',
        'dummy_creator'     : False,
        'solvers'           : ['SA', 'bSB'],
        'nb_runs'           : 5,
        'nb_cores'          : 4,
        'num_iterations_SA' : 8000,
        'num_iterations_SB' : 2000,
    }
    return config, problem_type


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run
    """)
    return


@app.cell
def _(config, pd, problem_type, run):
    ans = run(problem_type, config)

    rows = []
    for s in ans.config.solvers:
        pairs = zip(ans.energies[s], ans.computation_time[s])
        for i, (e, t) in enumerate(pairs):
            rows.append(
                {
                    'solver': s,
                    'run'   : i,
                    'energy': e,
                    'time_s': t
                }
            )

    results = pd.DataFrame(rows)
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sweep
    """)
    return


@app.cell
def _():
    BENCHMARKS = {
        'Maxcut': ['G/G1.txt', 'G/K2000.txt'],
        'TSP'   : ['TSP/burma14.tsp', 'TSP/bayg29.tsp'],
        'QKP'   : ['Knapsack/jeu_100_25_1.txt', 'Knapsack/jeu_300_25_1.txt']
    }

    SOLVERS = ['bSB', 'dSB', 'SA', 'inSituSA', 'SCA']
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
    return (bench_results,)


@app.cell
def _(bench_results):
    bench_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save Results
    """)
    return


@app.cell
def _(bench_results):
    bench_results.to_csv('../data/bench_results.csv', index=False)
    return


@app.cell
def _(BENCHMARKS, SOLVERS, pd, plt):
    _df  = pd.read_csv('../data/bench_results.csv')
    _df['eps'] = abs(_df.energy - _df.best_found) / abs(_df.best_found)

    _fig, _axs = plt.subplots(1, len(BENCHMARKS), figsize=(15, 4))

    for _ax, _problem in zip(_axs, BENCHMARKS):
        for _solver in SOLVERS:
            _d = _df[(_df.problem == _problem) & (_df.solver == _solver)]
            _m = _d.groupby('effort')[['ops', 'eps']].mean()
            _ax.loglog(_m.ops, _m.eps, marker='o', label=_solver)
        _ax.set_title(_problem)
        _ax.set_xlabel('operation count')

    _axs[0].legend()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
