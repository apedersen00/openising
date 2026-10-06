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
    import logging
    import tempfile

    import pandas as pd
    import yaml

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
