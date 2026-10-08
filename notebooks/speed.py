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
    import time
    import logging
    import tempfile
    from argparse import Namespace

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from ising import api
    from ising.stages import TOP
    from ising.stages.maxcut_parser_stage import MaxcutParserStage
    from ising.utils.flow import parse_hyperparameters
    from ising.stages.simulation_stage import SimulationStage
    from ising.stages.initialization_stage import InitializationStage

    return (
        InitializationStage,
        MaxcutParserStage,
        Namespace,
        SimulationStage,
        TOP,
        parse_hyperparameters,
        pd,
        time,
        yaml,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(MaxcutParserStage, Namespace, TOP, yaml):
    EFFORT = 0      # Select num_iterations from config
    NB_TRIALS = 10
    BENCHMARK = 'ising/benchmarks/G/K2000.txt'

    _mult = {
        'capacitance'                   : 1,
        'stop_criterion'                : 1e-6,
        'num_iterations_Multiplicative' : 1000,
        'nb_flipping'                   : 20
    }

    SOLVERS = {
        'SA'                    : {},
        'inSituSA'              : {},
        'SCA'                   : {},
        'bSB'                   : {},
        'dSB'                   : {},
        'BRIM'                  : {},
        'Multiplicative'        : _mult,
        'Hierarchical_solver'   : _mult | {
            'core_solver'                   : 'Multiplicative',
            'nb_sweeps_Hierarchical_solver' : 2,
            'partitioning_technique'        : 'modularity',
            'nb_partitions'                 : 4,
            'nb_meta_nodes'                 : 4
        }
    }

    def load_yaml(dir):
        with open(TOP / dir) as f:
            return yaml.safe_load(f)

    def make_configs(solvers, benchmark, effort):
        base_cfg = load_yaml('ising/inputs/config/example.yaml')
        opt_cfg  = load_yaml('Paper_OpenIsing/Pareto_curve_OPS/config/pareto_curves_Maxcut.yaml')

        common_cfg = base_cfg | opt_cfg | {
            'problem_type'  : 'Maxcut',
            'benchmark'     : BENCHMARK,
            'dummy_creater' : False,
            'gen_logfile'   : False,
        }

        for key, values in opt_cfg.items():
            if key.startswith('num_iterations_'):
                common_cfg[key] = values[effort]

        configs = {}
        for solver, overrides in solvers.items():
            configs[solver] = Namespace(**(common_cfg | overrides | {'solvers': [solver]}))
    
        return configs

    configs = make_configs(SOLVERS, BENCHMARK, EFFORT)
    graph, best_found = MaxcutParserStage.G_parser(TOP / BENCHMARK)
    model = MaxcutParserStage.generate_maxcut(graph)
    return NB_TRIALS, configs, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark
    """)
    return


@app.cell
def _(
    InitializationStage,
    NB_TRIALS,
    SimulationStage,
    configs,
    model,
    parse_hyperparameters,
    pd,
    time,
):
    def benchmark(configs, model, nb_trials):
        rows = []
        for solver, cfg in configs.items():
            hp    = parse_hyperparameters(cfg, model)
            stage = SimulationStage([], config=cfg, ising_model=model)

            for trial in range(nb_trials):
                s0, _ = InitializationStage([], trail_id=trial, config=cfg, ising_model=model).run()
                trial_hp = hp | {'seed': trial + int(cfg.seed)}

                t0 = time.perf_counter()
                state, energy, reported, ops, its = stage.run_solver(
                    solver, s0, model, None, stop_criterion_it=False, **trial_hp
                )
                wall = time.perf_counter() - t0

                rows.append(dict(
                    solver=solver, trial=trial, cold=(trial == 0),
                    wall_s=wall, reported_s=reported, energy=energy,
                    iterations=its, ops=ops,
                ))

                print(f'[{solver}] Completed trial {trial}')

        return pd.DataFrame(rows)

    results = benchmark(configs, model, NB_TRIALS)
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
