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
    import os
    import yaml
    import time
    import logging
    import tempfile
    import subprocess
    import platform
    import datetime
    import pathlib
    import numba
    from argparse import Namespace
    import cProfile
    import pstats
    import io
    import marimo as mo

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from line_profiler import LineProfiler

    from ising import api
    from ising.stages import TOP
    from ising.stages.maxcut_parser_stage import MaxcutParserStage
    from ising.utils.flow import parse_hyperparameters
    from ising.stages.simulation_stage import SimulationStage
    from ising.stages.initialization_stage import InitializationStage
    from ising.solvers.Multiplicative import Multiplicative
    from ising.solvers.Hierarchical_solver import HierarchicalSolver

    return (
        HierarchicalSolver,
        InitializationStage,
        LineProfiler,
        MaxcutParserStage,
        Multiplicative,
        Namespace,
        SimulationStage,
        TOP,
        cProfile,
        datetime,
        io,
        mo,
        np,
        numba,
        os,
        parse_hyperparameters,
        pathlib,
        pd,
        platform,
        plt,
        pstats,
        subprocess,
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
        # 'BRIM'                  : {},
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
    return BENCHMARK, EFFORT, NB_TRIALS, configs, model


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
def _(
    BENCHMARK,
    EFFORT,
    TOP,
    configs,
    datetime,
    np,
    numba,
    os,
    pathlib,
    platform,
    results,
    subprocess,
    yaml,
):
    def git(*args):
        return subprocess.run(['git', *args], cwd=TOP, capture_output=True, text=True).stdout.strip()

    def run_metadata(label, benchmark, effort):
        return {
            'label'      : label,
            'commit'     : git('rev-parse', '--short', 'HEAD'),
            'dirty'      : git('status', '--porcelain') != '',
            'date'       : datetime.datetime.now().isoformat(timespec='seconds'),
            'benchmark'  : pathlib.Path(benchmark).stem,
            'effort'     : effort,
            'cpu'        : next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')),
            'affinity'   : ','.join(map(str, sorted(os.sched_getaffinity(0)))),
            'omp_threads': os.environ.get('OMP_NUM_THREADS', ''),
            'python'     : platform.python_version(),
            'numpy'      : np.__version__,
            'numba'      : numba.__version__,
        }

    def save_results(results, configs, label, benchmark, effort):
        meta = run_metadata(label, benchmark, effort)
        out = TOP / 'data/speed' / f"{meta['benchmark']}_effort{effort}_{label}_{meta['commit']}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        results.assign(**meta).to_csv(out, index=False, float_format='%.17g')
        with open(out.with_suffix('.yaml'), 'w') as f:
            yaml.safe_dump({solver: vars(cfg) for solver, cfg in configs.items()}, f)
        return out

    save_results(results, configs, 'baseline', BENCHMARK, EFFORT)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Profiling
    """)
    return


@app.cell
def _(
    InitializationStage,
    SimulationStage,
    cProfile,
    configs,
    io,
    model,
    parse_hyperparameters,
    pstats,
):
    def call_solver(solver, configs, model, trial=0):
        cfg   = configs[solver]
        hp    = parse_hyperparameters(cfg, model) | {'seed': trial + int(cfg.seed)}
        stage = SimulationStage([], config=cfg, ising_model=model)
        s0, _ = InitializationStage([], trail_id=trial, config=cfg, ising_model=model).run()
        return lambda: stage.run_solver(solver, s0, model, None, stop_criterion_it=False, **hp)

    def profile_func(call, n=20):
        profiler = cProfile.Profile()
        profiler.runcall(call)
        out = io.StringIO()
        stats = pstats.Stats(profiler, stream=out)
        stats.sort_stats('cumulative').print_stats('ising/', n)
        stats.sort_stats('tottime').print_stats(n)
        return out.getvalue()

    print(profile_func(call_solver('Hierarchical_solver', configs, model)))
    return (call_solver,)


@app.cell
def _(
    HierarchicalSolver,
    LineProfiler,
    Multiplicative,
    call_solver,
    configs,
    io,
    model,
):
    def profile_lines(call, *functions):
        lp = LineProfiler(*functions)
        lp.runcall(call)
        out = io.StringIO()
        lp.print_stats(stream=out, output_unit=1e-3, stripzeros=True)
        return out.getvalue()

    print(profile_lines(call_solver('Hierarchical_solver', configs, model),
                        Multiplicative.inner_loop_FE, HierarchicalSolver.make_hierarchy))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Profile Results

    ### Hierarchical

    #### `Multiplicative.py`: 65.3% of time spent in np.block (line 154)

    np.block is replaced with:

    ```python
    # The below implementation replaces:
    # previous_states = np.block([[np.sign(new_state)], [previous_states]])[:-1, :]
    # np.block is slow as it copies and creates a new matrix. Instead we modify the same matrix
    # sliding the states and setting the first entry to the newest state
    previous_states[1:] = previous_states[:-1]
    previous_states[0] = np.sign(new_state)
    ```

    The new slowest are then:

    - `np.clip(np.sign(new_state)) != np.sign(state))`: 28.8% of the time (line 146)
    - `np.clip(state + self.dt * dv, -1, 1)`: 28.1% of the time (line 144)


    #### `HierarchicalSolver.py`: 97.4% of time spent in upper_J[i, j] loop (line 243)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Performance Results

    AI-generated plotting code!
    """)
    return


@app.cell
def _(TOP, mo, np, pd, plt):
    def load_run(name):
        return pd.read_csv(TOP / 'data/speed' / name)

    def compare_runs(old, new):
        """Per solver: do all trials give the same energy, and how much faster is the new run?"""
        both = old.merge(new, on=['solver', 'trial'], suffixes=('_old', '_new'))
        both['same_energy'] = both.energy_old == both.energy_new
        warm = both[~both.cold_old & ~both.cold_new]          # skip the first call (compile/cache load)

        by_solver = both.groupby('solver')
        summary = pd.DataFrame({
            'identical'   : by_solver.same_energy.sum().astype(str) + '/' + by_solver.size().astype(str),
            'all_same'    : by_solver.same_energy.all(),
            'energy_old'  : by_solver.energy_old.mean(),
            'energy_new'  : by_solver.energy_new.mean(),
            'old_s'       : warm.groupby('solver').wall_s_old.median(),
            'new_s'       : warm.groupby('solver').wall_s_new.median(),
        })
        summary['speedup'] = summary.old_s / summary.new_s
        return summary.sort_values('speedup', ascending=False)

    def plot_speedup(summary, title):
        INK, INK_2, MUTED, GRID, AXIS, BAR, SURFACE = (
            '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#2a78d6', '#fcfcfb')
        s = summary.sort_values('speedup')                     # largest ends up on top
        y = np.arange(len(s))

        fig, ax = plt.subplots(figsize=(7, 0.42 * len(s) + 1.3), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.barh(y, s.speedup - 1, left=1, height=0.5, color=BAR)   # bars grow from the 1x line
        ax.axvline(1, color=AXIS, linewidth=1)

        ax.set_xscale('log')
        lo, hi = min(0.8, s.speedup.min() / 1.3), s.speedup.max() * 2.2
        ticks = [t for t in (0.25, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= t <= hi]
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks, labels=[f'{t:g}×' for t in ticks])
        ax.minorticks_off()
        ax.grid(axis='x', color=GRID, linewidth=1)
        ax.set_axisbelow(True)

        for yi, value in zip(y, s.speedup):
            ax.annotate(f'{value:.2f}×', (value, yi), xytext=(4 if value >= 1 else -4, 0), textcoords='offset points',
                        ha='left' if value >= 1 else 'right', va='center', color=INK, fontsize=9)

        names = [name if same else f'{name}  (energies differ)' for name, same in zip(s.index, s.all_same)]
        ax.set_yticks(y, labels=names)
        ax.tick_params(axis='y', length=0, colors=INK_2)
        ax.tick_params(axis='x', length=0, colors=MUTED)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS)
        ax.set_title(title, loc='left', color=INK, fontsize=11)
        fig.tight_layout()
        return fig

    OLD = 'K2000_effort0_baseline_2c2995d.csv'
    NEW = 'K2000_effort0_baseline_47bb988.csv'      # ← your new file

    _old, _new = load_run(OLD), load_run(NEW)
    speedup = compare_runs(_old, _new)
    _same_machine = _old.cpu.iloc[0] == _new.cpu.iloc[0]
    mo.vstack([
        mo.md('' if _same_machine else '**Warning:** the runs come from different machines, so the times are not comparable.'),
        speedup,
        plot_speedup(speedup, f"Speedup of '{_new.label.iloc[0]}' over '{_old.label.iloc[0]}' "
                              f"(median wall time, {_old.benchmark.iloc[0]})"),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
