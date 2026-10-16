import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multiplicative vs MultiplicativeEvent

    Time per iteration (one analog run and its kick) of the two multiplicative solvers in openising:

    - **Multiplicative**: Forward Euler, a run stops when every voltage sits at a rail or after `num_iterations` steps
    - **MultiplicativeEvent**: jumps from one comparator flip to the next, a run is limited to the time of `num_iterations` Euler steps

    Both run through `SimulationStage.run_solver` with the same config, seeds and starting states, for a range of step budgets (`num_iterations`).
    The event solver also runs at its default budget (10¹² steps), where every run settles.

    Run it with one BLAS thread, set in the shell before Python starts (in script mode numpy is imported before the first cell runs):

    ```
    OMP_NUM_THREADS=1 uv run marimo edit notebooks/compare_solvers.py
    OMP_NUM_THREADS=1 uv run python notebooks/compare_solvers.py
    ```

    The results and the plot are saved to `data/compare_solvers/`.
    """)
    return


@app.cell
def _():
    import os
    import time
    import platform
    import datetime
    import subprocess
    from argparse import Namespace

    import yaml
    import numba
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import marimo as mo

    from ising.stages import TOP
    from ising.stages.maxcut_parser_stage import MaxcutParserStage
    from ising.stages.qkp_parser_stage import QKPParserStage
    from ising.stages.initialization_stage import InitializationStage
    from ising.stages.simulation_stage import SimulationStage
    from ising.utils.flow import parse_hyperparameters

    return (
        FuncFormatter,
        InitializationStage,
        MaxcutParserStage,
        Namespace,
        QKPParserStage,
        SimulationStage,
        TOP,
        datetime,
        mo,
        np,
        numba,
        os,
        parse_hyperparameters,
        pd,
        platform,
        plt,
        subprocess,
        time,
        yaml,
    )


@app.cell
def _(mo, os):
    _threads = os.environ.get('OMP_NUM_THREADS')
    if _threads != '1':
        print(f'Warning: OMP_NUM_THREADS = {_threads}, the timings may use more than one BLAS thread')
    mo.callout(mo.md(f'`OMP_NUM_THREADS = {_threads}`: the timings may use more than one BLAS thread. '
                     'Restart with `OMP_NUM_THREADS=1`.'), kind='warn') if _threads != '1' else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    The config is `example.yaml` with both solvers. The step budgets are chosen so the Forward Euler runs stay affordable:
    on QKP its runs need about 4·10⁷ steps to settle.
    """)
    return


@app.cell
def _(MaxcutParserStage, Namespace, QKPParserStage, TOP, yaml):
    NB_TRIALS   = 5             # seeds and starting states per point
    NB_FLIPPING = 20            # iterations per solve
    SOLVERS     = ['Multiplicative', 'MultiplicativeEvent']
    SETTLED     = 10**12        # default budget of MultiplicativeEvent: every run settles

    # name: (problem type, benchmark file, step budgets)
    BENCHMARKS = {
        'K2000'        : ('Maxcut', 'ising/benchmarks/G/K2000.txt',               [10**2, 10**3, 10**4, 10**5]),
        'jeu_100_25_1' : ('QKP',    'ising/benchmarks/Knapsack/jeu_100_25_1.txt', [10**2, 10**3, 10**4, 10**5, 10**6, 10**7]),
    }

    def make_setup(problem, benchmark):
        """Config and Ising model of a benchmark, built like openising's parser stages do."""
        with open(TOP / 'ising/inputs/config/example.yaml') as f:
            base = yaml.safe_load(f)
        cfg = Namespace(**(base | {
            'problem_type'  : problem,
            'benchmark'     : benchmark,
            'solvers'       : SOLVERS,
            'nb_flipping'   : NB_FLIPPING,
            'dummy_creator' : False,
            'gen_logfile'   : False,
        }))
        if problem == 'Maxcut':
            graph, _ = MaxcutParserStage.G_parser(TOP / benchmark)
            model = MaxcutParserStage.generate_maxcut(graph)
        else:
            graph, _ = QKPParserStage.QKP_parser(TOP / benchmark)
            model = QKPParserStage([], config=cfg).generate_knapsack(graph, float(cfg.weight_constant))
        return cfg, model

    setups = {name: make_setup(problem, path) for name, (problem, path, _) in BENCHMARKS.items()}
    return BENCHMARKS, NB_FLIPPING, NB_TRIALS, SETTLED, SOLVERS, setups


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Benchmark

    Every solver first runs once untimed, so numba compiles (or loads its cache) before the timing starts.
    A solve without iterations measures the fixed cost of a solve (model transform, `set_params`, ...), which is taken off
    every timed solve: time per iteration = (wall time − fixed cost) / `NB_FLIPPING`.
    """)
    return


@app.cell
def _(
    BENCHMARKS,
    InitializationStage,
    NB_FLIPPING,
    NB_TRIALS,
    SETTLED,
    SOLVERS,
    SimulationStage,
    parse_hyperparameters,
    pd,
    setups,
    time,
):
    def time_solve(stage, solver, s0, model, hp, budget):
        """Wall time and energy of one solve through openising's run_solver."""
        hp = hp | {f'num_iterations_{name}': budget for name in SOLVERS}
        t0 = time.perf_counter()
        _, energy, *_ = stage.run_solver(solver, s0, model, None, stop_criterion_it=False, **hp)
        return time.perf_counter() - t0, energy

    def run_benchmark():
        rows = []
        for name, (_, _, budgets) in BENCHMARKS.items():
            cfg, model = setups[name]
            hp     = parse_hyperparameters(cfg, model)
            stage  = SimulationStage([], config=cfg, ising_model=model)
            starts = [InitializationStage([], trail_id=t, config=cfg, ising_model=model).run()[0] for t in range(NB_TRIALS)]

            for solver in SOLVERS:
                time_solve(stage, solver, starts[0], model, hp | {'nb_flipping': 2}, budgets[0])
                fixed = min(time_solve(stage, solver, starts[0], model, hp | {'nb_flipping': 0}, budgets[0])[0]
                            for _ in range(3))

                for budget in budgets + ([SETTLED] if solver == 'MultiplicativeEvent' else []):
                    for trial, s0 in enumerate(starts):
                        wall, energy = time_solve(stage, solver, s0, model, hp | {'seed': int(cfg.seed) + trial}, budget)
                        rows.append(dict(
                            benchmark=name, solver=solver, budget=budget, trial=trial, wall_s=wall, fixed_s=fixed,
                            per_iteration_s=(wall - fixed) / NB_FLIPPING, energy=energy,
                        ))
                    _last = pd.DataFrame(rows[-NB_TRIALS:])
                    print(f'[{name}] {solver:20s} budget {budget:7.0e}: '
                          f'{_last.per_iteration_s.median() * 1e3:10.3f} ms per iteration')
        return pd.DataFrame(rows)

    results = run_benchmark()
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time per Iteration

    Median over the seeds. `same_energy` counts the seeds where both solvers end with the same energy.
    """)
    return


@app.cell
def _(pd, results):
    def summarize(results):
        """Median time per iteration per benchmark, budget and solver, the speed-up and whether the energies agree."""
        med = results.pivot_table(index=['benchmark', 'budget'], columns='solver', values='per_iteration_s', aggfunc='median')
        energies = results.pivot_table(index=['benchmark', 'budget', 'trial'], columns='solver', values='energy').dropna()
        same = (energies.Multiplicative == energies.MultiplicativeEvent).groupby(level=['benchmark', 'budget'])
        table = pd.DataFrame({
            'Multiplicative_s'      : med.Multiplicative,
            'MultiplicativeEvent_s' : med.MultiplicativeEvent,
            'speedup'               : med.Multiplicative / med.MultiplicativeEvent,
            'same_energy'           : same.sum().astype(int).astype(str) + '/' + same.size().astype(str),
        })
        return table

    summary = summarize(results)
    summary
    return (summary,)


@app.cell
def _(BENCHMARKS, FuncFormatter, NB_FLIPPING, NB_TRIALS, SETTLED, plt, setups, summary):
    def format_time(t):
        return f'{t * 1e6:g} µs' if t < 1e-3 else f'{t * 1e3:g} ms' if t < 1 else f'{t:g} s'

    def plot_comparison(summary):
        INK, INK_2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
        COLORS = {'Multiplicative': '#2a78d6', 'MultiplicativeEvent': '#eb6834'}     # categorical slots 1 and 2

        names = list(BENCHMARKS)
        fig, axes = plt.subplots(2, len(names), figsize=(5.6 * len(names), 6.6), sharex='col', squeeze=False,
                                 facecolor=SURFACE)
        for col, name in enumerate(names):
            s = summary.loc[name]
            settled = s.MultiplicativeEvent_s.get(SETTLED)
            s = s.drop(SETTLED, errors='ignore')
            top, bottom = axes[0, col], axes[1, col]

            # Time per iteration of both solvers
            for solver, color in COLORS.items():
                y = s[f'{solver}_s']
                top.plot(s.index, y, color=color, linewidth=2, marker='o', markersize=8, markeredgecolor=SURFACE,
                         markeredgewidth=2, solid_capstyle='round', solid_joinstyle='round', label=solver)
                top.annotate(solver, (s.index[-1], y.iloc[-1]), xytext=(8, 0), textcoords='offset points',
                             va='center', color=INK_2, fontsize=9)
            top.set_yscale('log')
            top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_time(v)))
            top.set_ylabel('time per iteration', color=INK_2)
            cfg, model = setups[name]
            top.set_title(f'{name} ({model.num_variables} spins)', loc='left', color=INK, fontsize=11, pad=20)
            if settled is not None:
                top.text(0, 1.03, f'MultiplicativeEvent at its default budget (settled): '
                                  f'{format_time(float(f"{settled:.2g}"))} per iteration',
                         transform=top.transAxes, va='bottom', color=INK_2, fontsize=8.5)

            # Speed-up of the event solver at the same budget
            bottom.plot(s.index, s.speedup, color=INK, linewidth=2, marker='o', markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=2, solid_capstyle='round', solid_joinstyle='round')
            bottom.annotate(f'{s.speedup.iloc[-1]:,.0f}×', (s.index[-1], s.speedup.iloc[-1]), xytext=(8, 0),
                            textcoords='offset points', va='center', color=INK, fontsize=9)
            bottom.set_yscale('log')
            bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}×'))
            bottom.set_ylabel('speed-up', color=INK_2)
            bottom.set_xlabel('step budget (num_iterations)', color=INK_2)

            for ax in (top, bottom):
                ax.set_facecolor(SURFACE)
                ax.set_xscale('log')
                ax.set_xlim(s.index[0] / 2, s.index[-1] * 40)       # room for the labels at the line ends
                ax.minorticks_off()
                ax.grid(color=GRID, linewidth=1)
                ax.set_axisbelow(True)
                ax.tick_params(length=0, colors=MUTED)
                for side in ('top', 'right'):
                    ax.spines[side].set_visible(False)
                for side in ('left', 'bottom'):
                    ax.spines[side].set_color(AXIS)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=2, frameon=False, labelcolor=INK_2, fontsize=9)
        fig.suptitle(f'Time per iteration at the same step budget ({NB_FLIPPING} iterations per solve, '
                     f'median of {NB_TRIALS} seeds)', x=0.01, ha='left', color=INK, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    fig = plot_comparison(summary)
    fig
    return (fig,)


@app.cell
def _(TOP, datetime, fig, mo, np, numba, os, platform, results, subprocess):
    def git(*args):
        return subprocess.run(['git', *args], cwd=TOP, capture_output=True, text=True).stdout.strip()

    _meta = {
        'commit'      : git('rev-parse', '--short', 'HEAD'),
        'dirty'       : git('status', '--porcelain') != '',
        'date'        : datetime.datetime.now().isoformat(timespec='seconds'),
        'cpu'         : next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')),
        'omp_threads' : os.environ.get('OMP_NUM_THREADS', ''),
        'python'      : platform.python_version(),
        'numpy'       : np.__version__,
        'numba'       : numba.__version__,
    }
    _out = TOP / 'data/compare_solvers' / f"compare_solvers_{_meta['commit']}.csv"
    _out.parent.mkdir(parents=True, exist_ok=True)
    results.assign(**_meta).to_csv(_out, index=False, float_format='%.17g')
    fig.savefig(_out.with_suffix('.png'), dpi=200, facecolor=fig.get_facecolor())
    print(f'Saved {_out} and {_out.with_suffix(".png")}')
    mo.md(f'Saved `{_out}` and the plot next to it.')
    return


if __name__ == "__main__":
    app.run()
