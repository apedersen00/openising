import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Event Solver Speed-up

    Time per multiplicative iteration (one analog run) on `jeu_100_25_1` for three solvers that should give the same result:

    - **basic**: Forward Euler in numpy
    - **njit**: the same Forward Euler compiled with numba
    - **event**: the njitted event-based solver from `scripts/lagrange.py`

    All three get the same start voltages and the same kicks.
    """)
    return


@app.cell
def _():
    import os
    os.environ['OMP_NUM_THREADS'] = '1'             # before numpy: threaded BLAS makes a 110x110 matvec 2-3x slower

    import sys
    import time

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from numba import njit
    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir().parent / 'scripts'))
    from lagrange import cluster_size, knapsack_to_qubo, load_qkp, max_field, paper_rule, qubo_to_ising, solver_event

    return (
        cluster_size,
        knapsack_to_qubo,
        load_qkp,
        max_field,
        mo,
        njit,
        np,
        paper_rule,
        pd,
        plt,
        qubo_to_ising,
        solver_event,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem

    QKP with a squared penalty, P from the paper rule (Delacour 2025), no Lagrange term.
    The Euler step is dt = 1 / max_field.
    """)
    return


@app.cell
def _(cluster_size, knapsack_to_qubo, load_qkp, max_field, np, paper_rule, qubo_to_ising):
    NAME = 'jeu_100_25_1'
    N_ITER = 50                                     # multiplicative iterations (analog runs)
    SEED = 0

    w, P, C = load_qkp(NAME)
    Q_obj, Q_pen, _, a = knapsack_to_qubo(w, P, C)
    J_obj, h_obj, _ = qubo_to_ising(Q_obj)
    J_pen, h_pen, _ = qubo_to_ising(Q_pen)
    P_pen, _ = paper_rule(P, a, C)
    J = J_obj + P_pen * J_pen
    h = h_obj + P_pen * h_pen
    n = len(h)
    dt = 1 / max_field(J, h)

    # Same start voltages and kicks for every solver (as in multiplicative_event)
    rng = np.random.default_rng(SEED)
    v0 = rng.uniform(-1, 1, n)
    sizes = np.array([cluster_size(it, N_ITER, int(0.4 * n), int(0.0625 * n)) for it in range(N_ITER)])
    clusters = np.zeros((N_ITER, n), dtype=np.int64)
    for _it in range(N_ITER):
        clusters[_it, :sizes[_it]] = rng.choice(n, size=sizes[_it], replace=False)
    return J, N_ITER, clusters, dt, h, sizes, v0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward Euler Solvers

    `analog_fe` runs a fixed number of Euler steps. `analog_fe_njit` is the same function compiled with numba.
    The outer loop is the same as `solver_event`.
    """)
    return


@app.cell
def _(njit, np):
    def analog_fe(J, h, v, dt, n_steps):
        for _ in range(n_steps):
            v = np.clip(v + dt * (J @ np.sign(v) + h), -1.0, 1.0)
        return np.where(v >= 0, 1.0, -1.0)             # a spin at 0 counts as +1, like in analog_event

    analog_fe_njit = njit(analog_fe)

    def multiplicative_fe(analog, J, h, v, sizes, clusters, dt, n_steps):
        n_iter = len(sizes)
        history = np.empty((n_iter, 2))                 # energy, best energy
        states = np.zeros((n_iter, len(h)))             # end state of every run
        best_sigma, best_E = None, np.inf
        for it in range(n_iter):
            sigma = analog(J, h, v, dt, n_steps)
            states[it] = sigma
            E = -0.5 * sigma @ J @ sigma - h @ sigma
            if E < best_E:
                best_sigma, best_E = sigma.copy(), E
            history[it] = E, best_E
            v = best_sigma.copy()
            v[clusters[it, :sizes[it]]] *= -1
        return best_sigma, best_E, history, states

    return analog_fe, analog_fe_njit, multiplicative_fe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Number of Euler Steps

    The event solver returns the settle time T of every run, so T / dt is the number of Euler steps that run needs.
    The njit solver is then swept over the number of steps to find where its runs end in the same states as the event solver.
    """)
    return


@app.cell
def _(J, N_ITER, analog_fe_njit, clusters, dt, h, multiplicative_fe, np, pd, sizes, solver_event, v0):
    _, _, hist_event, states_event = solver_event(J, h, v0.copy(), N_ITER, sizes, clusters, 10**6)
    steps_needed = hist_event[:, 2] / dt
    print(f'Euler steps to settle: median {np.median(steps_needed):.0f}, max {steps_needed.max():.0f}')

    def sweep_steps(step_counts):
        rows = []
        for n_steps in step_counts:
            *_, states = multiplicative_fe(analog_fe_njit, J, h, v0, sizes, clusters, dt, n_steps)
            rows.append(dict(n_steps=n_steps, runs_matching=(states == states_event).all(axis=1).sum()))
        return pd.DataFrame(rows)

    sweep = sweep_steps([1_000, 2_000, 4_000, 8_000, 16_000])
    sweep
    return (states_event,)


@app.cell
def _():
    N_STEPS = 8_000                                 # from the sweep above: all runs match from 8000 steps
    return (N_STEPS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time per Iteration
    """)
    return


@app.cell
def _(
    J,
    N_ITER,
    N_STEPS,
    analog_fe,
    analog_fe_njit,
    clusters,
    dt,
    h,
    multiplicative_fe,
    np,
    pd,
    sizes,
    solver_event,
    states_event,
    time,
    v0,
):
    solvers = {
        'basic' : lambda: multiplicative_fe(analog_fe, J, h, v0, sizes, clusters, dt, N_STEPS),
        'njit'  : lambda: multiplicative_fe(analog_fe_njit, J, h, v0, sizes, clusters, dt, N_STEPS),
        'event' : lambda: solver_event(J, h, v0.copy(), N_ITER, sizes, clusters, 10**6),
    }
    REPEATS = {'basic': 3, 'njit': 5, 'event': 50}  # min over repeats drops the numba compile of the first call

    def time_per_iteration(solve, repeats):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = solve()
            times.append(time.perf_counter() - t0)
        return min(times) / N_ITER, result

    rows = []
    for name, solve in solvers.items():
        t, (_, best_E, _, states) = time_per_iteration(solve, REPEATS[name])
        rows.append(dict(solver=name, s_per_iter=t, best_E=best_E,
                         runs_matching_event=(states == states_event).all(axis=1).sum()))
    timing = pd.DataFrame(rows).set_index('solver')
    timing['speedup'] = timing.s_per_iter['basic'] / timing.s_per_iter
    timing
    return (timing,)


@app.cell
def _(N_STEPS, np, plt, timing):
    def plot_speedup(timing):
        INK, INK_2, MUTED, GRID, AXIS, BAR, SURFACE = (
            '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#2a78d6', '#fcfcfb')
        s = timing.drop('basic').sort_values('speedup')
        y = np.arange(len(s))

        fig, ax = plt.subplots(figsize=(7, 2.2), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.barh(y, s.speedup - 1, left=1, height=0.4, color=BAR)   # bars grow from the 1x line
        ax.axvline(1, color=AXIS, linewidth=1)

        ax.set_xscale('log')
        ticks = [10.0**k for k in range(int(np.log10(s.speedup.max())) + 2)]
        ax.set_xlim(1, ticks[-1])
        ax.set_xticks(ticks, labels=[f'{t:g}×' for t in ticks])
        ax.minorticks_off()
        ax.grid(axis='x', color=GRID, linewidth=1)
        ax.set_axisbelow(True)

        for yi, value in zip(y, s.speedup):
            ax.annotate(f'{value:,.0f}×' if value >= 10 else f'{value:.1f}×', (value, yi), xytext=(4, 0), textcoords='offset points',
                        ha='left', va='center', color=INK, fontsize=9)

        ax.set_yticks(y, labels=s.index)
        ax.tick_params(axis='y', length=0, colors=INK_2)
        ax.tick_params(axis='x', length=0, colors=MUTED)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS)
        ax.set_title(f'Speed-up over basic Forward Euler (time per iteration, {N_STEPS:,} Euler steps)',
                     loc='left', color=INK, fontsize=11)
        fig.tight_layout()
        return fig

    plot_speedup(timing)
    return


if __name__ == "__main__":
    app.run()
