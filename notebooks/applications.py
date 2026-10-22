import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from numba import njit
    import marimo as mo

    from scipy.optimize import linprog

    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from lagrange import multiplicative_lagrange, penalty_qubo, qubo_to_ising
    from mkp import BENCH, load_mkp, mkp_to_qubo, mkp_optimum, load_best, mkp_rule

    return (
        BENCH,
        linprog,
        load_best,
        load_mkp,
        mkp_optimum,
        mkp_rule,
        mkp_to_qubo,
        mo,
        multiplicative_lagrange,
        np,
        penalty_qubo,
        plt,
        qubo_to_ising,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-Dimensional Knapsack (MKP)

    One profit per item, but multiple constraints e.g. price and volume.
    """)
    return


@app.cell
def _(np):
    p = np.array([10, 7, 8, 4, 6, 5])   # profit per item
    W = np.array([[3, 4, 5, 2, 3, 1],   # resource 1 e.g. price
                  [5, 2, 4, 3, 1, 4]])  # resource 2 e.g. volume
    C = np.array([10, 9])               # Capacities for price/volume

    # Brute force
    N = len(p)
    X = (np.arange(2**N)[:, None] >> np.arange(N)[::-1]) & 1    # binary digits (64, 6) i.e. all possible states

    profit = X @ p
    used = X @ W.T
    feasible = np.all(used <= C, axis=1)
    best = np.argmax(np.where(feasible, profit, -1))

    print(f'Best items    : {X[best]}')
    print(f'Resources used: {used[best]}')
    print(f'Profit        : {profit[best]}')
    return C, N, W, X, best, p


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inequalities

    MKP has $Wx \leq C$. One variable per resource:

    $\sum_j w_{ij} x_j + y_i = C_i, \qquad y_i \ge 0$

    With C = [10, 9], each resource needs floor(log2 C) + 1 = 4 bits.

    $$\begin{aligned}
    3x_0 + 4x_1 + 5x_2 + 2x_3 + 3x_4 + 1x_5 \;+\; (1y_{1,0} + 2y_{1,1} + 4y_{1,2} + 8y_{1,3}) &= 10 \\
    5x_0 + 2x_1 + 4x_2 + 3x_3 + 1x_4 + 4x_5 \;+\; (1y_{2,0} + 2y_{2,1} + 4y_{2,2} + 8y_{2,3}) &= 9
    \end{aligned}$$

    The above is A for the constraint $Ax = b$
    """)
    return


@app.cell
def _(C, W, np):
    n_bits = [int(np.log2(c)) + 1 for c in C]
    S = np.zeros((len(C), sum(n_bits)))
    _start = 0
    for _i, _k in enumerate(n_bits):
        S[_i, _start:_start + _k] = 2.0 ** np.arange(_k)
        _start += _k
    A = np.hstack([W, S])
    b = C.astype(float)

    print('A (resources and slack):')
    for _i in range(A.shape[0]):
        print(A[_i, :])

    print(f'\nb: {b}')
    return A, b


@app.cell
def _(A, C, W, X, b, best, np):
    x_best = X[best]
    slack = C - W @ x_best
    y = (slack[:, None] >> np.arange(4)) & 1
    z_best = np.concatenate([x_best, y.ravel()])

    print(f'Slack at best: {slack}')
    print(A @ z_best, b)
    return (z_best,)


@app.cell
def _(A, N, b, np, p, penalty_qubo, z_best):
    Q_obj = np.zeros((A.shape[1], A.shape[1]))
    Q_obj[:N, :N] = np.diag(-p)
    Q_pen = penalty_qubo(A, b)

    print(z_best @ Q_obj @ z_best)          # Energy
    print(z_best @ Q_pen @ z_best + b @ b)  # Constraints (0 = met)
    return Q_obj, Q_pen


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Solver
    """)
    return


@app.cell
def _(
    A,
    C,
    N,
    Q_obj,
    Q_pen,
    W,
    b,
    multiplicative_lagrange,
    np,
    p,
    qubo_to_ising,
):
    J_obj, h_obj, _ = qubo_to_ising(Q_obj)
    J_pen, h_pen, _ = qubo_to_ising(Q_pen)
    n = A.shape[1]
    print(f'All couplings from the penalty? {np.all(J_obj == 0)}')   ## all coupling come from the penalty

    states, lams = multiplicative_lagrange(
        J_obj,
        h_obj,
        J_pen,
        h_pen,
        A,
        b,
        P=0.5,
        eta=0.5,
        n_iter=50,
        init_size=int(0.8 * n),
        end_size=1,
        seed=0
    )

    z = (1 + states) / 2                            # spins -> 0/1, shape (50 runs, 14)
    x = z[:, :N]                                    # items only, slack thrown away
    run_profit = x @ p
    run_feasible = np.all(x @ W.T <= C, axis=1)
    return lams, run_feasible, run_profit


@app.cell
def _(lams, np, plt, run_feasible, run_profit):
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(run_profit, 'k.-')
    ax[0].plot(np.where(~run_feasible)[0], run_profit[~run_feasible], 'r.', label='infeasible')
    ax[0].axhline(23, ls='--', c='gray', label='optimum')
    ax[0].set_ylabel('profit')
    ax[0].legend()
    ax[1].plot(lams, '.-')                          # one line per resource
    ax[1].legend(['$\\lambda_1$ price', '$\\lambda_2$ volume'])
    ax[1].set_xlabel('run')
    ax[1].set_ylabel('$\\lambda$')
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Real Benchmarks and optima
    """)
    return


@app.cell
def _(BENCH, load_mkp, mkp_optimum, mo):
    from multiprocessing import Pool

    mo.stop((BENCH / 'best_values.txt').exists(), mo.md('`best_values.txt` exists, delete it to recompute'))

    _files = ['mknap1'] + [f'mknapcb{i}' for i in range(1, 10)]
    _jobs = [(name, k) for name in _files for k in range(7 if name == 'mknap1' else 30)]           # 277 instances
    _fits = ['mknap1', 'mknapcb1', 'mknapcb4']                                                    # near 256 spins: solve to the end
    _args = [(*load_mkp(name, k), None if name in _fits else 300) for name, k in _jobs]           # p, W, C, time limit [s]

    with Pool(20) as _pool:
        _results = _pool.starmap(mkp_optimum, _args)                                              # one instance per core

    with open(BENCH / 'best_values.txt', 'w') as _f:
        _f.write('# name k value gap. scipy.optimize.milp (HiGHS); gap 0 = proven optimal, else stopped after 300 s\n')
        for (_name, _k), (_value, _gap) in zip(_jobs, _results):
            _f.write(f'{_name} {_k} {_value:.10g} {_gap:.2g}\n')
    return


@app.cell
def _(
    linprog,
    load_best,
    load_mkp,
    mkp_rule,
    mkp_to_qubo,
    multiplicative_lagrange,
    np,
    penalty_qubo,
    plt,
    qubo_to_ising,
):
    _p, _W, _C = load_mkp('mknapcb1', 0)
    _Q_obj, _A, _b = mkp_to_qubo(_p, _W, _C)
    _J_obj, _h_obj, _ = qubo_to_ising(_Q_obj)
    _J_pen, _h_pen, _ = qubo_to_ising(penalty_qubo(_A, _b))
    _n, _N = _A.shape[1], len(_p)
    _P, _eta_paper = mkp_rule(_p, _A, _b)
    _lp_price = -linprog(-_p, A_ub=_W, b_ub=_C, bounds=(0, 1)).ineqlin.marginals   # profit per extra unit of capacity, items fractional
    _best = load_best('mknapcb1', 0)

    fig_real, ax = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
    for _col, (_label, _eta) in enumerate([('paper η', _eta_paper), ('η = 2P', 2 * _P)]):
        _states, _lams = multiplicative_lagrange(_J_obj, _h_obj, _J_pen, _h_pen, _A, _b, P=_P, eta=_eta, n_iter=200,
                                                    init_size=int(0.8 * _n), end_size=int(0.0625 * _n), seed=0)
        _x = (1 + _states[:, :_N]) / 2                  # items only
        _profit = _x @ _p
        _feasible = np.all(_x @ _W.T <= _C, axis=1)
        ax[0, _col].plot(_profit, 'k.', ms=3)
        ax[0, _col].plot(np.where(~_feasible)[0], _profit[~_feasible], 'r.', ms=3, label='infeasible')
        ax[0, _col].axhline(_best, ls='--', c='gray', label='optimum')
        ax[0, _col].set_title(f'{_label}: {_feasible.sum()} feasible, best {_profit[_feasible].max(initial=0):.0f}')
        for _i, _price in enumerate(_lp_price):
            ax[1, _col].plot(_lams[:, _i], c=f'C{_i}', lw=1)
            ax[1, _col].axhline(_price, c=f'C{_i}', ls='--', lw=1)
        ax[1, _col].set_xlabel('run')
    ax[0, 0].set_ylabel('profit')
    ax[0, 0].legend()
    ax[1, 0].set_ylabel('$\\lambda$ (dashed: LP price)')
    fig_real
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
