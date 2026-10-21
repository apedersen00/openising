import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo

    return Path, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(Path, pd):
    RAW      = Path(__file__).parent.parent / 'data' / 'qkp_sweep'
    BUDGETS  = [50, 200]
    settings = pd.read_csv(RAW / 'settings.csv', index_col='setting')
    names    = sorted(p.stem for p in (RAW / 'n200').glob('*.npz'))
    return BUDGETS, RAW, names, settings


@app.cell
def _(RAW, np, pd, settings):
    def first_run(gap, tol):
        """
        Median over the seeds of the first run within tol of the optimum (inf: under half got there).
        """
        hit = gap <= tol
        return np.median(np.where(hit.any(axis=2), hit.argmax(axis=2) + 1, np.inf), axis=1)


    def reduce(budget, name):
        """
        One row per setting: how the 200 seeds did on one instance with one budget.
        """
        d = np.load(RAW / f'n{budget}' / f'{name}.npz')
        gap   = d['gap']                                                    # (setting, seed, run)
        best  = gap.min(axis=2)                                             # best gap of each seed
        return pd.DataFrame({
            'budget'       : budget,
            'instance'     : name,
            'setting'      : settings.index,
            'P_used'       : d['P'],                                        # actual P and η on this instance
            'eta_used'     : d['eta'],
            'gap'          : best.mean(axis=1),                             # mean best gap over the seeds
            'solved'       : (best == 0).mean(axis=1),                      # share of seeds that found the optimum
            'infeasible'   : (best == 1).mean(axis=1),                      # share of seeds that never found a feasible solution
            'within_1pct'  : (best <= 0.01).mean(axis=1),                   # share of seeds within 1%
            'within_0.1pct': (best <= 0.001).mean(axis=1),                  # share of seeds within 0.1%
            'run_1pct'     : first_run(gap, 0.01),                          # median first run within 1%
            'run_0.1pct'   : first_run(gap, 0.001),                         # median first run within 0.1%
            'lambda_end'   : np.median(d['lam'][:, :, -1], axis=1),         # median λ in the last run
        })

    return (reduce,)


@app.cell
def _(BUDGETS, mo, names, pd, reduce, settings):
    _tasks = [(budget, name) for budget in BUDGETS for name in names]
    stats = pd.concat([reduce(b, n) for b, n in mo.status.progress_bar(_tasks, title='Reducing')], ignore_index=True)
    stats = stats.join(settings, on='setting')
    stats['size']    = stats.instance.str.split('_').str[1].astype(int)   # jeu_<size>_<density>_<number>
    stats['density'] = stats.instance.str.split('_').str[2].astype(int)
    stats
    return (stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plain Lagrange

    What are good values of $P$ and $r = n/P$?
    """)
    return


@app.cell
def _(BUDGETS, np, plt, stats):
    plain = stats[(stats['update'] == 'plain') & stats.r.notna()]   # η = rP

    _n50  = plain[plain.budget == 50].groupby(['P_factor', 'r']).gap.mean().unstack()
    _n200 = plain[plain.budget == 200].groupby(['P_factor', 'r']).gap.mean().unstack()

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))

    for _a, _t, _budget in zip(_ax, [_n50, _n200], BUDGETS):
        _im = _a.imshow(_t, norm='log', vmin=1e-3, vmax=1, cmap='Blues', origin='lower')

        for (_i, _j), _g in np.ndenumerate(_t.to_numpy()):
            _a.text(_j, _i, f'{100 * _g:.3g}', ha='center', va='center', fontsize=7,
                    color='white' if _g > 0.03 else 'black')

        _a.set_xticks(range(len(_t.columns)), [f'{x:g}' for x in _t.columns], rotation=45)
        _a.set_yticks(range(len(_t.index)), _t.index)
        _a.set_xlabel('r')
        _a.set_ylabel('P_factor')
        _a.set_title(f'{_budget} runs')

    print('Average gap to optimum between the best seed of all benchmarks')
    print('Best values of P and r transfer well between benchmarks')
    _fig
    return (plain,)


@app.cell
def _(BUDGETS, np, plain, plt):
    _n50  = plain[plain.budget == 50].groupby(['P_factor', 'r'])['within_0.1pct'].mean().unstack()
    _n200 = plain[plain.budget == 200].groupby(['P_factor', 'r'])['within_0.1pct'].mean().unstack()

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))
    for _a, _t, _budget in zip(_ax, [_n50, _n200], BUDGETS):
        _a.imshow(_t, vmin=0, vmax=1, cmap='Blues', origin='lower')
        for (_i, _j), _g in np.ndenumerate(_t.to_numpy()):
            _a.text(_j, _i, f'{100 * _g:.0f}', ha='center', va='center', fontsize=7,
                    color='white' if _g > 0.5 else 'black')
        _a.set_xticks(range(len(_t.columns)), [f'{x:g}' for x in _t.columns], rotation=45)
        _a.set_yticks(range(len(_t.index)), _t.index)
        _a.set_xlabel('r')
        _a.set_ylabel('P_factor')
        _a.set_title(f'{_budget} runs: % of seeds within 0.1%')
    _fig
    return


@app.cell
def _(BUDGETS, np, plt, stats):
    penalty = stats[stats.eta == 0].copy()                                    # penalty only: η = 0 keeps λ at 0
    penalty['size']    = penalty.instance.str.split('_').str[1].astype(int)   # jeu_<size>_<density>_<number>
    penalty['density'] = penalty.instance.str.split('_').str[2].astype(int)

    _n50  = penalty[penalty.budget == 50].groupby(['size', 'density', 'P_factor']).gap.mean().unstack()
    _n200 = penalty[penalty.budget == 200].groupby(['size', 'density', 'P_factor']).gap.mean().unstack()

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for _a, _t, _budget in zip(_ax, [_n50, _n200], BUDGETS):
        _a.imshow(_t, norm='log', vmin=1e-3, vmax=1, cmap='Blues', origin='lower', aspect='auto')
        for (_i, _j), _g in np.ndenumerate(_t.to_numpy()):
            _a.text(_j, _i, f'{100 * _g:.3g}', ha='center', va='center', fontsize=7, color='white' if _g > 0.03 else 'black',
                    fontweight='bold' if _g == _t.to_numpy()[_i].min() else 'normal')
        _a.set_xticks(range(len(_t.columns)), _t.columns)
        _a.set_yticks(range(len(_t.index)), [f'{s} items, {d}%' for s, d in _t.index])
        _a.set_xlabel('P_factor')
        _a.set_title(f'Penalty only, {_budget} runs')

    print('Penalty method requires adjustment of penalty weight between benchmarks.')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No benchmark vary much from the average
    """)
    return


@app.cell
def _(BUDGETS, np, plain, plt):
    _n50  = plain[(plain.budget == 50)  & (plain.r == 2)].groupby(['size', 'density', 'P_factor']).gap.mean().unstack()
    _n200 = plain[(plain.budget == 200) & (plain.r == 2)].groupby(['size', 'density', 'P_factor']).gap.mean().unstack()

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for _a, _t, _budget in zip(_ax, [_n50, _n200], BUDGETS):
        _a.imshow(_t, norm='log', vmin=1e-3, vmax=1, cmap='Blues', origin='lower', aspect='auto')
        for (_i, _j), _g in np.ndenumerate(_t.to_numpy()):
            _a.text(_j, _i, f'{100 * _g:.3g}', ha='center', va='center', fontsize=7, color='white' if _g > 0.03 else 'black')
        _a.set_xticks(range(len(_t.columns)), _t.columns)
        _a.set_yticks(range(len(_t.index)), [f'{s} items, {d}%' for s, d in _t.index])
        _a.set_xlabel('P_factor')
        _a.set_title(f'Plain Lagrange, r = 2, {_budget} runs')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Momemtum
    """)
    return


@app.cell
def _(BUDGETS, plt, stats):
    _p16 = stats[stats.r.notna() & (stats.P_factor == 16) & stats['update'].isin(['plain', 'momentum'])]

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout='constrained')
    for _a, _budget in zip(_ax, BUDGETS):
        _t = _p16[_p16.budget == _budget].groupby(['update', 'beta1', 'r']).gap.mean().unstack(['update', 'beta1'])
        (100 * _t).plot(ax=_a, logx=True, logy=True, marker='o')
        _a.set_xlabel('r')
        _a.set_title(f'P_factor = 16, {_budget} runs')
    _ax[0].set_ylabel('mean gap [%]')
    _fig
    return


@app.cell
def _(pd, plain, stats):
    def _best_per_instance(group):
        return group.loc[group.groupby(['budget', 'instance']).gap.idxmin()]

    _rows = {
        'penalty, P×1024'              : stats[(stats.eta == 0) & (stats.P_factor == 1024)],
        'penalty, best P per instance' : _best_per_instance(stats[stats.eta == 0]),
        'paper text (1, 1)'            : stats[stats.n_factor == 1],
        'paper code (4, 4)'            : stats[stats.n_factor == 4],
        'plain, P16 r2'                : plain[(plain.P_factor == 16) & (plain.r == 2)],
        'plain, best per instance'     : _best_per_instance(plain),
        'momentum β.875, P16 r4'       : stats[(stats['update'] == 'momentum') & (stats.beta1 == 0.875) & (stats.P_factor == 16) & (stats.r == 4)],
    }
    _cols = ['gap', 'infeasible', 'within_1pct', 'within_0.1pct', 'solved']
    summary = pd.concat({_name: _g.groupby('budget')[_cols].mean() for _name, _g in _rows.items()})
    (100 * summary).round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convergence
    """)
    return


@app.cell
def _(BUDGETS, RAW, names, np, plt, settings):
    _pick = {
        'penalty, Px1024'        : 'eta == 0 and P_factor == 1024',
        'plain, P16 r2'          : 'update == "plain" and P_factor == 16 and r == 2',
        'momentum β.875, P16 r4' : 'update == "momentum" and beta1 == 0.875 and P_factor == 16 and r == 4',
    }
    _ids = [settings.query(_q).index[0] for _q in _pick.values()]


    def _best_so_far(budget):
        """
        Mean best-so-far gap per run over seeds and instances: (setting, run).
        """
        return np.mean([np.minimum.accumulate(np.load(RAW / f'n{budget}' / f'{name}.npz')['gap'][_ids], axis=2).mean(axis=1)
                        for name in names], axis=0)


    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout='constrained')
    for _a, _budget in zip(_ax, BUDGETS):
        for _label, _curve in zip(_pick, _best_so_far(_budget)):
            _a.plot(np.arange(1, _budget + 1), 100 * _curve, label=_label)
        _a.set_yscale('log')
        _a.set_xlabel('run')
        _a.set_title(f'{_budget} runs')
    _ax[0].set_ylabel('mean best-so-far gap [%]')
    _ax[0].legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lambda value

    $\lambda$ always ends up near best_profit / C.

    Perhaps a penalty method run could be used to then initialize $\lambda$ to that value?
    """)
    return


@app.cell
def _(Path, names, pd, plain, plt):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from lagrange import load_optimum, load_qkp

    price = pd.Series({_name: load_optimum(_name) / load_qkp(_name)[2] for _name in names})   # best profit per unit of capacity

    _s = plain[(plain.budget == 200) & (plain.r == 2)]

    _fig, _ax = plt.subplots(figsize=(5, 5), layout='constrained')
    for _P, _g in _s.groupby('P_factor'):
        _ax.scatter(_g.instance.map(price), _g.lambda_end, s=10, label=f'P_factor {_P}')
    _ax.axline((10, 10), (100, 100), color='gray', ls='--')     # λ = best / C
    _ax.set_xscale('log')
    _ax.set_yscale('log')
    _ax.set_xlabel('best profit / C')
    _ax.set_ylabel('median λ in the last run')
    _ax.legend()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
