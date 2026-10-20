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

    return Path, mo, np, pd


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
    stats
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
